"""
core/session.py

Key design decisions in this version:

TRACE ZONES
-----------
Every aim position is recorded regardless of whether it's on-target,
as long as it's within the "approach zone" (2× the scoring radius).
This gives you the full picture of the shot approach.

Each TracePoint carries an `on_target` flag so the renderer can colour
approach points differently from hold points.

    approach_radius_mm = scoring_radius_mm * 2.0
    on_target_radius_mm = scoring_radius_mm (outer ring)

FALSE-POSITIVE REJECTION
------------------------
A shot is only registered if aim_mm is within valid_shot_radius_mm
(same as approach_radius_mm = 2× scoring radius). Any audio trigger
while the camera is pointing well outside the target is discarded.

LIVE SESSION FILE
-----------------
When start_series() is called, a CSV file is opened immediately.
Every shot is appended as a line the moment it fires — no data is
lost even if the app crashes. The file is named:
    session_YYYYMMDD_HHMMSS_<name>.csv

FADING TRACE FIX
----------------
fading_trace_display_start is set at the moment get_fading_trace()
is first called after a shot, not when record_shot() runs. This means
the 2-second window starts from when the UI actually sees it.
"""

import os
import csv
import json
import time
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


# ── Zone constants ────────────────────────────────────────────────────────────
APPROACH_ZONE_FACTOR = 2.0    # approach zone = scoring_radius * this
DEFAULT_SCORING_RADIUS_MM = 22.75   # 10m ISSF outer ring radius

# Trace point zone labels
ZONE_APPROACH  = "approach"   # outside scoring rings but within approach zone
ZONE_ON_TARGET = "on_target"  # inside scoring rings


@dataclass
class TracePoint:
    timestamp: float
    aim_mm: Tuple[float, float]
    zone: str = ZONE_ON_TARGET   # ZONE_APPROACH or ZONE_ON_TARGET


@dataclass
class ShotTrace:
    """Complete trace for one shot — approach + hold + fire."""
    points: List[TracePoint] = field(default_factory=list)
    fired_time: Optional[float] = None
    state: str = "active"
    # Pre-computed colours — one entry per point, updated at append time.
    # Avoids recalculating colour_for_point() for every point on every frame.
    cached_colours: List[Tuple[int,int,int]] = field(default_factory=list)
    # Renderer params used for the cache — stored so we can detect changes
    _cache_params: Optional[tuple] = field(default=None, repr=False)

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def on_target_points(self) -> List[TracePoint]:
        return [p for p in self.points if p.zone == ZONE_ON_TARGET]

    @property
    def approach_points(self) -> List[TracePoint]:
        return [p for p in self.points if p.zone == ZONE_APPROACH]

    def on_target_duration_s(self) -> float:
        pts = self.on_target_points
        if len(pts) < 2:
            return 0.0
        return pts[-1].timestamp - pts[0].timestamp

    def total_duration_s(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return self.points[-1].timestamp - self.points[0].timestamp

    def aim_centrepoint(self, fraction: float = 0.40) -> Optional[Tuple[float, float]]:
        """
        Mean aim position over the last `fraction` of ON-TARGET points only.
        This is the most meaningful measure of where you were holding.
        """
        pts = self.on_target_points
        if not pts:
            pts = self.points      # fallback if no on-target points
        if not pts:
            return None
        n = max(1, int(len(pts) * fraction))
        pts = pts[-n:]
        return (float(np.mean([p.aim_mm[0] for p in pts])),
                float(np.mean([p.aim_mm[1] for p in pts])))

    def recompute_colours(
        self,
        col_approach, col_hold, col_preshot, col_final,
        preshot_s: float = 1.0,
        final_s: float = 0.2,
        tail_only: bool = False,
    ):
        """
        (Re)compute cached_colours for all or just the tail of the trace.
        Call with tail_only=True after fired_time is set — only the last
        few seconds of points change colour on firing, the rest stay the same.
        """
        params = (col_approach, col_hold, col_preshot, col_final,
                  preshot_s, final_s)
        n = len(self.points)
        if not self.cached_colours or len(self.cached_colours) != n:
            tail_only = False  # full recompute if lengths mismatch

        if tail_only and self.fired_time is not None:
            # Only recompute points within preshot_s + 0.5s of fired_time
            recompute_from = 0
            for i in range(n - 1, -1, -1):
                if self.fired_time - self.points[i].timestamp > preshot_s + 0.5:
                    recompute_from = i
                    break
            indices = range(recompute_from, n)
        else:
            self.cached_colours = [None] * n
            indices = range(n)

        for i in indices:
            self.cached_colours[i] = self.colour_for_point(
                i, col_approach, col_hold, col_preshot, col_final,
                preshot_s, final_s)
        self._cache_params = params

    def colour_for_point(
        self, idx: int,
        col_approach: Tuple[int,int,int] = (60, 60, 60),
        col_hold:     Tuple[int,int,int] = (80, 190, 40),
        col_preshot:  Tuple[int,int,int] = (0, 208, 240),
        col_final:    Tuple[int,int,int] = (32, 48, 227),
        preshot_s: float = 1.0,
        final_s:   float = 0.2,
    ) -> Tuple[int, int, int]:
        """
        BGR colour per trace point. Colour params come from the renderer
        (which reads them from user config) so they're fully customisable.

        Approach zone : col_approach (default dark grey)
        Hold > 1.0s   : col_hold (default green), brightness ramps up
        Hold 1.0–0.2s : lerp col_hold → col_preshot (default yellow)
        Hold < 0.2s   : lerp col_preshot → col_final (default red)
        """
        if idx >= len(self.points):
            return col_hold

        pt = self.points[idx]

        if pt.zone == ZONE_APPROACH:
            n = len(self.points)
            a = 0.3 + 0.5 * (idx / max(n - 1, 1))
            return tuple(int(c * a) for c in col_approach)

        if self.fired_time is None:
            n = max(len(self.on_target_points), 1)
            ot_idx = len([p for p in self.points[:idx+1]
                          if p.zone == ZONE_ON_TARGET])
            a = 0.25 + 0.75 * (ot_idx + 1) / n
            return tuple(int(c * a) for c in col_hold)

        t_before = max(0.0, self.fired_time - pt.timestamp)

        def _lerp(c1, c2, t):
            return tuple(int(a + t * (b - a)) for a, b in zip(c1, c2))

        if t_before > preshot_s:
            total = self.fired_time - (self.on_target_points[0].timestamp
                                       if self.on_target_points
                                       else self.points[0].timestamp)
            age = (pt.timestamp - self.points[0].timestamp) / max(total, 0.001)
            a = 0.25 + 0.75 * age
            return tuple(int(c * a) for c in col_hold)
        elif t_before > final_s:
            band = preshot_s - final_s
            t = (preshot_s - t_before) / band if band > 0 else 1.0
            return _lerp(col_hold, col_preshot, t)
        else:
            t = (final_s - t_before) / final_s if final_s > 0 else 1.0
            return _lerp(col_preshot, col_final, t)


@dataclass
class Shot:
    index: int
    timestamp: float
    aim_mm: Tuple[float, float]
    score: float
    ring_index: int
    series: int = 1
    trace: Optional[ShotTrace] = None
    aim_centrepoint: Optional[Tuple[float, float]] = None
    # Shot flags (matching Scatt data model)
    match_shot: bool = True    # counts toward result (vs sighter)
    deleted: bool = False      # soft-deleted / false positive
    favourite: bool = False    # marked for review
    missed: bool = False       # missed the target entirely
    comments: str = ""
    mark_index: int = 0    # which aiming mark (0 for single-mark targets)

    @property
    def radius_mm(self) -> float:
        return math.sqrt(self.aim_mm[0] ** 2 + self.aim_mm[1] ** 2)

    @property
    def on_target_duration_s(self) -> float:
        return self.trace.on_target_duration_s() if self.trace else 0.0

    @property
    def clock_position(self) -> int:
        """Clock position (1-12) where 12 is top, 3 is right, etc."""
        x, y = self.aim_mm
        angle = math.degrees(math.atan2(-y, x))  # Invert y for correct orientation
        clock_angle = (90 - angle) % 360
        clock_hour = int(round(clock_angle / 30)) % 12
        return 12 if clock_hour == 0 else clock_hour


# ── Live CSV writer ───────────────────────────────────────────────────────────

class LiveSessionWriter:
    """
    Opens a CSV file at series-start and appends each shot immediately.
    Safe against crashes — every shot is flushed to disk as it's recorded.
    """
    HEADER = ["Shot", "Series", "Timestamp", "X_mm", "Y_mm", "Radius_mm",
              "Score", "OnTarget_s", "ApproachTotal_s",
              "ACP_X", "ACP_Y", "TracePoints"]

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._f = open(path, "w", newline="", buffering=1)  # line-buffered
        self._w = csv.writer(self._f)
        self._w.writerow(self.HEADER)
        self._f.flush()

    def write_shot(self, shot: Shot):
        acp = shot.aim_centrepoint
        trace_pts = ""
        if shot.trace:
            # Compact encoding: "t:x:y:zone|t:x:y:zone|..."
            trace_pts = "|".join(
                f"{p.timestamp:.3f}:{p.aim_mm[0]:.2f}:{p.aim_mm[1]:.2f}:{p.zone[0]}"
                for p in shot.trace.points
            )
        self._w.writerow([
            shot.index, shot.series,
            f"{shot.timestamp:.3f}",
            f"{shot.aim_mm[0]:.3f}", f"{shot.aim_mm[1]:.3f}",
            f"{shot.radius_mm:.3f}",
            shot.score,
            f"{shot.on_target_duration_s:.3f}",
            f"{shot.trace.total_duration_s():.3f}" if shot.trace else "0",
            f"{acp[0]:.3f}" if acp else "",
            f"{acp[1]:.3f}" if acp else "",
            trace_pts,
        ])
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

    @property
    def is_open(self) -> bool:
        return not self._f.closed


# ── Session ───────────────────────────────────────────────────────────────────

class Session:
    """
    Manages trace recording, shot registration, and live file I/O.

    Zones:
        approach_radius_mm  = scoring_radius_mm * APPROACH_ZONE_FACTOR
        on_target_radius_mm = scoring_radius_mm

    A shot is rejected (returns None) if aim_mm is outside approach_radius_mm.
    """

    def __init__(
        self,
        name: str = "Session",
        shots_per_series: int = 10,
        scoring_radius_mm: float = DEFAULT_SCORING_RADIUS_MM,
    ):
        self.name = name
        self.shots_per_series = shots_per_series
        self.scoring_radius_mm = scoring_radius_mm
        self.fading_trace_duration_s = 2.0   # overridden by app from cfg
        self.acp_fraction = 0.40              # overridden by app from cfg
        self.approach_radius_mm = scoring_radius_mm * APPROACH_ZONE_FACTOR
        self.on_target_radius_mm = scoring_radius_mm

        self.current_series = 1
        self.shots: List[Shot] = []
        self.start_time = time.time()
        self._shot_counter = 0

        # Live trace state
        self.active_trace: ShotTrace = ShotTrace()
        self._in_approach_zone: bool = False

        # Post-shot fading trace
        self.fading_trace: Optional[ShotTrace] = None
        self._fading_first_seen: float = 0.0   # when UI first saw it
        self._fading_pending: bool = False       # True = new fading, not yet displayed

        # Live file writer (None until start_series() called)
        self._writer: Optional[LiveSessionWriter] = None
        self.series_file_path: Optional[str] = None

    # ── Series file management ────────────────────────────────────────────────

    def start_series(self, save_dir: str) -> str:
        """
        Open a new live CSV file for this series.
        Files are organised into daily subfolders: save_dir/YYYY-MM-DD/
        Returns the file path.
        """
        if self._writer and self._writer.is_open:
            self._writer.close()
        # Daily subfolder
        day_folder = time.strftime("%Y-%m-%d")
        day_dir = os.path.join(save_dir, day_folder)
        os.makedirs(day_dir, exist_ok=True)
        # File name: HH-MM-SS_name_seriesN  (date already in folder)
        ts = time.strftime("%H-%M-%S")
        safe_name = self.name.replace(" ", "_")
        fname = f"{ts}_{safe_name}_series{self.current_series}.csv"
        path = os.path.join(day_dir, fname)
        self._writer = LiveSessionWriter(path)
        self.series_file_path = path
        self.start_time = time.time()
        return path

    def end_series(self):
        """Close the live writer and write a JSON archive alongside."""
        if self._writer:
            self._writer.close()
            self._writer = None
        # Write companion JSON so the history viewer can load rich data
        if self.series_file_path and self.shots:
            try:
                json_path = self.series_file_path.replace(".csv", ".json")
                self.save_json(json_path)
            except Exception as e:
                print(f"[Session] JSON archive write failed: {e}")
        self.series_file_path = None

    @property
    def series_active(self) -> bool:
        return self._writer is not None and self._writer.is_open

    # ── Trace / aim management ────────────────────────────────────────────────

    def update_aim(self, aim_mm: Tuple[float, float]) -> Tuple[bool, bool]:
        """
        Feed the current zeroed aim position.
        Records to the active trace (approach + on-target).
        Returns (in_approach_zone, on_target).
        """
        radius = math.sqrt(aim_mm[0] ** 2 + aim_mm[1] ** 2)
        in_approach = radius <= self.approach_radius_mm
        on_target   = radius <= self.on_target_radius_mm

        if in_approach:
            if not self._in_approach_zone:
                # Entered approach zone — start fresh trace for this shot
                self.active_trace = ShotTrace()
                self._in_approach_zone = True

            zone = ZONE_ON_TARGET if on_target else ZONE_APPROACH
            self.active_trace.points.append(
                TracePoint(timestamp=time.time(), aim_mm=aim_mm, zone=zone))
            # Pre-compute colour for new point using cached renderer params.
            # Falls back to a default if params not yet set (first frame).
            cp = self.active_trace._cache_params
            if cp:
                col = self.active_trace.colour_for_point(
                    len(self.active_trace.points) - 1, *cp[:4],
                    preshot_s=cp[4], final_s=cp[5])
            else:
                col = (80, 190, 40)   # default green until params arrive
            self.active_trace.cached_colours.append(col)
        else:
            if self._in_approach_zone:
                self._in_approach_zone = False
            # Outside approach zone — don't record, but don't reset trace either
            # (preserves partial approach if tracking briefly loses the markers)

        return in_approach, on_target

    def record_shot(
        self,
        aim_mm: Tuple[float, float],
        score: float,
        ring_index: int,
        shot_timestamp: Optional[float] = None,
        mark_index: int = 0,
        defer_write: bool = False,
    ) -> Optional[Shot]:
        """
        Register a shot.

        If shot_timestamp is provided (the exact time.time() when the audio
        callback fired), the aim position is looked up retroactively from the
        trace — finding the trace point whose timestamp is closest to
        shot_timestamp, then linearly interpolating between the two nearest
        points for sub-frame accuracy.  This eliminates software lag between
        audio detection and camera frame capture.

        Falls back to the supplied aim_mm if the trace has no points yet.
        Returns None (rejected) if aim is outside the approach zone.
        """
        # ── Retroactive position lookup ──────────────────────────────────────
        if shot_timestamp is not None and len(self.active_trace.points) >= 2:
            pts = self.active_trace.points
            times = [p.timestamp for p in pts]
            # Find the two bracketing points
            # Clamp to trace range (shot may be right at the end)
            t = max(times[0], min(times[-1], shot_timestamp))
            # Find insertion index
            lo, hi = 0, len(times) - 1
            for i in range(len(times) - 1):
                if times[i] <= t <= times[i + 1]:
                    lo, hi = i, i + 1
                    break
            p0, p1 = pts[lo], pts[hi]
            dt = times[hi] - times[lo]
            if dt > 0:
                frac = (t - times[lo]) / dt
                x = p0.aim_mm[0] + frac * (p1.aim_mm[0] - p0.aim_mm[0])
                y = p0.aim_mm[1] + frac * (p1.aim_mm[1] - p0.aim_mm[1])
                aim_mm = (x, y)
            else:
                aim_mm = p0.aim_mm
        elif shot_timestamp is not None and len(self.active_trace.points) == 1:
            aim_mm = self.active_trace.points[0].aim_mm

        radius = math.sqrt(aim_mm[0] ** 2 + aim_mm[1] ** 2)
        if radius > self.approach_radius_mm:
            return None   # reject — camera not on or near target

        self._shot_counter += 1
        trace = self.active_trace
        trace.fired_time = shot_timestamp or time.time()
        trace.state = "fired"
        # Recompute only the tail of the cached colours — the preshot/final
        # colour zones only affect the last ~1-2 seconds before the shot.
        # The bulk of the trace (hold zone) is already correctly cached.
        if trace._cache_params:
            cp = trace._cache_params
            trace.recompute_colours(*cp[:4], preshot_s=cp[4], final_s=cp[5],
                                    tail_only=True)

        shot = Shot(
            index=self._shot_counter,
            timestamp=shot_timestamp or time.time(),
            aim_mm=aim_mm,
            score=score,
            ring_index=ring_index,
            series=self.current_series,
            trace=trace,
            aim_centrepoint=trace.aim_centrepoint(self.acp_fraction),
            mark_index=mark_index,
        )
        self.shots.append(shot)

        # Write to live file immediately (unless caller wants to rescore first)
        if not defer_write and self._writer and self._writer.is_open:
            try:
                self._writer.write_shot(shot)
            except Exception as e:
                print(f"[LiveWriter] {e}")

        # Set up fading trace — mark as pending, timer starts on first display
        self.fading_trace = trace
        self._fading_pending = True
        self._fading_first_seen = 0.0

        # Fresh trace for the next shot
        self.active_trace = ShotTrace()
        self._in_approach_zone = False
        return shot

    def get_fading_trace(self) -> Optional[ShotTrace]:
        """
        Return the post-shot fading trace, or None if expired.
        The 2-second timer starts from the FIRST call after a shot —
        i.e. from when the UI actually renders it, not when it was recorded.
        """
        if self.fading_trace is None:
            return None

        now = time.time()
        if self._fading_pending:
            # First time the UI is asking — start the clock now
            self._fading_first_seen = now
            self._fading_pending = False

        age = now - self._fading_first_seen
        fade_dur = getattr(self, "fading_trace_duration_s", 2.0)
        if age > fade_dur:
            self.fading_trace = None
            return None
        return self.fading_trace

    @property
    def fading_age_s(self) -> float:
        """Current age of the fading trace in seconds (0 if not active)."""
        if self.fading_trace is None or self._fading_first_seen == 0.0:
            return 0.0
        return time.time() - self._fading_first_seen

    def undo_last_shot(self) -> Optional[Shot]:
        if self.shots:
            s = self.shots.pop()
            self._shot_counter -= 1
            self.fading_trace = None
            self._fading_pending = False
            return s
        return None

    def clear_series(self):
        self.end_series()
        self.current_series += 1
        self.active_trace = ShotTrace()
        self.fading_trace = None
        self._in_approach_zone = False

    def reset(self):
        self.end_series()
        self.shots = []
        self.current_series = 1
        self._shot_counter = 0
        self.start_time = time.time()
        self.active_trace = ShotTrace()
        self.fading_trace = None
        self._in_approach_zone = False


    # ── Statistics ────────────────────────────────────────────────────────────

    @property
    def total_score(self) -> float:
        return sum(s.score for s in self.shots if not s.deleted)

    @property
    def shot_count(self) -> int:
        return len([s for s in self.shots if not s.deleted])

    @property
    def match_shots(self) -> List[Shot]:
        return [s for s in self.shots if s.match_shot and not s.deleted]

    @property
    def series_shots(self) -> List[Shot]:
        return [s for s in self.shots
                if s.series == self.current_series and not s.deleted]

    @property
    def series_match_shots(self) -> List[Shot]:
        return [s for s in self.series_shots if s.match_shot]

    @property
    def series_score(self) -> float:
        return sum(s.score for s in self.series_match_shots)

    @property
    def series_avg(self) -> Optional[float]:
        ss = self.series_match_shots
        return sum(s.score for s in ss) / len(ss) if ss else None

    def _scored_coords(self) -> Optional[np.ndarray]:
        ss = self.series_match_shots
        if not ss:
            return None
        return np.array([s.aim_mm for s in ss], dtype=float)

    @property
    def mean_radius_mm(self) -> Optional[float]:
        """MR — Mean Radius."""
        c = self._scored_coords()
        if c is None:
            return None
        return float(np.mean(np.sqrt(c[:,0]**2 + c[:,1]**2)))

    @property
    def extreme_spread_mm(self) -> Optional[float]:
        """ES — Extreme Spread: largest distance between any two shots."""
        c = self._scored_coords()
        if c is None or len(c) < 2:
            return None
        return float(max(
            np.linalg.norm(c[i] - c[j])
            for i in range(len(c)) for j in range(i+1, len(c))
        ))

    @property
    def figure_of_merit_mm(self) -> Optional[float]:
        """FOM — (ES_x + ES_y) / 2."""
        c = self._scored_coords()
        if c is None or len(c) < 2:
            return None
        return (float(np.max(c[:,0]) - np.min(c[:,0])) +
                float(np.max(c[:,1]) - np.min(c[:,1]))) / 2.0

    @property
    def std_x_mm(self) -> Optional[float]:
        c = self._scored_coords()
        return float(np.std(c[:,0])) if c is not None and len(c) > 1 else None

    @property
    def std_y_mm(self) -> Optional[float]:
        c = self._scored_coords()
        return float(np.std(c[:,1])) if c is not None and len(c) > 1 else None

    @property
    def cep_mm(self) -> Optional[float]:
        """CEP — radius containing 50% of shots."""
        c = self._scored_coords()
        if c is None or len(c) < 2:
            return None
        return float(np.percentile(np.sqrt(c[:,0]**2 + c[:,1]**2), 50))

    @property
    def mean_point_of_impact(self) -> Optional[Tuple[float, float]]:
        c = self._scored_coords()
        if c is None:
            return None
        return (float(np.mean(c[:,0])), float(np.mean(c[:,1])))

    @property
    def group_size_mm(self) -> Optional[float]:
        return self.extreme_spread_mm

    @property
    def best_shot(self) -> Optional[Shot]:
        ss = self.series_match_shots
        return max(ss, key=lambda s: s.score) if ss else None

    @property
    def worst_shot(self) -> Optional[Shot]:
        ss = self.series_match_shots
        return min(ss, key=lambda s: s.score) if ss else None

    @property
    def bbox_shots_mm(self) -> Optional[Tuple[float, float]]:
        """W x H bounding box of match shots."""
        c = self._scored_coords()
        if c is None or len(c) < 2:
            return None
        return (float(np.max(c[:,0]) - np.min(c[:,0])),
                float(np.max(c[:,1]) - np.min(c[:,1])))

    @property
    def avg_on_target_s(self) -> Optional[float]:
        vals = [s.on_target_duration_s for s in self.shots
                if s.on_target_duration_s > 0 and not s.deleted]
        return float(np.mean(vals)) if vals else None

    @property
    def duration_s(self) -> float:
        return time.time() - self.start_time

    # ── Full JSON save (end of session) ──────────────────────────────────────

    def save_json(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": time.time(),
            "duration_s": self.duration_s,
            "shots": [
                {
                    "index": s.index,
                    "series": s.series,
                    "timestamp": s.timestamp,
                    "aim_mm": list(s.aim_mm),
                    "score": s.score,
                    "ring_index": s.ring_index,
                    "on_target_s": s.on_target_duration_s,
                    "aim_centrepoint": list(s.aim_centrepoint) if s.aim_centrepoint else None,
                    "trace": [
                        {"t": p.timestamp, "x": p.aim_mm[0],
                         "y": p.aim_mm[1], "z": p.zone}
                        for p in (s.trace.points if s.trace else [])
                    ],
                }
                for s in self.shots
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def save_csv(self, path: str):
        """Summary CSV (one row per shot, no trace points)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Shot", "Series", "X_mm", "Y_mm", "Radius_mm",
                        "Score", "OnTarget_s", "ACP_X", "ACP_Y", "Timestamp"])
            for s in self.shots:
                acp = s.aim_centrepoint
                w.writerow([
                    s.index, s.series,
                    f"{s.aim_mm[0]:.3f}", f"{s.aim_mm[1]:.3f}",
                    f"{s.radius_mm:.3f}", s.score,
                    f"{s.on_target_duration_s:.3f}",
                    f"{acp[0]:.3f}" if acp else "",
                    f"{acp[1]:.3f}" if acp else "",
                    f"{s.timestamp:.3f}",
                ])

    def summary_dict(self) -> dict:
        return {
            "name": self.name,
            "shots": self.shot_count,
            "total_score": self.total_score,
            "avg_score": round(self.total_score / self.shot_count, 2) if self.shot_count else 0,
            "mean_radius_mm": self.mean_radius_mm,
            "group_size_mm": self.group_size_mm,
            "mean_poi": self.mean_point_of_impact,
            "duration_s": self.duration_s,
            "avg_on_target_s": self.avg_on_target_s,
        }


# ── History helpers ───────────────────────────────────────────────────────────

def _load_session_file(path: str, fname: str, day_label: str) -> Optional[dict]:
    """Load a single .json or .csv session file. Returns a dict or None."""
    base = fname.rsplit(".", 1)[0]

    if fname.endswith(".json"):
        try:
            with open(path) as f:
                data = json.load(f)
            shots  = data.get("shots", [])
            scores = [s["score"] for s in shots]
            ot     = [s["on_target_s"] for s in shots if s.get("on_target_s", 0) > 0]
            return {
                "filename": fname, "path": path, "base": base,
                "day": day_label,
                "name": data.get("name", fname),
                "date": time.strftime("%H:%M",
                          time.localtime(data.get("start_time", 0))),
                "shot_count": len(shots),
                "total_score": round(sum(scores), 1),
                "avg_score": round(sum(scores)/len(scores), 2) if scores else 0,
                "duration_s": round(data.get("duration_s", 0)),
                "avg_on_target_s": round(sum(ot)/len(ot), 2) if ot else 0,
                "raw": data, "source": "json",
            }
        except Exception:
            return None

    elif fname.endswith(".csv"):
        try:
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                return None
            scores, ot_vals = [], []
            for r in rows:
                try:
                    sc = float(r.get("Score", 0))
                    if sc > 0: scores.append(sc)
                    ot = float(r.get("OnTarget_s", 0))
                    if ot > 0: ot_vals.append(ot)
                except (ValueError, TypeError):
                    pass
            # Time from filename.
            # New format:  HH-MM-SS_name_seriesN.csv  → parts[0] = HH-MM-SS
            # Old format:  session_YYYYMMDD_HHMMSS_name_seriesN.csv
            time_str = "—"
            try:
                parts = fname.split("_")
                if parts[0] == "session" and len(parts) >= 3:
                    # Old format — extract HHMMSS from parts[2]
                    h = parts[2][:2]; m = parts[2][2:4]; s = parts[2][4:6]
                    time_str = f"{h}:{m}:{s}"
                else:
                    time_str = parts[0].replace("-", ":")   # HH:MM:SS
            except Exception:
                pass
            name = fname.replace(".csv", "")
            return {
                "filename": fname, "path": path, "base": base,
                "day": day_label,
                "name": name, "date": time_str,
                "shot_count": len(rows),
                "total_score": round(sum(scores), 1),
                "avg_score": round(sum(scores)/len(scores), 2) if scores else 0,
                "duration_s": 0,
                "avg_on_target_s": round(sum(ot_vals)/len(ot_vals), 2) if ot_vals else 0,
                "raw": {"shots": [
                    {"index": r.get("Shot",""), "score": float(r.get("Score",0)),
                     "aim_mm": [float(r.get("X_mm",0)), float(r.get("Y_mm",0))],
                     "on_target_s": float(r.get("OnTarget_s",0)),
                     "aim_centrepoint": (
                         [float(r.get("ACP_X",0)), float(r.get("ACP_Y",0))]
                         if r.get("ACP_X") else None),
                     "series": int(r.get("Series",1)),
                     "timestamp": float(r.get("Timestamp", 0)),
                     "trace": [],
                    } for r in rows
                ]},
                "source": "csv",
            }
        except Exception:
            return None
    return None


def load_session_history(save_dir: str) -> dict:
    """
    Load all saved sessions from save_dir, organised by day.

    Walks save_dir recursively:
      - YYYY-MM-DD/ subfolders → day-organised sessions (new format)
      - files directly in save_dir → legacy flat sessions

    Returns a dict:
        {
            "YYYY-MM-DD": [session_dict, ...],   # newest day first
            ...
        }
    Each session_dict has: filename, path, base, day, name, date,
    shot_count, total_score, avg_score, duration_s, raw, source.

    Deduplicates: .json takes priority over .csv with the same base name.
    """
    if not os.path.isdir(save_dir):
        return {}

    # Collect all files grouped by day label
    day_files: dict = {}   # day_label → {base: entry}

    def _scan_dir(dirpath: str, day_label: str):
        seen_bases = set()
        entries = {}
        # Pass 1: JSON files (preferred)
        for fname in sorted(os.listdir(dirpath)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(dirpath, fname)
            if not os.path.isfile(path):
                continue
            e = _load_session_file(path, fname, day_label)
            if e:
                entries[e["base"]] = e
                seen_bases.add(e["base"])
        # Pass 2: CSV files (only if no matching JSON)
        for fname in sorted(os.listdir(dirpath)):
            if not fname.endswith(".csv"):
                continue
            path = os.path.join(dirpath, fname)
            if not os.path.isfile(path):
                continue
            base = fname.rsplit(".", 1)[0]
            if base in seen_bases:
                continue
            e = _load_session_file(path, fname, day_label)
            if e:
                entries[base] = e
        return entries

    # Scan day subfolders (YYYY-MM-DD pattern)
    for entry in os.listdir(save_dir):
        full = os.path.join(save_dir, entry)
        if os.path.isdir(full):
            # Accept any folder as a day label
            day_files[entry] = _scan_dir(full, entry)

    # Scan legacy flat files directly in save_dir
    flat = _scan_dir(save_dir, "Legacy")
    if flat:
        day_files.setdefault("Legacy", {}).update(flat)

    # Build ordered result: newest day first, entries newest first within day
    result = {}
    for day in sorted(day_files.keys(), reverse=True):
        entries = sorted(day_files[day].values(),
                         key=lambda e: e["filename"], reverse=True)
        if entries:
            result[day] = entries

    return result


def reconstruct_shot_traces(session_data: dict) -> List[ShotTrace]:
    """Rebuild ShotTrace objects from saved JSON for the history viewer."""
    traces = []
    for s in session_data.get("shots", []):
        tr = ShotTrace()
        for pt in s.get("trace", []):
            zone = pt.get("z", ZONE_ON_TARGET)
            tr.points.append(TracePoint(
                timestamp=pt["t"],
                aim_mm=(pt["x"], pt["y"]),
                zone=zone,
            ))
        tr.fired_time = s.get("timestamp")
        tr.state = "fired"
        traces.append(tr)
    return traces
