"""Shot data, trace recording and per-series file I/O.

A :class:`Session` owns the active aim trace, the list of recorded
:class:`Shot` objects, the live CSV writer, and statistics (mean radius,
extreme spread, CEP, MPI). Shots fired while the aim is well outside
the target are rejected as false positives. Shots fired retroactively
against a known timestamp are interpolated against the trace history so
audio-detection latency does not skew the recorded position.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


APPROACH_ZONE_FACTOR = 2.0
"""Approach zone radius = scoring radius × this factor."""

DEFAULT_SCORING_RADIUS_MM = 22.75
"""Outer ring radius for the 10 m ISSF target."""

ZONE_APPROACH = "approach"
ZONE_ON_TARGET = "on_target"


@dataclass
class TracePoint:
    """A single sampled aim position with a zone classification."""
    timestamp: float
    aim_mm: Tuple[float, float]
    zone: str = ZONE_ON_TARGET


@dataclass
class ShotTrace:
    """The full pre-shot trace plus the firing event for one shot."""

    points: List[TracePoint] = field(default_factory=list)
    fired_time: Optional[float] = None
    state: str = "active"
    # Pre-computed BGR colours (one per point) refreshed by the renderer.
    cached_colours: List[Tuple[int, int, int]] = field(default_factory=list)
    # Renderer params last used to populate ``cached_colours``.
    _cache_params: Optional[tuple] = field(default=None, repr=False)

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

    def aim_centrepoint(
        self, fraction: float = 0.40,
    ) -> Optional[Tuple[float, float]]:
        """Mean aim position over the final ``fraction`` of on-target points.

        Falls back to the full trace when no on-target points exist.
        """
        pts = self.on_target_points or self.points
        if not pts:
            return None
        n = max(1, int(len(pts) * fraction))
        tail = pts[-n:]
        return (
            float(np.mean([p.aim_mm[0] for p in tail])),
            float(np.mean([p.aim_mm[1] for p in tail])),
        )

    def recompute_colours(
        self,
        col_approach, col_hold, col_preshot, col_final,
        preshot_s: float = 1.0,
        final_s: float = 0.2,
        tail_only: bool = False,
    ) -> None:
        """Refresh ``cached_colours`` for all or just the trailing points."""
        params = (col_approach, col_hold, col_preshot, col_final,
                  preshot_s, final_s)
        n = len(self.points)
        if not self.cached_colours or len(self.cached_colours) != n:
            tail_only = False

        if tail_only and self.fired_time is not None:
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
        col_approach: Tuple[int, int, int] = (60, 60, 60),
        col_hold: Tuple[int, int, int] = (80, 190, 40),
        col_preshot: Tuple[int, int, int] = (0, 208, 240),
        col_final: Tuple[int, int, int] = (32, 48, 227),
        preshot_s: float = 1.0,
        final_s: float = 0.2,
    ) -> Tuple[int, int, int]:
        """BGR colour for a trace point based on zone and time-to-fire.

        Approach points are dimmed grey. Hold points fade from dim to
        full ``col_hold``. Once the shot has fired, points within
        ``preshot_s`` of fire are interpolated towards ``col_preshot``,
        and points within ``final_s`` are interpolated towards
        ``col_final``.
        """
        if idx >= len(self.points):
            return col_hold

        pt = self.points[idx]

        if pt.zone == ZONE_APPROACH:
            n = len(self.points)
            alpha = 0.3 + 0.5 * (idx / max(n - 1, 1))
            return tuple(int(c * alpha) for c in col_approach)

        if self.fired_time is None:
            on_target_total = max(len(self.on_target_points), 1)
            ot_idx = sum(1 for p in self.points[:idx + 1]
                         if p.zone == ZONE_ON_TARGET)
            alpha = 0.25 + 0.75 * (ot_idx + 1) / on_target_total
            return tuple(int(c * alpha) for c in col_hold)

        t_before = max(0.0, self.fired_time - pt.timestamp)

        def lerp(c1, c2, t):
            return tuple(int(a + t * (b - a)) for a, b in zip(c1, c2))

        if t_before > preshot_s:
            anchor = (self.on_target_points[0].timestamp
                      if self.on_target_points else self.points[0].timestamp)
            total = self.fired_time - anchor
            age = (pt.timestamp - self.points[0].timestamp) / max(total, 0.001)
            alpha = 0.25 + 0.75 * age
            return tuple(int(c * alpha) for c in col_hold)
        if t_before > final_s:
            band = preshot_s - final_s
            t = (preshot_s - t_before) / band if band > 0 else 1.0
            return lerp(col_hold, col_preshot, t)
        t = (final_s - t_before) / final_s if final_s > 0 else 1.0
        return lerp(col_preshot, col_final, t)


@dataclass
class Shot:
    """One recorded shot with optional trace and review flags."""

    index: int
    timestamp: float
    aim_mm: Tuple[float, float]
    score: float
    ring_index: int
    series: int = 1
    trace: Optional[ShotTrace] = None
    aim_centrepoint: Optional[Tuple[float, float]] = None
    match_shot: bool = True
    deleted: bool = False
    favourite: bool = False
    missed: bool = False
    comments: str = ""
    mark_index: int = 0

    @property
    def radius_mm(self) -> float:
        return math.hypot(self.aim_mm[0], self.aim_mm[1])

    @property
    def on_target_duration_s(self) -> float:
        return self.trace.on_target_duration_s() if self.trace else 0.0


class LiveSessionWriter:
    """CSV writer that flushes every shot immediately.

    Opening one of these at series start guarantees that all completed
    shots survive even if the app crashes mid-series.
    """

    HEADER = [
        "Shot", "Series", "Timestamp", "X_mm", "Y_mm", "Radius_mm",
        "Score", "OnTarget_s", "ApproachTotal_s",
        "ACP_X", "ACP_Y", "TracePoints",
    ]

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._f = open(path, "w", newline="", buffering=1)
        self._w = csv.writer(self._f)
        self._w.writerow(self.HEADER)
        self._f.flush()

    def write_shot(self, shot: Shot) -> None:
        acp = shot.aim_centrepoint
        trace_pts = ""
        if shot.trace:
            trace_pts = "|".join(
                f"{p.timestamp:.3f}:{p.aim_mm[0]:.2f}:"
                f"{p.aim_mm[1]:.2f}:{p.zone[0]}"
                for p in shot.trace.points
            )
        approach_total = (
            f"{shot.trace.total_duration_s():.3f}" if shot.trace else "0")
        self._w.writerow([
            shot.index, shot.series,
            f"{shot.timestamp:.3f}",
            f"{shot.aim_mm[0]:.3f}", f"{shot.aim_mm[1]:.3f}",
            f"{shot.radius_mm:.3f}",
            shot.score,
            f"{shot.on_target_duration_s:.3f}",
            approach_total,
            f"{acp[0]:.3f}" if acp else "",
            f"{acp[1]:.3f}" if acp else "",
            trace_pts,
        ])
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass

    @property
    def is_open(self) -> bool:
        return not self._f.closed


def _interpolate_aim(
    points: List[TracePoint], target_time: float,
) -> Tuple[float, float]:
    """Aim position at ``target_time`` interpolated between trace points."""
    if len(points) == 1:
        return points[0].aim_mm

    times = [p.timestamp for p in points]
    t = max(times[0], min(times[-1], target_time))

    lo, hi = 0, len(times) - 1
    for i in range(len(times) - 1):
        if times[i] <= t <= times[i + 1]:
            lo, hi = i, i + 1
            break

    p0, p1 = points[lo], points[hi]
    dt = times[hi] - times[lo]
    if dt <= 0:
        return p0.aim_mm
    frac = (t - times[lo]) / dt
    return (
        p0.aim_mm[0] + frac * (p1.aim_mm[0] - p0.aim_mm[0]),
        p0.aim_mm[1] + frac * (p1.aim_mm[1] - p0.aim_mm[1]),
    )


class Session:
    """Aim trace recording, shot registration, statistics and live I/O.

    Args:
        name: Session name used in saved file names.
        shots_per_series: Default series length.
        scoring_radius_mm: ``card_radius + calibre_radius`` in mm. Drives
            the on-target and approach zones used to classify trace
            points and reject false positives.
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
        self.fading_trace_duration_s = 2.0
        self.acp_fraction = 0.40
        self.approach_radius_mm = scoring_radius_mm * APPROACH_ZONE_FACTOR
        self.on_target_radius_mm = scoring_radius_mm

        self.current_series = 1
        self.shots: List[Shot] = []
        self.start_time = time.time()
        self._shot_counter = 0

        self.active_trace: ShotTrace = ShotTrace()
        self._in_approach_zone: bool = False

        self.fading_trace: Optional[ShotTrace] = None
        self._fading_first_seen: float = 0.0
        self._fading_pending: bool = False

        self._writer: Optional[LiveSessionWriter] = None
        self.series_file_path: Optional[str] = None

    def start_series(self, save_dir: str) -> str:
        """Open a new live CSV file for the current series.

        Files are organised into per-day subfolders
        (``YYYY-MM-DD/HH-MM-SS_<name>_seriesN.csv``).
        """
        if self._writer and self._writer.is_open:
            self._writer.close()
        day_dir = os.path.join(save_dir, time.strftime("%Y-%m-%d"))
        os.makedirs(day_dir, exist_ok=True)
        ts = time.strftime("%H-%M-%S")
        safe_name = self.name.replace(" ", "_")
        path = os.path.join(
            day_dir, f"{ts}_{safe_name}_series{self.current_series}.csv")
        self._writer = LiveSessionWriter(path)
        self.series_file_path = path
        self.start_time = time.time()
        return path

    def end_series(self) -> None:
        """Close the live writer and persist a JSON archive alongside."""
        if self._writer:
            self._writer.close()
            self._writer = None
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

    def update_aim(
        self, aim_mm: Tuple[float, float],
    ) -> Tuple[bool, bool]:
        """Feed the latest zeroed aim position to the active trace.

        Returns ``(in_approach_zone, on_target)``.
        """
        radius = math.hypot(aim_mm[0], aim_mm[1])
        in_approach = radius <= self.approach_radius_mm
        on_target = radius <= self.on_target_radius_mm

        if not in_approach:
            self._in_approach_zone = False
            return in_approach, on_target

        if not self._in_approach_zone:
            self.active_trace = ShotTrace()
            self._in_approach_zone = True

        zone = ZONE_ON_TARGET if on_target else ZONE_APPROACH
        self.active_trace.points.append(
            TracePoint(timestamp=time.time(), aim_mm=aim_mm, zone=zone))

        cp = self.active_trace._cache_params
        if cp:
            colour = self.active_trace.colour_for_point(
                len(self.active_trace.points) - 1, *cp[:4],
                preshot_s=cp[4], final_s=cp[5])
        else:
            colour = (80, 190, 40)
        self.active_trace.cached_colours.append(colour)

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
        """Register a shot, returning ``None`` if the aim was off-target.

        When ``shot_timestamp`` is provided, the trace history is
        interpolated to find the aim at the exact audio-trigger moment.
        This eliminates the lag between audio detection and the next
        camera frame.
        """
        if shot_timestamp is not None and self.active_trace.points:
            aim_mm = _interpolate_aim(
                self.active_trace.points, shot_timestamp)

        radius = math.hypot(aim_mm[0], aim_mm[1])
        if radius > self.approach_radius_mm:
            return None

        self._shot_counter += 1
        trace = self.active_trace
        trace.fired_time = shot_timestamp or time.time()
        trace.state = "fired"
        if trace._cache_params:
            cp = trace._cache_params
            trace.recompute_colours(
                *cp[:4], preshot_s=cp[4], final_s=cp[5], tail_only=True)

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

        if not defer_write and self._writer and self._writer.is_open:
            try:
                self._writer.write_shot(shot)
            except Exception as e:
                print(f"[LiveWriter] {e}")

        self.fading_trace = trace
        self._fading_pending = True
        self._fading_first_seen = 0.0

        self.active_trace = ShotTrace()
        self._in_approach_zone = False
        return shot

    def get_fading_trace(self) -> Optional[ShotTrace]:
        """Post-shot fading trace, or ``None`` once the fade window ends.

        The fade timer starts on the first call after a shot is recorded,
        so the window measures wall-clock UI display time rather than
        the shot timestamp.
        """
        if self.fading_trace is None:
            return None

        now = time.time()
        if self._fading_pending:
            self._fading_first_seen = now
            self._fading_pending = False

        age = now - self._fading_first_seen
        fade_duration = getattr(self, "fading_trace_duration_s", 2.0)
        if age > fade_duration:
            self.fading_trace = None
            return None
        return self.fading_trace

    @property
    def fading_age_s(self) -> float:
        """Age of the fading trace in seconds, or ``0`` when inactive."""
        if self.fading_trace is None or self._fading_first_seen == 0.0:
            return 0.0
        return time.time() - self._fading_first_seen

    def undo_last_shot(self) -> Optional[Shot]:
        if not self.shots:
            return None
        shot = self.shots.pop()
        self._shot_counter -= 1
        self.fading_trace = None
        self._fading_pending = False
        return shot

    def clear_series(self) -> None:
        self.end_series()
        self.current_series += 1
        self.active_trace = ShotTrace()
        self.fading_trace = None
        self._in_approach_zone = False

    def reset(self) -> None:
        self.end_series()
        self.shots = []
        self.current_series = 1
        self._shot_counter = 0
        self.start_time = time.time()
        self.active_trace = ShotTrace()
        self.fading_trace = None
        self._in_approach_zone = False

    @property
    def total_score(self) -> float:
        return sum(s.score for s in self.shots if not s.deleted)

    @property
    def shot_count(self) -> int:
        return sum(1 for s in self.shots if not s.deleted)

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
        c = self._scored_coords()
        if c is None:
            return None
        return float(np.mean(np.sqrt(c[:, 0] ** 2 + c[:, 1] ** 2)))

    @property
    def extreme_spread_mm(self) -> Optional[float]:
        c = self._scored_coords()
        if c is None or len(c) < 2:
            return None
        return float(max(
            np.linalg.norm(c[i] - c[j])
            for i in range(len(c)) for j in range(i + 1, len(c))
        ))

    @property
    def figure_of_merit_mm(self) -> Optional[float]:
        c = self._scored_coords()
        if c is None or len(c) < 2:
            return None
        return (
            float(np.max(c[:, 0]) - np.min(c[:, 0]))
            + float(np.max(c[:, 1]) - np.min(c[:, 1]))
        ) / 2.0

    @property
    def std_x_mm(self) -> Optional[float]:
        c = self._scored_coords()
        return float(np.std(c[:, 0])) if c is not None and len(c) > 1 else None

    @property
    def std_y_mm(self) -> Optional[float]:
        c = self._scored_coords()
        return float(np.std(c[:, 1])) if c is not None and len(c) > 1 else None

    @property
    def cep_mm(self) -> Optional[float]:
        """Circular error probable: radius enclosing 50% of shots."""
        c = self._scored_coords()
        if c is None or len(c) < 2:
            return None
        return float(
            np.percentile(np.sqrt(c[:, 0] ** 2 + c[:, 1] ** 2), 50))

    @property
    def mean_point_of_impact(self) -> Optional[Tuple[float, float]]:
        c = self._scored_coords()
        if c is None:
            return None
        return float(np.mean(c[:, 0])), float(np.mean(c[:, 1]))

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
        """Width and height of the axis-aligned bounding box (mm)."""
        c = self._scored_coords()
        if c is None or len(c) < 2:
            return None
        return (
            float(np.max(c[:, 0]) - np.min(c[:, 0])),
            float(np.max(c[:, 1]) - np.min(c[:, 1])),
        )

    @property
    def avg_on_target_s(self) -> Optional[float]:
        values = [
            s.on_target_duration_s for s in self.shots
            if s.on_target_duration_s > 0 and not s.deleted
        ]
        return float(np.mean(values)) if values else None

    @property
    def duration_s(self) -> float:
        return time.time() - self.start_time

    def save_json(self, path: str) -> None:
        """Write a full session archive (including trace points) as JSON."""
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
                    "aim_centrepoint": (
                        list(s.aim_centrepoint) if s.aim_centrepoint else None),
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

    def save_csv(self, path: str) -> None:
        """Summary CSV with one row per shot and no trace points."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".",
                    exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "Shot", "Series", "X_mm", "Y_mm", "Radius_mm",
                "Score", "OnTarget_s", "ACP_X", "ACP_Y", "Timestamp",
            ])
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
            "avg_score": (
                round(self.total_score / self.shot_count, 2)
                if self.shot_count else 0
            ),
            "mean_radius_mm": self.mean_radius_mm,
            "group_size_mm": self.group_size_mm,
            "mean_poi": self.mean_point_of_impact,
            "duration_s": self.duration_s,
            "avg_on_target_s": self.avg_on_target_s,
        }


def _time_str_from_filename(fname: str) -> str:
    """Extract the time portion of either the new or legacy filename format.

    New: ``HH-MM-SS_name_seriesN.csv``
    Legacy: ``session_YYYYMMDD_HHMMSS_name_seriesN.csv``
    """
    try:
        parts = fname.split("_")
        if parts[0] == "session" and len(parts) >= 3:
            stamp = parts[2]
            return f"{stamp[:2]}:{stamp[2:4]}:{stamp[4:6]}"
        return parts[0].replace("-", ":")
    except Exception:
        return "—"


def _load_json_session(
    path: str, fname: str, base: str, day_label: str,
) -> Optional[dict]:
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return None

    shots = data.get("shots", [])
    scores = [s["score"] for s in shots]
    ot = [s["on_target_s"] for s in shots if s.get("on_target_s", 0) > 0]
    return {
        "filename": fname, "path": path, "base": base, "day": day_label,
        "name": data.get("name", fname),
        "date": time.strftime("%H:%M",
                              time.localtime(data.get("start_time", 0))),
        "shot_count": len(shots),
        "total_score": round(sum(scores), 1),
        "avg_score": round(sum(scores) / len(scores), 2) if scores else 0,
        "duration_s": round(data.get("duration_s", 0)),
        "avg_on_target_s": round(sum(ot) / len(ot), 2) if ot else 0,
        "raw": data, "source": "json",
    }


def _load_csv_session(
    path: str, fname: str, base: str, day_label: str,
) -> Optional[dict]:
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    if not rows:
        return None

    scores: list = []
    ot_vals: list = []
    for row in rows:
        try:
            score = float(row.get("Score", 0))
            if score > 0:
                scores.append(score)
            ot = float(row.get("OnTarget_s", 0))
            if ot > 0:
                ot_vals.append(ot)
        except (ValueError, TypeError):
            pass

    return {
        "filename": fname, "path": path, "base": base, "day": day_label,
        "name": fname.replace(".csv", ""),
        "date": _time_str_from_filename(fname),
        "shot_count": len(rows),
        "total_score": round(sum(scores), 1),
        "avg_score": round(sum(scores) / len(scores), 2) if scores else 0,
        "duration_s": 0,
        "avg_on_target_s": (
            round(sum(ot_vals) / len(ot_vals), 2) if ot_vals else 0),
        "raw": {"shots": [
            {
                "index": row.get("Shot", ""),
                "score": float(row.get("Score", 0)),
                "aim_mm": [float(row.get("X_mm", 0)),
                           float(row.get("Y_mm", 0))],
                "on_target_s": float(row.get("OnTarget_s", 0)),
                "aim_centrepoint": (
                    [float(row.get("ACP_X", 0)), float(row.get("ACP_Y", 0))]
                    if row.get("ACP_X") else None),
                "series": int(row.get("Series", 1)),
                "timestamp": float(row.get("Timestamp", 0)),
                "trace": [],
            }
            for row in rows
        ]},
        "source": "csv",
    }


def _load_session_file(
    path: str, fname: str, day_label: str,
) -> Optional[dict]:
    """Load a single session file (.json or .csv) into a dict."""
    base = fname.rsplit(".", 1)[0]
    if fname.endswith(".json"):
        return _load_json_session(path, fname, base, day_label)
    if fname.endswith(".csv"):
        return _load_csv_session(path, fname, base, day_label)
    return None


def _scan_session_dir(dirpath: str, day_label: str) -> dict:
    """Load all JSON sessions in ``dirpath`` and any CSV sessions whose
    base name has no JSON counterpart.
    """
    entries: dict = {}
    seen_bases: set = set()
    for fname in sorted(os.listdir(dirpath)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(dirpath, fname)
        if not os.path.isfile(path):
            continue
        entry = _load_session_file(path, fname, day_label)
        if entry:
            entries[entry["base"]] = entry
            seen_bases.add(entry["base"])
    for fname in sorted(os.listdir(dirpath)):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(dirpath, fname)
        if not os.path.isfile(path):
            continue
        base = fname.rsplit(".", 1)[0]
        if base in seen_bases:
            continue
        entry = _load_session_file(path, fname, day_label)
        if entry:
            entries[base] = entry
    return entries


def load_session_history(save_dir: str) -> dict:
    """Load saved sessions grouped by day.

    Walks ``save_dir`` recursively. Subdirectories are treated as days
    (``YYYY-MM-DD``); files directly in ``save_dir`` are grouped under
    ``"Legacy"``. Returns a mapping of day label to the list of session
    entries for that day, newest day first.
    """
    if not os.path.isdir(save_dir):
        return {}

    day_files: dict = {}
    for entry in os.listdir(save_dir):
        full = os.path.join(save_dir, entry)
        if os.path.isdir(full):
            day_files[entry] = _scan_session_dir(full, entry)

    flat = _scan_session_dir(save_dir, "Legacy")
    if flat:
        day_files.setdefault("Legacy", {}).update(flat)

    result: dict = {}
    for day in sorted(day_files.keys(), reverse=True):
        entries = sorted(
            day_files[day].values(),
            key=lambda e: e["filename"], reverse=True,
        )
        if entries:
            result[day] = entries
    return result


def reconstruct_shot_traces(session_data: dict) -> List[ShotTrace]:
    """Rebuild :class:`ShotTrace` objects from a saved JSON session."""
    traces: List[ShotTrace] = []
    for s in session_data.get("shots", []):
        trace = ShotTrace()
        for pt in s.get("trace", []):
            trace.points.append(TracePoint(
                timestamp=pt["t"],
                aim_mm=(pt["x"], pt["y"]),
                zone=pt.get("z", ZONE_ON_TARGET),
            ))
        trace.fired_time = s.get("timestamp")
        trace.state = "fired"
        traces.append(trace)
    return traces
