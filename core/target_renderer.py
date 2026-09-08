"""
core/target_renderer.py
Renders the target face with per-shot colour traces, aim centrepoints,
fading post-shot traces, and shot holes.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

from core.session import Shot, ShotTrace
from core.config import DEFAULT_CONFIG


# ── Colour palette (BGR) ──────────────────────────────────────────────────────
C_BG         = (22, 22, 28)
C_RING_OUTER = (180, 180, 180)
C_SHOT_RING  = (255, 255, 255)
C_MPI        = (255, 180, 80)
C_GROUP      = (255, 100, 100)

def _hex_to_bgr(h: str) -> Tuple[int, int, int]:
    """Convert #rrggbb hex string to OpenCV BGR tuple."""
    h = h.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return (b, g, r)

# Defaults (overridden by display_cfg passed to TargetRenderer)
C_SHOT_FILL  = (80,  80,  255)
C_CROSSHAIR  = (0,   220, 100)
C_MISS       = (60,  60,  200)
C_ACP        = (255, 200,   0)


class TargetRenderer:
    def __init__(self, canvas_size: Tuple[int, int], target_cfg: dict,
                 display_calibre_mm: float = None,
                 display_cfg: dict = None,
                 zoom: float = 1.0):
        self.cw, self.ch = canvas_size
        self.target_cfg = target_cfg
        self.calibre_mm = display_calibre_mm or target_cfg.get("calibre_mm", 4.5)
        self.zoom = max(0.1, float(zoom))
        dc = display_cfg or {}
        
        # Target appearance — defaults are defined in config.py.
        self.target_colours = {}
        for name in ("outer", "inner", "outer_lines", "inner_lines"):
            key = f"colour_target_{name}"
            value = dc.get(key, DEFAULT_CONFIG[key])

            try:
                self.target_colours[name] = _hex_to_bgr(value)
            except (AttributeError, TypeError, ValueError):
                self.target_colours[name] = _hex_to_bgr(DEFAULT_CONFIG[key])

        try:
            count = int(dc.get(
                "target_inner_rings",
                DEFAULT_CONFIG["target_inner_rings"],
            ))
        except (TypeError, ValueError):
            count = DEFAULT_CONFIG["target_inner_rings"]

        self.inner_ring_count = max(
            0, min(count, len(target_cfg["rings_mm"]))
        )
        
        try:
            count = int(dc.get(
                "target_score_rings",
                DEFAULT_CONFIG["target_score_rings"],
            ))
        except (TypeError, ValueError):
            count = DEFAULT_CONFIG["target_score_rings"]

        self.score_ring_count = max(
            0, min(count, len(target_cfg["rings_mm"]))
        )

        self.score_directions = []
        for name, offset in [
            ("top", (0, -1)),
            ("right", (1, 0)),
            ("bottom", (0, 1)),
            ("left", (-1, 0)),
        ]:
            key = f"target_score_{name}"
            enabled = dc.get(key, DEFAULT_CONFIG[key])
            if str(enabled).lower() == "true":
                self.score_directions.append(offset)

        # Colours from display config (fall back to hardcoded defaults)
        self.C_shot_fill = _hex_to_bgr(dc.get("colour_shot_fill",  "#5050ff"))
        self.C_acp       = _hex_to_bgr(dc.get("colour_acp",        "#ffc800"))
        self.C_crosshair = _hex_to_bgr(dc.get("colour_crosshair",  "#00dc64"))
        self.C_miss      = _hex_to_bgr(dc.get("colour_miss",       "#3c3cd0"))
        self.C_mpi       = _hex_to_bgr(dc.get("colour_mpi",        "#50b4ff"))
        self.C_group     = _hex_to_bgr(dc.get("colour_group",      "#6464ff"))
        self.trace_width = int(dc.get("trace_width", 1))
        # Store colour config so ShotTrace.colour_for_point can use it
        self.col_approach  = _hex_to_bgr(dc.get("colour_trace_approach", "#3c3c3c"))
        self.col_hold      = _hex_to_bgr(dc.get("colour_trace_hold",     "#28be50"))
        self.col_preshot   = _hex_to_bgr(dc.get("colour_trace_preshot",  "#f0d000"))
        self.col_final     = _hex_to_bgr(dc.get("colour_trace_final",    "#e03020"))
        self.trace_preshot_s = float(dc.get("trace_preshot_s", 1.0))
        self.trace_final_s   = float(dc.get("trace_final_s",   0.2))

        # For multi-mark targets, the effective visual diameter is larger than
        # the single card diameter — use the full span of all marks.
        target_dia = target_cfg["diameter_mm"]
        mark_offsets = target_cfg.get("mark_offsets")
        if mark_offsets:
            import math as _m
            max_reach = max(_m.sqrt(mx**2 + my**2) for mx, my in mark_offsets)
            # Full span = furthest mark centre + one card radius on each side
            target_dia = (max_reach + target_dia / 2) * 2
        usable = min(self.cw, self.ch) * 0.88 * self.zoom
        self.scale = usable / target_dia

        self.cx = self.cw // 2
        self.cy = self.ch // 2

        self._static = self._render_static()

    # ── Public API ────────────────────────────────────────────────────────────

    def render(
        self,
        shots: List[Shot],
        active_trace: Optional[ShotTrace] = None,
        fading_trace: Optional[ShotTrace] = None,
        fading_age_s: float = 0.0,
        live_aim_mm: Optional[Tuple[float, float]] = None,
        show_mpi: bool = True,
        show_group: bool = True,
        current_series: int = 1,
        zero_mode: bool = False,
        show_acp: bool = True,
        show_traces: bool = True,
        highlighted_shot_trace: Optional[ShotTrace] = None,
        show_bbox_shots: bool = False,
        show_bbox_acp: bool = False,
        show_dot_only: bool = False,   # True = tiny dot; False = full calibre circle
        trace_alpha: float = 0.30,      # alpha for past traces (1.0 = full in review)
    ) -> np.ndarray:
        canvas = self._static.copy()

        # 1. Past shot traces (dim) — drawn BELOW holes
        if show_traces:
            for shot in shots:
                if shot.series == current_series and shot.trace:
                    self._draw_shot_trace(canvas, shot.trace, alpha=trace_alpha)

        # 2. Fading post-shot trace
        if fading_trace and fading_trace.points:
            fade_alpha = max(0.05, 1.0 - fading_age_s / 2.0)
            self._draw_shot_trace(canvas, fading_trace, alpha=fade_alpha, width=2)

        # 3. Active live trace
        if active_trace and active_trace.points:
            self._draw_shot_trace(canvas, active_trace, alpha=1.0, width=1)

        # 4. Shot holes
        for shot in shots:
            self._draw_shot_hole(canvas, shot, current_series,
                                  dot_only=show_dot_only)

        # 5. Highlighted trace — drawn ON TOP of shot holes
        if highlighted_shot_trace:
            self._draw_shot_trace(canvas, highlighted_shot_trace,
                                   alpha=1.0, width=2)

        # 6. Aim centrepoints
        if show_acp:
            for shot in shots:
                if shot.series == current_series and shot.aim_centrepoint:
                    self._draw_acp(canvas, shot.aim_centrepoint)

        # 7. Bounding boxes
        series_shots = [s for s in shots if s.series == current_series]
        if show_bbox_shots and len(series_shots) >= 2:
            self._draw_bbox(canvas, [s.aim_mm for s in series_shots],
                             color=(100, 200, 255))
        if show_bbox_acp:
            acps = [s.aim_centrepoint for s in series_shots
                    if s.aim_centrepoint]
            if len(acps) >= 2:
                self._draw_bbox(canvas, acps, color=self.C_acp)

        # 8. MPI + group
        if show_mpi and len(series_shots) >= 2:
            self._draw_group(canvas, series_shots)

        # 9. Live crosshair
        if live_aim_mm is not None:
            self._draw_live_aim(canvas, live_aim_mm)

        # 10. Zero mode overlay
        if zero_mode:
            self._draw_zero_overlay(canvas)

        return canvas

    def mm_to_px(self, mm: Tuple[float, float]) -> Tuple[int, int]:
        return (int(self.cx + mm[0] * self.scale),
                int(self.cy + mm[1] * self.scale))

    def radius_to_px(self, r_mm: float) -> int:
        return max(1, int(r_mm * self.scale))

    # ── Static target face ────────────────────────────────────────────────────

    def _render_static(self) -> np.ndarray:
        img    = np.full((self.ch, self.cw, 3), C_BG, dtype=np.uint8)
        rings  = self.target_cfg["rings_mm"]
        scores = self.target_cfg["ring_scores"]
        # Multi-mark: draw rings at each mark centre; single-mark: draw at canvas centre
        mark_offsets = self.target_cfg.get("mark_offsets")
        centres = [self.mm_to_px((mx, my)) for mx, my in mark_offsets] \
                  if mark_offsets else [(self.cx, self.cy)]

        colours = self.target_colours

        for ci, (ocx, ocy) in enumerate(centres):
            # Draw from outside to inside.
            for i in reversed(range(len(rings))):
                r = self.radius_to_px(rings[i])
                is_inner = i < self.inner_ring_count

                fill = colours["inner" if is_inner else "outer"]
                line = colours[
                    "inner_lines" if is_inner else "outer_lines"
                ]

                cv2.circle(img, (ocx, ocy), r, fill, -1)
                cv2.circle(
                    img, (ocx, ocy), r, line, 1, cv2.LINE_AA
                )


        # Crosshair lines across full canvas (only for single-mark targets)
        if not mark_offsets:
            hl = min(self.cw, self.ch) // 2
            cv2.line(img, (self.cx - hl, self.cy), (self.cx + hl, self.cy), (40, 40, 45), 1)
            cv2.line(img, (self.cx, self.cy - hl), (self.cx, self.cy + hl), (40, 40, 45), 1)

        # Approach zone boundary (2× scoring radius) — dashed grey circle
        approach_r = self.radius_to_px(self.target_cfg["rings_mm"][-1] * 2.0)
        n_dash = 48
        for i in range(n_dash):
            if i % 2 == 0:
                a1 = 2 * np.pi * i / n_dash
                a2 = 2 * np.pi * (i + 1) / n_dash
                for a in np.linspace(a1, a2, 4):
                    px = int(self.cx + approach_r * np.cos(a))
                    py = int(self.cy + approach_r * np.sin(a))
                    cv2.circle(img, (px, py), 1, (50, 50, 60), -1)

        # Draw labels last so ring fills cannot cover them.
        # Keep the existing behaviour: labels on the first mark only.
        ocx, ocy = centres[0]
        font = cv2.FONT_HERSHEY_SIMPLEX

        for i in range(self.score_ring_count):
            score = scores[i]
            label = (
                str(int(score))
                if score == int(score)
                else str(score)
            )

            inner_r = rings[i - 1] if i > 0 else 0.0
            mid_r_px = (inner_r + rings[i]) * 0.5 * self.scale

            colour = self.target_colours[
                "inner_lines"
                if i < self.inner_ring_count
                else "outer_lines"
            ]

            font_scale = 0.35
            (width, height), baseline = cv2.getTextSize(
                label, font, font_scale, 1
            )

            for dx, dy in self.score_directions:
                x = round(ocx + dx * mid_r_px - width / 2)
                y = round(ocy + dy * mid_r_px + height / 2)

                cv2.putText(
                    img, label, (x, y),
                    font, font_scale, colour, 1, cv2.LINE_AA,
                )
                
        return img

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_shot_trace(self, img, trace: ShotTrace,
                         alpha: float = 1.0, width: int = 1):
        pts = trace.points
        if len(pts) < 2:
            return
        n = len(pts)

        # Pass renderer params to trace so it can pre-compute colours.
        # If params changed (user updated colours in settings), trigger
        # a full recompute; otherwise colours are already cached.
        params = (self.col_approach, self.col_hold, self.col_preshot,
                  self.col_final, self.trace_preshot_s, self.trace_final_s)
        if trace._cache_params != params or len(trace.cached_colours) != n:
            trace.recompute_colours(*params[:4],
                                    preshot_s=params[4], final_s=params[5])

        # Draw using cached colours — O(n) pixel ops, zero Python colour math
        colours = trace.cached_colours
        for i in range(1, n):
            col = tuple(int(c * alpha) for c in colours[i])
            p1 = self.mm_to_px(pts[i - 1].aim_mm)
            p2 = self.mm_to_px(pts[i].aim_mm)
            cv2.line(img, p1, p2, col, width, cv2.LINE_AA)

    def _draw_shot_hole(self, img, shot: Shot, current_series: int,
                         dot_only: bool = False):
        px = self.mm_to_px(shot.aim_mm)

        if shot.score == 0:
            hr = max(3, self.radius_to_px(self.calibre_mm / 2))
            cv2.line(img, (px[0]-hr, px[1]-hr), (px[0]+hr, px[1]+hr), self.C_miss, 2, cv2.LINE_AA)
            cv2.line(img, (px[0]+hr, px[1]-hr), (px[0]-hr, px[1]+hr), self.C_miss, 2, cv2.LINE_AA)
            return

        a = 1.0 if shot.series == current_series else 0.45
        fill = tuple(int(c * a) for c in self.C_shot_fill)

        if dot_only:
            # Just a small 3px red dot — clean minimal view
            cv2.circle(img, px, 3, fill, -1, cv2.LINE_AA)
        else:
            # Full calibre-sized circle
            hr = max(2, self.radius_to_px(self.calibre_mm / 2))
            cv2.circle(img, px, hr, fill, -1, cv2.LINE_AA)
            cv2.circle(img, px, hr, C_SHOT_RING, 1, cv2.LINE_AA)
            if shot.series == current_series:
                lbl = (str(int(shot.score)) if shot.score == int(shot.score)
                       else f"{shot.score:.1f}")
                cv2.putText(img, lbl, (px[0] + hr + 2, px[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 100),
                            1, cv2.LINE_AA)

    def _draw_acp(self, img, acp: Tuple[float, float]):
        """Draw aim centrepoint — gold diamond marker."""
        px = self.mm_to_px(acp)
        s = 6
        pts = np.array([[px[0], px[1]-s], [px[0]+s, px[1]],
                         [px[0], px[1]+s], [px[0]-s, px[1]]], np.int32)
        cv2.polylines(img, [pts], True, self.C_acp, 1, cv2.LINE_AA)
        cv2.drawMarker(img, px, self.C_acp, cv2.MARKER_CROSS, 5, 1, cv2.LINE_AA)

    def _draw_bbox(self, img, points, color=(100, 200, 255)):
        """Draw axis-aligned bounding box around a list of mm-coordinate points."""
        if len(points) < 2:
            return
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        tl = self.mm_to_px((min(xs), min(ys)))
        br = self.mm_to_px((max(xs), max(ys)))
        cv2.rectangle(img, tl, br, color, 1, cv2.LINE_AA)
        # Dimension labels
        w_mm = max(xs) - min(xs)
        h_mm = max(ys) - min(ys)
        mid_x = (tl[0] + br[0]) // 2
        cv2.putText(img, f"{w_mm:.1f}mm", (mid_x - 20, tl[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        mid_y = (tl[1] + br[1]) // 2
        cv2.putText(img, f"{h_mm:.1f}mm", (br[0] + 4, mid_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    def _draw_group(self, img, shots: List[Shot]):
        coords = np.array([s.aim_mm for s in shots])
        mpi = (float(np.mean(coords[:, 0])), float(np.mean(coords[:, 1])))
        mpx = self.mm_to_px(mpi)
        cv2.drawMarker(img, mpx, self.C_mpi, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        max_d = max(
            (float(np.linalg.norm(coords[i] - coords[j]))
             for i in range(len(coords)) for j in range(i+1, len(coords))),
            default=0.0,
        )
        if max_d > 0:
            cv2.circle(img, mpx, self.radius_to_px(max_d / 2), self.C_group, 1, cv2.LINE_AA)

    def _draw_live_aim(self, img, aim_mm: Tuple[float, float]):
        px = self.mm_to_px(aim_mm)
        s = 14
        cv2.line(img, (px[0]-s, px[1]), (px[0]+s, px[1]), self.C_crosshair, 1, cv2.LINE_AA)
        cv2.line(img, (px[0], px[1]-s), (px[0], px[1]+s), self.C_crosshair, 1, cv2.LINE_AA)
        cv2.circle(img, px, 5, self.C_crosshair, 1, cv2.LINE_AA)

    def _draw_zero_overlay(self, img: np.ndarray):
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (4, 4), (w-4, h-4), (0, 165, 255), 8)
        img[:] = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
        label = "ZERO MODE"
        sub   = "Fire one shot to set zero point"
        lw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
        sw = cv2.getTextSize(sub,   cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        cv2.putText(img, label, ((w-lw)//2, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(img, sub, ((w-sw)//2, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)
