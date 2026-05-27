"""Target canvas renderer.

Composes the static target face once at construction time and overlays
shot traces, holes, aim centrepoints, bounding boxes, the live aim
crosshair and the optional zero-mode banner on each call to
:meth:`TargetRenderer.render`.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.session import Shot, ShotTrace


# Static palette (BGR).
C_BG = (22, 22, 28)
C_RING_OUTER = (180, 180, 180)
C_SHOT_RING = (255, 255, 255)
C_MPI = (255, 180, 80)
C_GROUP = (255, 100, 100)

# Supersampling factor for the static target face. Rings, ring labels
# and the canvas crosshair are drawn this many times larger and then
# downscaled with INTER_AREA, which avoids the blocky aliasing OpenCV
# produces when filling small circles directly at canvas resolution.
_STATIC_SUPERSAMPLE = 3


def _hex_to_bgr(value: str) -> Tuple[int, int, int]:
    """Convert ``#rrggbb`` to an OpenCV BGR tuple."""
    h = value.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _effective_diameter_mm(target_cfg: dict) -> float:
    """Visual diameter that must fit on the canvas.

    For multi-mark targets this is the full span across all marks, not
    the diameter of a single card.
    """
    diameter = target_cfg["diameter_mm"]
    offsets = target_cfg.get("mark_offsets")
    if not offsets:
        return diameter
    max_reach = max(math.hypot(mx, my) for mx, my in offsets)
    return (max_reach + diameter / 2) * 2


class TargetRenderer:
    """Render a digital target face with shot data overlaid."""

    def __init__(
        self,
        canvas_size: Tuple[int, int],
        target_cfg: dict,
        display_calibre_mm: Optional[float] = None,
        display_cfg: Optional[dict] = None,
        zoom: float = 1.0,
    ):
        self.cw, self.ch = canvas_size
        self.target_cfg = target_cfg
        self.calibre_mm = display_calibre_mm or target_cfg.get("calibre_mm", 4.5)
        self.zoom = max(0.1, float(zoom))

        self._configure_colours(display_cfg or {})

        usable = min(self.cw, self.ch) * 0.88 * self.zoom
        self.scale = usable / _effective_diameter_mm(target_cfg)

        self.cx = self.cw // 2
        self.cy = self.ch // 2

        self._static = self._render_static()

    def _configure_colours(self, dc: dict) -> None:
        self.C_shot_fill = _hex_to_bgr(dc.get("colour_shot_fill", "#5050ff"))
        self.C_acp = _hex_to_bgr(dc.get("colour_acp", "#ffc800"))
        self.C_crosshair = _hex_to_bgr(dc.get("colour_crosshair", "#00dc64"))
        self.C_miss = _hex_to_bgr(dc.get("colour_miss", "#3c3cd0"))
        self.C_mpi = _hex_to_bgr(dc.get("colour_mpi", "#50b4ff"))
        self.C_group = _hex_to_bgr(dc.get("colour_group", "#6464ff"))
        self.trace_width = int(dc.get("trace_width", 1))
        self.col_approach = _hex_to_bgr(
            dc.get("colour_trace_approach", "#3c3c3c"))
        self.col_hold = _hex_to_bgr(
            dc.get("colour_trace_hold", "#28be50"))
        self.col_preshot = _hex_to_bgr(
            dc.get("colour_trace_preshot", "#f0d000"))
        self.col_final = _hex_to_bgr(
            dc.get("colour_trace_final", "#e03020"))
        self.trace_preshot_s = float(dc.get("trace_preshot_s", 1.0))
        self.trace_final_s = float(dc.get("trace_final_s", 0.2))

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
        show_dot_only: bool = False,
        trace_alpha: float = 0.30,
    ) -> np.ndarray:
        """Compose and return the final BGR target frame."""
        canvas = self._static.copy()

        if show_traces:
            for shot in shots:
                if shot.series == current_series and shot.trace:
                    self._draw_shot_trace(canvas, shot.trace, alpha=trace_alpha)

        if fading_trace and fading_trace.points:
            fade_alpha = max(0.05, 1.0 - fading_age_s / 2.0)
            self._draw_shot_trace(canvas, fading_trace,
                                  alpha=fade_alpha, width=2)

        if active_trace and active_trace.points:
            self._draw_shot_trace(canvas, active_trace, alpha=1.0, width=1)

        for shot in shots:
            self._draw_shot_hole(canvas, shot, current_series,
                                 dot_only=show_dot_only)

        if highlighted_shot_trace:
            self._draw_shot_trace(canvas, highlighted_shot_trace,
                                  alpha=1.0, width=2)

        if show_acp:
            for shot in shots:
                if shot.series == current_series and shot.aim_centrepoint:
                    self._draw_acp(canvas, shot.aim_centrepoint)

        series_shots = [s for s in shots if s.series == current_series]
        if show_bbox_shots and len(series_shots) >= 2:
            self._draw_bbox(canvas, [s.aim_mm for s in series_shots],
                            color=(100, 200, 255))
        if show_bbox_acp:
            acps = [s.aim_centrepoint for s in series_shots
                    if s.aim_centrepoint]
            if len(acps) >= 2:
                self._draw_bbox(canvas, acps, color=self.C_acp)

        if show_mpi and len(series_shots) >= 2:
            self._draw_group(canvas, series_shots)

        if live_aim_mm is not None:
            self._draw_live_aim(canvas, live_aim_mm)

        if zero_mode:
            self._draw_zero_overlay(canvas)

        return canvas

    def mm_to_px(self, point_mm: Tuple[float, float]) -> Tuple[int, int]:
        """Convert mm offsets from target centre to canvas pixels."""
        return (
            int(self.cx + point_mm[0] * self.scale),
            int(self.cy + point_mm[1] * self.scale),
        )

    def radius_to_px(self, r_mm: float) -> int:
        """Convert a millimetre radius to canvas pixels (minimum 1)."""
        return max(1, int(r_mm * self.scale))

    def _render_static(self) -> np.ndarray:
        """Render the static target face once at supersampled resolution.

        Drawing at ``_STATIC_SUPERSAMPLE`` times the canvas resolution
        and downsampling with ``INTER_AREA`` produces smooth ring edges
        and legible small text without resorting to a true vector
        backend. The downscaled image is cached for reuse on every
        :meth:`render` call.
        """
        ss = _STATIC_SUPERSAMPLE
        big_w, big_h = self.cw * ss, self.ch * ss
        big = np.full((big_h, big_w, 3), C_BG, dtype=np.uint8)

        rings = self.target_cfg["rings_mm"]
        scores = self.target_cfg["ring_scores"]
        mark_offsets = self.target_cfg.get("mark_offsets")

        big_cx, big_cy = self.cx * ss, self.cy * ss
        if mark_offsets:
            centres = [
                (int(big_cx + mx * self.scale * ss),
                 int(big_cy + my * self.scale * ss))
                for mx, my in mark_offsets
            ]
        else:
            centres = [(big_cx, big_cy)]

        for ci, (ocx, ocy) in enumerate(centres):
            self._draw_rings(big, ocx, ocy, rings, scores,
                             label=(ci == 0), ss=ss)

        if not mark_offsets:
            self._draw_canvas_crosshair(big, ss=ss)

        return cv2.resize(big, (self.cw, self.ch),
                          interpolation=cv2.INTER_AREA)

    def _draw_rings(
        self, img: np.ndarray, ocx: int, ocy: int,
        rings: list, scores: list, label: bool, ss: int = 1,
    ) -> None:
        n_rings = len(rings)
        # The "bull" is the lighter-coloured central disc. Targets that
        # carry an explicit ``bull_dia_mm`` use it directly; everything
        # else falls back to the historical heuristic of treating the
        # outer four rings as the dark surround. A bull diameter of 0
        # means there is no bull — the whole card is dark.
        bull_dia = self.target_cfg.get("bull_dia_mm")
        if bull_dia is None:
            bull_cutoff = n_rings - 4
            in_bull = lambda i: i < bull_cutoff  # noqa: E731
        else:
            limit = float(bull_dia) + 1e-6
            in_bull = lambda i: rings[i] * 2 <= limit  # noqa: E731

        scale_px = self.scale * ss
        ring_thickness = max(1, ss // 2)

        for i in reversed(range(n_rings)):
            r = max(1, int(rings[i] * scale_px))
            light = in_bull(i)
            fill = (240, 240, 240) if light else (15, 15, 15)
            cv2.circle(img, (ocx, ocy), r, fill, -1, cv2.LINE_AA)
            cv2.circle(img, (ocx, ocy), r, C_RING_OUTER,
                       ring_thickness, cv2.LINE_AA)
            if not label or i >= n_rings - 1:
                continue
            score = scores[i]
            text = str(int(score)) if score == int(score) else str(score)
            colour = (80, 80, 80) if light else (200, 200, 200)
            mid_r = (rings[i] + (rings[i - 1] if i > 0 else 0)) / 2
            lx = int(ocx + mid_r * scale_px * 0.6)
            ly = int(ocy + 4 * ss)
            cv2.putText(img, text, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35 * ss, colour,
                        max(1, ss // 2), cv2.LINE_AA)
        cv2.circle(img, (ocx, ocy),
                   max(2 * ss, int(0.5 * scale_px)),
                   (0, 0, 0), -1, cv2.LINE_AA)

    def _draw_canvas_crosshair(self, img: np.ndarray, ss: int = 1) -> None:
        h, w = img.shape[:2]
        cx, cy = self.cx * ss, self.cy * ss
        hl = min(w, h) // 2
        thickness = max(1, ss // 2)
        cv2.line(img, (cx - hl, cy), (cx + hl, cy),
                 (40, 40, 45), thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy - hl), (cx, cy + hl),
                 (40, 40, 45), thickness, cv2.LINE_AA)

    def _draw_shot_trace(
        self, img: np.ndarray, trace: ShotTrace,
        alpha: float = 1.0, width: int = 1,
    ) -> None:
        pts = trace.points
        n = len(pts)
        if n < 2:
            return

        params = (self.col_approach, self.col_hold, self.col_preshot,
                  self.col_final, self.trace_preshot_s, self.trace_final_s)
        if trace._cache_params != params or len(trace.cached_colours) != n:
            trace.recompute_colours(*params[:4],
                                    preshot_s=params[4], final_s=params[5])

        colours = trace.cached_colours
        for i in range(1, n):
            colour = tuple(int(c * alpha) for c in colours[i])
            p1 = self.mm_to_px(pts[i - 1].aim_mm)
            p2 = self.mm_to_px(pts[i].aim_mm)
            cv2.line(img, p1, p2, colour, width, cv2.LINE_AA)

    def _draw_shot_hole(
        self, img: np.ndarray, shot: Shot, current_series: int,
        dot_only: bool = False,
    ) -> None:
        px = self.mm_to_px(shot.aim_mm)

        if shot.score == 0:
            hr = max(3, self.radius_to_px(self.calibre_mm / 2))
            cv2.line(img, (px[0] - hr, px[1] - hr),
                     (px[0] + hr, px[1] + hr), self.C_miss, 2, cv2.LINE_AA)
            cv2.line(img, (px[0] + hr, px[1] - hr),
                     (px[0] - hr, px[1] + hr), self.C_miss, 2, cv2.LINE_AA)
            return

        alpha = 1.0 if shot.series == current_series else 0.45
        fill = tuple(int(c * alpha) for c in self.C_shot_fill)

        if dot_only:
            cv2.circle(img, px, 3, fill, -1, cv2.LINE_AA)
            return

        hr = max(2, self.radius_to_px(self.calibre_mm / 2))
        cv2.circle(img, px, hr, fill, -1, cv2.LINE_AA)
        cv2.circle(img, px, hr, C_SHOT_RING, 1, cv2.LINE_AA)
        if shot.series != current_series:
            return

        score = shot.score
        label = str(int(score)) if score == int(score) else f"{score:.1f}"
        cv2.putText(img, label, (px[0] + hr + 2, px[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 100),
                    1, cv2.LINE_AA)

    def _draw_acp(
        self, img: np.ndarray, acp: Tuple[float, float],
    ) -> None:
        px = self.mm_to_px(acp)
        s = 6
        diamond = np.array([
            [px[0], px[1] - s],
            [px[0] + s, px[1]],
            [px[0], px[1] + s],
            [px[0] - s, px[1]],
        ], np.int32)
        cv2.polylines(img, [diamond], True, self.C_acp, 1, cv2.LINE_AA)
        cv2.drawMarker(img, px, self.C_acp,
                       cv2.MARKER_CROSS, 5, 1, cv2.LINE_AA)

    def _draw_bbox(
        self, img: np.ndarray, points: list,
        color: Tuple[int, int, int] = (100, 200, 255),
    ) -> None:
        if len(points) < 2:
            return
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        tl = self.mm_to_px((min(xs), min(ys)))
        br = self.mm_to_px((max(xs), max(ys)))
        cv2.rectangle(img, tl, br, color, 1, cv2.LINE_AA)

        w_mm = max(xs) - min(xs)
        h_mm = max(ys) - min(ys)
        mid_x = (tl[0] + br[0]) // 2
        cv2.putText(img, f"{w_mm:.1f}mm", (mid_x - 20, tl[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        mid_y = (tl[1] + br[1]) // 2
        cv2.putText(img, f"{h_mm:.1f}mm", (br[0] + 4, mid_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    def _draw_group(self, img: np.ndarray, shots: List[Shot]) -> None:
        coords = np.array([s.aim_mm for s in shots])
        mpi = (float(np.mean(coords[:, 0])), float(np.mean(coords[:, 1])))
        mpx = self.mm_to_px(mpi)
        cv2.drawMarker(img, mpx, self.C_mpi,
                       cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        max_d = max(
            (float(np.linalg.norm(coords[i] - coords[j]))
             for i in range(len(coords)) for j in range(i + 1, len(coords))),
            default=0.0,
        )
        if max_d > 0:
            cv2.circle(img, mpx, self.radius_to_px(max_d / 2),
                       self.C_group, 1, cv2.LINE_AA)

    def _draw_live_aim(
        self, img: np.ndarray, aim_mm: Tuple[float, float],
    ) -> None:
        px = self.mm_to_px(aim_mm)
        s = 14
        cv2.line(img, (px[0] - s, px[1]), (px[0] + s, px[1]),
                 self.C_crosshair, 1, cv2.LINE_AA)
        cv2.line(img, (px[0], px[1] - s), (px[0], px[1] + s),
                 self.C_crosshair, 1, cv2.LINE_AA)
        cv2.circle(img, px, 5, self.C_crosshair, 1, cv2.LINE_AA)

    def _draw_zero_overlay(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (4, 4), (w - 4, h - 4), (0, 165, 255), 8)
        img[:] = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
        label = "ZERO MODE"
        sub = "Fire one shot to set zero point"
        lw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
        sw = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        cv2.putText(img, label, ((w - lw) // 2, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(img, sub, ((w - sw) // 2, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1,
                    cv2.LINE_AA)
