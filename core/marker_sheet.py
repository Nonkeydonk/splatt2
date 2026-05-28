"""Generator for the printable A4 ArUco marker sheet.

The sheet carries the configured number of ArUco markers around its
edges and one or more aiming marks in the centre. Each mark scales
linearly with the print distance::

    aiming_mark_dia_mm = reference_dia_mm * (print_distance_m
                                              / reference_distance_m)

Printing at 100% on A4 (no fit-to-page scaling) is required.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np


# A4 at 300 DPI.
A4_W_MM = 210.0
A4_H_MM = 297.0
DPI = 300
A4_W_PX = 2480
A4_H_PX = 3508
MM_TO_PX = DPI / 25.4

DEFAULT_MARGIN_MM = 8.0
DEFAULT_MARKER_MM = 40.0


def _build_aiming_marks() -> dict:
    """Build the aiming-mark dict from the loaded target catalogue."""
    from core.config import TARGETS

    marks = {}
    for key, t in TARGETS.items():
        marks[key] = {
            "name": t["name"],
            "reference_dist_m": t.get("reference_dist_m", 10.0),
            "aiming_mark_dia_mm": t.get(
                "aiming_mark_dia_mm", t["diameter_mm"] * 0.67),
            "outer_ring_dia_mm": t["diameter_mm"],
            "rings_dia_mm": t.get(
                "rings_dia_mm", [d * 2 for d in t["rings_mm"]]),
            "ring_labels": t.get(
                "ring_labels",
                [str(int(s)) if s == int(s) else str(s)
                 for s in t["ring_scores"]],
            ),
            "mark_offsets": t.get("mark_offsets"),
        }
    return marks


# Backwards-compatible alias used by the UI module.
_get_aiming_marks = _build_aiming_marks


# Computed once at import time. Includes any user-added target CSVs.
AIMING_MARKS = _build_aiming_marks()


def mm(value: float) -> int:
    """Convert millimetres to image pixels at the sheet DPI."""
    return int(value * MM_TO_PX)


def _marker_positions(
    marker_count: int, marker_px: int, margin_px: int,
) -> dict:
    """Pixel positions for each marker ID at the chosen layout."""
    positions = {
        0: (margin_px, margin_px),
        1: (A4_W_PX - margin_px - marker_px, margin_px),
        2: (A4_W_PX - margin_px - marker_px, A4_H_PX - margin_px - marker_px),
        3: (margin_px, A4_H_PX - margin_px - marker_px),
    }
    if marker_count >= 6:
        positions[4] = (margin_px, A4_H_PX // 2 - marker_px // 2)
        positions[5] = (A4_W_PX - margin_px - marker_px,
                        A4_H_PX // 2 - marker_px // 2)
    if marker_count >= 8:
        positions[6] = (A4_W_PX // 2 - marker_px // 2, margin_px)
        positions[7] = (A4_W_PX // 2 - marker_px // 2,
                        A4_H_PX - margin_px - marker_px)
    return positions


_MARKER_LABELS = {
    0: "TL(0)", 1: "TR(1)", 2: "BR(2)", 3: "BL(3)",
    4: "LM(4)", 5: "RM(5)", 6: "TM(6)", 7: "BM(7)",
}


def _draw_markers(
    img: np.ndarray, aruco_dict, positions: dict, marker_px: int,
) -> None:
    """Render each ArUco marker at its position with a label underneath."""
    border = 4
    for mid, (x, y) in positions.items():
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, mid, marker_px)
        bordered = np.full(
            (marker_px + 2 * border, marker_px + 2 * border),
            255, dtype=np.uint8,
        )
        bordered[border:border + marker_px, border:border + marker_px] = (
            marker_img)
        h, w = bordered.shape
        img[y - border:y - border + h, x - border:x - border + w] = bordered

        label_x = x if mid in (0, 3) else x - mm(8)
        label_y = y + marker_px + mm(5) if mid in (0, 1) else y - mm(2)
        cv2.putText(img, _MARKER_LABELS[mid], (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)


def _draw_dashed_circle(
    img: np.ndarray, cx: int, cy: int, radius: int,
    shade: int = 150, n_dashes: int = 48,
) -> None:
    """Draw a dashed circle in greyscale at the given centre and radius."""
    for i in range(0, n_dashes, 2):
        a1 = 2 * np.pi * i / n_dashes
        a2 = 2 * np.pi * (i + 1) / n_dashes
        points = [
            (int(cx + radius * np.cos(a)), int(cy + radius * np.sin(a)))
            for a in np.linspace(a1, a2, 6)
        ]
        for p1, p2 in zip(points, points[1:]):
            cv2.line(img, p1, p2, shade, 2)


def _draw_aiming_mark(
    img: np.ndarray, cx: int, cy: int, mark_cfg: dict,
    scale_pct: float, outer_dia_scaled: float, aiming_dia_scaled: float,
    show_ring_guides: bool, show_labels: bool,
) -> None:
    """Draw rings, outer boundary, black aiming bull and centre cross."""
    if show_ring_guides:
        for i, ring_dia in enumerate(mark_cfg["rings_dia_mm"]):
            r_px = mm(ring_dia / 2 * scale_pct)
            if r_px < 4:
                continue
            _draw_dashed_circle(img, cx, cy, r_px, shade=160, n_dashes=48)
            if not show_labels:
                continue
            labels = mark_cfg["ring_labels"]
            label = labels[i] if i < len(labels) else ""
            if label:
                cv2.putText(img, label, (cx + r_px + mm(1), cy + mm(1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 120, 1)

    outer_r_px = mm(outer_dia_scaled / 2)
    if outer_r_px > 4:
        cv2.circle(img, (cx, cy), outer_r_px, 100, 2)

    aim_r_px = mm(aiming_dia_scaled / 2)
    if aim_r_px > 2:
        cv2.circle(img, (cx, cy), aim_r_px, 0, -1)
    else:
        cv2.circle(img, (cx, cy), max(3, mm(0.5)), 0, -1)

    cross = max(mm(2), aim_r_px // 4)
    cv2.line(img, (cx - cross, cy), (cx + cross, cy), 255, 2)
    cv2.line(img, (cx, cy - cross), (cx, cy + cross), 255, 2)


def _aiming_mark_centres(
    mark_offsets: Optional[list], scale_pct: float,
) -> list:
    """Pixel centres for each aiming mark on the sheet."""
    cx, cy = A4_W_PX // 2, A4_H_PX // 2
    if not mark_offsets:
        return [(cx, cy)]
    return [
        (cx + mm(mx * scale_pct), cy + mm(my * scale_pct))
        for mx, my in mark_offsets
    ]


def _draw_instructions(
    img: np.ndarray, lines: list,
) -> None:
    """Render the print instructions strip at the foot of the sheet."""
    for i, line in enumerate(lines):
        cv2.putText(img, line,
                    (mm(10), A4_H_PX - mm(36) + i * mm(9)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, 60, 1)


def generate_marker_sheet(
    output_path: str = "aruco_sheet.png",
    target_key: str = "10m_air_rifle",
    print_distance_m: Optional[float] = None,
    show_ring_guides: bool = True,
    aruco_dict_name: str = "DICT_4X4_50",
    marker_size_mm: Optional[float] = None,
    margin_mm: Optional[float] = None,
    marker_count: int = 4,
) -> str:
    """Generate the marker sheet PNG and return its path.

    Args:
        output_path: Destination PNG path.
        target_key: Key into the target catalogue.
        print_distance_m: Distance the sheet will be printed for. Defaults
            to the target's reference distance, i.e. 100% scale.
        show_ring_guides: Draw dashed scoring-ring guides around the bull.
        aruco_dict_name: ``cv2.aruco`` dictionary name.
        marker_size_mm: Printed marker size. Defaults to ``DEFAULT_MARKER_MM``.
        margin_mm: Distance from sheet edge to marker. Defaults to
            ``DEFAULT_MARGIN_MM``.
        marker_count: Number of markers on the sheet (4, 6 or 8).
    """
    mark_cfg = AIMING_MARKS.get(target_key, AIMING_MARKS["10m_air_rifle"])
    ref_dist = mark_cfg["reference_dist_m"]
    distance = print_distance_m if print_distance_m else ref_dist
    scale_pct = distance / ref_dist

    aiming_dia_scaled = mark_cfg["aiming_mark_dia_mm"] * scale_pct
    outer_dia_scaled = mark_cfg["outer_ring_dia_mm"] * scale_pct
    marker = marker_size_mm if marker_size_mm else DEFAULT_MARKER_MM
    margin = margin_mm if margin_mm else DEFAULT_MARGIN_MM

    img = np.full((A4_H_PX, A4_W_PX), 255, dtype=np.uint8)

    dict_id = getattr(cv2.aruco, aruco_dict_name, cv2.aruco.DICT_4X4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    marker_px = mm(marker)
    margin_px = mm(margin)

    _draw_markers(
        img, aruco_dict,
        _marker_positions(marker_count, marker_px, margin_px),
        marker_px,
    )

    centres = _aiming_mark_centres(mark_cfg.get("mark_offsets"), scale_pct)
    for i, (cx, cy) in enumerate(centres):
        _draw_aiming_mark(
            img, cx, cy, mark_cfg, scale_pct,
            outer_dia_scaled, aiming_dia_scaled,
            show_ring_guides, show_labels=(i == 0),
        )

    _draw_instructions(img, [
        f"SPLATT2  —  {mark_cfg['name']}",
        (f"Print distance: {distance:.1f}m  |  Scale: {scale_pct*100:.0f}%  |"
         f"  Markers: {marker_count}  |  "
         f"Aiming mark: {aiming_dia_scaled:.1f}mm  |  "
         f"Card: {outer_dia_scaled:.1f}mm  |  Markers: {marker:.0f}mm"),
        "Print at 100% on A4 - NO fit-to-page scaling.",
        "Verify printed aiming mark diameter with ruler before use.",
    ])

    _save_with_dpi(output_path, img, DPI)
    print(f"[MarkerSheet] Saved: {output_path}  (scale={scale_pct*100:.0f}%, "
          f"aiming mark={aiming_dia_scaled:.1f}mm)")
    return output_path


def _save_with_dpi(path: str, img_bgr: np.ndarray, dpi: int) -> None:
    """Write a PNG/JPEG with the physical print resolution embedded.

    ``cv2.imwrite`` produces a bare bitmap with no DPI tag, so when
    the file is opened by a print pipeline the page size defaults to
    whatever DPI the printer assumes (often 72 or 96), producing a
    sheet several times larger than A4. Round-tripping through PIL
    lets us write the standard ``pHYs`` chunk for PNG and the JFIF
    density fields for JPEG.
    """
    from PIL import Image
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path, dpi=(dpi, dpi))


if __name__ == "__main__":
    generate_marker_sheet(print_distance_m=10.0)
    generate_marker_sheet("aruco_sheet_5m.png", print_distance_m=5.0)
