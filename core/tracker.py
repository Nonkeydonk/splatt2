"""ArUco-based aim tracking and shot scoring.

A printed sheet carries four (or six, or eight) ArUco markers at known
positions. Each frame we:

1. Detect the markers in the camera image.
2. Solve a homography from image pixels to board millimetres.
3. Map the image centre - i.e. where the camera is pointing - through
   that homography to get the aim point relative to the target centre.

If markers are briefly lost, the most recent homography is reused for a
short window so the aim point does not jitter to the centre on every
dropped frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# Marker IDs 0-3 form the corners; 4-5 add the left/right edge midpoints
# for 6-marker layouts; 6-7 add top/bottom for 8-marker layouts.
_MARKER_LAYOUTS = {
    4: [0, 1, 2, 3],
    6: [0, 1, 2, 3, 4, 5],
    8: [0, 1, 2, 3, 4, 5, 6, 7],
}


@dataclass
class TrackFrame:
    """Tracking output for one camera frame."""
    aim_mm: Optional[Tuple[float, float]] = None
    aim_px: Optional[Tuple[int, int]] = None
    markers_found: int = 0
    frame_display: Optional[np.ndarray] = None
    homography: Optional[np.ndarray] = None
    quality: float = 0.0


def _build_board_corners(
    board_w: float, board_h: float, marker: float, margin: float,
) -> dict:
    """Return the board-space (mm) corners for every supported marker ID."""
    m, mg = marker, margin
    bw, bh = board_w, board_h
    return {
        # Corners (TL, TR, BR, BL).
        0: np.array([[mg, mg], [mg + m, mg], [mg + m, mg + m], [mg, mg + m]],
                    dtype=np.float32),
        1: np.array([[bw - mg - m, mg], [bw - mg, mg],
                     [bw - mg, mg + m], [bw - mg - m, mg + m]],
                    dtype=np.float32),
        2: np.array([[bw - mg - m, bh - mg - m], [bw - mg, bh - mg - m],
                     [bw - mg, bh - mg], [bw - mg - m, bh - mg]],
                    dtype=np.float32),
        3: np.array([[mg, bh - mg - m], [mg + m, bh - mg - m],
                     [mg + m, bh - mg], [mg, bh - mg]],
                    dtype=np.float32),
        # Left and right edge midpoints (6-marker layout).
        4: np.array([[mg, bh / 2 - m / 2], [mg + m, bh / 2 - m / 2],
                     [mg + m, bh / 2 + m / 2], [mg, bh / 2 + m / 2]],
                    dtype=np.float32),
        5: np.array([[bw - mg - m, bh / 2 - m / 2],
                     [bw - mg, bh / 2 - m / 2],
                     [bw - mg, bh / 2 + m / 2],
                     [bw - mg - m, bh / 2 + m / 2]],
                    dtype=np.float32),
        # Top and bottom edge midpoints (8-marker layout).
        6: np.array([[bw / 2 - m / 2, mg], [bw / 2 + m / 2, mg],
                     [bw / 2 + m / 2, mg + m], [bw / 2 - m / 2, mg + m]],
                    dtype=np.float32),
        7: np.array([[bw / 2 - m / 2, bh - mg - m],
                     [bw / 2 + m / 2, bh - mg - m],
                     [bw / 2 + m / 2, bh - mg],
                     [bw / 2 - m / 2, bh - mg]],
                    dtype=np.float32),
    }


class ArucoTracker:
    """Track the aim point against a printed multi-marker board.

    Args:
        board_width_mm: Real-world width of the printed sheet (mm).
        board_height_mm: Real-world height of the printed sheet (mm).
        marker_size_mm: Size of each printed ArUco marker (mm).
        aruco_dict_name: Name of the ``cv2.aruco`` dictionary, e.g.
            ``'DICT_4X4_50'``.
        margin_mm: Distance from sheet edge to marker corner (mm).
        use_clahe: Apply CLAHE contrast enhancement before detection.
        clahe_clip: CLAHE clip limit (higher is more aggressive).
        marker_count: Number of markers on the sheet (``4``, ``6`` or ``8``).
        brightness_target: Mean brightness target for software gain
            normalisation, used before CLAHE.
    """

    MAX_HOMOGRAPHY_AGE = 5
    """Frames a stale homography may be reused after detection drops out."""

    def __init__(
        self,
        board_width_mm: float = 210.0,
        board_height_mm: float = 297.0,
        marker_size_mm: float = 40.0,
        aruco_dict_name: str = "DICT_4X4_50",
        margin_mm: float = 8.0,
        use_clahe: bool = True,
        clahe_clip: float = 4.0,
        marker_count: int = 4,
        brightness_target: float = 128.0,
        sharpen: float = 0.0,
    ):
        self.board_width_mm = board_width_mm
        self.board_height_mm = board_height_mm
        self.marker_size_mm = marker_size_mm
        self.margin_mm = margin_mm
        self.use_clahe = use_clahe
        self.brightness_target = float(brightness_target)
        # Unsharp-mask amount applied after CLAHE. 0 disables; the
        # useful range is roughly 0.3 - 1.5 for crisping up soft
        # marker edges at distance.
        self.sharpen = max(0.0, float(sharpen))

        dict_id = getattr(cv2.aruco, aruco_dict_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = (
            cv2.aruco.CORNER_REFINE_SUBPIX)
        # Allow markers slightly smaller than the 3% default so 4-marker
        # boards stay detected when zoomed/cropped tight at distance.
        self.detector_params.minMarkerPerimeterRate = 0.02
        # Widen the adaptive threshold sweep so a marker isn't missed
        # purely because the default window size is wrong for the
        # current frame size.
        self.detector_params.adaptiveThreshWinSizeMin = 3
        self.detector_params.adaptiveThreshWinSizeMax = 23
        self.detector_params.adaptiveThreshWinSizeStep = 10
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, self.detector_params)

        self._clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip), tileGridSize=(8, 8))

        all_corners = _build_board_corners(
            board_width_mm, board_height_mm, marker_size_mm, margin_mm)
        active = _MARKER_LAYOUTS.get(marker_count, _MARKER_LAYOUTS[4])
        self._board_corners = {k: all_corners[k] for k in active}

        self.target_centre_mm = np.array(
            [board_width_mm / 2, board_height_mm / 2], dtype=np.float32)

        self._last_homography: Optional[np.ndarray] = None
        self._homography_age: int = 0

    def process_frame(self, frame: np.ndarray) -> TrackFrame:
        """Run detection on one BGR frame and return the resulting state."""
        result = TrackFrame()
        result.frame_display = frame.copy()

        gray = self._preprocess(frame)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            self._homography_age += 1
            self._try_reuse_homography(result, frame)
            return result

        ids_flat = ids.flatten()
        result.markers_found = len(ids_flat)
        cv2.aruco.drawDetectedMarkers(result.frame_display, corners, ids)

        img_pts: List[np.ndarray] = []
        brd_pts: List[np.ndarray] = []
        for i, mid in enumerate(ids_flat):
            if mid in self._board_corners:
                img_pts.append(corners[i][0])
                brd_pts.append(self._board_corners[mid])

        if not img_pts:
            self._homography_age += 1
            self._try_reuse_homography(result, frame)
            return result

        H, _ = cv2.findHomography(
            np.concatenate(img_pts, axis=0),
            np.concatenate(brd_pts, axis=0),
            cv2.RANSAC, 5.0,
        )
        if H is None:
            self._homography_age += 1
            self._try_reuse_homography(result, frame)
            return result

        self._last_homography = H
        self._homography_age = 0
        result.homography = H
        result.quality = min(1.0, len(img_pts) / len(self._board_corners))
        self._compute_aim(result, frame)
        return result

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Greyscale, software gain normalisation, then CLAHE.

        An optional unsharp mask runs after CLAHE to tighten marker
        edges that have been softened by lens blur or downscaling.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.use_clahe:
            mean = float(np.mean(gray))
            if mean > 1.0:
                scale = self.brightness_target / mean
                gray = np.clip(
                    gray.astype(np.float32) * scale, 0, 255).astype(np.uint8)
            gray = self._clahe.apply(gray)
        if self.sharpen > 0.0:
            gray = self._unsharp_mask(gray, self.sharpen)
        return gray

    @staticmethod
    def _unsharp_mask(gray: np.ndarray, amount: float) -> np.ndarray:
        """Sharpen by subtracting a Gaussian-blurred copy of the image."""
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5, sigmaY=1.5)
        return cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)

    def _try_reuse_homography(
        self, result: TrackFrame, frame: np.ndarray,
    ) -> None:
        if self._last_homography is None:
            return
        if self._homography_age > self.MAX_HOMOGRAPHY_AGE:
            return
        result.homography = self._last_homography
        result.quality = max(0.1, 0.5 - self._homography_age * 0.1)
        self._compute_aim(result, frame)

    def _compute_aim(
        self, result: TrackFrame, frame: np.ndarray,
    ) -> None:
        H = result.homography
        if H is None:
            return

        h, w = frame.shape[:2]
        img_centre = np.array([[[w / 2, h / 2]]], dtype=np.float32)
        board_pt = cv2.perspectiveTransform(img_centre, H)[0][0]

        result.aim_mm = (
            float(board_pt[0] - self.target_centre_mm[0]),
            float(board_pt[1] - self.target_centre_mm[1]),
        )
        result.aim_px = (int(w / 2), int(h / 2))

        cx, cy = result.aim_px
        colour = (0, 255, 0) if result.quality > 0.5 else (0, 165, 255)
        cv2.line(result.frame_display,
                 (cx - 20, cy), (cx + 20, cy), colour, 2)
        cv2.line(result.frame_display,
                 (cx, cy - 20), (cx, cy + 20), colour, 2)
        cv2.circle(result.frame_display, (cx, cy), 8, colour, 1)

        text = f"Aim: ({result.aim_mm[0]:+.1f}, {result.aim_mm[1]:+.1f}) mm"
        cv2.putText(result.frame_display, text, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def _nearest_mark(
    aim_mm: Tuple[float, float], mark_offsets: list,
) -> Tuple[int, Tuple[float, float]]:
    """Return ``(index, local_aim)`` for the nearest mark."""
    best_idx = 0
    best_local = aim_mm
    best_dist = float("inf")
    for idx, (mx, my) in enumerate(mark_offsets):
        dx = aim_mm[0] - mx
        dy = aim_mm[1] - my
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
            best_local = (dx, dy)
    return best_idx, best_local


def score_shot(
    aim_mm: Tuple[float, float],
    scoring_radius_mm: float,
    decimal: bool = False,
    mark_offsets: Optional[list] = None,
) -> Tuple[float, int, int]:
    """Score a shot purely from geometry.

    Args:
        aim_mm: Aim position ``(x, y)`` in mm relative to sheet centre.
        scoring_radius_mm: ``card_radius + calibre_radius``.
        decimal: ``True`` for 99 ISSF decimal bands; ``False`` for 10
            integer bands.
        mark_offsets: Optional list of mark centres for multi-mark
            targets. When provided, the shot is assigned to the nearest
            mark and scored relative to that mark's centre.

    Returns:
        ``(score, band_index, mark_index)``. A miss returns
        ``(0.0, -1, mark_index)``. ``mark_index`` is always ``0`` for
        single-mark targets.
    """
    if mark_offsets and len(mark_offsets) > 1:
        mark_idx, aim_local = _nearest_mark(aim_mm, mark_offsets)
    else:
        mark_idx, aim_local = 0, aim_mm

    aim_r = math.sqrt(aim_local[0] ** 2 + aim_local[1] ** 2)
    if aim_r > scoring_radius_mm:
        return 0.0, -1, mark_idx

    if decimal:
        n_bands = 99
        step = 9.9 / 98
        band = min(int(aim_r / (scoring_radius_mm / n_bands)), n_bands - 1)
        return round(10.9 - band * step, 1), band, mark_idx

    n_bands = 10
    band = min(int(aim_r / (scoring_radius_mm / n_bands)), n_bands - 1)
    return float(10 - band), band, mark_idx


def aim_to_display(
    aim_mm: Tuple[float, float], target_cfg: dict, display_size_px: tuple,
) -> Tuple[int, int]:
    """Convert an aim offset (mm) to canvas pixels for the target view."""
    dw, dh = display_size_px
    scale = min(dw, dh) / target_cfg["diameter_mm"]
    cx, cy = dw / 2, dh / 2
    return int(cx + aim_mm[0] * scale), int(cy + aim_mm[1] * scale)
