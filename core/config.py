"""User configuration and target catalogue.

Targets are loaded from CSV files in two locations: a read-only seed
directory bundled with the app, and a user-writable directory under the
app data folder. User files override seeds with the same key, so a
shooter can tweak a built-in target without losing the original.

The runtime config is a plain JSON dict persisted to the user data dir.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Iterable, Optional, Tuple

from core.paths import config_path, resource_path, user_targets_dir

VERSION = "1.1.0"

CONFIG_FILE = str(config_path())


def _migrate_legacy_config() -> None:
    """Copy any pre-1.2 config beside ``main.py`` into the user data dir."""
    if os.path.exists(CONFIG_FILE):
        return
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    legacy = os.path.join(project_root, "splatt2_config.json")
    if not os.path.isfile(legacy):
        return
    try:
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        shutil.copy2(legacy, CONFIG_FILE)
        print(f"[Config] Migrated legacy config -> {CONFIG_FILE}")
    except OSError as e:
        print(f"[Config] Could not migrate legacy config: {e}")


_migrate_legacy_config()


# Target CSV layout
# -----------------
# Header rows in ``key=value`` form, then a separator header, then data
# rows. Two header forms are accepted::
#
#     score,ring_diameter_mm
#     score_integer,score_decimal,ring_diameter_mm
#
# Ring diameters are visual only; scoring geometry is computed at runtime
# from ``card_diameter_mm`` and the configured pellet calibre.

_DATA_HEADERS = (
    "score,ring_diameter_mm",
    "score_integer,score_decimal,ring_diameter_mm",
)


def bundled_targets_dir() -> str:
    """Read-only directory of bundled target CSVs."""
    return str(resource_path("targets"))


def writable_targets_dir() -> str:
    """User-writable directory of target CSVs."""
    return str(user_targets_dir())


# Backwards-compatible aliases used by the UI module.
_targets_dir = bundled_targets_dir
_user_targets_dir = writable_targets_dir


def _parse_csv_row(parts: list) -> Optional[Tuple[float, float]]:
    """Return ``(score, diameter_mm)`` for a CSV data row, or ``None``."""
    if len(parts) == 2:
        return float(parts[0]), float(parts[1])
    if len(parts) == 3:
        # Legacy three-column format: score_integer, score_decimal, diameter.
        return float(parts[0]), float(parts[2])
    return None


def _quincunx_offsets(spacing_mm: float) -> list:
    """Five mark centres in a quincunx pattern at the given spacing."""
    h = spacing_mm / 2
    return [
        (-h, -h), (+h, -h),
        (0.0, 0.0),
        (-h, +h), (+h, +h),
    ]


def _resolve_mark_offsets(meta: dict) -> Optional[list]:
    """Build mark centres for a multi-mark target, or ``None``."""
    mark_count = int(meta.get("mark_count", 1))
    if mark_count <= 1:
        return None
    spacing = float(meta.get("mark_spacing_mm", 75.0))
    if mark_count == 5:
        return _quincunx_offsets(spacing)
    return None


def load_target_csv(path: str) -> Optional[dict]:
    """Parse a single target CSV into a target dict, or ``None`` on error."""
    meta: dict = {}
    scores: list = []
    diameters: list = []
    in_data = False

    try:
        with open(path, newline="", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower() in _DATA_HEADERS:
                    in_data = True
                    continue
                if in_data:
                    row = _parse_csv_row([p.strip() for p in line.split(",")])
                    if row is not None:
                        scores.append(row[0])
                        diameters.append(row[1])
                else:
                    parts = line.split(",", 1)
                    if len(parts) == 2:
                        meta[parts[0].strip().lower()] = parts[1].strip()
    except Exception as e:
        print(f"[Targets] Could not load {path}: {e}")
        return None

    if not scores or "key" not in meta or "name" not in meta:
        return None

    rings_mm = [d / 2.0 for d in diameters]
    outer_dia = float(meta.get("card_diameter_mm", diameters[-1]))
    aiming_dia = float(meta.get("aiming_mark_dia_mm", diameters[0]))
    unique_dias = list(dict.fromkeys(diameters))
    ring_labels = [str(int(s)) if s == int(s) else str(s) for s in scores]
    mark_count = int(meta.get("mark_count", 1))

    # Optional explicit bull diameter for the renderer; targets that
    # don't set this fall back to the historical "outer 4 rings dark,
    # inner 6 rings light" rule applied in TargetRenderer.
    bull_dia = meta.get("bull_dia_mm")
    bull_dia = float(bull_dia) if bull_dia is not None else None

    return {
        "name": meta["name"],
        "key": meta["key"],
        "diameter_mm": outer_dia,
        "rings_mm": rings_mm,
        "ring_scores": scores,
        "gauging": meta.get("gauging", "outward"),
        "calibre_mm": float(meta.get("calibre_mm", 4.5)),
        "reference_dist_m": float(meta.get("reference_dist_m", 10.0)),
        "aiming_mark_dia_mm": aiming_dia,
        "bull_dia_mm": bull_dia,
        "outer_ring_dia_mm": outer_dia,
        "rings_dia_mm": unique_dias,
        "ring_labels": ring_labels,
        "a4_target_width_mm": float(
            meta.get("a4_target_width_mm", min(outer_dia * 1.1, 170))),
        "mark_count": mark_count,
        "mark_offsets": _resolve_mark_offsets(meta),
        "mark_spacing_mm": float(meta.get("mark_spacing_mm", 0)),
    }


# Backwards-compatible alias.
_load_target_csv = load_target_csv


def _iter_target_files(directories: Iterable[str]):
    """Yield ``(directory, filename)`` for every CSV in the given dirs."""
    for tdir in directories:
        if not os.path.isdir(tdir):
            continue
        for fname in sorted(os.listdir(tdir)):
            if fname.lower().endswith(".csv"):
                yield tdir, fname


def load_all_targets() -> dict:
    """Merge target CSVs from the bundle and the user dir into a dict."""
    targets: dict = {}
    for tdir, fname in _iter_target_files(
            (bundled_targets_dir(), writable_targets_dir())):
        target = load_target_csv(os.path.join(tdir, fname))
        if target:
            targets[target["key"]] = target
    return targets


_load_all_targets = load_all_targets

TARGETS = load_all_targets()


DEFAULT_CONFIG = {
    # Target & scoring
    "target_key": "10m_air_rifle",
    "real_range_m": 10.0,
    "shot_circle_calibre_mm": 4.5,
    "scoring_calibre_mm": 4.5,
    "decimal_scoring": False,
    "ignore_misses": False,

    # Camera
    "camera_index": 0,
    "video_width": 1920,
    "video_height": 1080,
    "video_fps": 30,
    "camera_rotation": 0,
    "flip_image": False,
    "flip_mode": -1,
    "no_video_mode": False,
    "use_clahe": True,
    "clahe_clip": 4.0,
    "brightness_target": 128.0,
    "spike_velocity_mm": 25.0,
    "spike_reversal": 0.7,
    # Unsharp-mask amount applied after CLAHE. 0 disables; 0.5 - 1.5
    # is the useful range for crisping up soft marker edges.
    "sharpen": 0.0,
    # Cap for the frame size handed to the ArUco detector. Lower is
    # faster; higher preserves more pixels per marker, which matters
    # when the camera is far from the printed sheet.
    "detection_max_width": 1920,
    "detection_max_height": 1080,
    # Digital zoom applied before detection. 1.0 disables; values up
    # to 4.0 centre-crop the frame so distant markers fill more pixels.
    "camera_zoom": 1.0,

    # ArUco tracking
    "aruco_dict": "DICT_4X4_50",
    "aruco_marker_count": 4,
    "camera_pixel_format": "Auto",
    "aruco_marker_mm": 40.0,
    "aruco_margin_mm": 8.0,

    # Smoothing
    "smooth_mode": "ema",
    "smooth_alpha": 0.35,
    "smooth_window": 11,
    "smooth_poly": 2,

    # Audio detection
    "audio_device_index": None,
    "audio_sample_rate": 44100,
    "audio_trigger_threshold": 0.4,
    "audio_transient_ratio": 6.0,
    "audio_trigger_cooldown_ms": 800,
    "post_shot_cooldown_s": 2.0,

    # Trace and shot colours (hex; converted to BGR at render time)
    "colour_trace_approach": "#3c3c3c",
    "colour_trace_hold":     "#28be50",
    "colour_trace_preshot":  "#f0d000",
    "colour_trace_final":    "#e03020",
    "colour_shot_fill":      "#5050ff",
    "colour_acp":            "#ffc800",
    "colour_crosshair":      "#00dc64",
    "colour_mpi":            "#50b4ff",
    "colour_group":          "#6464ff",
    "colour_miss":           "#3c3cd0",

    # Trace behaviour
    "trace_width": 1,
    "trace_preshot_s": 1.0,
    "trace_final_s": 0.2,
    "fading_trace_duration_s": 2.0,
    "acp_fraction": 0.40,
    "approach_zone_factor": 2.0,

    # Persisted zero offset
    "zero_offset_x": 0.0,
    "zero_offset_y": 0.0,

    # Session & files
    "session_name": "Session",
    "shooter_name": "",
    "shots_per_series": 10,
    "save_directory": "",
}


def load_config() -> Tuple[dict, bool]:
    """Load the persisted config or return defaults.

    Returns:
        ``(cfg, is_first_run)``. ``is_first_run`` is true when no config
        file exists or the existing one could not be parsed.
    """
    cfg = DEFAULT_CONFIG.copy()
    if not os.path.exists(CONFIG_FILE):
        return cfg, True
    try:
        with open(CONFIG_FILE, "r") as f:
            cfg.update(json.load(f))
        return cfg, False
    except Exception:
        return cfg, True


def save_config(cfg: dict) -> None:
    """Write the config to the user data dir, ignoring write errors."""
    try:
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"[Config] Could not save config: {e}")
