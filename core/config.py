"""
Splatt2 Configuration
All user-configurable settings and target definitions.
"""

import json
import os

VERSION = "1.1.0"

def _config_path() -> str:
    """Config file lives in the project root (next to main.py)."""
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    base = os.path.dirname(base)  # up from core/ to project root
    return os.path.join(base, "splatt2_config.json")

CONFIG_FILE = _config_path()

# ── Target definitions — loaded from targets/ folder ────────────────────────
# Each .csv in the targets/ folder defines one target.
# Format: header rows (key=value), then a blank line,
#         then "score,ring_diameter_mm" header, then data rows.
# Ring diameters are in mm (not radii). Innermost ring first.

def _targets_dir() -> str:
    """Return the absolute path to the targets/ folder."""
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    base = os.path.dirname(base)  # up from core/ to project root
    return os.path.join(base, "targets")


def _load_target_csv(path: str) -> dict:
    """
    Parse a single target CSV file.

    Header rows (key=value), then a blank line, then:
        score,ring_diameter_mm
        10,0.5
        9,5.5
        ...

    ring_diameter_mm values are the VISUAL ring boundaries (mm diameter).
    Scoring geometry is computed at runtime from card_diameter_mm + calibre.
    """
    meta = {}
    scores    = []
    diameters = []
    in_data   = False

    try:
        with open(path, newline="", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                low = line.lower()
                # Accept both old and new header names
                if low in ("score,ring_diameter_mm",
                           "score_integer,score_decimal,ring_diameter_mm"):
                    in_data = True
                    continue
                if in_data:
                    parts = [p.strip() for p in line.split(",")]
                    # Old two-column format: take only first and last columns
                    if len(parts) == 3:
                        scores.append(float(parts[0]))
                        diameters.append(float(parts[2]))
                    elif len(parts) == 2:
                        scores.append(float(parts[0]))
                        diameters.append(float(parts[1]))
                else:
                    parts = line.split(",", 1)
                    if len(parts) == 2:
                        meta[parts[0].strip().lower()] = parts[1].strip()
    except Exception as e:
        print(f"[Targets] Could not load {path}: {e}")
        return None

    if not scores or "key" not in meta or "name" not in meta:
        return None

    # Visual ring radii (for rendering only — NOT used for scoring)
    rings_mm    = [d / 2.0 for d in diameters]
    outer_dia   = float(meta.get("card_diameter_mm", diameters[-1]))
    aiming_dia  = float(meta.get("aiming_mark_dia_mm", diameters[0]))
    unique_dias = list(dict.fromkeys(diameters))

    ring_labels = [str(int(s)) if s == int(s) else str(s) for s in scores]

    # ── Multi-mark support ───────────────────────────────────────────────────
    # mark_count and mark_spacing_mm are optional — single-mark targets omit them.
    # mark_offsets: list of (x_mm, y_mm) centres relative to sheet centre.
    # For a 5-mark quincunx at spacing s:
    #   (-s,-s)  (+s,-s)     (Y+ = down, matching OpenCV)
    #      (0, 0)
    #   (-s,+s)  (+s,+s)
    mark_count = int(meta.get("mark_count", 1))
    mark_offsets = None   # None = single mark at (0,0)
    if mark_count > 1:
        s = float(meta.get("mark_spacing_mm", 75.0))
        if mark_count == 5:
            h = s / 2      # half the square side = offset on each axis
            mark_offsets = [
                (-h, -h),   # top-left
                (+h, -h),   # top-right
                ( 0,  0),   # centre
                (-h, +h),   # bottom-left
                (+h, +h),   # bottom-right
            ]
        # Future: other mark_count values can be added here

    return {
        "name":               meta["name"],
        "key":                meta["key"],
        "diameter_mm":        outer_dia,
        "rings_mm":           rings_mm,
        "ring_scores":        scores,
        "gauging":            meta.get("gauging", "outward"),
        "calibre_mm":         float(meta.get("calibre_mm", 4.5)),
        "reference_dist_m":   float(meta.get("reference_dist_m", 10.0)),
        "aiming_mark_dia_mm": aiming_dia,
        "outer_ring_dia_mm":  outer_dia,
        "rings_dia_mm":       unique_dias,
        "ring_labels":        ring_labels,
        "a4_target_width_mm": float(meta.get("a4_target_width_mm",
                                             min(outer_dia * 1.1, 170))),
        "mark_count":         mark_count,
        "mark_offsets":       mark_offsets,   # None for single-mark targets
        "mark_spacing_mm":    float(meta.get("mark_spacing_mm", 0)),
    }

def _load_all_targets() -> dict:
    """Load all .csv files from the targets/ folder. Returns {key: target_dict}."""
    tdir = _targets_dir()
    targets = {}
    if not os.path.isdir(tdir):
        print(f"[Targets] Folder not found: {tdir}")
        return targets
    for fname in sorted(os.listdir(tdir)):
        if not fname.lower().endswith(".csv"):
            continue
        path = os.path.join(tdir, fname)
        t = _load_target_csv(path)
        if t:
            targets[t["key"]] = t
    return targets


TARGETS = _load_all_targets()


# ── Default runtime config ──────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # ── Target & scoring ─────────────────────────────────────────────────────
    "target_key":              "10m_air_rifle",
    "real_range_m":            10.0,         # shooting distance (m)
    "shot_circle_calibre_mm":  4.5,          # displayed shot hole diameter (mm)
    "scoring_calibre_mm":      4.5,          # pellet diameter for scoring geometry (.177=4.5 .22=5.6)
    "decimal_scoring":         False,        # ISSF decimal scoring mode
    "ignore_misses":           False,        # discard shots scoring 0

    # ── Camera ───────────────────────────────────────────────────────────────
    "camera_index":    0,
    "video_width":     640,
    "video_height":    480,
    "video_fps":       30,
    "camera_rotation": 0,       # 0 / 90 / 180 / 270 degrees
    "flip_image":      False,
    "flip_mode":       -1,
    "no_video_mode":   False,   # skip camera preview (faster on slow machines)
    "use_clahe":       True,    # adaptive contrast enhancement for ArUco detection
    "clahe_clip":      4.0,     # CLAHE clip limit (2=mild, 4=moderate, 8=aggressive)
    "brightness_target": 128.0, # software gain target brightness (0-255, 128=mid)
    "spike_velocity_mm":  25.0, # min mm/frame for spike candidate
    "spike_reversal":     0.7,  # min reversal ratio to confirm spike (0-1)

    # ── ArUco tracking ───────────────────────────────────────────────────────
    "aruco_dict":       "DICT_4X4_50",
    "aruco_marker_count": 4,        # number of ArUco markers: 4, 6, or 8
    "camera_pixel_format": "Auto",  # Auto / MJPEG / YUY2
    "aruco_marker_mm":  40.0,   # printed size of each ArUco marker (mm)
    "aruco_margin_mm":  8.0,    # margin from sheet edge to marker corner (mm)

    # ── Smoothing ────────────────────────────────────────────────────────────
    "smooth_mode":   "ema",     # "none" / "ema" / "savgol"
    "smooth_alpha":  0.35,      # EMA alpha (0.05=heavy smooth, 0.8=light)
    "smooth_window": 11,        # Savitzky-Golay window size (odd number)
    "smooth_poly":   2,         # Savitzky-Golay polynomial degree

    # ── Audio detection ──────────────────────────────────────────────────────
    "audio_device_index":       None,
    "audio_sample_rate":        44100,
    "audio_trigger_threshold":  0.4,
    "audio_transient_ratio":    6.0,
    "audio_trigger_cooldown_ms":800,
    "post_shot_cooldown_s":     2.0,   # ignore audio N seconds after a shot

    # ── Trace colours (hex, BGR-converted at render time) ────────────────────
    "colour_trace_approach": "#3c3c3c",  # approach zone: dark grey
    "colour_trace_hold":     "#28be50",  # early hold: green
    "colour_trace_preshot":  "#f0d000",  # pre-shot window: yellow
    "colour_trace_final":    "#e03020",  # final window: red
    "colour_shot_fill":      "#5050ff",  # shot hole fill
    "colour_acp":            "#ffc800",  # ACP marker
    "colour_crosshair":      "#00dc64",  # live crosshair
    "colour_mpi":            "#50b4ff",  # MPI cross
    "colour_group":          "#6464ff",  # group circle
    "colour_miss":           "#3c3cd0",  # miss X marker

    # ── Trace behaviour ──────────────────────────────────────────────────────
    "trace_width":            1,
    "trace_preshot_s":        1.0,   # seconds before shot: trace turns yellow
    "trace_final_s":          0.2,   # seconds before shot: trace turns red
    "fading_trace_duration_s":2.0,   # how long post-shot trace lingers
    "acp_fraction":           0.40,  # fraction of hold used for ACP
    "approach_zone_factor":   2.0,   # approach zone = scoring_radius × this

    # ── Zero offset (persistent across restarts) ─────────────────────────────
    "zero_offset_x": 0.0,
    "zero_offset_y": 0.0,

    # ── Session & files ──────────────────────────────────────────────────────
    "session_name":       "Session",
    "shooter_name":       "",
    "shots_per_series":   10,
    "save_directory":     "",    # empty = sessions/ folder next to the app

    # ── Voice feedback ───────────────────────────────────────────────────────
    "voice_enabled":      True, # speak scores aloud
}


def load_config():
    """Load config. Returns (cfg, is_first_run)."""
    cfg = DEFAULT_CONFIG.copy()
    first_run = not os.path.exists(CONFIG_FILE)
    if not first_run:
        try:
            with open(CONFIG_FILE, "r") as f:
                saved = json.load(f)
            cfg.update(saved)
        except Exception:
            first_run = True  # corrupt config treated as first run
    return cfg, first_run


def save_config(cfg: dict):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"[Config] Could not save config: {e}")
