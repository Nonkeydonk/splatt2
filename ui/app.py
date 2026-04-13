"""
ui/app.py — Splatt2 main window.

Layout:
  Top bar    : title, session name, Sessions history button, status
  Left       : camera feed + camera selector + tracking quality
  Centre     : target canvas
  Right      : score panel OR series-complete editor (swappable)
  Bottom     : pause / zero / decimal / mic sensitivity slider / on-target indicator
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import queue
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageTk

from core.config import load_config, save_config, TARGETS, VERSION
from core.tracker import ArucoTracker, score_shot
from core.audio import AudioDetector
from core.session import (Session, Shot, ShotTrace,
                          load_session_history, reconstruct_shot_traces)
from core.target_renderer import TargetRenderer
from core.marker_sheet import generate_marker_sheet
from core.smoother import make_smoother

# - Added by RP: Text-to-speech for audio feedback on scoring. Optional dependency.
import pyttsx3

# ── Palette ───────────────────────────────────────────────────────────────────
BG_DARK  = "#0f0f13"
BG_MID   = "#16161d"
BG_PANEL = "#1c1c25"
BG_CARD  = "#22222e"
ACCENT   = "#00e5a0"
ACCENT2  = "#ff4f6d"
TEXT_PRI = "#f0f0f8"
TEXT_SEC = "#c0c0d8"
TEXT_DIM = "#9090b0"
BORDER   = "#2a2a3a"
GOLD     = "#ffd060"

FM = ("Consolas", 10)
FT = ("Segoe UI", 9, "bold")
FS = ("Consolas", 42, "bold")
FL = ("Segoe UI", 9)
FB = ("Segoe UI", 10)
FH = ("Segoe UI", 11)


def _mk_btn(parent, text, cmd, accent=False, width=None):
    fg = BG_DARK if accent else TEXT_SEC
    bg = ACCENT if accent else BG_CARD
    kw = dict(bg=bg, fg=fg, activebackground=ACCENT if accent else BORDER,
              activeforeground=fg, font=FB, relief="flat", bd=0,
              padx=8, pady=5, cursor="hand2", text=text, command=cmd)
    if width:
        kw["width"] = width
    return tk.Button(parent, **kw)


def _default_save_dir() -> str:
    """sessions/ subfolder in the project root (next to main.py)."""
    try:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, "sessions")
    except Exception:
        return os.path.join(os.path.expanduser("~"), "Documents", "Splatt2")


class SplattApp:
    def __init__(self):
        self.cfg, self._first_run = load_config()
        self.target_cfg = TARGETS[self.cfg["target_key"]]
        _card_r   = self.target_cfg["diameter_mm"] / 2.0
        _calibre  = float(self.cfg.get("scoring_calibre_mm",
                          self.target_cfg.get("calibre_mm", 4.5)))
        _R        = _card_r + _calibre / 2.0
        self.session = Session(
            name=self.cfg["session_name"],
            shots_per_series=self.cfg["shots_per_series"],
            scoring_radius_mm=_R,
        )
        self._apply_session_cfg()  # wire cfg into session at startup
        self.tracker = ArucoTracker(
            aruco_dict_name=self.cfg.get("aruco_dict", "DICT_4X4_50"),
            marker_size_mm=float(self.cfg.get("aruco_marker_mm", 40.0)),
            margin_mm=float(self.cfg.get("aruco_margin_mm", 8.0)),
            use_clahe=bool(self.cfg.get("use_clahe", True)),
            clahe_clip=float(self.cfg.get("clahe_clip", 4.0)),
            marker_count=int(self.cfg.get("aruco_marker_count", 4)),
            brightness_target=float(self.cfg.get("brightness_target", 128.0)),
        )
        self.target_renderer = None

        self.audio = AudioDetector(
            threshold=self.cfg.get("audio_trigger_threshold", 0.15),
            transient_ratio=self.cfg.get("audio_transient_ratio", 6.0),
            cooldown_ms=self.cfg.get("audio_trigger_cooldown_ms", 800),
            sample_rate=self.cfg.get("audio_sample_rate", 44100),
            chunk_size=512,          # ~12ms chunks — low latency, still enough for transient detection
            device_index=self.cfg.get("audio_device_index"),
            on_shot=self._on_shot_detected,
        )

        # State
        self._cap = None
        self._running = False
        self._paused = False
        self._shot_queue = queue.Queue()
        self._current_aim_mm  = None
        self._raw_aim_prev     = None   # previous raw position (pre-smoother)
        self._raw_aim_prev2    = None   # two frames ago (for velocity reversal)
        self._spike_hold       = None   # buffered position during spike candidate
        self._tracking_quality = 0.0
        self._zero_offset = (
            float(self.cfg.get("zero_offset_x", 0.0)),
            float(self.cfg.get("zero_offset_y", 0.0)),
        )
        self._zero_mode = False
        self._decimal_scoring = self.cfg.get("decimal_scoring", False)
        self._zoom_factor      = 1.0   # target canvas zoom (0.3–1.5)
        self._live_fps         = 0.0   # measured camera FPS
        self._sharpness        = 0.0   # Laplacian variance (focus measure)
        self._sharpness_peak   = 0.0   # peak hold
        self._sharpness_peak_t = 0.0   # time of peak
        self._show_acp = True
        self._show_bbox_shots = False
        self._show_bbox_acp = False
        self._show_group = False    # show group circle (blue) and MPI cross
        self._highlighted_trace: ShotTrace = None
        self._on_target_status = False
        self._in_approach_zone = False
        self._series_started = False
        self._post_shot_cooldown_s = float(self.cfg.get("post_shot_cooldown_s", 2.0))
        self._last_shot_fired_time: float = 0.0
        self._last_shot_info = None
        self._current_markers_found: int = 0   # updated every camera frame
        self._camera_rotation = int(self.cfg.get("camera_rotation", 0))
        self._fine_zero_mode = False   # click-on-canvas zero fine-tune
        self._shot_dot_only = False       # True=dot, False=full circle
        self._smoother = make_smoother(
            self.cfg.get("smooth_mode", "ema"),
            alpha=self.cfg.get("smooth_alpha", 0.35),
            window=self.cfg.get("smooth_window", 11),
            poly=self.cfg.get("smooth_poly", 2),
        )
        self._selected_shot: Shot = None   # for shot log selection
        self._latest_cam_frame = None

        # Series editor state
        self._in_series_editor = False
        self._editor_shot_vars = {}      # shot.index -> BooleanVar
        # BooleanVars must be created AFTER the root window exists
        self._editor_show_trace = None
        self._editor_show_acp   = None
        self._editor_show_dur   = None

        self._build_window()   # creates self.root
        # Start update loop — runs always, not just when camera active
        self.root.after(100, self._update_loop)
        # First-run wizard — shown after window is ready
        if self._first_run:
            self.root.after(300, self._show_first_run_wizard)
        # Now safe to create tk variables
        self._editor_show_trace = tk.BooleanVar(value=True)
        self._editor_show_acp   = tk.BooleanVar(value=True)
        self._editor_show_dur   = tk.BooleanVar(value=True)
        self._apply_styles()

    # =========================================================================
    # WINDOW BUILD
    # =========================================================================

    def _build_window(self):
        self.root = tk.Tk()
        self.root.title(f"SPLATT2 v{VERSION} — Target Shooting Trainer")
        self.root.configure(bg=BG_DARK)
        self.root.minsize(1200, 750)
        self.root.geometry("1440x840")

        # Top bar
        top = tk.Frame(self.root, bg=BG_MID, height=46)
        top.pack(fill="x", side="top")
        top.pack_propagate(False)
        tk.Label(top, text=f"◎  SPLATT2  v{VERSION}", bg=BG_MID, fg=ACCENT,
                 font=("Consolas", 15, "bold")).pack(side="left", padx=16, pady=10)
        self._session_lbl = tk.Label(top, text=f"Session: {self.session.name}",
                                     bg=BG_MID, fg=TEXT_SEC, font=FH)
        self._session_lbl.pack(side="left", padx=8)
        self._status_lbl = tk.Label(top, text="● READY", bg=BG_MID,
                                    fg=TEXT_SEC, font=("Consolas", 9))
        self._status_lbl.pack(side="right", padx=16)
        self._tracking_lbl = tk.Label(top, text="TRACKING: —", bg=BG_MID,
                                      fg=TEXT_DIM, font=("Consolas", 9))
        self._tracking_lbl.pack(side="right", padx=8)
        _mk_btn(top, "📋  Series", self._open_series_tab).pack(
            side="right", padx=4, pady=8)

        # Body
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=6, pady=(0, 4))

        # Left column (camera)
        self._left_col = tk.Frame(body, bg=BG_PANEL, width=400)
        self._left_col.pack(side="left", fill="y", padx=(0, 4))
        self._left_col.pack_propagate(False)
        self._build_camera_panel(self._left_col)

        # Right column (score panel or series editor — swappable)
        self._right_col = tk.Frame(body, bg=BG_PANEL, width=270)
        self._right_col.pack(side="right", fill="y", padx=(4, 0))
        self._right_col.pack_propagate(False)
        self._score_panel_frame = tk.Frame(self._right_col, bg=BG_PANEL)
        self._score_panel_frame.pack(fill="both", expand=True)
        self._build_score_panel(self._score_panel_frame)

        # Centre (target)
        centre = tk.Frame(body, bg=BG_PANEL)
        centre.pack(side="left", fill="both", expand=True, padx=4)
        self._build_target_panel(centre)

        # Bottom bar
        bottom = tk.Frame(self.root, bg=BG_MID, height=42)
        bottom.pack(fill="x", side="bottom")
        bottom.pack_propagate(False)
        self._build_bottom_bar(bottom)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<KeyPress>", self._on_key)

    # ── Camera panel ──────────────────────────────────────────────────────────
    def _build_camera_panel(self, parent):
        tk.Label(parent, text="CAMERA FEED", bg=BG_PANEL, fg=TEXT_DIM,
                 font=FL).pack(anchor="nw", padx=8, pady=(6, 0))

        self._cam_label = tk.Label(parent, bg=BG_DARK, text="No camera",
                                   fg=TEXT_DIM, font=FH)
        self._cam_label.pack(fill="both", expand=True, padx=6, pady=4)

        # Focus sharpness — toggle button + collapsible bar
        self._focus_active = False
        ff_btn = tk.Frame(parent, bg=BG_PANEL)
        ff_btn.pack(fill="x", padx=6, pady=(0, 2))
        self._btn_focus = tk.Button(ff_btn, text="◎ Focus assist: OFF",
                                     command=self._toggle_focus_assist,
                                     bg=BG_CARD, fg=TEXT_DIM, font=FL,
                                     relief="flat", padx=6, pady=2, cursor="hand2")
        self._btn_focus.pack(side="left")

        self._focus_frame = tk.Frame(parent, bg=BG_PANEL)
        # Not packed yet — shown only when active
        tk.Label(self._focus_frame, text="FOCUS", bg=BG_PANEL,
                 fg=TEXT_DIM, font=FL).pack(side="left")
        self._sharpness_var = tk.DoubleVar()
        ttk.Progressbar(self._focus_frame, variable=self._sharpness_var,
                        maximum=100, length=140,
                        style="Quality.Horizontal.TProgressbar").pack(side="left", padx=4)
        self._focus_lbl = tk.Label(self._focus_frame, text="—", bg=BG_PANEL,
                                    fg=ACCENT, font=("Consolas", 9), width=12)
        self._focus_lbl.pack(side="left")

        # Tracking quality bar
        qf = tk.Frame(parent, bg=BG_PANEL)
        qf.pack(fill="x", padx=6, pady=(0, 2))
        tk.Label(qf, text="TRACKING", bg=BG_PANEL, fg=TEXT_DIM, font=FL).pack(side="left")
        self._quality_var = tk.DoubleVar()
        ttk.Progressbar(qf, variable=self._quality_var, maximum=100, length=160,
                        style="Quality.Horizontal.TProgressbar").pack(side="left", padx=4)

        # Camera selector
        sf = tk.Frame(parent, bg=BG_PANEL)
        sf.pack(fill="x", padx=6, pady=(0, 2))
        tk.Label(sf, text="CAM", bg=BG_PANEL, fg=TEXT_DIM, font=FL).pack(side="left")
        self._cam_var = tk.StringVar()
        self._cam_combo = ttk.Combobox(sf, textvariable=self._cam_var,
                                        state="readonly", width=22, font=FL)
        self._cam_combo.pack(side="left", padx=4)
        self._cam_combo.bind("<<ComboboxSelected>>", self._on_cam_selected)
        _mk_btn(sf, "⟳", self._scan_cameras).pack(side="left")
        self.root.after(200, self._scan_cameras)

        # Camera controls
        cf = tk.Frame(parent, bg=BG_PANEL)
        cf.pack(fill="x", padx=6, pady=(0, 6))
        self._btn_cam = _mk_btn(cf, "▶  Start Camera", self._start_camera, accent=True)
        self._btn_cam.pack(side="left")
        _mk_btn(cf, "⚙  Settings", self._open_settings).pack(side="right")
        _mk_btn(cf, "🎛  Cam Props", self._open_camera_properties).pack(side="right", padx=4)

    # ── Target panel ──────────────────────────────────────────────────────────
    def _build_target_panel(self, parent):
        tk.Label(parent, text="TARGET", bg=BG_PANEL, fg=TEXT_DIM,
                 font=FL).pack(anchor="nw", padx=8, pady=(6, 0))

        cf = tk.Frame(parent, bg=BG_DARK)
        cf.pack(fill="both", expand=True, padx=6, pady=4)
        self._tgt_canvas = tk.Canvas(cf, bg=BG_DARK, highlightthickness=0, bd=0)
        self._tgt_canvas.pack(fill="both", expand=True)
        self._tgt_img_id = None
        self._tgt_canvas.create_text(200, 150,
            text="Start camera to begin tracking", fill=TEXT_DIM, font=FH, tags="ph")

        info = tk.Frame(parent, bg=BG_PANEL)
        info.pack(fill="x", padx=6, pady=(0, 4))
        self._last_shot_lbl = tk.Label(info, text="Last shot: —",
                                       bg=BG_PANEL, fg=TEXT_SEC, font=FM)
        self._last_shot_lbl.pack(side="left")
        self._aim_lbl = tk.Label(info, text="Aim: — mm",
                                 bg=BG_PANEL, fg=TEXT_DIM, font=FM)
        self._aim_lbl.pack(side="right")

    # ── Score panel ───────────────────────────────────────────────────────────
    def _build_score_panel(self, parent):
        # Score card
        sc = tk.Frame(parent, bg=BG_CARD)
        sc.pack(fill="x", padx=6, pady=(6, 3))
        tk.Label(sc, text="SERIES SCORE", bg=BG_CARD, fg=TEXT_DIM, font=FL
                 ).pack(anchor="nw", padx=8, pady=(6, 0))
        self._score_big = tk.Label(sc, text="0", bg=BG_CARD, fg=ACCENT, font=FS)
        self._score_big.pack()
        sub = tk.Frame(sc, bg=BG_CARD)
        sub.pack(fill="x", padx=8, pady=(0, 4))
        self._shots_lbl  = tk.Label(sub, text="0 shots", bg=BG_CARD,
                                    fg=TEXT_SEC, font=FB)
        self._shots_lbl.pack(side="left")
        self._avg_lbl = tk.Label(sub, text="Avg: —", bg=BG_CARD,
                                 fg=TEXT_SEC, font=FB)
        self._avg_lbl.pack(side="right")
        self._total_lbl = tk.Label(sc, text="Total: 0", bg=BG_CARD,
                                   fg=TEXT_SEC, font=FB)
        self._total_lbl.pack(anchor="e", padx=8, pady=(0, 6))

        # Stats card
        stc = tk.Frame(parent, bg=BG_CARD)
        stc.pack(fill="x", padx=6, pady=(0, 3))
        tk.Label(stc, text="STATISTICS", bg=BG_CARD, fg=TEXT_DIM,
                 font=FL).pack(anchor="nw", padx=8, pady=(6, 2))
        sf = tk.Frame(stc, bg=BG_CARD)
        sf.pack(fill="x", padx=8, pady=(0, 4))
        self._stat_labels = {}
        stat_rows = [
            ("mr",    "MR"),
            ("es",    "ES"),
            ("fom",   "FOM"),
            ("cep",   "CEP"),
            ("std_x", "Std X"),
            ("std_y", "Std Y"),
            ("mpi_x", "MPI X"),
            ("mpi_y", "MPI Y"),
            ("best",  "Best"),
            ("worst", "Worst"),
            ("bbox_s","Shot box"),
            ("bbox_a","ACP box"),
        ]
        for key, lbl in stat_rows:
            row = tk.Frame(sf, bg=BG_CARD)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=lbl, bg=BG_CARD, fg=TEXT_DIM,
                     font=FL, width=9, anchor="w").pack(side="left")
            v = tk.Label(row, text="—", bg=BG_CARD, fg=TEXT_SEC,
                         font=FM, anchor="e")
            v.pack(side="right")
            self._stat_labels[key] = v

        # Display toggles — two rows so nothing overflows 270px panel
        def _tog_state(text_on, text_off, state):
            return (text_on if state else text_off,
                    ACCENT if state else BG_CARD,
                    BG_DARK if state else TEXT_SEC)

        def _tbtn(parent, text_on, text_off, state, cmd):
            t, bg, fg = _tog_state(text_on, text_off, state)
            return tk.Button(parent, text=t, bg=bg, fg=fg, font=FL,
                             relief="flat", bd=0, padx=6, pady=4,
                             cursor="hand2", command=cmd,
                             activebackground=ACCENT, activeforeground=BG_DARK)

        # Row 1: ACP  Shots Box  ACP Box
        row1 = tk.Frame(parent, bg=BG_PANEL)
        row1.pack(fill="x", padx=6, pady=(2, 1))
        self._btn_acp = _tbtn(row1, "◈ ACP", "◈ ACP",
                               self._show_acp, self._toggle_acp)
        self._btn_acp.config(bg=ACCENT if self._show_acp else BG_CARD,
                              fg=BG_DARK if self._show_acp else TEXT_SEC)
        self._btn_acp.pack(side="left", padx=(0, 2))
        self._btn_bbox_s = _tbtn(row1, "⊡ Shots", "⊡ Shots",
                                  self._show_bbox_shots, self._toggle_bbox_shots)
        if self._show_bbox_shots:
            self._btn_bbox_s.config(bg=ACCENT, fg=BG_DARK)
        self._btn_bbox_s.pack(side="left", padx=(0, 2))
        self._btn_bbox_a = _tbtn(row1, "◇ ACP Box", "◇ ACP Box",
                                  self._show_bbox_acp, self._toggle_bbox_acp)
        if self._show_bbox_acp:
            self._btn_bbox_a.config(bg=ACCENT, fg=BG_DARK)
        self._btn_bbox_a.pack(side="left")

        # Row 2: Dot  Group circle
        row2 = tk.Frame(parent, bg=BG_PANEL)
        row2.pack(fill="x", padx=6, pady=(0, 3))
        self._btn_dot = _tbtn(row2, "● Dot", "● Dot",
                               self._shot_dot_only, self._toggle_dot_mode)
        if self._shot_dot_only:
            self._btn_dot.config(bg=ACCENT, fg=BG_DARK)
        self._btn_dot.pack(side="left", padx=(0, 2))
        self._btn_group = _tbtn(row2, "○ Group", "○ Group",
                                 self._show_group, self._toggle_group)
        if self._show_group:
            self._btn_group.config(bg=ACCENT, fg=BG_DARK)
        self._btn_group.pack(side="left")

        # Shot log
        lc = tk.Frame(parent, bg=BG_CARD)
        lc.pack(fill="both", expand=True, padx=6, pady=(0, 3))
        hdr = tk.Frame(lc, bg=BG_CARD)
        hdr.pack(fill="x", padx=8, pady=(6, 2))
        tk.Label(hdr, text="SHOT LOG", bg=BG_CARD, fg=TEXT_DIM, font=FL).pack(side="left")
        _mk_btn(hdr, "✕ Del", self._delete_selected_shot).pack(side="right")

        li = tk.Frame(lc, bg=BG_CARD)
        li.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self._shot_log = tk.Text(li, bg=BG_DARK, fg=TEXT_SEC, font=FM,
                                 relief="flat", state="disabled",
                                 height=10, wrap="none", cursor="hand2")
        sb = tk.Scrollbar(li, command=self._shot_log.yview,
                          bg=BG_DARK, troughcolor=BG_DARK)
        self._shot_log.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._shot_log.pack(fill="both", expand=True)
        self._shot_log.bind("<ButtonRelease-1>", self._on_shot_log_click)
        self._shot_log.bind("<Button-3>",         self._on_shot_log_rclick)
        for tag, col in [("ten", GOLD),("nine", ACCENT),("mid", TEXT_PRI),
                         ("low", TEXT_SEC),("miss", ACCENT2),
                         ("hdr", TEXT_DIM),("sel", ACCENT)]:
            self._shot_log.tag_config(tag, foreground=col)
        self._shot_log.tag_config("sel_bg", background=BG_CARD)

        # Action buttons
        bf = tk.Frame(parent, bg=BG_PANEL)
        bf.pack(fill="x", padx=6, pady=(0, 4))
        _mk_btn(bf, "⟲  Undo Shot",    self._undo_shot).pack(fill="x", pady=1)
        self._btn_start_series = _mk_btn(bf, "▶  Start Series",
                                              self._start_series, accent=True)
        self._btn_start_series.pack(fill="x", pady=1)
        _mk_btn(bf, "📋  Series Review", self._open_series_tab).pack(fill="x", pady=1)
        _mk_btn(bf, "◻  Reset All",    self._reset_all).pack(fill="x", pady=1)
        tk.Frame(bf, bg=BG_PANEL, height=3).pack()
        _mk_btn(bf, "⊞  Marker Sheet", self._print_markers).pack(fill="x", pady=1)
        _mk_btn(bf, "💾  Save CSV",     self._save_csv).pack(fill="x", pady=1)

    # ── Bottom bar ────────────────────────────────────────────────────────────
    def _build_bottom_bar(self, parent):
        _mk_btn(parent, "❙❙  Pause", self._toggle_pause).pack(
            side="left", padx=(8, 3), pady=5)
        self._btn_pause = parent.winfo_children()[-1]

        self._btn_zero = _mk_btn(parent, "◎  Zero", self._toggle_zero_mode)
        self._btn_zero.pack(side="left", padx=(0, 2), pady=5)
        self._btn_fine_zero = _mk_btn(parent, "⊕  Fine Zero",
                                       self._toggle_fine_zero_mode)
        self._btn_fine_zero.pack(side="left", padx=(0, 3), pady=5)

        self._btn_decimal = _mk_btn(
            parent, "DEC ON" if self._decimal_scoring else "DEC OFF",
            self._toggle_decimal_scoring)
        self._btn_decimal.pack(side="left", padx=(0, 8), pady=5)

        rot = self.cfg.get("camera_rotation", 0)
        self._btn_rotate = _mk_btn(
            parent, f"↻ {rot}°", self._cycle_rotation)
        self._btn_rotate.pack(side="left", padx=(0, 8), pady=5)

        # Zoom slider
        tk.Label(parent, text="ZOOM", bg=BG_MID, fg=TEXT_DIM,
                 font=FL).pack(side="left", padx=(4, 2))
        self._zoom_var = tk.DoubleVar(value=self._zoom_factor)
        ttk.Scale(parent, from_=0.3, to=1.5, variable=self._zoom_var,
                  orient="horizontal", length=80,
                  command=self._on_zoom_change).pack(side="left", pady=5)
        self._zoom_lbl = tk.Label(parent, text="1.00×", bg=BG_MID, fg=ACCENT,
                                   font=("Consolas", 9), width=5)
        self._zoom_lbl.pack(side="left", padx=(0, 6))

        # Mic section
        tk.Label(parent, text="MIC", bg=BG_MID, fg=TEXT_DIM,
                 font=FL).pack(side="left", padx=(6, 2))
        self._audio_var = tk.DoubleVar()
        ttk.Progressbar(parent, variable=self._audio_var, maximum=100,
                        length=90, style="Audio.Horizontal.TProgressbar"
                        ).pack(side="left", pady=5)

        # Threshold slider
        tk.Label(parent, text="THRESH", bg=BG_MID, fg=TEXT_DIM,
                 font=FL).pack(side="left", padx=(8, 2))
        self._thresh_var = tk.DoubleVar(
            value=self.cfg.get("audio_trigger_threshold", 0.15))
        thresh_slider = ttk.Scale(parent, from_=0.01, to=1.0,
                                  variable=self._thresh_var, orient="horizontal",
                                  length=100, command=self._on_thresh_change)
        thresh_slider.pack(side="left", pady=5)
        self._thresh_lbl = tk.Label(
            parent,
            text=f"{self.cfg.get('audio_trigger_threshold', 0.15):.2f}",
            bg=BG_MID, fg=ACCENT, font=("Consolas", 9), width=4)
        self._thresh_lbl.pack(side="left")

        # Transient ratio slider
        tk.Label(parent, text="SENS", bg=BG_MID, fg=TEXT_DIM,
                 font=FL).pack(side="left", padx=(8, 2))
        self._ratio_var = tk.DoubleVar(
            value=self.cfg.get("audio_transient_ratio", 6.0))
        ttk.Scale(parent, from_=1.5, to=15.0, variable=self._ratio_var,
                  orient="horizontal", length=80,
                  command=self._on_ratio_change).pack(side="left", pady=5)
        self._ratio_lbl = tk.Label(
            parent,
            text=f"{self.cfg.get('audio_transient_ratio', 6.0):.1f}x",
            bg=BG_MID, fg=ACCENT, font=("Consolas", 9), width=5)
        self._ratio_lbl.pack(side="left")

        # On-target indicator
        self._ontarget_lbl = tk.Label(parent, text="○ OFF", bg=BG_MID,
                                      fg=TEXT_DIM, font=("Consolas", 9))
        self._ontarget_lbl.pack(side="left", padx=8)

        tk.Label(parent, text="P=Pause  Spc=Undo  R=Reset  Q=Quit",
                 bg=BG_MID, fg=TEXT_DIM, font=FL).pack(side="right", padx=10)

    def _apply_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        for name, col in [("Quality", ACCENT), ("Audio", "#4f8fff")]:
            s.configure(f"{name}.Horizontal.TProgressbar",
                        troughcolor=BG_DARK, background=col,
                        bordercolor=BG_DARK, lightcolor=col, darkcolor=col)
        s.configure("TScale", background=BG_MID, troughcolor=BG_CARD)
        s.configure("TNotebook", background=BG_DARK, borderwidth=0)
        s.configure("TNotebook.Tab", background=BG_CARD, foreground=TEXT_SEC,
                    padding=[8, 3])
        s.map("TNotebook.Tab", background=[("selected", BG_PANEL)],
              foreground=[("selected", ACCENT)])

    # =========================================================================
    # CAMERA
    # =========================================================================

    def _scan_cameras(self):
        self._cam_combo.config(state="disabled")
        self._cam_combo["values"] = ["Scanning..."]
        self._cam_var.set("Scanning...")
        self.root.after(10, self._do_scan)

    def _do_scan(self):
        found = []
        for i in range(8):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                found.append((i, f"{i}: Camera {i}  ({w}x{h})"))
                cap.release()
            else:
                cap.release()
        if not found:
            found = [(i, f"{i}: Camera {i}") for i in range(4)]
        self._cam_entries = {lbl: idx for idx, lbl in found}
        labels = [lbl for _, lbl in found]
        self._cam_combo["values"] = labels
        self._cam_combo.config(state="readonly")
        cur = self.cfg.get("camera_index", 0)
        for lbl in labels:
            if lbl.startswith(str(cur) + ":"):
                self._cam_var.set(lbl)
                return
        if labels:
            self._cam_var.set(labels[0])

    def _on_cam_selected(self, event=None):
        lbl = self._cam_var.get()
        if hasattr(self, "_cam_entries") and lbl in self._cam_entries:
            self.cfg["camera_index"] = self._cam_entries[lbl]
            save_config(self.cfg)
            if self._running:
                self._stop_camera()
                self.root.after(400, self._start_camera)

    def _start_camera(self):
        if self._running:
            self._stop_camera()
            return
        idx = self.cfg.get("camera_index", 0)
        self._cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(idx)
        if not self._cap.isOpened():
            messagebox.showerror("Camera Error",
                                  f"Cannot open camera {idx}. Try another index.")
            return
        w = int(self.cfg.get("video_width", 640))
        h = int(self.cfg.get("video_height", 480))
        target_fps = int(self.cfg.get("video_fps", 30))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._cap.set(cv2.CAP_PROP_FPS, target_fps)
        # Pixel format — MJPEG prevents static-scene frame rate throttling
        _fmt = self.cfg.get("camera_pixel_format", "Auto")
        if _fmt == "MJPEG":
            _accepted = self._cap.set(cv2.CAP_PROP_FOURCC,
                                      cv2.VideoWriter.fourcc('M','J','P','G'))
            _rb = int(self._cap.get(cv2.CAP_PROP_FOURCC))
            _rb_str = ''.join([chr((_rb >> (8*i)) & 0xFF) for i in range(4)])
            print(f"[Camera] MJPEG requested — fourcc readback: {_rb_str!r}")
        elif _fmt == "YUY2":
            self._cap.set(cv2.CAP_PROP_FOURCC,
                          cv2.VideoWriter.fourcc('Y','U','Y','2'))
        # Minimise buffer — always get freshest frame, never queue up stale ones
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._camera_fps = actual_fps if actual_fps > 0 else target_fps
        self._running = True
        self._btn_cam.config(text="■  Stop Camera", bg=ACCENT2, fg=BG_DARK)
        self._set_status("LIVE", ACCENT)
        self.audio.start()
        threading.Thread(target=self._camera_loop, daemon=True).start()
        # _update_loop is already running (started at init); no need to restart

    def _stop_camera(self):
        self._running = False
        self.audio.stop()
        if self._cap:
            self._cap.release()
            self._cap = None
        self._btn_cam.config(text="▶  Start Camera", bg=BG_CARD, fg=TEXT_SEC)
        self._set_status("STOPPED", TEXT_DIM)



    def _camera_loop(self):
        """
        Optimised camera loop:
        - BUFFERSIZE=1 means cap.read() always returns the freshest frame
        - Frame is downscaled to max 480p for ArUco detection regardless of
          capture resolution — this is the single biggest speed gain
        - UI preview is only updated every N frames to save tkinter overhead
        - 'no_video_mode': skip annotated frame entirely, pure tracking only
        """
        no_video   = self.cfg.get("no_video_mode", False)
        ui_every   = 3        # send frame to UI every 3rd detection
        ui_counter = 0
        detect_w   = 640      # ArUco detection width (never larger than capture)
        detect_h   = 480

        # Live FPS measurement
        _fps_frame_count = 0
        _fps_last_time   = time.time()
        _fps_display     = 0.0

        while self._running:
            if not self._cap or not self._cap.isOpened():
                break
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.005)
                continue


            if self.cfg.get("flip_image"):
                frame = cv2.flip(frame, self.cfg.get("flip_mode", -1))

            # Camera rotation (0 / 90 / 180 / 270 degrees)
            rot = self._camera_rotation
            if rot == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rot == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Downscale for detection — major speed improvement
            fh, fw = frame.shape[:2]
            if fw > detect_w or fh > detect_h:
                scale = min(detect_w / fw, detect_h / fh)
                small = cv2.resize(frame,
                                   (int(fw * scale), int(fh * scale)),
                                   interpolation=cv2.INTER_LINEAR)
            else:
                small = frame

            # Laplacian variance — only computed when focus assist is on
            if self._focus_active:
                _lap = cv2.Laplacian(small if len(small.shape)==2
                                     else cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
                                     cv2.CV_64F)
                self._sharpness = float(_lap.var())

            result = self.tracker.process_frame(small)
            self._tracking_quality = result.quality
            self._current_markers_found = result.markers_found

            if result.aim_mm is not None and not self._paused:
                raw = result.aim_mm
                zeroed = (raw[0] - self._zero_offset[0],
                          raw[1] - self._zero_offset[1])
                if result.quality >= 0.25:
                    # ── Velocity spike filter (raw positions, pre-smoother) ──
                    # Detects bad homography spikes: a large jump that immediately
                    # reverses. Genuine fast movement (recoil, swing) continues
                    # in the same direction — it does not sharply reverse.
                    filtered = zeroed
                    if self._raw_aim_prev is not None:
                        import math as _m
                        spk_vel  = float(self.cfg.get("spike_velocity_mm", 25.0))
                        spk_rev  = float(self.cfg.get("spike_reversal", 0.7))
                        px, py   = self._raw_aim_prev
                        vx = zeroed[0] - px
                        vy = zeroed[1] - py
                        speed = _m.sqrt(vx*vx + vy*vy)
                        if speed > spk_vel and self._raw_aim_prev2 is not None:
                            # Check if previous frame also had a large jump
                            # and this frame reverses it
                            p2x, p2y = self._raw_aim_prev2
                            v2x = px - p2x
                            v2y = py - p2y
                            dot = vx*v2x + vy*v2y
                            mag2 = _m.sqrt(v2x*v2x + v2y*v2y)
                            if mag2 > spk_vel and dot < -spk_rev * speed * mag2:
                                # Sharp reversal detected — this frame is the
                                # return from a spike. Discard both spike frames.
                                filtered = self._raw_aim_prev2
                                self._raw_aim_prev  = self._raw_aim_prev2
                    self._raw_aim_prev2 = self._raw_aim_prev
                    self._raw_aim_prev  = zeroed
                    smoothed = self._smoother.update(filtered)
                    self._current_aim_mm = smoothed
                    in_appr, on_tgt = self.session.update_aim(smoothed)
                    self._in_approach_zone = in_appr
                    self._on_target_status = on_tgt

            # Measure actual delivered FPS
            _fps_frame_count += 1
            _fps_now = time.time()
            _fps_elapsed = _fps_now - _fps_last_time
            if _fps_elapsed >= 0.5:   # update every half second
                _fps_display     = _fps_frame_count / _fps_elapsed
                _fps_frame_count = 0
                _fps_last_time   = _fps_now
            self._live_fps = _fps_display
            # Update sharpness peak hold (only when focus assist is on)
            if self._focus_active:
                _now = time.time()
                if self._sharpness >= self._sharpness_peak:
                    self._sharpness_peak   = self._sharpness
                    self._sharpness_peak_t = _now
                elif _now - self._sharpness_peak_t > 3.0:
                    self._sharpness_peak = self._sharpness

            # Only update UI preview every N frames and not in no_video mode
            ui_counter += 1
            if not no_video and ui_counter >= ui_every:
                # Overlay FPS on the camera frame
                disp = result.frame_display.copy() if result.frame_display is not None else None
                if disp is not None:
                    fps_txt = f"{_fps_display:.1f} fps"
                    cv2.putText(disp, fps_txt, (6, 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (80, 220, 80), 1, cv2.LINE_AA)
                    self._latest_cam_frame = disp
                ui_counter = 0

            try:
                shot_ts = self._shot_queue.get_nowait()
                if self._current_aim_mm is not None:
                    if self._zero_mode:
                        if self._tracking_quality > 0:
                            # Markers visible — apply zero
                            self._apply_zero(self._current_aim_mm)
                        else:
                            # No markers — ignore trigger, stay in zero mode
                            self.root.after(0, lambda: self._set_status(
                                "ZERO MODE — no markers, try again", GOLD))
                    elif self._tracking_quality > 0:
                        # quality > 0 means we have a valid homography
                        # Pass shot_ts so record_shot can retroactively look up
                        # the camera position at the exact audio trigger moment
                        self._register_shot(self._current_aim_mm, shot_ts)
                    else:
                        # quality == 0: no homography at all, camera lost
                        self.root.after(0, lambda: self._set_status(
                            "SHOT REJECTED — no tracking", ACCENT2))
            except queue.Empty:
                pass

    def _on_shot_detected(self, ts):
        if self._paused:
            return
        # Post-shot cooldown: ignore audio triggers for N seconds after a shot
        if ts - self._last_shot_fired_time < self._post_shot_cooldown_s:
            return
        self._shot_queue.put(ts)

    def _scoring_radius_mm(self) -> float:
        """
        R = card_radius + pellet_radius.
        This is the single number that drives all scoring geometry.
        Changing calibre in settings instantly changes all scoring bands.
        """
        card_r    = self.target_cfg["diameter_mm"] / 2.0
        calibre   = float(self.cfg.get("scoring_calibre_mm",
                          self.target_cfg.get("calibre_mm", 4.5)))
        return card_r + calibre / 2.0

    def _register_shot(self, aim_mm, shot_ts: float = None):
        if not self._series_started:
            return
        R            = self._scoring_radius_mm()
        mark_offsets = self.target_cfg.get("mark_offsets")  # None for single-mark

        # Step 1: record the shot — this does the retroactive position lookup,
        # correcting aim_mm to where the crosshair actually was at shot_ts.
        # We pass a placeholder score of 0.0 and rescore after using the
        # corrected position, so scoring and drawing use identical coordinates.
        shot = self.session.record_shot(aim_mm, 0.0, 0,
                                         shot_timestamp=shot_ts,
                                         mark_index=0,
                                         defer_write=True)
        if shot is None:
            self.root.after(0, lambda: self._set_status(
                "SHOT REJECTED — outside approach zone", ACCENT2))
            return

        # Step 2: score from the retroactively corrected position
        score, ring, mark_idx = score_shot(shot.aim_mm, R,
                                            decimal=self._decimal_scoring,
                                            mark_offsets=mark_offsets)

        # Step 3: update the shot with the correct score and mark
        shot.score      = score
        shot.ring_index = ring
        shot.mark_index = mark_idx

        def _speek():
            pyttsx3.speak(f"{score:.1f}, {shot.clock_position} o'clock" if self._decimal_scoring else f"{int(score)} at {shot.clock_position} o'clock")

        if self.cfg.get("voice_enabled", True):
            threading.Thread(target=_speek).start()

        # Step 4: rewrite the live CSV row with the correct score
        if self.session._writer and self.session._writer.is_open:
            self.session._writer.write_shot(shot)

        if score == 0 and self.cfg.get("ignore_misses", False):
            self.session.shots.remove(shot)
            self.root.after(0, lambda: self._set_status(
                "Miss ignored (score 0)", TEXT_DIM))
            return
        if shot is None:
            self.root.after(0, lambda: self._set_status(
                "SHOT REJECTED — outside approach zone", ACCENT2))
            return
        import time as _t
        self._last_shot_fired_time = _t.time()
        self._last_shot_info = shot
        sc = shot.score
        sc_s = f"{sc:.1f}" if sc != int(sc) else str(int(sc))
        col = GOLD if sc >= 10 else (ACCENT if sc >= 9 else
              TEXT_PRI if sc >= 7 else ACCENT2)
        _lbl = (f"Shot #{shot.index} (mark {mark_idx+1}): {sc_s} pts"
                if mark_offsets else f"Shot #{shot.index}: {sc_s} pts")
        self.root.after(0, lambda lbl=_lbl, c=col: self._set_status(lbl, c))


    def _open_camera_properties(self):
        """Open Windows DirectShow camera properties dialog.
        Adjust brightness, contrast, saturation, sharpness, hue,
        gamma, white balance. Start the camera first.
        """
        if self._cap is None or not self._cap.isOpened():
            messagebox.showinfo("Camera Properties",
                "Start the camera first, then open Camera Properties.")
            return
        try:
            self._cap.set(cv2.CAP_PROP_SETTINGS, 1)
        except Exception as e:
            messagebox.showerror("Camera Properties",
                f"Could not open properties dialog:\n{e}")

    def _apply_zero(self, aim_mm):
        self._zero_offset = (self._zero_offset[0] + aim_mm[0],
                              self._zero_offset[1] + aim_mm[1])
        self._zero_mode = False
        self.session.active_trace = ShotTrace()
        self.session._in_approach_zone = False
        self._smoother.reset()
        self._last_shot_fired_time = time.time()
        self._save_zero_offset()
        self.root.after(0, lambda: (
            self._btn_zero.config(text="◎  Zero", bg=BG_CARD, fg=TEXT_SEC),
            self._set_status("ZEROED", ACCENT)
        ))

    # =========================================================================
    # UPDATE LOOP
    # =========================================================================

    def _update_loop(self):
        # Target display and scores run always (even without camera)
        # so colour changes / shot edits reflect immediately
        self._update_target_display()
        self._update_scores()
        if self._running:
            self._update_cam_display()
            self._update_audio_meter()
        self.root.after(33, self._update_loop)

    def _update_cam_display(self):
        frame = self._latest_cam_frame
        if frame is None:
            return
        lw = max(10, self._cam_label.winfo_width())
        lh = max(10, self._cam_label.winfo_height())
        fh, fw = frame.shape[:2]
        sc = min(lw / fw, lh / fh)
        nw, nh = max(1, int(fw * sc)), max(1, int(fh * sc))
        disp = cv2.resize(frame, (nw, nh))
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        self._cam_label.config(image=img, text="")
        self._cam_label._img = img

        q = int(self._tracking_quality * 100)
        self._quality_var.set(q)
        col = ACCENT if q > 60 else (GOLD if q > 30 else ACCENT2)
        self._tracking_lbl.config(text=f"TRACKING: {q}%", fg=col)
        if self._current_aim_mm:
            x, y = self._current_aim_mm
            self._aim_lbl.config(text=f"Aim: ({x:+.1f}, {y:+.1f}) mm")

        # Sharpness / focus bar (only when active)
        if self._focus_active and hasattr(self, "_sharpness_var") and self._sharpness > 0:
            # Normalise: Laplacian variance is unbounded; use peak as 100%
            # Cap at 5× the rolling peak so the bar doesn't stay pegged at 0
            peak = max(self._sharpness_peak, 1.0)
            norm = min(100.0, self._sharpness / peak * 100.0)
            self._sharpness_var.set(norm)
            # Colour: green when near peak, amber when dropping, dim when far off
            at_peak = (self._sharpness >= self._sharpness_peak * 0.97)
            near    = (self._sharpness >= self._sharpness_peak * 0.90)
            fc = ACCENT if at_peak else (GOLD if near else TEXT_DIM)
            peak_txt = "PEAK" if at_peak else f"pk:{self._sharpness_peak:.0f}"
            self._focus_lbl.config(
                text=f"{norm:4.0f}% {peak_txt}", fg=fc)

    def _update_target_display(self):
        fw = self._tgt_canvas.winfo_width()
        fh = self._tgt_canvas.winfo_height()
        if fw < 50 or fh < 50:
            return

        if self.target_renderer is None or \
           (self.target_renderer.cw, self.target_renderer.ch) != (fw, fh):
            cal = float(self.cfg.get("shot_circle_calibre_mm",
                              self.target_cfg.get("calibre_mm", 4.5)))
            self.target_renderer = TargetRenderer((fw, fh), self.target_cfg,
                                                   display_calibre_mm=cal,
                                                   display_cfg=self._make_display_cfg(),
                                                   zoom=self._zoom_factor)
            self._tgt_canvas.delete("ph")

        fading = self.session.get_fading_trace()
        fade_age = self.session.fading_age_s

        # In editor mode pass only selected shots
        if self._in_series_editor and self._editor_show_trace is not None:
            sel_shots = [s for s in self.session.shots
                         if self._editor_shot_vars.get(s.index,
                             tk.BooleanVar(value=True)).get()]
            show_tr  = self._editor_show_trace.get()
            show_acp = self._editor_show_acp.get()
        else:
            sel_shots = self.session.shots
            show_tr   = True
            show_acp  = self._show_acp

        img = self.target_renderer.render(
            shots=sel_shots,
            active_trace=self.session.active_trace if not self._paused else None,
            fading_trace=fading,
            fading_age_s=fade_age,
            live_aim_mm=self._current_aim_mm if not self._paused else None,
            show_mpi=self._show_group,
            show_group=self._show_group,
            current_series=self.session.current_series,
            zero_mode=self._zero_mode,
            show_acp=show_acp,
            show_traces=show_tr,
            highlighted_shot_trace=self._highlighted_trace,
            show_bbox_shots=self._show_bbox_shots,
            show_bbox_acp=self._show_bbox_acp,
            show_dot_only=self._shot_dot_only,
        )
        photo = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        if self._tgt_img_id is None:
            self._tgt_img_id = self._tgt_canvas.create_image(0, 0, anchor="nw",
                                                               image=photo)
        else:
            self._tgt_canvas.itemconfig(self._tgt_img_id, image=photo)
        self._tgt_canvas._img = photo

    def _update_scores(self):
        ss   = self.session.series_shots
        ssm  = self.session.series_match_shots
        n    = len(ss)
        nm   = len(ssm)
        series_score = self.session.series_score
        avg  = self.session.series_avg

        sc_str = (f"{series_score:.1f}"
                  if series_score != int(series_score)
                  else str(int(series_score)))
        self._score_big.config(text=sc_str)
        sighter_str = f" ({n-nm}S)" if n > nm else ""
        self._shots_lbl.config(
            text=f"{nm} match{sighter_str}")
        self._avg_lbl.config(text=f"Avg: {avg:.2f}" if avg else "Avg: —")
        self._total_lbl.config(text=f"Total: {self.session.total_score:.1f}")

        def _f(v, fmt=".2f", unit="mm"):
            return f"{v:{fmt}}{unit}" if v is not None else "—"

        ses = self.session
        self._stat_labels["mr"   ].config(text=_f(ses.mean_radius_mm))
        self._stat_labels["es"   ].config(text=_f(ses.extreme_spread_mm))
        self._stat_labels["fom"  ].config(text=_f(ses.figure_of_merit_mm))
        self._stat_labels["cep"  ].config(text=_f(ses.cep_mm))
        self._stat_labels["std_x"].config(text=_f(ses.std_x_mm))
        self._stat_labels["std_y"].config(text=_f(ses.std_y_mm))
        mpi = ses.mean_point_of_impact
        self._stat_labels["mpi_x"].config(text=f"{mpi[0]:+.2f}mm" if mpi else "—")
        self._stat_labels["mpi_y"].config(text=f"{mpi[1]:+.2f}mm" if mpi else "—")
        b  = ses.best_shot
        w  = ses.worst_shot
        self._stat_labels["best" ].config(
            text=f"#{b.index} {b.score:.1f}" if b else "—",
            fg=GOLD if b else TEXT_SEC)
        self._stat_labels["worst"].config(
            text=f"#{w.index} {w.score:.1f}" if w else "—",
            fg=ACCENT2 if w else TEXT_SEC)
        bbox = ses.bbox_shots_mm
        self._stat_labels["bbox_s"].config(
            text=f"{bbox[0]:.1f}×{bbox[1]:.1f}mm" if bbox and self._show_bbox_shots else "—")
        acps = [s.aim_centrepoint for s in ss if s.aim_centrepoint]
        if acps and self._show_bbox_acp:
            ax = [a[0] for a in acps]; ay = [a[1] for a in acps]
            self._stat_labels["bbox_a"].config(
                text=f"{max(ax)-min(ax):.1f}×{max(ay)-min(ay):.1f}mm")
        else:
            self._stat_labels["bbox_a"].config(text="—")

        self._refresh_shot_log()

        # On-target indicator
        if self._on_target_status:
            self._ontarget_lbl.config(text="● ON TARGET", fg=ACCENT)
        elif self._in_approach_zone:
            self._ontarget_lbl.config(text="◉ APPROACH", fg=GOLD)
        else:
            self._ontarget_lbl.config(text="○ OFF", fg=TEXT_DIM)

        # Last shot
        last = getattr(self, "_last_shot_info", None)
        if last and isinstance(last, Shot):
            sc = last.score
            sc_str = f"{sc:.1f}" if sc != int(sc) else str(int(sc))
            acp = last.aim_centrepoint
            acp_s = f" ACP({acp[0]:+.1f},{acp[1]:+.1f})" if acp else ""
            ot_s  = f" {last.on_target_duration_s:.1f}s"
            self._last_shot_lbl.config(
                text=f"#{last.index} {sc_str}pts ({last.aim_mm[0]:+.1f},{last.aim_mm[1]:+.1f}){acp_s}{ot_s}",
                fg=GOLD if sc >= 9 else (ACCENT if sc >= 7 else TEXT_SEC))

    def _refresh_shot_log(self):
        log = self._shot_log
        log.config(state="normal")
        log.delete("1.0", "end")
        log.insert("end", f"{'#':>3}  {'Sc':>5}  {'X':>6}  {'Y':>6}  {'T':>4}  Flg\n", "hdr")
        log.insert("end", "─" * 38 + "\n", "hdr")
        shots_rev = list(reversed(self.session.shots[-40:]))
        for shot in shots_rev:
            if shot.deleted:
                continue   # don't show deleted shots in log
            sc   = shot.score
            sc_s = f"{sc:.1f}" if sc != int(sc) else str(int(sc))
            ot   = f"{shot.on_target_duration_s:.1f}"
            # Build flag string
            flags = ""
            if not shot.match_shot: flags += "S"   # S = Sighter
            if shot.favourite:      flags += "★"
            if shot.missed:         flags += "M"
            if shot.comments:       flags += "✎"
            line = (f"{shot.index:>3}  {sc_s:>5}  "
                    f"{shot.aim_mm[0]:>+6.1f}  {shot.aim_mm[1]:>+6.1f}  "
                    f"{ot:>4}  {flags}\n")
            tag  = ("ten"  if sc >= 10 else "nine" if sc >= 9 else
                    "mid"  if sc >= 7  else "low"  if sc > 0  else "miss")
            if self._selected_shot and self._selected_shot.index == shot.index:
                log.insert("end", line, ("sel", "sel_bg"))
            else:
                log.insert("end", line, tag)
        log.config(state="disabled")

    def _update_audio_meter(self):
        level = min(1.0, self.audio.current_level * 12)
        self._audio_var.set(level * 100)

    # =========================================================================
    # CONTROLS
    # =========================================================================

    def _on_exp_mode(self):
        """Auto exposure checkbox changed — apply live."""
        if hasattr(self, '_exp_auto_var'):
            self.cfg["camera_auto_exposure"] = self._exp_auto_var.get()

    def _toggle_focus_assist(self):
        """Toggle focus sharpness meter on/off."""
        self._focus_active = not self._focus_active
        if self._focus_active:
            # Reset peak so it builds fresh from current focus position
            self._sharpness_peak   = 0.0
            self._sharpness_peak_t = 0.0
            self._sharpness        = 0.0
            self._focus_frame.pack(fill="x", padx=6, pady=(0, 2),
                                   after=self._btn_focus.master)
            self._btn_focus.config(text="◎ Focus assist: ON", fg=ACCENT, bg=BG_MID)
        else:
            self._focus_frame.pack_forget()
            self._sharpness_var.set(0)
            self._focus_lbl.config(text="—", fg=TEXT_DIM)
            self._btn_focus.config(text="◎ Focus assist: OFF", fg=TEXT_DIM, bg=BG_CARD)

    def _on_zoom_change(self, val=None):
        """Zoom slider changed — rebuild renderer at new scale."""
        self._zoom_factor = round(float(self._zoom_var.get()), 2)
        if hasattr(self, "_zoom_lbl"):
            self._zoom_lbl.config(text=f"{self._zoom_factor:.2f}×")
        self.target_renderer = None   # force rebuild on next frame

    def _on_thresh_change(self, val=None):
        v = self._thresh_var.get()
        self.audio.set_threshold(v)
        self.cfg["audio_trigger_threshold"] = v
        self._thresh_lbl.config(text=f"{v:.2f}")

    def _on_ratio_change(self, val=None):
        v = self._ratio_var.get()
        self.audio.set_transient_ratio(v)
        self.cfg["audio_transient_ratio"] = v
        self._ratio_lbl.config(text=f"{v:.1f}x")

    def _toggle_pause(self):
        self._paused = not self._paused
        self.audio.pause(self._paused)
        btn = self._btn_pause
        if self._paused:
            btn.config(text="▶  Resume", bg=GOLD, fg=BG_DARK)
            self._set_status("PAUSED", GOLD)
        else:
            btn.config(text="❙❙  Pause", bg=BG_CARD, fg=TEXT_SEC)
            self._set_status("LIVE" if self._running else "READY", ACCENT)

    def _save_zero_offset(self):
        """Persist zero offset to config file so it survives restarts."""
        self.cfg["zero_offset_x"] = self._zero_offset[0]
        self.cfg["zero_offset_y"] = self._zero_offset[1]
        save_config(self.cfg)

    def _toggle_zero_mode(self):
        if not self._running:
            messagebox.showinfo("Zero", "Start the camera first.")
            return
        self._zero_mode = not self._zero_mode
        if self._zero_mode:
            self._btn_zero.config(text="◎  Waiting…", bg=GOLD, fg=BG_DARK)
            self._set_status("ZERO MODE — fire one shot", GOLD)
            # Cancel fine-zero if active
            if self._fine_zero_mode:
                self._cancel_fine_zero()
        else:
            self._btn_zero.config(text="◎  Zero", bg=BG_CARD, fg=TEXT_SEC)
            self._set_status("LIVE" if self._running else "READY", ACCENT)

    def _toggle_fine_zero_mode(self):
        """Enter/exit fine-zero mode: next click on target canvas adjusts offset."""
        if self._fine_zero_mode:
            self._cancel_fine_zero()
            return
        self._fine_zero_mode = True
        self._btn_fine_zero.config(
            text="⊕  Click centre…", bg=GOLD, fg=BG_DARK)
        self._set_status(
            "FINE ZERO — click your shot group centre on the target", GOLD)
        # Cancel normal zero mode if active
        if self._zero_mode:
            self._zero_mode = False
            self._btn_zero.config(text="◎  Zero", bg=BG_CARD, fg=TEXT_SEC)
        # Bind click on target canvas
        self._tgt_canvas.bind("<Button-1>", self._on_fine_zero_click)
        self._tgt_canvas.config(cursor="crosshair")

    def _cancel_fine_zero(self):
        self._fine_zero_mode = False
        self._btn_fine_zero.config(
            text="⊕  Fine Zero", bg=BG_CARD, fg=TEXT_SEC)
        self._tgt_canvas.unbind("<Button-1>")
        self._tgt_canvas.config(cursor="")
        self._set_status("LIVE" if self._running else "READY", ACCENT)

    def _on_fine_zero_click(self, event):
        """Convert canvas click to mm offset and add to _zero_offset."""
        if not self._fine_zero_mode or self.target_renderer is None:
            self._cancel_fine_zero()
            return
        r = self.target_renderer
        # Inverse of mm_to_px: mm = (px - centre) / scale
        clicked_mm_x = (event.x - r.cx) / r.scale
        clicked_mm_y = (event.y - r.cy) / r.scale
        # Add this offset to the current zero offset
        # (clicked point should now be treated as centre)
        self._zero_offset = (
            self._zero_offset[0] + clicked_mm_x,
            self._zero_offset[1] + clicked_mm_y,
        )
        self._save_zero_offset()
        self._cancel_fine_zero()
        self._set_status(
            f"Fine zero applied: ({clicked_mm_x:+.1f}, {clicked_mm_y:+.1f}) mm",
            ACCENT)
        self._smoother.reset()

    def _toggle_decimal_scoring(self):
        self._decimal_scoring = not self._decimal_scoring
        self.cfg["decimal_scoring"] = self._decimal_scoring
        save_config(self.cfg)
        self._btn_decimal.config(
            text="DEC ON"  if self._decimal_scoring else "DEC OFF",
            bg=ACCENT if self._decimal_scoring else BG_CARD,
            fg=BG_DARK if self._decimal_scoring else TEXT_SEC)

    def _cycle_rotation(self):
        """Cycle camera rotation: 0 → 90 → 180 → 270 → 0."""
        self._camera_rotation = (self._camera_rotation + 90) % 360
        self.cfg["camera_rotation"] = self._camera_rotation
        save_config(self.cfg)
        if hasattr(self, "_btn_rotate"):
            self._btn_rotate.config(text=f"↻ {self._camera_rotation}°",
                                    bg=BG_CARD if self._camera_rotation == 0 else ACCENT,
                                    fg=TEXT_SEC if self._camera_rotation == 0 else BG_DARK)

    def _toggle_acp(self):
        self._show_acp = not self._show_acp
        self._btn_acp.config(bg=ACCENT if self._show_acp else BG_CARD,
                              fg=BG_DARK if self._show_acp else TEXT_SEC)

    def _toggle_bbox_shots(self):
        self._show_bbox_shots = not self._show_bbox_shots
        self._btn_bbox_s.config(bg=ACCENT if self._show_bbox_shots else BG_CARD,
                                 fg=BG_DARK if self._show_bbox_shots else TEXT_SEC)

    def _toggle_bbox_acp(self):
        self._show_bbox_acp = not self._show_bbox_acp
        self._btn_bbox_a.config(bg=ACCENT if self._show_bbox_acp else BG_CARD,
                                 fg=BG_DARK if self._show_bbox_acp else TEXT_SEC)

    def _toggle_dot_mode(self):
        self._shot_dot_only = not self._shot_dot_only
        self._btn_dot.config(bg=ACCENT if self._shot_dot_only else BG_CARD,
                              fg=BG_DARK if self._shot_dot_only else TEXT_SEC)

    def _toggle_group(self):
        self._show_group = not self._show_group
        self._btn_group.config(bg=ACCENT if self._show_group else BG_CARD,
                                fg=BG_DARK if self._show_group else TEXT_SEC)

    def _undo_shot(self):
        shot = self.session.undo_last_shot()
        if shot:
            self._last_shot_info = None
            self._selected_shot = None
            self._highlighted_trace = None
            self._last_shot_lbl.config(text=f"Undid shot #{shot.index}", fg=GOLD)

    def _delete_selected_shot(self):
        if not self._selected_shot:
            messagebox.showinfo("Delete", "Click a shot in the log first.")
            return
        shot = self._selected_shot
        if messagebox.askyesno("Delete Shot",
                                f"Delete shot #{shot.index} "
                                f"(score {shot.score})?"):
            try:
                self.session.shots.remove(shot)
            except ValueError:
                pass
            self._selected_shot = None
            self._highlighted_trace = None

    def _start_series(self):
        """Open the live CSV file and begin accepting shots."""
        # If already active, this is a second press — treat as toggle off
        if self._series_started:
            if messagebox.askyesno("Stop Series",
                    "Series is active. Stop recording and finish?"):
                self.session.end_series()
                self._series_started = False
                self._btn_start_series.config(
                    text="▶  Start Series", bg=ACCENT, fg=BG_DARK)
                self._set_status("Series stopped — file saved", GOLD)
            return

        # Resolve save directory — config value or default sessions folder
        save_dir = (self.cfg.get("save_directory") or "").strip()
        if not save_dir:
            save_dir = _default_save_dir()
        save_dir = os.path.abspath(save_dir)

        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Save Directory Error",
                f"Cannot create save directory:\n{save_dir}\n\n{e}\n\n"
                "Change the Save Directory in Settings.")
            return

        try:
            path = self.session.start_series(save_dir)
        except Exception as e:
            messagebox.showerror("Series Start Error",
                f"Could not open session file:\n{e}")
            return

        self._series_started = True
        self._btn_start_series.config(
            text="■  Series Active — click to stop",
            bg=ACCENT2, fg=BG_DARK)
        sn = self.session.current_series
        self._set_status(f"● REC  Series {sn} — recording to {os.path.basename(path)}", ACCENT2)

    def _new_series(self):
        self.session.clear_series()
        self._series_started = False
        if hasattr(self, '_btn_start_series'):
            self._btn_start_series.config(
                text="▶  Start Series", bg=ACCENT, fg=BG_DARK)
        self._selected_shot = None
        self._highlighted_trace = None
        self._set_status("READY — press Start Series", TEXT_SEC)

    def _reset_all(self):
        if messagebox.askyesno("Reset", "Clear all shots and restart?"):
            self.session.reset()
            self.target_renderer = None
            self._last_shot_info = None
            self._selected_shot = None
            self._highlighted_trace = None
            # zero offset intentionally preserved — reset via Settings
            self._zero_mode = False
            self._series_started = False
            self._in_approach_zone = False
            self._in_series_editor = False
            self._btn_zero.config(text="◎  Zero", bg=BG_CARD, fg=TEXT_SEC)
            if hasattr(self, '_btn_start_series'):
                self._btn_start_series.config(
                    text="▶  Start Series", bg=ACCENT, fg=BG_DARK)
            self._last_shot_lbl.config(text="Last shot: —", fg=TEXT_SEC)
            self._set_status("READY — press Start Series", TEXT_SEC)
            self._exit_series_editor()

    def _print_markers(self):
        MarkerSheetDialog(self.root, self.cfg)

    def _save_csv(self):
        if not self.session.shots:
            messagebox.showinfo("Save CSV", "No shots yet.")
            return
        out = filedialog.asksaveasfilename(
            title="Save shot data", defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
            initialfile=f"{self.session.name}_shots.csv")
        if out:
            self.session.save_csv(out)
            messagebox.showinfo("Saved", f"Saved to:\n{out}")

    def _open_settings(self):
        SettingsDialog(self.root, self.cfg, self._apply_settings)

    def _apply_settings(self, new_cfg):
        cam_keys = {"video_width", "video_height", "video_fps",
                    "camera_index", "no_video_mode"}
        cam_changed = any(new_cfg.get(k) != self.cfg.get(k) for k in cam_keys)

        self.cfg.update(new_cfg)
        save_config(self.cfg)
        self.target_cfg = TARGETS[self.cfg["target_key"]]
        self.target_renderer = None

        # Audio — applies live
        self.audio.set_threshold(self.cfg["audio_trigger_threshold"])
        self.audio.set_transient_ratio(self.cfg.get("audio_transient_ratio", 6.0))
        self.audio.set_cooldown(self.cfg["audio_trigger_cooldown_ms"])
        self._thresh_var.set(self.cfg["audio_trigger_threshold"])
        self._ratio_var.set(self.cfg.get("audio_transient_ratio", 6.0))

        # Cooldown — live
        self._post_shot_cooldown_s = float(self.cfg.get("post_shot_cooldown_s", 2.0))

        # Smoother — live, no restart needed
        self._smoother = make_smoother(
            self.cfg.get("smooth_mode", "ema"),
            alpha=float(self.cfg.get("smooth_alpha", 0.35)),
            window=int(self.cfg.get("smooth_window", 11)),
            poly=int(self.cfg.get("smooth_poly", 2)),
        )
        self._smoother.reset()

        # Rebuild tracker if marker size changed
        marker_keys = {"aruco_marker_mm", "aruco_margin_mm", "aruco_dict", "use_clahe", "clahe_clip", "aruco_marker_count", "brightness_target"}
        if any(new_cfg.get(k) != self.cfg.get(k) for k in marker_keys):
            self.tracker = ArucoTracker(
                aruco_dict_name=self.cfg["aruco_dict"],
                marker_size_mm=float(self.cfg.get("aruco_marker_mm", 40.0)),
                margin_mm=float(self.cfg.get("aruco_margin_mm", 8.0)),
                use_clahe=bool(self.cfg.get("use_clahe", True)),
                clahe_clip=float(self.cfg.get("clahe_clip", 4.0)),
                marker_count=int(self.cfg.get("aruco_marker_count", 4)),
                brightness_target=float(self.cfg.get("brightness_target", 128.0)),
            )

        # Rotation — applies immediately to camera loop
        self._camera_rotation = int(self.cfg.get("camera_rotation", 0))
        self._fine_zero_mode = False   # click-on-canvas zero fine-tune
        if hasattr(self, "_btn_rotate"):
            rot = self._camera_rotation
            self._btn_rotate.config(text=f"↻ {rot}°",
                bg=BG_CARD if rot == 0 else ACCENT,
                fg=TEXT_SEC if rot == 0 else BG_DARK)

        # Sync zero offset from config (reset via Settings takes effect now)
        self._zero_offset = (
            float(self.cfg.get("zero_offset_x", 0.0)),
            float(self.cfg.get("zero_offset_y", 0.0)),
        )
        self._apply_session_cfg()  # push new cfg values into live session

        # Camera resolution/fps — restart to take effect
        if cam_changed and self._running:
            self._set_status("Restarting camera with new settings…", GOLD)
            self._stop_camera()
            self.root.after(600, self._start_camera)

    def _apply_session_cfg(self):
        """Push config values into the live session object."""
        self.session.fading_trace_duration_s = float(
            self.cfg.get("fading_trace_duration_s", 2.0))
        self.session.acp_fraction = float(
            self.cfg.get("acp_fraction", 0.40))
        from core.session import APPROACH_ZONE_FACTOR
        factor = float(self.cfg.get("approach_zone_factor",
                                     APPROACH_ZONE_FACTOR))
        R = self._scoring_radius_mm()
        self.session.scoring_radius_mm = R
        # For multi-mark targets, the approach zone must cover the furthest mark
        mark_offsets = self.target_cfg.get("mark_offsets")
        if mark_offsets:
            import math as _m
            max_mark_r = max(_m.sqrt(mx**2 + my**2) for mx, my in mark_offsets)
            self.session.approach_radius_mm = (R + max_mark_r) * factor
        else:
            self.session.approach_radius_mm = R * factor

    def _make_display_cfg(self) -> dict:
        """Extract display-relevant keys from cfg for TargetRenderer."""
        keys = ["colour_trace_approach","colour_trace_hold",
                "colour_trace_preshot","colour_trace_final",
                "colour_shot_fill","colour_acp","colour_crosshair",
                "colour_mpi","colour_group","colour_miss",
                "trace_width","fading_trace_duration_s",
                "trace_preshot_s","trace_final_s"]
        return {k: self.cfg[k] for k in keys if k in self.cfg}

    def _open_series_tab(self):
        """Open the Series Review window for the current session."""
        # End series recording if still active
        if self._series_started:
            self.session.end_series()
            self._series_started = False
            if hasattr(self, '_btn_start_series'):
                self._btn_start_series.config(
                    text="▶  Start Series", bg=ACCENT, fg=BG_DARK)
        SeriesReviewWindow(self.root, self.session, self.cfg,
                           self.target_cfg,
                           on_next_series=self._start_next_series,
                           on_close_refresh=self._on_series_review_close)

    def _on_series_review_close(self):
        """Called when Series Review window closes — refresh display."""
        self.target_renderer = None   # force re-render

    def _set_status(self, text, color=TEXT_SEC):
        self._status_lbl.config(text=f"● {text}", fg=color)

    # ── Shot log interactions ─────────────────────────────────────────────────
    def _shot_at_log_line(self, event):
        try:
            line_no = int(self._shot_log.index(
                f"@{event.x},{event.y}").split(".")[0])
            idx = line_no - 3   # 2 header lines, 1-indexed
            shots_rev = list(reversed(self.session.shots[-40:]))
            if 0 <= idx < len(shots_rev):
                return shots_rev[idx]
        except Exception:
            pass
        return None

    def _on_shot_log_click(self, event):
        shot = self._shot_at_log_line(event)
        if shot:
            if self._selected_shot and self._selected_shot.index == shot.index:
                # Deselect
                self._selected_shot = None
                self._highlighted_trace = None
            else:
                self._selected_shot = shot
                self._highlighted_trace = shot.trace

    def _on_shot_log_rclick(self, event):
        shot = self._shot_at_log_line(event)
        if not shot:
            return
        m = tk.Menu(self.root, tearoff=0, bg=BG_CARD, fg=TEXT_SEC,
                    activebackground=ACCENT, activeforeground=BG_DARK,
                    font=("Segoe UI", 9))
        sc = shot.score
        sc_s = f"{sc:.1f}" if sc != int(sc) else str(int(sc))
        m.add_command(label=f"Shot #{shot.index}  —  {sc_s} pts  "
                            f"({shot.aim_mm[0]:+.1f}, {shot.aim_mm[1]:+.1f}) mm",
                      state="disabled")
        m.add_separator()

        # View trace
        m.add_command(label="🔍  View trace",
                      command=lambda s=shot: self._highlight_shot(s))
        m.add_separator()

        # Flag toggles — checkmark shows current state
        fav_lbl  = "★  Unfavourite" if shot.favourite else "☆  Mark favourite"
        mtch_lbl = "●  Mark as sighter" if shot.match_shot else "●  Mark as match shot"
        miss_lbl = "✗  Unmark missed" if shot.missed else "✗  Mark as missed"

        m.add_command(label=fav_lbl,
                      command=lambda s=shot: self._toggle_flag(s, "favourite"))
        m.add_command(label=mtch_lbl,
                      command=lambda s=shot: self._toggle_flag(s, "match_shot"))
        m.add_command(label=miss_lbl,
                      command=lambda s=shot: self._toggle_flag(s, "missed"))
        m.add_separator()

        # Comment
        m.add_command(label="💬  Add comment…",
                      command=lambda s=shot: self._edit_shot_comment(s))
        m.add_separator()

        # Delete
        m.add_command(label="🗑  Delete shot",
                      command=lambda s=shot: self._delete_shot(s),
                      foreground=ACCENT2)
        m.tk_popup(event.x_root, event.y_root)

    def _highlight_shot(self, shot):
        self._selected_shot = shot
        self._highlighted_trace = shot.trace

    def _toggle_flag(self, shot, flag: str):
        """Toggle a boolean flag on a shot (favourite, match_shot, missed)."""
        current = getattr(shot, flag)
        setattr(shot, flag, not current)
        # If toggling match_shot, update the live CSV row (best effort)
        # Stats will refresh automatically on next _update_scores call

    def _edit_shot_comment(self, shot):
        """Open a small dialog to add/edit a comment on a shot."""
        dlg = tk.Toplevel(self.root)
        dlg.title(f"Comment — Shot #{shot.index}")
        dlg.configure(bg=BG_DARK)
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.geometry("340x120")

        tk.Label(dlg, text=f"Shot #{shot.index}  ({shot.score} pts)",
                 bg=BG_DARK, fg=TEXT_SEC, font=("Segoe UI", 10)).pack(
                     anchor="nw", padx=12, pady=(10, 4))
        var = tk.StringVar(value=shot.comments or "")
        tk.Entry(dlg, textvariable=var, width=38, bg=BG_CARD,
                 fg=TEXT_PRI, insertbackground=ACCENT,
                 relief="flat", font=FM).pack(padx=12, pady=4, fill="x")

        bf = tk.Frame(dlg, bg=BG_DARK)
        bf.pack(fill="x", padx=12, pady=(0, 10))
        def _save():
            shot.comments = var.get().strip()
            dlg.destroy()
        tk.Button(bf, text="Save", command=_save,
                  bg=ACCENT, fg=BG_DARK, font=("Segoe UI", 9),
                  relief="flat", padx=10, pady=4).pack(side="right", padx=4)
        tk.Button(bf, text="Cancel", command=dlg.destroy,
                  bg=BG_CARD, fg=TEXT_SEC, font=("Segoe UI", 9),
                  relief="flat", padx=10, pady=4).pack(side="right")

    def _delete_shot(self, shot):
        if messagebox.askyesno("Delete", f"Delete shot #{shot.index}?"):
            try:
                self.session.shots.remove(shot)
            except ValueError:
                pass
            if self._selected_shot and self._selected_shot.index == shot.index:
                self._selected_shot = None
                self._highlighted_trace = None

    # =========================================================================
    # SERIES COMPLETE EDITOR
    # =========================================================================

    def _series_complete(self):
        """Finish shooting, save, enter review mode."""
        # End live file (also writes JSON)
        self.session.end_series()
        self._series_started = False
        if hasattr(self, '_btn_start_series'):
            self._btn_start_series.config(
                text="▶  Start Series", bg=ACCENT, fg=BG_DARK)
        # Enter review mode
        self._in_series_editor = True
        for w in self._score_panel_frame.winfo_children():
            w.destroy()
        self._build_series_editor(self._score_panel_frame)
        self._set_status(f"REVIEW — Series {self.session.current_series}", GOLD)

    def _start_next_series(self):
        """From review mode: clear target and start a new series."""
        self._in_series_editor = False
        # Advance series counter — shots from previous series are kept in memory
        # but the target display will only show current series
        self.session.clear_series()
        self._series_started = False
        self._selected_shot = None
        self._highlighted_trace = None
        self._last_shot_fired_time = 0.0
        for w in self._score_panel_frame.winfo_children():
            w.destroy()
        self._build_score_panel(self._score_panel_frame)
        self._set_status("READY — press Start Series", TEXT_SEC)
        if hasattr(self, '_btn_start_series'):
            self._btn_start_series.config(
                text="▶  Start Series", bg=ACCENT, fg=BG_DARK)

    def _exit_series_editor(self):
        self._in_series_editor = False
        for w in self._score_panel_frame.winfo_children():
            w.destroy()
        self._build_score_panel(self._score_panel_frame)

    def _build_series_editor(self, parent):
        tk.Label(parent, text="SERIES EDITOR", bg=BG_PANEL, fg=ACCENT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="nw", padx=8, pady=(6, 2))

        # Data toggles
        tog = tk.Frame(parent, bg=BG_PANEL)
        tog.pack(fill="x", padx=6, pady=(0, 4))

        def _tog_btn(p, text, var, col=ACCENT):
            def cb():
                var.set(not var.get())
                btn.config(bg=col if var.get() else BG_CARD,
                           fg=BG_DARK if var.get() else TEXT_SEC)
            btn = _mk_btn(p, text, cb)
            btn.config(bg=col if var.get() else BG_CARD,
                       fg=BG_DARK if var.get() else TEXT_SEC)
            return btn

        _tog_btn(tog, "Trace",   self._editor_show_trace).pack(side="left", padx=(0,2))
        _tog_btn(tog, "ACP",     self._editor_show_acp,  col="#4f8fff").pack(side="left", padx=(0,2))
        _tog_btn(tog, "Dur",     self._editor_show_dur,  col=GOLD).pack(side="left")

        # Shot checkboxes with scrollable list
        tk.Label(parent, text="Select shots to display:",
                 bg=BG_PANEL, fg=TEXT_DIM, font=FL).pack(anchor="nw", padx=8)

        cb_outer = tk.Frame(parent, bg=BG_PANEL)
        cb_outer.pack(fill="both", expand=True, padx=6, pady=(0, 4))
        canvas = tk.Canvas(cb_outer, bg=BG_DARK, highlightthickness=0, bd=0)
        vsb = tk.Scrollbar(cb_outer, orient="vertical", command=canvas.yview,
                           bg=BG_DARK, troughcolor=BG_DARK)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=BG_DARK)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        self._editor_shot_vars = {}

        # Select all / none buttons
        sel_row = tk.Frame(inner, bg=BG_DARK)
        sel_row.pack(fill="x", pady=(2, 4))
        _mk_btn(sel_row, "All",  lambda: self._editor_select_all(True)).pack(side="left", padx=2)
        _mk_btn(sel_row, "None", lambda: self._editor_select_all(False)).pack(side="left", padx=2)

        for shot in self.session.series_shots:
            var = tk.BooleanVar(value=True)
            self._editor_shot_vars[shot.index] = var
            row = tk.Frame(inner, bg=BG_DARK)
            row.pack(fill="x", pady=1)
            sc = shot.score
            sc_s = f"{sc:.1f}" if sc != int(sc) else str(int(sc))
            col = GOLD if sc >= 10 else (ACCENT if sc >= 9 else
                  TEXT_PRI if sc >= 7 else TEXT_SEC)
            cb_text = f"#{shot.index:>2} {sc_s:>5}  ({shot.aim_mm[0]:>+5.1f},{shot.aim_mm[1]:>+5.1f})"
            if self._editor_show_dur.get():
                cb_text += f"  {shot.on_target_duration_s:.1f}s"
            chk = tk.Checkbutton(row, text=cb_text, variable=var,
                                  bg=BG_DARK, fg=col, selectcolor=BG_CARD,
                                  activebackground=BG_DARK, font=FM,
                                  anchor="w")
            chk.pack(fill="x")
            # Right-click to delete
            chk.bind("<Button-3>", lambda e, s=shot: self._delete_shot(s))

        def _on_inner_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(win_id, width=canvas.winfo_width())
        inner.bind("<Configure>", _on_inner_configure)

        # Stats summary
        stats_frame = tk.Frame(parent, bg=BG_CARD)
        stats_frame.pack(fill="x", padx=6, pady=(0, 4))
        tk.Label(stats_frame, text="SERIES STATS", bg=BG_CARD, fg=TEXT_DIM,
                 font=FL).pack(anchor="nw", padx=8, pady=(4, 2))
        ss = self.session.series_shots
        n = len(ss)
        total = sum(s.score for s in ss)
        avg = total / n if n else 0
        ot_vals = [s.on_target_duration_s for s in ss if s.on_target_duration_s > 0]
        avg_ot = sum(ot_vals) / len(ot_vals) if ot_vals else 0
        for label, val in [
            ("Shots", str(n)),
            ("Total", f"{total:.1f}"),
            ("Average", f"{avg:.2f}"),
            ("Avg on-target", f"{avg_ot:.1f}s"),
        ]:
            r = tk.Frame(stats_frame, bg=BG_CARD)
            r.pack(fill="x", padx=8, pady=1)
            tk.Label(r, text=label, bg=BG_CARD, fg=TEXT_DIM,
                     font=FL, width=14, anchor="w").pack(side="left")
            tk.Label(r, text=val, bg=BG_CARD, fg=TEXT_PRI,
                     font=FM, anchor="e").pack(side="right")
        tk.Frame(stats_frame, bg=BG_CARD, height=4).pack()

        # Action buttons
        bf = tk.Frame(parent, bg=BG_PANEL)
        bf.pack(fill="x", padx=6, pady=(0, 6))
        _mk_btn(bf, "💾  Save CSV",        self._save_csv).pack(fill="x", pady=1)
        _mk_btn(bf, "▶  Next Series",     self._start_next_series,
                accent=True).pack(fill="x", pady=1)
        _mk_btn(bf, "✕  Exit Editor",     self._exit_series_editor
                ).pack(fill="x", pady=1)

    def _editor_select_all(self, value: bool):
        for var in self._editor_shot_vars.values():
            var.set(value)

    # =========================================================================
    # KEYBOARD
    # =========================================================================

    def _on_key(self, event):
        k = event.keysym.lower()
        if k == "p":         self._toggle_pause()
        elif k == "space":   self._undo_shot()
        elif k == "r":       self._reset_all()
        elif k == "q":       self._on_close()
        elif k == "s":       self._save_csv()

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def _on_close(self):
        self._running = False
        self.audio.stop()
        if self._cap:
            self._cap.release()
        # Close live writer first
        self.session.end_series()
        # Save full JSON archive
        if self.session.shots:
            save_dir = (self.cfg.get("save_directory") or "").strip() or _default_save_dir()
            save_dir = os.path.abspath(save_dir)
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(save_dir,
                                  f"archive_{ts}_{self.session.name}.json")
            try:
                self.session.save_json(fname)
            except Exception as e:
                print(f"[AutoSave] {e}")
        save_config(self.cfg)
        self.root.destroy()

    def _show_first_run_wizard(self):
        """Simple first-run setup wizard shown when no config file exists."""
        from core.config import TARGETS, save_config
        dlg = tk.Toplevel(self.root)
        dlg.title("Welcome to Splatt2!")
        dlg.configure(bg=BG_DARK)
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.geometry("480x400")

        tk.Label(dlg, text="Welcome to Splatt2", bg=BG_DARK, fg=ACCENT,
                 font=("Segoe UI", 16, "bold")).pack(pady=(20, 4))
        tk.Label(dlg, text="Let's get you set up in three quick steps.",
                 bg=BG_DARK, fg=TEXT_SEC, font=("Segoe UI", 10)).pack(pady=(0, 16))

        body = tk.Frame(dlg, bg=BG_DARK)
        body.pack(fill="x", padx=32)

        def row(label):
            f = tk.Frame(body, bg=BG_DARK)
            f.pack(fill="x", pady=6)
            tk.Label(f, text=label, bg=BG_DARK, fg=TEXT_SEC,
                     font=("Segoe UI", 10, "bold"), width=20, anchor="w").pack(side="left")
            return f

        # Step 1: target type
        r1 = row("1.  Target type:")
        target_var = tk.StringVar(value="10m_air_rifle")
        ttk.Combobox(r1, textvariable=target_var,
                     values=list(TARGETS.keys()),
                     state="readonly", width=24,
                     font=("Segoe UI", 9)).pack(side="left")

        # Step 2: shooting distance
        r2 = row("2.  Shooting distance:")
        dist_var = tk.StringVar(value="10")
        tk.Entry(r2, textvariable=dist_var, width=6, bg=BG_CARD, fg=TEXT_PRI,
                 insertbackground=ACCENT, relief="flat",
                 font=("Segoe UI", 10)).pack(side="left")
        tk.Label(r2, text=" metres", bg=BG_DARK, fg=TEXT_DIM,
                 font=("Segoe UI", 9)).pack(side="left")

        # Step 3: shooter name (optional)
        r3 = row("3.  Your name (optional):")
        name_var = tk.StringVar()
        tk.Entry(r3, textvariable=name_var, width=20, bg=BG_CARD, fg=TEXT_PRI,
                 insertbackground=ACCENT, relief="flat",
                 font=("Segoe UI", 10)).pack(side="left")

        tk.Label(dlg,
                 text="You can change all of these later in Settings.",
                 bg=BG_DARK, fg=TEXT_DIM, font=("Segoe UI", 9)).pack(pady=(16, 0))

        def _done():
            try:
                dist = float(dist_var.get())
            except ValueError:
                dist = 10.0
            self.cfg["target_key"] = target_var.get()
            self.cfg["real_range_m"] = dist
            name = name_var.get().strip()
            if name:
                self.cfg["shooter_name"] = name
                self.cfg["session_name"] = name
            self.target_cfg = TARGETS[self.cfg["target_key"]]
            self.target_renderer = None
            save_config(self.cfg)
            dlg.destroy()
            self._set_status("Setup complete — print your marker sheet and start the camera!", ACCENT)

        tk.Button(dlg, text="Let's go  ▶", command=_done,
                  bg=ACCENT, fg=BG_DARK, font=("Segoe UI", 11, "bold"),
                  relief="flat", padx=20, pady=8, cursor="hand2").pack(pady=20)

        # Prevent closing without completing
        dlg.protocol("WM_DELETE_WINDOW", _done)

    def run(self):
        self.root.mainloop()


# =============================================================================
# SETTINGS DIALOG
# =============================================================================

class MarkerSheetDialog(tk.Toplevel):
    def __init__(self, parent, cfg):
        super().__init__(parent)
        self.cfg = cfg
        self.title("Marker Sheet & Target Creator")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)
        self.grab_set()
        self._build()
        self.geometry("500x520")

    def _build(self):
        from core.marker_sheet import AIMING_MARKS, A4_W_MM
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        tab1 = tk.Frame(nb, bg=BG_DARK)
        nb.add(tab1, text="Print Marker Sheet")
        self._build_sheet_tab(tab1)

        tab2 = tk.Frame(nb, bg=BG_DARK)
        nb.add(tab2, text="Target Creator")
        self._build_creator_tab(tab2)

        pad = {"padx": 14, "pady": 5}  # keep for legacy refs

    def _build_sheet_tab(self, parent):
        from core.marker_sheet import AIMING_MARKS, A4_W_MM
        pad = {"padx": 14, "pady": 5}

        tk.Label(parent, text="Generate Marker Sheet", bg=BG_DARK, fg=ACCENT,
                 font=("Segoe UI", 12, "bold")).pack(anchor="nw", **pad)

        r1 = tk.Frame(parent, bg=BG_DARK); r1.pack(fill="x", **pad)
        tk.Label(r1, text="Target type:", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=22, anchor="w").pack(side="left")
        self._tvar = tk.StringVar(value=self.cfg.get("target_key", "10m_air_rifle"))
        cb = ttk.Combobox(r1, textvariable=self._tvar,
                           values=list(AIMING_MARKS.keys()),
                           state="readonly", width=26, font=FL)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", self._preview)

        r2 = tk.Frame(parent, bg=BG_DARK); r2.pack(fill="x", **pad)
        tk.Label(r2, text="Print distance (m):", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=22, anchor="w").pack(side="left")
        self._dvar = tk.StringVar(value=str(self.cfg.get("real_range_m", 10.0)))
        tk.Entry(r2, textvariable=self._dvar, width=7, bg=BG_CARD, fg=TEXT_PRI,
                 insertbackground=ACCENT, relief="flat", font=FM).pack(side="left")
        self._dvar.trace_add("write", lambda *_: self._preview())

        r4 = tk.Frame(parent, bg=BG_DARK); r4.pack(fill="x", **pad)
        tk.Label(r4, text="Marker size (mm):", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=22, anchor="w").pack(side="left")
        self._mvar = tk.StringVar(value=str(self.cfg.get("aruco_marker_mm", 40.0)))
        tk.Entry(r4, textvariable=self._mvar, width=6, bg=BG_CARD, fg=TEXT_PRI,
                 insertbackground=ACCENT, relief="flat", font=FM).pack(side="left")
        for sz in [25, 35, 40, 50]:
            tk.Button(r4, text=str(sz),
                      command=lambda v=sz: (self._mvar.set(str(v)), self._preview()),
                      bg=BG_CARD, fg=TEXT_DIM, font=FL, relief="flat",
                      padx=4, pady=2, cursor="hand2").pack(side="left", padx=2)
        self._mvar.trace_add("write", lambda *_: self._preview())

        r5 = tk.Frame(parent, bg=BG_DARK); r5.pack(fill="x", **pad)
        tk.Label(r5, text="Margin (mm):", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=22, anchor="w").pack(side="left")
        self._mgvar = tk.StringVar(value=str(self.cfg.get("aruco_margin_mm", 8.0)))
        tk.Entry(r5, textvariable=self._mgvar, width=6, bg=BG_CARD, fg=TEXT_PRI,
                 insertbackground=ACCENT, relief="flat", font=FM).pack(side="left")
        self._mgvar.trace_add("write", lambda *_: self._preview())

        r7 = tk.Frame(parent, bg=BG_DARK); r7.pack(fill="x", **pad)
        tk.Label(r7, text="Pellet calibre (mm):", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=22, anchor="w").pack(side="left")
        self._sheet_calibre = tk.StringVar(
            value=str(self.cfg.get("scoring_calibre_mm", 4.5)))
        for label, val in [("4.5 (.177)", "4.5"), ("5.6 (.22)", "5.6")]:
            tk.Button(r7, text=label,
                      command=lambda v=val: (self._sheet_calibre.set(v),
                                            self._preview()),
                      bg=BG_CARD, fg=TEXT_DIM, font=FL, relief="flat",
                      padx=4, pady=2, cursor="hand2").pack(side="left", padx=2)
        tk.Entry(r7, textvariable=self._sheet_calibre, width=5, bg=BG_CARD,
                 fg=TEXT_PRI, insertbackground=ACCENT, relief="flat",
                 font=FM).pack(side="left", padx=(4,0))

        r6 = tk.Frame(parent, bg=BG_DARK); r6.pack(fill="x", **pad)
        tk.Label(r6, text="ArUco dictionary:", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=22, anchor="w").pack(side="left")
        self._dictvar = tk.StringVar(value=self.cfg.get("aruco_dict", "DICT_4X4_50"))
        ttk.Combobox(r6, textvariable=self._dictvar,
                     values=["DICT_4X4_50","DICT_4X4_100",
                             "DICT_5X5_50","DICT_5X5_100",
                             "DICT_6X6_50","DICT_6X6_100"],
                     state="readonly", width=18, font=FL).pack(side="left")
        tk.Label(r6, text="  (must match Settings → Camera)",
                 bg=BG_DARK, fg=TEXT_DIM, font=FL).pack(side="left", padx=4)

        r8 = tk.Frame(parent, bg=BG_DARK); r8.pack(fill="x", **pad)
        tk.Label(r8, text="Marker count:", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=22, anchor="w").pack(side="left")
        self._sheet_marker_count = tk.StringVar(
            value=str(self.cfg.get("aruco_marker_count", 4)))
        ttk.Combobox(r8, textvariable=self._sheet_marker_count,
                     values=["4", "6", "8"], state="readonly",
                     width=6, font=FL).pack(side="left")
        tk.Label(r8, text="  must match Settings → Camera → Marker count",
                 bg=BG_DARK, fg=TEXT_DIM, font=FL).pack(side="left", padx=4)

        r3 = tk.Frame(parent, bg=BG_DARK); r3.pack(fill="x", **pad)
        tk.Label(r3, text="Show ring guides:", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=22, anchor="w").pack(side="left")
        self._rvar = tk.BooleanVar(value=True)
        tk.Checkbutton(r3, variable=self._rvar, bg=BG_DARK, selectcolor=BG_CARD,
                       command=self._preview).pack(side="left")

        self._info = tk.Label(parent, text="", bg=BG_CARD, fg=TEXT_PRI,
                               font=FM, justify="left", anchor="nw",
                               padx=12, pady=8)
        self._info.pack(fill="x", padx=14, pady=8)

        bf = tk.Frame(parent, bg=BG_DARK); bf.pack(fill="x", padx=14, pady=(4, 14))
        tk.Button(bf, text="Generate & Open", command=self._generate,
                  bg=ACCENT, fg=BG_DARK, font=FB, relief="flat",
                  padx=12, pady=6).pack(side="right", padx=4)
        tk.Button(bf, text="Cancel", command=self.destroy,
                  bg=BG_CARD, fg=TEXT_SEC, font=FB, relief="flat",
                  padx=12, pady=6).pack(side="right", padx=4)
        self._preview()

    def _get(self):
        from core.marker_sheet import AIMING_MARKS
        key  = self._tvar.get()
        mark = AIMING_MARKS.get(key, AIMING_MARKS["10m_air_rifle"])
        try:    dist = float(self._dvar.get())
        except: dist = mark["reference_dist_m"]
        try:    marker_sz = max(10.0, float(self._mvar.get()))
        except: marker_sz = 40.0
        try:    margin_sz = max(3.0,  float(self._mgvar.get()))
        except: margin_sz = 8.0
        return key, mark, dist, dist / mark["reference_dist_m"], marker_sz, margin_sz

    def _preview(self, *_):
        from core.marker_sheet import A4_W_MM
        key, mark, dist, scale, marker_sz, margin_sz = self._get()
        # Centre space = A4 width - 2 margins - 2 markers (left+right)
        centre_space = A4_W_MM - 2 * margin_sz - 2 * marker_sz
        outer_scaled = mark["outer_ring_dia_mm"] * scale
        fits = "✓ fits" if centre_space > outer_scaled else "✗ too small — reduce marker size"
        warn = "" if centre_space > outer_scaled else f"  ← need {outer_scaled:.0f}mm"
        self._info.config(text=(
            f"Target        : {mark['name']}\n"
            f"Distance      : {dist:.1f}m  (ref {mark['reference_dist_m']:.0f}m)  Scale: {scale*100:.0f}%\n"
            f"Aiming mark   : {mark['aiming_mark_dia_mm']*scale:.1f}mm  dia\n"
            f"Scoring area  : {outer_scaled:.1f}mm  dia\n"
            f"Marker size   : {marker_sz:.0f}mm  ×4 corners\n"
            f"Centre space  : {centre_space:.0f}mm  {fits}{warn}\n"
            f"Markers take  : {2*marker_sz + 2*margin_sz:.0f}mm  of {A4_W_MM:.0f}mm width"
        ))

    def _generate(self):
        key, mark, dist, scale, _, _ = self._get()
        out = filedialog.asksaveasfilename(
            title="Save marker sheet", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All", "*.*")],
            initialfile=f"splatt2_{key}_{dist:.0f}m.png")
        if not out:
            return
        from core.marker_sheet import generate_marker_sheet
        _, _, _, _, marker_sz, margin_sz = self._get()
        # Save marker/margin back to cfg so tracker stays in sync
        self.cfg["aruco_marker_mm"] = marker_sz
        self.cfg["aruco_margin_mm"] = margin_sz
        from core.config import save_config
        save_config(self.cfg)
        generate_marker_sheet(out, target_key=key, print_distance_m=dist,
                               show_ring_guides=self._rvar.get(),
                               aruco_dict_name=self._dictvar.get(),
                               marker_size_mm=marker_sz,
                               margin_mm=margin_sz,
                               marker_count=int(self._sheet_marker_count.get()))
        try:
            os.startfile(out)
        except Exception:
            pass
        self.destroy()

    # ── Target Creator tab ────────────────────────────────────────────────────

    def _build_creator_tab(self, parent):
        """Target editor — create, edit and delete targets from the targets/ folder."""
        from core.config import TARGETS, _targets_dir
        pad = {"padx": 10, "pady": 3}

        # ── Mode selector: New or edit existing ──────────────────────────────
        top = tk.Frame(parent, bg=BG_DARK); top.pack(fill="x", **pad)
        tk.Label(top, text="Mode:", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=8, anchor="w").pack(side="left")
        self._c_mode = tk.StringVar(value="New target")
        modes = ["New target"] + [f"Edit: {t['name']}" for t in TARGETS.values()]
        self._c_mode_combo = ttk.Combobox(top, textvariable=self._c_mode,
                                           values=modes, state="readonly",
                                           width=30, font=FL)
        self._c_mode_combo.pack(side="left", padx=(0,6))
        self._c_mode_combo.bind("<<ComboboxSelected>>", self._on_creator_mode)
        tk.Button(top, text="Delete", command=self._delete_target,
                  bg=ACCENT2, fg=BG_DARK, font=FL, relief="flat",
                  padx=6, pady=3, cursor="hand2").pack(side="left")

        # ── Metadata fields ───────────────────────────────────────────────────
        meta = tk.Frame(parent, bg=BG_DARK); meta.pack(fill="x", **pad)

        def _field(frame, label, var, width=20):
            r = tk.Frame(frame, bg=BG_DARK); r.pack(fill="x", pady=1)
            tk.Label(r, text=label, bg=BG_DARK, fg=TEXT_SEC, font=FB,
                     width=20, anchor="w").pack(side="left")
            tk.Entry(r, textvariable=var, width=width, bg=BG_CARD, fg=TEXT_PRI,
                     insertbackground=ACCENT, relief="flat",
                     font=FM).pack(side="left")
            return r

        self._c_key     = tk.StringVar()
        self._c_name    = tk.StringVar()
        self._c_calibre = tk.StringVar(value="4.5")
        self._c_dist    = tk.StringVar(value="10")
        self._c_aiming  = tk.StringVar(value="")
        self._c_card    = tk.StringVar(value="")

        _field(meta, "Key (no spaces):", self._c_key)
        _field(meta, "Display name:", self._c_name)

        r_g = tk.Frame(meta, bg=BG_DARK); r_g.pack(fill="x", pady=1)
        tk.Label(r_g, text="Gauging:", bg=BG_DARK, fg=TEXT_SEC, font=FB,
                 width=20, anchor="w").pack(side="left")
        self._c_gauging = tk.StringVar(value="outward")
        ttk.Combobox(r_g, textvariable=self._c_gauging,
                     values=["outward","inward"], state="readonly",
                     width=10, font=FL).pack(side="left")
        tk.Label(r_g, text="  Ref dist (m):", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB).pack(side="left", padx=(10,0))
        tk.Entry(r_g, textvariable=self._c_dist, width=5, bg=BG_CARD,
                 fg=TEXT_PRI, insertbackground=ACCENT, relief="flat",
                 font=FM).pack(side="left")

        r_cal = tk.Frame(meta, bg=BG_DARK); r_cal.pack(fill="x", pady=1)
        tk.Label(r_cal, text="Default calibre (mm):", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=20, anchor="w").pack(side="left")
        tk.Entry(r_cal, textvariable=self._c_calibre, width=5, bg=BG_CARD,
                 fg=TEXT_PRI, insertbackground=ACCENT, relief="flat",
                 font=FM).pack(side="left")
        for lbl, val in [("4.5",4.5),("5.6",5.6)]:
            tk.Button(r_cal, text=lbl,
                      command=lambda v=val: (self._c_calibre.set(str(v)),
                                            self._update_creator_preview()),
                      bg=BG_CARD, fg=TEXT_DIM, font=FL, relief="flat",
                      padx=4, pady=1, cursor="hand2").pack(side="left", padx=2)
        tk.Label(r_cal, text="  Card dia (mm):", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB).pack(side="left", padx=(8,0))
        tk.Entry(r_cal, textvariable=self._c_card, width=6, bg=BG_CARD,
                 fg=TEXT_PRI, insertbackground=ACCENT, relief="flat",
                 font=FM).pack(side="left")
        tk.Label(r_cal, text=" (outer card edge)", bg=BG_DARK, fg=TEXT_DIM,
                 font=FL).pack(side="left", padx=2)

        # Trace changes to update live preview
        for v in (self._c_calibre, self._c_card):
            v.trace_add("write", lambda *_: self._update_creator_preview())

        # ── Ring data text area ───────────────────────────────────────────────
        tk.Label(parent,
                 text="Visual rings — score, ring_diameter_mm (innermost first):",
                 bg=BG_DARK, fg=TEXT_DIM, font=FL).pack(anchor="nw", padx=10, pady=(4,1))
        txt_frame = tk.Frame(parent, bg=BG_CARD)
        txt_frame.pack(fill="both", expand=True, padx=10, pady=(0,2))
        self._c_rings_txt = tk.Text(txt_frame, height=6, bg=BG_CARD, fg=TEXT_PRI,
                                     insertbackground=ACCENT, relief="flat",
                                     font=("Consolas", 9), wrap="none")
        self._c_rings_txt.pack(fill="both", expand=True, padx=4, pady=4)
        self._c_rings_txt.insert("1.0",
            "10, 0.5\n9, 5.5\n8, 10.5\n7, 15.5\n6, 20.5\n"
            "5, 25.5\n4, 30.5\n3, 35.5\n2, 40.5\n1, 45.5")
        self._c_rings_txt.bind("<KeyRelease>", lambda *_: self._update_creator_preview())

        # ── Live scoring preview ──────────────────────────────────────────────
        self._c_preview = tk.Label(parent, text="", bg=BG_CARD, fg=ACCENT,
                                    font=("Consolas", 9), justify="left",
                                    anchor="nw", padx=8, pady=4)
        self._c_preview.pack(fill="x", padx=10, pady=(0,2))

        # ── Status + buttons ─────────────────────────────────────────────────
        self._c_status = tk.Label(parent, text="", bg=BG_DARK, fg=ACCENT,
                                   font=FM, anchor="w")
        self._c_status.pack(fill="x", padx=10)
        bf = tk.Frame(parent, bg=BG_DARK); bf.pack(fill="x", padx=10, pady=(2,8))
        tk.Button(bf, text="Save Target", command=self._save_target,
                  bg=ACCENT, fg=BG_DARK, font=FB, relief="flat",
                  padx=12, pady=5, cursor="hand2").pack(side="right")

        self._update_creator_preview()

    def _update_creator_preview(self, *_):
        """Update the live scoring geometry preview."""
        try:
            calibre = float(self._c_calibre.get())
            card_d  = float(self._c_card.get()) if self._c_card.get().strip() else None
            if card_d is None:
                # Try to infer from last ring diameter in text area
                lines = self._c_rings_txt.get("1.0","end").strip().splitlines()
                for ln in reversed(lines):
                    ln = ln.strip()
                    if ln and not ln.startswith("#"):
                        parts = ln.split(",")
                        if len(parts) == 2:
                            card_d = float(parts[1].strip())
                            break
            if card_d is None:
                self._c_preview.config(text="Enter card diameter or ring data to preview.")
                return
            R = card_d / 2 + calibre / 2
            bw_int = R / 10
            bw_dec = R / 99
            txt = (f"  R = {card_d:.2f}/2 + {calibre:.2f}/2 = {R:.3f} mm  "
                   f"| Int: 10 bands x {bw_int:.3f}mm -> 10..1  "
                   f"| Dec: 99 bands x {bw_dec:.4f}mm -> 10.9..1.0")
            self._c_preview.config(text=txt)
        except (ValueError, ZeroDivisionError):
            self._c_preview.config(text="  Enter valid numbers to see scoring preview.")

    def _on_creator_mode(self, event=None):
        """Load an existing target into the editor fields."""
        from core.config import TARGETS
        mode = self._c_mode.get()
        if mode == "New target":
            self._c_key.set(""); self._c_name.set("")
            self._c_gauging.set("outward"); self._c_calibre.set("4.5")
            self._c_dist.set("10"); self._c_card.set("")
            self._c_rings_txt.delete("1.0","end")
            self._c_rings_txt.insert("1.0",
                "10, 0.5\n9, 5.5\n8, 10.5\n7, 15.5\n6, 20.5\n"
                "5, 25.5\n4, 30.5\n3, 35.5\n2, 40.5\n1, 45.5")
            self._c_status.config(text="")
            self._update_creator_preview()
            return
        # Find matching target
        key = mode.replace("Edit: ","").strip()
        t   = next((t for t in TARGETS.values() if t["name"] == key), None)
        if t is None:
            return
        self._c_key.set(t["key"])
        self._c_name.set(t["name"])
        self._c_gauging.set(t.get("gauging","outward"))
        self._c_calibre.set(str(t.get("calibre_mm",4.5)))
        self._c_dist.set(str(t.get("reference_dist_m",10.0)))
        self._c_card.set(str(t.get("diameter_mm", "")))
        # Rebuild ring text from visual rings
        self._c_rings_txt.delete("1.0","end")
        scores = t.get("ring_scores",[])
        dias   = [r*2 for r in t.get("rings_mm",[])]
        lines  = "\n".join(f"{int(s) if s==int(s) else s}, {d}" for s,d in zip(scores,dias))
        self._c_rings_txt.insert("1.0", lines)
        self._c_status.config(text=f"Loaded '{t['name']}' for editing.")
        self._update_creator_preview()

    def _delete_target(self):
        """Delete the selected existing target CSV."""
        from core.config import TARGETS, _targets_dir
        mode = self._c_mode.get()
        if mode == "New target":
            self._c_status.config(text="Select an existing target to delete.", fg=GOLD)
            return
        key = mode.replace("Edit: ","").strip()
        t   = next((t for t in TARGETS.values() if t["name"] == key), None)
        if t is None:
            return
        if not messagebox.askyesno("Delete Target",
                f"Permanently delete '{t['name']}' ({t['key']}.csv)?\n"
                "This cannot be undone."):
            return
        import os as _os
        path = _os.path.join(_targets_dir(), f"{t['key']}.csv")
        try:
            _os.remove(path)
            import core.config as _cc, core.marker_sheet as _ms
            _cc.TARGETS.pop(t["key"], None)
            from core.marker_sheet import _get_aiming_marks
            _ms.AIMING_MARKS = _get_aiming_marks()
            self._c_status.config(text=f"Deleted '{t['name']}'.", fg=ACCENT)
            # Refresh mode dropdown
            modes = ["New target"] + [f"Edit: {tgt['name']}" for tgt in _cc.TARGETS.values()]
            self._c_mode_combo["values"] = modes
            self._c_mode.set("New target")
        except Exception as e:
            self._c_status.config(text=f"Delete failed: {e}", fg=ACCENT2)

    def _save_target(self):
        """Validate and save the target as a CSV in the targets/ folder."""
        from core.config import _targets_dir, _load_target_csv
        import core.config as _cc, core.marker_sheet as _ms

        key  = self._c_key.get().strip().replace(" ","_")
        name = self._c_name.get().strip()
        if not key or not name:
            self._c_status.config(text="Key and name are required.", fg=ACCENT2)
            return

        lines = self._c_rings_txt.get("1.0","end").strip().splitlines()
        scores, diameters = [], []
        for ln in lines:
            ln = ln.strip()
            if not ln or ln.startswith("#"): continue
            parts = ln.split(",")
            if len(parts) != 2:
                self._c_status.config(text=f"Bad line: {ln!r} — use  score, diameter_mm",
                                       fg=ACCENT2); return
            try:
                scores.append(float(parts[0].strip()))
                diameters.append(float(parts[1].strip()))
            except ValueError:
                self._c_status.config(text=f"Not a number: {ln!r}", fg=ACCENT2); return
        if len(scores) < 2:
            self._c_status.config(text="Need at least 2 rings.", fg=ACCENT2); return

        try:
            calibre = float(self._c_calibre.get())
            dist    = float(self._c_dist.get())
        except ValueError:
            self._c_status.config(text="Calibre and distance must be numbers.", fg=ACCENT2)
            return

        # Card diameter — explicit field or inferred from last ring
        try:
            card_d = float(self._c_card.get()) if self._c_card.get().strip() else diameters[-1]
        except ValueError:
            card_d = diameters[-1]

        # Check if overwriting a file
        tdir  = _targets_dir()
        os.makedirs(tdir, exist_ok=True)
        fname = f"{key}.csv"
        path  = os.path.join(tdir, fname)
        if os.path.exists(path):
            existing_name = _cc.TARGETS.get(key,{}).get("name","")
            if existing_name and existing_name != name:
                if not messagebox.askyesno("Overwrite?",
                        f"A target with key '{key}' already exists ({existing_name}).\n"
                        "Overwrite it?"):
                    return

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                f.write(f"# Splatt2 Target Definition — {name}\n")
                f.write(f"key,{key}\n")
                f.write(f"name,{name}\n")
                f.write(f"gauging,{self._c_gauging.get()}\n")
                f.write(f"calibre_mm,{calibre}\n")
                f.write(f"reference_dist_m,{dist}\n")
                f.write(f"aiming_mark_dia_mm,{diameters[0]}\n")
                f.write(f"card_diameter_mm,{card_d}\n")
                f.write("\n")
                f.write("score,ring_diameter_mm\n")
                for s, d in zip(scores, diameters):
                    f.write(f"{int(s) if s==int(s) else s},{d}\n")
        except Exception as e:
            self._c_status.config(text=f"Save failed: {e}", fg=ACCENT2); return

        # Hot-reload into running app
        t = _load_target_csv(path)
        if t:
            _cc.TARGETS[t["key"]] = t
            from core.marker_sheet import _get_aiming_marks
            _ms.AIMING_MARKS = _get_aiming_marks()
            # Refresh mode dropdown
            modes = ["New target"] + [f"Edit: {tgt['name']}" for tgt in _cc.TARGETS.values()]
            self._c_mode_combo["values"] = modes

        self._c_status.config(
            text=f"Saved '{name}' → {fname}  (available immediately)",
            fg=ACCENT)

# =============================================================================
# SESSION HISTORY WINDOW
# =============================================================================

class SessionHistoryWindow(tk.Toplevel):
    def __init__(self, parent, save_dir, cfg):
        super().__init__(parent)
        self.save_dir = save_dir
        self.cfg = cfg
        self.title("Session History — Splatt2")
        self.configure(bg=BG_DARK)
        self.geometry("920x640")
        self._sel = None
        self._renderer = None
        self._tgt_img_id = None
        self._build()
        self._load()

    def _build(self):
        left = tk.Frame(self, bg=BG_PANEL, width=320)
        left.pack(side="left", fill="y", padx=(6, 3), pady=6)
        left.pack_propagate(False)
        tk.Label(left, text="SESSIONS", bg=BG_PANEL, fg=TEXT_DIM,
                 font=FL).pack(anchor="nw", padx=8, pady=(6, 2))
        lf = tk.Frame(left, bg=BG_DARK)
        lf.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self._slist = tk.Text(lf, bg=BG_DARK, fg=TEXT_SEC, font=FM,
                               relief="flat", state="disabled",
                               cursor="hand2", width=34, wrap="none")
        sb = tk.Scrollbar(lf, command=self._slist.yview,
                          bg=BG_DARK, troughcolor=BG_DARK)
        self._slist.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._slist.pack(fill="both", expand=True)
        self._slist.bind("<ButtonRelease-1>", self._on_click)
        self._slist.tag_config("hdr", foreground=TEXT_DIM)
        self._slist.tag_config("row", foreground=TEXT_SEC)

        right = tk.Frame(self, bg=BG_PANEL)
        right.pack(fill="both", expand=True, padx=(3, 6), pady=6)
        self._stats_lbl = tk.Label(right, text="Select a session",
                                    bg=BG_PANEL, fg=TEXT_SEC, font=FM,
                                    justify="left", anchor="nw")
        self._stats_lbl.pack(fill="x", padx=8, pady=(6, 4))
        tf = tk.Frame(right, bg=BG_DARK)
        tf.pack(fill="both", expand=True, padx=6, pady=4)
        self._tgt = tk.Canvas(tf, bg=BG_DARK, highlightthickness=0, bd=0)
        self._tgt.pack(fill="both", expand=True)
        df = tk.Frame(right, bg=BG_DARK)
        df.pack(fill="x", padx=6, pady=(0, 6))
        self._detail = tk.Text(df, bg=BG_DARK, fg=TEXT_SEC, font=FM,
                                relief="flat", state="disabled",
                                height=7, wrap="none")
        dsb = tk.Scrollbar(df, command=self._detail.yview,
                           bg=BG_DARK, troughcolor=BG_DARK)
        self._detail.config(yscrollcommand=dsb.set)
        dsb.pack(side="right", fill="y")
        self._detail.pack(fill="x")
        self._detail.tag_config("hdr",  foreground=TEXT_DIM)
        self._detail.tag_config("high", foreground=GOLD)
        self._detail.tag_config("mid",  foreground=ACCENT)
        self._detail.tag_config("low",  foreground=TEXT_SEC)
        self._detail.tag_config("miss", foreground=ACCENT2)
        self.after(200, self._init_renderer)

    def _init_renderer(self):
        from core.target_renderer import TargetRenderer
        w = self._tgt.winfo_width()
        h = self._tgt.winfo_height()
        if w < 50 or h < 50:
            self.after(100, self._init_renderer)
            return
        tkey = self.cfg.get("target_key", "10m_air_rifle")
        self._renderer = TargetRenderer((w, h), TARGETS[tkey])
        self._redraw()

    def _load(self):
        from core.session import load_session_history
        # _history is {day_label: [entry, ...]} newest day first
        self._history_dict = load_session_history(self.save_dir)
        # Flatten to a list for display, newest first
        self._history = []
        for day, entries in self._history_dict.items():
            for e in entries:
                e["_day"] = day   # stash day label for display
                self._history.append(e)

        self._slist.config(state="normal")
        self._slist.delete("1.0", "end")
        if not self._history:
            self._slist.insert("end",
                f"No saved sessions.\nSave dir:\n{self.save_dir}", "hdr")
        else:
            self._slist.insert("end",
                f"{'Day':<12} {'Time':<8} {'N':>3} {'Score':>6}\n", "hdr")
            self._slist.insert("end", "─" * 34 + "\n", "hdr")
            for e in self._history:
                self._slist.insert("end",
                    f"{e.get('_day',''):<12} {e['date']:<8} "
                    f"{e['shot_count']:>3} {e['total_score']:>6.1f}\n", "row")
        self._slist.config(state="disabled")

    def _on_click(self, event):
        try:
            ln = int(self._slist.index(f"@{event.x},{event.y}").split(".")[0])
            idx = ln - 3
            if 0 <= idx < len(self._history):
                self._show(idx)
        except Exception:
            pass

    def _show(self, idx):
        from core.session import reconstruct_shot_traces
        e = self._history[idx]
        self._sel = e
        dur = e["duration_s"]
        self._stats_lbl.config(text=(
            f"{e['name']}  ·  {e['date']}\n"
            f"Shots: {e['shot_count']}  Total: {e['total_score']:.1f}  "
            f"Avg: {e['avg_score']:.2f}  "
            f"Dur: {dur//60:.0f}m{dur%60:.0f}s  "
            f"Avg on-tgt: {e.get('avg_on_target_s',0):.1f}s"
        ))
        self._detail.config(state="normal")
        self._detail.delete("1.0", "end")
        self._detail.insert("end",
            f"{'#':>3}  {'Score':>5}  {'X':>6}  {'Y':>6}  {'OT':>5}\n", "hdr")
        self._detail.insert("end", "─" * 34 + "\n", "hdr")
        for s in e["raw"].get("shots", []):
            sc = s["score"]
            line = (f"{s['index']:>3}  {sc:>5.1f}  "
                    f"{s['aim_mm'][0]:>+6.1f}  {s['aim_mm'][1]:>+6.1f}  "
                    f"{s.get('on_target_s',0):>5.1f}\n")
            tag = "high" if sc >= 10 else "mid" if sc >= 8 else \
                  "low" if sc > 0 else "miss"
            self._detail.insert("end", line, tag)
        self._detail.config(state="disabled")
        self._redraw()

    def _redraw(self):
        if self._renderer is None:
            return
        from core.session import Shot, ShotTrace, TracePoint, reconstruct_shot_traces
        shots = []
        if self._sel:
            traces = reconstruct_shot_traces(self._sel["raw"])
            for i, s in enumerate(self._sel["raw"].get("shots", [])):
                acp = s.get("aim_centrepoint")
                shots.append(Shot(
                    index=s["index"], timestamp=s["timestamp"],
                    aim_mm=tuple(s["aim_mm"]), score=s["score"],
                    ring_index=s.get("ring_index", 0), series=1,
                    trace=traces[i] if i < len(traces) else None,
                    aim_centrepoint=tuple(acp) if acp else None,
                ))
        img = self._renderer.render(shots=shots, show_mpi=True,
                                     show_group=True, show_acp=True)
        photo = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        if self._tgt_img_id is None:
            self._tgt_img_id = self._tgt.create_image(0, 0, anchor="nw",
                                                        image=photo)
        else:
            self._tgt.itemconfig(self._tgt_img_id, image=photo)
        self._tgt._img = photo


# =============================================================================
# SERIES REVIEW WINDOW
# =============================================================================



# =============================================================================
# SETTINGS DIALOG  (scrollable tabs, resizable)
# =============================================================================

class SettingsDialog(tk.Toplevel):
    """Scrollable settings dialog — each tab has a canvas+scrollbar so nothing
    is ever clipped regardless of screen size or font scaling."""

    def __init__(self, parent, cfg, apply_cb):
        super().__init__(parent)
        self.cfg      = cfg.copy()
        self.apply_cb = apply_cb
        self.title("Settings — Splatt2")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)
        self.grab_set()
        self.geometry("560x680")
        self.minsize(480, 500)
        self._build()

    # ── shell ─────────────────────────────────────────────────────────────────

    def _build(self):
        # Fixed button row at BOTTOM (always visible)
        btn_row = tk.Frame(self, bg=BG_DARK)
        btn_row.pack(side="bottom", fill="x", padx=8, pady=8)
        tk.Button(btn_row, text="Apply & Close", command=self._apply_and_close,
                  bg=ACCENT, fg=BG_DARK, font=FB, relief="flat",
                  padx=12, pady=7).pack(side="right", padx=4)
        tk.Button(btn_row, text="Apply", command=self._apply,
                  bg=BG_CARD, fg=ACCENT, font=FB, relief="flat",
                  padx=12, pady=7).pack(side="right", padx=4)
        tk.Button(btn_row, text="Cancel", command=self.destroy,
                  bg=BG_CARD, fg=TEXT_SEC, font=FB, relief="flat",
                  padx=12, pady=7).pack(side="right", padx=4)
        self._status_lbl = tk.Label(btn_row, text="", bg=BG_DARK,
                                     fg=ACCENT, font=FL)
        self._status_lbl.pack(side="left", padx=8)

        # Notebook fills the rest
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=(8, 0))
        for name, builder in [("Camera",   self._build_cam),
                               ("Audio",    self._build_audio),
                               ("Target",   self._build_target),
                               ("Colours",  self._build_colours),
                               ("Advanced", self._build_advanced)]:
            outer = tk.Frame(nb, bg=BG_DARK)
            nb.add(outer, text=name)
            canvas = tk.Canvas(outer, bg=BG_DARK, highlightthickness=0, bd=0)
            vsb    = tk.Scrollbar(outer, orient="vertical",
                                   command=canvas.yview,
                                   bg=BG_DARK, troughcolor=BG_CARD)
            canvas.configure(yscrollcommand=vsb.set)
            vsb.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)
            inner = tk.Frame(canvas, bg=BG_DARK)
            win   = canvas.create_window((0, 0), window=inner, anchor="nw")

            def _refresh(c=canvas, w=win, inn=inner):
                """Force scroll region update — called on resize AND on tab map."""
                inn.update_idletasks()
                c.configure(scrollregion=c.bbox("all"))
                c.itemconfig(w, width=c.winfo_width())

            inner.bind("<Configure>", lambda e, r=_refresh: r())
            # <Map> fires when a tab becomes visible — forces first-paint update
            canvas.bind("<Map>", lambda e, r=_refresh: r())

            def _wheel(e, c=canvas):
                c.yview_scroll(int(-1*(e.delta/120)), "units")
            # Bind wheel only to this canvas (not bind_all — that leaks)
            canvas.bind("<Enter>", lambda e, c=canvas, wh=_wheel:
                         c.bind_all("<MouseWheel>", wh))
            canvas.bind("<Leave>", lambda e, c=canvas:
                         c.unbind_all("<MouseWheel>"))

            builder(inner)
            # Force scroll region now that content is populated
            self.after(50, _refresh)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _row(self, parent, label, widget_fn):
        r = tk.Frame(parent, bg=BG_DARK)
        r.pack(fill="x", padx=12, pady=4)
        tk.Label(r, text=label, bg=BG_DARK, fg=TEXT_SEC, font=FB,
                 width=24, anchor="w").pack(side="left")
        w = widget_fn(r)
        w.pack(side="left")
        return w

    def _section(self, parent, title):
        tk.Label(parent, text=title, bg=BG_DARK, fg=ACCENT,
                 font=FT).pack(anchor="nw", padx=12, pady=(14, 4))

    def _note(self, parent, text):
        tk.Label(parent, text=text, bg=BG_DARK, fg=TEXT_DIM,
                 font=FL, justify="left").pack(anchor="nw", padx=16, pady=(0, 4))

    def _entry(self, parent, key, width=8):
        v = tk.StringVar(value=str(self.cfg.get(key, "")))
        setattr(self, f"_v_{key}", v)
        return tk.Entry(parent, textvariable=v, width=width, bg=BG_CARD,
                        fg=TEXT_PRI, insertbackground=ACCENT, relief="flat",
                        font=FM)

    def _combo(self, parent, key, values):
        v = tk.StringVar(value=str(self.cfg.get(key, values[0])))
        setattr(self, f"_v_{key}", v)
        return ttk.Combobox(parent, textvariable=v, values=values,
                             width=18, state="readonly", font=FM)

    # ── Camera tab ────────────────────────────────────────────────────────────

    def _build_cam(self, tab):
        self._section(tab, "Camera")
        self._row(tab, "Camera index", lambda p: self._entry(p, "camera_index", 4))

        # Resolution
        r = tk.Frame(tab, bg=BG_DARK); r.pack(fill="x", padx=12, pady=4)
        tk.Label(r, text="Resolution:", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=24, anchor="w").pack(side="left")
        self._v_video_width  = tk.StringVar(value=str(self.cfg.get("video_width",  640)))
        self._v_video_height = tk.StringVar(value=str(self.cfg.get("video_height", 480)))
        tk.Entry(r, textvariable=self._v_video_width,  width=5, bg=BG_CARD,
                 fg=TEXT_PRI, insertbackground=ACCENT, relief="flat", font=FM).pack(side="left")
        tk.Label(r, text="×", bg=BG_DARK, fg=TEXT_DIM, font=FB).pack(side="left", padx=2)
        tk.Entry(r, textvariable=self._v_video_height, width=5, bg=BG_CARD,
                 fg=TEXT_PRI, insertbackground=ACCENT, relief="flat", font=FM).pack(side="left")
        for lbl, (w, h) in [("480×360",(480,360)),("640×480",(640,480)),
                             ("1280×720",(1280,720))]:
            tk.Button(r, text=lbl,
                      command=lambda w=w,h=h: (self._v_video_width.set(str(w)),
                                               self._v_video_height.set(str(h))),
                      bg=BG_CARD, fg=TEXT_DIM, font=FL, relief="flat",
                      padx=4, pady=2, cursor="hand2").pack(side="left", padx=2)

        # FPS
        r2 = tk.Frame(tab, bg=BG_DARK); r2.pack(fill="x", padx=12, pady=4)
        tk.Label(r2, text="Target FPS:", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=24, anchor="w").pack(side="left")
        self._v_video_fps = tk.StringVar(value=str(self.cfg.get("video_fps", 30)))
        tk.Entry(r2, textvariable=self._v_video_fps, width=5, bg=BG_CARD,
                 fg=TEXT_PRI, insertbackground=ACCENT, relief="flat", font=FM).pack(side="left")
        for fps in [30, 60, 120]:
            tk.Button(r2, text=str(fps),
                      command=lambda f=fps: self._v_video_fps.set(str(f)),
                      bg=BG_CARD, fg=TEXT_DIM, font=FL, relief="flat",
                      padx=4, pady=2, cursor="hand2").pack(side="left", padx=2)
        tk.Button(r2, text="Detect", command=self._detect_camera_caps,
                  bg=BG_CARD, fg=ACCENT, font=FL, relief="flat",
                  padx=6, pady=2, cursor="hand2").pack(side="left", padx=(8, 0))
        self._cam_caps_lbl = tk.Label(r2, text="", bg=BG_DARK, fg=TEXT_DIM, font=FL)
        self._cam_caps_lbl.pack(side="left", padx=4)

        self._section(tab, "Trace Smoothing")
        self._note(tab, "EMA: fast, good for most use.\n"
                        "Savitzky-Golay: better shape, needs scipy.\n"
                        "None: raw (shows camera jitter).")
        self._row(tab, "Smooth mode",
                  lambda p: self._combo(p, "smooth_mode", ["none","ema","savgol"]))

        r_a = tk.Frame(tab, bg=BG_DARK); r_a.pack(fill="x", padx=12, pady=4)
        tk.Label(r_a, text="EMA alpha (0.05–0.8):", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=24, anchor="w").pack(side="left")
        self._v_smooth_alpha = tk.StringVar(value=str(self.cfg.get("smooth_alpha", 0.35)))
        tk.Entry(r_a, textvariable=self._v_smooth_alpha, width=6, bg=BG_CARD,
                 fg=TEXT_PRI, insertbackground=ACCENT, relief="flat", font=FM).pack(side="left")
        tk.Label(r_a, text="  lower=smoother", bg=BG_DARK, fg=TEXT_DIM, font=FL).pack(side="left", padx=4)

        r_w = tk.Frame(tab, bg=BG_DARK); r_w.pack(fill="x", padx=12, pady=4)
        tk.Label(r_w, text="SavGol window (odd):", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=24, anchor="w").pack(side="left")
        self._v_smooth_window = tk.StringVar(value=str(self.cfg.get("smooth_window", 11)))
        tk.Entry(r_w, textvariable=self._v_smooth_window, width=4, bg=BG_CARD,
                 fg=TEXT_PRI, insertbackground=ACCENT, relief="flat", font=FM).pack(side="left")
        tk.Label(r_w, text="  7=snappy  11=smooth  15=max",
                 bg=BG_DARK, fg=TEXT_DIM, font=FL).pack(side="left", padx=4)

        self._section(tab, "ArUco")
        self._flip = tk.BooleanVar(value=self.cfg.get("flip_image", False))
        self._row(tab, "Flip image",
                  lambda p: tk.Checkbutton(p, variable=self._flip,
                                           bg=BG_DARK, selectcolor=BG_CARD))
        self._row(tab, "Dictionary",
                  lambda p: self._combo(p, "aruco_dict",
                                        ["DICT_4X4_50","DICT_4X4_100",
                                         "DICT_5X5_50","DICT_ARUCO_ORIGINAL"]))
        self._row(tab, "Marker size (mm)",  lambda p: self._entry(p, "aruco_marker_mm", 6))
        self._row(tab, "Margin (mm)",       lambda p: self._entry(p, "aruco_margin_mm", 6))
        self._note(tab, "Larger markers → better detection at low resolution.\n"
                        "Default 40mm with 8mm margin leaves 114mm centre space.")

        # Marker count
        r_mc = tk.Frame(tab, bg=BG_DARK); r_mc.pack(fill="x", padx=12, pady=4)
        tk.Label(r_mc, text="Marker count:", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=24, anchor="w").pack(side="left")
        self._v_aruco_marker_count = tk.StringVar(
            value=str(self.cfg.get("aruco_marker_count", 4)))
        ttk.Combobox(r_mc, textvariable=self._v_aruco_marker_count,
                     values=["4", "6", "8"], state="readonly",
                     width=6, font=FL).pack(side="left")
        tk.Label(r_mc, text="  4=corners only  6=+left/right midpoints  8=+top/bottom",
                 bg=BG_DARK, fg=TEXT_DIM, font=FL).pack(side="left", padx=6)

        # Pixel format (MJPEG prevents static-scene fps throttling)
        r_pf = tk.Frame(tab, bg=BG_DARK); r_pf.pack(fill="x", padx=12, pady=4)
        tk.Label(r_pf, text="Pixel format:", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=24, anchor="w").pack(side="left")
        self._v_camera_pixel_format = tk.StringVar(
            value=self.cfg.get("camera_pixel_format", "Auto"))
        ttk.Combobox(r_pf, textvariable=self._v_camera_pixel_format,
                     values=["Auto", "MJPEG", "YUY2"], state="readonly",
                     width=8, font=FL).pack(side="left")
        tk.Label(r_pf, text="  MJPEG prevents fps throttling on static scenes",
                 bg=BG_DARK, fg=TEXT_DIM, font=FL).pack(side="left", padx=6)

        self._section(tab, "Performance")
        r_nv = tk.Frame(tab, bg=BG_DARK); r_nv.pack(fill="x", padx=12, pady=4)
        tk.Label(r_nv, text="No video preview:", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=24, anchor="w").pack(side="left")
        self._nv_var = tk.BooleanVar(value=bool(self.cfg.get("no_video_mode", False)))
        self._v_no_video_mode = tk.StringVar(value=str(self.cfg.get("no_video_mode", False)))
        tk.Checkbutton(r_nv, variable=self._nv_var, bg=BG_DARK, selectcolor=BG_CARD,
                       command=lambda: self._v_no_video_mode.set(str(self._nv_var.get()))
                       ).pack(side="left")
        tk.Label(r_nv, text="Pure tracking, no display (faster)",
                 bg=BG_DARK, fg=TEXT_DIM, font=FL).pack(side="left", padx=6)
        self._note(tab, "For best fps: 480×360, No video, EMA smoothing.")

        r_cl = tk.Frame(tab, bg=BG_DARK); r_cl.pack(fill="x", padx=12, pady=4)
        tk.Label(r_cl, text="CLAHE enhancement:", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=24, anchor="w").pack(side="left")
        self._clahe_var = tk.BooleanVar(value=bool(self.cfg.get("use_clahe", True)))
        tk.Checkbutton(r_cl, variable=self._clahe_var, bg=BG_DARK, selectcolor=BG_CARD,
                       command=lambda: None).pack(side="left")
        tk.Label(r_cl,
                 text="Adaptive contrast — improves tracking in uneven lighting",
                 bg=BG_DARK, fg=TEXT_DIM, font=FL).pack(side="left", padx=6)

        r_cc = tk.Frame(tab, bg=BG_DARK); r_cc.pack(fill="x", padx=12, pady=4)
        tk.Label(r_cc, text="CLAHE clip limit:", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=24, anchor="w").pack(side="left")
        self._v_clahe_clip = tk.StringVar(value=str(self.cfg.get("clahe_clip", 4.0)))
        tk.Entry(r_cc, textvariable=self._v_clahe_clip, width=5, bg=BG_CARD,
                 fg=TEXT_PRI, insertbackground=ACCENT, relief="flat",
                 font=FM).pack(side="left")
        for lbl, val in [("2",2.0),("4",4.0),("6",6.0),("8",8.0),("12",12.0)]:
            tk.Button(r_cc, text=lbl,
                      command=lambda v=val: self._v_clahe_clip.set(str(v)),
                      bg=BG_CARD, fg=TEXT_DIM, font=FL, relief="flat",
                      padx=4, pady=2, cursor="hand2").pack(side="left", padx=2)
        tk.Label(r_cc, text="  2=mild  4=balanced  8=aggressive  12=extreme",
                 bg=BG_DARK, fg=TEXT_DIM, font=FL).pack(side="left", padx=4)

        # Brightness target (software gain normalisation)
        r_bt = tk.Frame(tab, bg=BG_DARK); r_bt.pack(fill="x", padx=12, pady=4)
        tk.Label(r_bt, text="Brightness target:", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=24, anchor="w").pack(side="left")
        self._v_brightness_target = tk.StringVar(
            value=str(self.cfg.get("brightness_target", 128.0)))
        tk.Entry(r_bt, textvariable=self._v_brightness_target, width=5,
                 bg=BG_CARD, fg=TEXT_PRI, insertbackground=ACCENT,
                 relief="flat", font=FM).pack(side="left")
        for lbl, val in [("64",64),("96",96),("128",128),("160",160),("192",192)]:
            tk.Button(r_bt, text=lbl,
                      command=lambda v=val: self._v_brightness_target.set(str(v)),
                      bg=BG_CARD, fg=TEXT_DIM, font=FL, relief="flat",
                      padx=4, pady=2, cursor="hand2").pack(side="left", padx=2)
        tk.Label(r_bt, text="  lower=darker/faster  128=balanced  higher=brighter",
                 bg=BG_DARK, fg=TEXT_DIM, font=FL).pack(side="left", padx=4)

        # Spike filter
        self._section(tab, "Spike Filter")
        self._note(tab, "Rejects bad homography spikes (marker briefly lost).\n"
                        "Spike velocity: min mm/frame to flag as candidate.\n"
                        "Reversal ratio: how sharply it must reverse (0-1).")
        r_sv = tk.Frame(tab, bg=BG_DARK); r_sv.pack(fill="x", padx=12, pady=4)
        tk.Label(r_sv, text="Spike velocity (mm/frame):", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=24, anchor="w").pack(side="left")
        self._v_spike_velocity_mm = tk.StringVar(
            value=str(self.cfg.get("spike_velocity_mm", 25.0)))
        tk.Entry(r_sv, textvariable=self._v_spike_velocity_mm, width=5,
                 bg=BG_CARD, fg=TEXT_PRI, insertbackground=ACCENT,
                 relief="flat", font=FM).pack(side="left")
        for lbl, val in [("15",15),("20",20),("25",25),("35",35),("50",50)]:
            tk.Button(r_sv, text=lbl,
                      command=lambda v=val: self._v_spike_velocity_mm.set(str(v)),
                      bg=BG_CARD, fg=TEXT_DIM, font=FL, relief="flat",
                      padx=4, pady=2, cursor="hand2").pack(side="left", padx=2)
        tk.Label(r_sv, text="  lower=more sensitive  25=default  higher=less sensitive",
                 bg=BG_DARK, fg=TEXT_DIM, font=FL).pack(side="left", padx=4)

        r_sr = tk.Frame(tab, bg=BG_DARK); r_sr.pack(fill="x", padx=12, pady=4)
        tk.Label(r_sr, text="Reversal ratio (0-1):", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=24, anchor="w").pack(side="left")
        self._v_spike_reversal = tk.StringVar(
            value=str(self.cfg.get("spike_reversal", 0.7)))
        tk.Entry(r_sr, textvariable=self._v_spike_reversal, width=5,
                 bg=BG_CARD, fg=TEXT_PRI, insertbackground=ACCENT,
                 relief="flat", font=FM).pack(side="left")
        for lbl, val in [("0.5",0.5),("0.6",0.6),("0.7",0.7),("0.8",0.8),("0.9",0.9)]:
            tk.Button(r_sr, text=lbl,
                      command=lambda v=val: self._v_spike_reversal.set(str(v)),
                      bg=BG_CARD, fg=TEXT_DIM, font=FL, relief="flat",
                      padx=4, pady=2, cursor="hand2").pack(side="left", padx=2)
        tk.Label(r_sr, text="  0.5=loose  0.7=default  0.9=strict",
                 bg=BG_DARK, fg=TEXT_DIM, font=FL).pack(side="left", padx=4)

        tk.Frame(tab, bg=BG_DARK, height=16).pack()   # bottom padding

    # ── Audio tab ─────────────────────────────────────────────────────────────

    def _build_audio(self, tab):
        self._section(tab, "Shot Detection")
        self._note(tab, "Threshold: absolute peak floor (0.01–1.0)\n"
                        "Sensitivity: how many times louder than ambient\n"
                        "  the click must be. Higher = more selective.\n"
                        "Dry-fire: threshold 0.05–0.15, sensitivity 5–8×")
        self._row(tab, "Threshold (0.01–1.0)", lambda p: self._entry(p, "audio_trigger_threshold", 6))
        self._row(tab, "Sensitivity ratio (×)", lambda p: self._entry(p, "audio_transient_ratio", 6))
        self._row(tab, "Audio cooldown (ms)",   lambda p: self._entry(p, "audio_trigger_cooldown_ms", 6))
        self._row(tab, "Post-shot cooldown (s)", lambda p: self._entry(p, "post_shot_cooldown_s", 6))
        self._note(tab, "Post-shot cooldown: ignore audio for N seconds after a shot.")

        self._section(tab, "Device")
        devs = AudioDetector.list_devices()
        dev_txt = "\n".join(f"  {i}: {n}" for i, n in devs) or "  (none found)"
        tk.Label(tab, text="Available inputs:\n" + dev_txt,
                 bg=BG_DARK, fg=TEXT_SEC, font=FM, justify="left"
                 ).pack(anchor="nw", padx=16)
        self._row(tab, "Device index (blank=auto)", lambda p: self._entry(p, "audio_device_index", 4))
        tk.Frame(tab, bg=BG_DARK, height=16).pack()

    # ── Target tab ────────────────────────────────────────────────────────────

    def _build_target(self, tab):
        self._section(tab, "Target & Range")
        self._row(tab, "Target type",
                  lambda p: self._combo(p, "target_key", list(TARGETS.keys())))
        self._row(tab, "Real range (m)",  lambda p: self._entry(p, "real_range_m", 6))
        # Session name and shots per series are in the Advanced tab

        tk.Label(tab, text="Zero Offset", bg=BG_DARK, fg=ACCENT,
                 font=FT).pack(anchor="nw", padx=12, pady=(10, 4))
        r_zo = tk.Frame(tab, bg=BG_DARK)
        r_zo.pack(fill="x", padx=12, pady=4)
        self._zero_x_lbl = tk.Label(r_zo, bg=BG_DARK, fg=TEXT_SEC,
                                     font=FM)
        self._zero_x_lbl.pack(side="left")
        self._update_zero_display()
        tk.Button(r_zo, text="Reset to (0, 0)",
                  command=self._reset_zero_offset,
                  bg=ACCENT2, fg=BG_DARK, font=FL, relief="flat",
                  padx=8, pady=3, cursor="hand2"
                  ).pack(side="right")

        tk.Label(tab, text="Shot Handling", bg=BG_DARK, fg=ACCENT,
                 font=FT).pack(anchor="nw", padx=12, pady=(10, 4))

        # Pellet calibre for scoring — drives ALL scoring geometry dynamically
        r_cal = tk.Frame(tab, bg=BG_DARK); r_cal.pack(fill="x", padx=12, pady=4)
        tk.Label(r_cal, text="Pellet diameter (mm):", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=24, anchor="w").pack(side="left")
        cal_vals = ["4.5 (.177 air)", "5.6 (.22 air)", "6.35 (.25)", "7.26 (.284)"]
        self._v_scoring_calibre = tk.StringVar(
            value=str(self.cfg.get("scoring_calibre_mm", 4.5)))
        cal_entry = tk.Entry(r_cal, textvariable=self._v_scoring_calibre,
                             width=6, bg=BG_CARD, fg=TEXT_PRI,
                             insertbackground=ACCENT, relief="flat", font=FM)
        cal_entry.pack(side="left")
        for label, val in [("4.5",4.5),("5.6",5.6)]:
            tk.Button(r_cal, text=label,
                      command=lambda v=val: self._v_scoring_calibre.set(str(v)),
                      bg=BG_CARD, fg=TEXT_DIM, font=FL, relief="flat",
                      padx=4, pady=2, cursor="hand2").pack(side="left", padx=2)
        tk.Label(r_cal, text="  affects scoring bands & approach zone",
                 bg=BG_DARK, fg=TEXT_DIM, font=FL).pack(side="left", padx=4)

        r_im = tk.Frame(tab, bg=BG_DARK)
        r_im.pack(fill="x", padx=12, pady=4)
        tk.Label(r_im, text="Ignore misses (score 0):", bg=BG_DARK,
                 fg=TEXT_SEC, font=FB, width=24, anchor="w").pack(side="left")
        self._im_var = tk.BooleanVar(
            value=bool(self.cfg.get("ignore_misses", False)))
        self._v_ignore_misses = tk.StringVar(
            value=str(self.cfg.get("ignore_misses", False)))
        tk.Checkbutton(r_im, variable=self._im_var, bg=BG_DARK,
                       selectcolor=BG_CARD,
                       command=lambda: self._v_ignore_misses.set(
                           str(self._im_var.get()))
                       ).pack(side="left")
        tk.Label(r_im, text="Discard shots that miss the scoring area",
                 bg=BG_DARK, fg=TEXT_DIM, font=FL).pack(side="left", padx=6)

        self._section(tab, "Display")
        self._row(tab, "Shot circle dia (mm)", lambda p: self._entry(p, "shot_circle_calibre_mm", 6))
        self._note(tab, "4.5=.177  5.6=.22  set to your actual calibre")

        self._section(tab, "Files")
        default_dir = _default_save_dir()
        self._note(tab, f"Default location: {default_dir}")
        r = tk.Frame(tab, bg=BG_DARK); r.pack(fill="x", padx=12, pady=4)
        tk.Label(r, text="Save directory:", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=24, anchor="w").pack(side="left")
        v = tk.StringVar(value=str(self.cfg.get("save_directory", "")))
        setattr(self, "_v_save_directory", v)
        tk.Entry(r, textvariable=v, width=22, bg=BG_CARD, fg=TEXT_PRI,
                 insertbackground=ACCENT, relief="flat", font=FM).pack(side="left", padx=(0,4))
        tk.Button(r, text="Browse…", command=self._browse_save_dir,
                  bg=BG_CARD, fg=TEXT_SEC, font=FL, relief="flat",
                  padx=6, pady=3, cursor="hand2").pack(side="left")
        def _use_default():
            v.set("")
            self._status_lbl.config(text="Reset to default location")
        tk.Button(r, text="↺ Default", command=_use_default,
                  bg=BG_CARD, fg=TEXT_DIM, font=FL, relief="flat",
                  padx=6, pady=3, cursor="hand2").pack(side="left", padx=2)
        tk.Frame(tab, bg=BG_DARK, height=16).pack()

    # ── Collect / Apply ───────────────────────────────────────────────────────

    # ── Colours & Display tab ────────────────────────────────────────────────

    def _build_colours(self, tab):
        self._note(tab, "Click a swatch to pick a colour. Changes apply on Apply.")

        self._section(tab, "Trace Colours")
        for key, label in [
            ("colour_trace_approach", "Approach zone"),
            ("colour_trace_hold",     "Hold (early)"),
            ("colour_trace_preshot",  "Pre-shot (<1s)"),
            ("colour_trace_final",    "Final (<0.2s)"),
        ]:
            self._colour_row(tab, label, key)

        self._section(tab, "Shot & UI Colours")
        for key, label in [
            ("colour_shot_fill",  "Shot hole fill"),
            ("colour_acp",        "ACP marker"),
            ("colour_crosshair",  "Crosshair"),
            ("colour_mpi",        "MPI cross"),
            ("colour_group",      "Group circle"),
            ("colour_miss",       "Miss marker"),
        ]:
            self._colour_row(tab, label, key)

        self._section(tab, "Trace Appearance")
        r = tk.Frame(tab, bg=BG_DARK); r.pack(fill="x", padx=12, pady=4)
        tk.Label(r, text="Trace line width:", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=24, anchor="w").pack(side="left")
        self._v_trace_width = tk.StringVar(value=str(self.cfg.get("trace_width", 1)))
        for w in [1, 2, 3, 4]:
            tk.Button(r, text=str(w),
                      command=lambda v=w: self._v_trace_width.set(str(v)),
                      bg=BG_CARD, fg=TEXT_DIM, font=FB, relief="flat",
                      padx=8, pady=3, cursor="hand2").pack(side="left", padx=2)
        tk.Entry(r, textvariable=self._v_trace_width, width=3,
                 bg=BG_CARD, fg=TEXT_PRI, insertbackground=ACCENT,
                 relief="flat", font=FM).pack(side="left", padx=4)

        r2 = tk.Frame(tab, bg=BG_DARK); r2.pack(fill="x", padx=12, pady=4)
        tk.Label(r2, text="Fading trace duration (s):", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=24, anchor="w").pack(side="left")
        self._v_fading_trace_duration_s = tk.StringVar(
            value=str(self.cfg.get("fading_trace_duration_s", 2.0)))
        tk.Entry(r2, textvariable=self._v_fading_trace_duration_s, width=5,
                 bg=BG_CARD, fg=TEXT_PRI, insertbackground=ACCENT,
                 relief="flat", font=FM).pack(side="left")
        for s in [1, 2, 4, 8]:
            tk.Button(r2, text=str(s),
                      command=lambda v=s: self._v_fading_trace_duration_s.set(str(v)),
                      bg=BG_CARD, fg=TEXT_DIM, font=FL, relief="flat",
                      padx=4, pady=2, cursor="hand2").pack(side="left", padx=2)
        tk.Frame(tab, bg=BG_DARK, height=16).pack()

    def _colour_row(self, parent, label: str, key: str):
        """A row with a colour swatch button that opens a colour picker."""
        r = tk.Frame(parent, bg=BG_DARK); r.pack(fill="x", padx=12, pady=3)
        tk.Label(r, text=label, bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=24, anchor="w").pack(side="left")
        # Store the hex value
        v = tk.StringVar(value=self.cfg.get(key, "#888888"))
        setattr(self, f"_v_{key}", v)

        swatch = tk.Label(r, width=4, relief="flat", cursor="hand2")
        swatch.pack(side="left", padx=(0, 4))

        hex_entry = tk.Entry(r, textvariable=v, width=9, bg=BG_CARD,
                              fg=TEXT_PRI, insertbackground=ACCENT,
                              relief="flat", font=FM)
        hex_entry.pack(side="left")

        def _update_swatch(*_):
            try:
                swatch.config(bg=v.get())
            except Exception:
                swatch.config(bg="#888888")
        v.trace_add("write", _update_swatch)
        _update_swatch()

        def _pick():
            from tkinter.colorchooser import askcolor
            current = v.get()
            result = askcolor(color=current, parent=self,
                               title=f"Choose colour — {label}")
            if result and result[1]:
                v.set(result[1].upper())
        tk.Button(r, text="Pick…", command=_pick,
                  bg=BG_CARD, fg=TEXT_SEC, font=FL, relief="flat",
                  padx=6, pady=2, cursor="hand2").pack(side="left", padx=(4, 0))

    # ── Advanced tab ──────────────────────────────────────────────────────────

    def _build_advanced(self, tab):
        self._section(tab, "Shooter Profile")
        self._row(tab, "Shooter name",
                  lambda p: self._entry(p, "shooter_name", 18))
        self._row(tab, "Position",
                  lambda p: self._combo(p, "shooting_position",
                                        ["standing","prone","kneeling","benchrest"]))

        self._section(tab, "Hold Analysis")
        r = tk.Frame(tab, bg=BG_DARK); r.pack(fill="x", padx=12, pady=4)
        tk.Label(r, text="ACP fraction (0.1–0.8):", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=24, anchor="w").pack(side="left")
        self._v_acp_fraction = tk.StringVar(
            value=str(self.cfg.get("acp_fraction", 0.40)))
        tk.Entry(r, textvariable=self._v_acp_fraction, width=5,
                 bg=BG_CARD, fg=TEXT_PRI, insertbackground=ACCENT,
                 relief="flat", font=FM).pack(side="left")
        tk.Label(r, text="  fraction of hold used for ACP",
                 bg=BG_DARK, fg=TEXT_DIM, font=FL).pack(side="left", padx=6)

        r2 = tk.Frame(tab, bg=BG_DARK); r2.pack(fill="x", padx=12, pady=4)
        tk.Label(r2, text="Approach zone factor:", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=24, anchor="w").pack(side="left")
        self._v_approach_zone_factor = tk.StringVar(
            value=str(self.cfg.get("approach_zone_factor", 2.0)))
        tk.Entry(r2, textvariable=self._v_approach_zone_factor, width=5,
                 bg=BG_CARD, fg=TEXT_PRI, insertbackground=ACCENT,
                 relief="flat", font=FM).pack(side="left")
        tk.Label(r2, text="  ×scoring radius = approach zone size",
                 bg=BG_DARK, fg=TEXT_DIM, font=FL).pack(side="left", padx=6)

        self._section(tab, "Trace Timing")
        r_ps = tk.Frame(tab, bg=BG_DARK); r_ps.pack(fill="x", padx=12, pady=4)
        tk.Label(r_ps, text="Pre-shot window (s):", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=24, anchor="w").pack(side="left")
        self._v_trace_preshot_s = tk.StringVar(
            value=str(self.cfg.get("trace_preshot_s", 1.0)))
        tk.Entry(r_ps, textvariable=self._v_trace_preshot_s, width=5,
                 bg=BG_CARD, fg=TEXT_PRI, insertbackground=ACCENT,
                 relief="flat", font=FM).pack(side="left")
        tk.Label(r_ps, text="  trace turns yellow this far before shot",
                 bg=BG_DARK, fg=TEXT_DIM, font=FL).pack(side="left", padx=4)

        r_fs = tk.Frame(tab, bg=BG_DARK); r_fs.pack(fill="x", padx=12, pady=4)
        tk.Label(r_fs, text="Final window (s):", bg=BG_DARK, fg=TEXT_SEC,
                 font=FB, width=24, anchor="w").pack(side="left")
        self._v_trace_final_s = tk.StringVar(
            value=str(self.cfg.get("trace_final_s", 0.2)))
        tk.Entry(r_fs, textvariable=self._v_trace_final_s, width=5,
                 bg=BG_CARD, fg=TEXT_PRI, insertbackground=ACCENT,
                 relief="flat", font=FM).pack(side="left")
        tk.Label(r_fs, text="  trace turns red this far before shot",
                 bg=BG_DARK, fg=TEXT_DIM, font=FL).pack(side="left", padx=4)

        self._section(tab, "Session")
        self._row(tab, "Shots per series",
                  lambda p: self._entry(p, "shots_per_series", 4))
        self._row(tab, "Session name",
                  lambda p: self._entry(p, "session_name", 16))

        self._note(tab, "Session name is used for saved file names.")
        tk.Frame(tab, bg=BG_DARK, height=16).pack()

        self._section(tab, "Voice Feedback")
        self._voice_enabled = tk.BooleanVar(value=self.cfg.get("voice_enabled", False))
        self._row(tab, "Voice enabled",
                  lambda p: tk.Checkbutton(p, variable=self._voice_enabled,
                                           bg=BG_DARK, selectcolor=BG_CARD))

    def _collect(self):
        for attr in dir(self):
            if attr.startswith("_v_"):
                key = attr[3:]
                val = getattr(self, attr).get()
                orig = self.cfg.get(key)
                try:
                    if val in ("True", "False"):     self.cfg[key] = (val == "True")
                    elif isinstance(orig, bool):     self.cfg[key] = bool(val)
                    elif isinstance(orig, int):      self.cfg[key] = int(val)
                    elif isinstance(orig, float):    self.cfg[key] = float(val)
                    elif val in ("", "None"):         self.cfg[key] = None
                    else:                             self.cfg[key] = val
                except (ValueError, TypeError):      self.cfg[key] = val
        if hasattr(self, "_flip"):
            self.cfg["flip_image"] = self._flip.get()
        if hasattr(self, "_nv_var"):
            self.cfg["no_video_mode"] = self._nv_var.get()
        if hasattr(self, "_v_scoring_calibre"):
            try:
                self.cfg["scoring_calibre_mm"] = float(self._v_scoring_calibre.get())
            except ValueError:
                pass
        if hasattr(self, "_clahe_var"):
            self.cfg["use_clahe"] = self._clahe_var.get()
        if hasattr(self, "_im_var"):
            self.cfg["ignore_misses"] = self._im_var.get()
        if hasattr(self, "_voice_enabled"):
            self.cfg["voice_enabled"] = self._voice_enabled.get()

    def _apply(self):
        self._collect()
        self.apply_cb(self.cfg)
        self._status_lbl.config(text="✓ Applied")
        self.after(2000, lambda: self._status_lbl.config(text=""))

    def _apply_and_close(self):
        self._collect()
        self.apply_cb(self.cfg)
        self.destroy()

    def _update_zero_display(self):
        x = self.cfg.get("zero_offset_x", 0.0)
        y = self.cfg.get("zero_offset_y", 0.0)
        if hasattr(self, "_zero_x_lbl"):
            active = (x != 0 or y != 0)
            self._zero_x_lbl.config(
                text=f"Current: ({x:+.2f}, {y:+.2f}) mm",
                fg=GOLD if active else TEXT_DIM)

    def _reset_zero_offset(self):
        self.cfg["zero_offset_x"] = 0.0
        self.cfg["zero_offset_y"] = 0.0
        self._update_zero_display()
        self.apply_cb(self.cfg)
        self._status_lbl.config(text="Zero reset to (0, 0)")
        self.after(2000, lambda: self._status_lbl.config(text=""))

    def _browse_save_dir(self):
        from tkinter.filedialog import askdirectory
        v = getattr(self, "_v_save_directory", None)
        init = v.get() if v else ""
        if not init or not os.path.isdir(init):
            init = os.path.expanduser("~")
        chosen = askdirectory(title="Choose save directory",
                               initialdir=init, mustexist=False)
        if chosen and v:
            v.set(os.path.abspath(chosen))

    def _detect_camera_caps(self):
        try:
            idx = int(self.cfg.get("camera_index", 0))
        except (ValueError, TypeError):
            idx = 0
        self._cam_caps_lbl.config(text="Probing…")
        self.update()
        import cv2 as _cv2
        cap = _cv2.VideoCapture(idx, _cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = _cv2.VideoCapture(idx)
        if not cap.isOpened():
            self._cam_caps_lbl.config(text="Not found")
            return
        results = []
        for (w, h) in [(480,360),(640,480),(1280,720),(1920,1080)]:
            cap.set(_cv2.CAP_PROP_FRAME_WIDTH,  w)
            cap.set(_cv2.CAP_PROP_FRAME_HEIGHT, h)
            for fps_try in (60, 30):
                cap.set(_cv2.CAP_PROP_FPS, fps_try)
                aw = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
                ah = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
                af = cap.get(_cv2.CAP_PROP_FPS)
                if aw == w and ah == h:
                    results.append(f"{w}×{h}@{af:.0f}")
                    break
        cap.release()
        if results:
            self._cam_caps_lbl.config(text="  ".join(results))
            parts = results[-1].replace("×","@").split("@")
            if len(parts) == 3:
                self._v_video_width.set(parts[0])
                self._v_video_height.set(parts[1])
                self._v_video_fps.set(parts[2])
        else:
            self._cam_caps_lbl.config(text="No modes found")


# =============================================================================
# SERIES REVIEW WINDOW  (series picker, working toggles, scrollable shot list)
# =============================================================================

class SeriesReviewWindow(tk.Toplevel):
    """
    Review window with:
    - Series picker at top (current session's series + any saved sessions)
    - Target canvas on left
    - Right panel: stats, display toggles (working), scrollable shot list
      with per-shot checkbox (show/hide) and delete button
    """

    def __init__(self, parent, session, cfg, target_cfg,
                 on_next_series=None, on_close_refresh=None):
        super().__init__(parent)
        self.live_session   = session
        self.cfg            = cfg
        self.target_cfg     = target_cfg
        self.on_next_series = on_next_series
        self.on_close_refresh = on_close_refresh

        # Active view state
        self._view_session  = session          # Session object being viewed
        self._view_series   = session.current_series
        self._is_live       = True             # True = viewing live session

        self._renderer    = None
        self._tgt_img_id  = None
        self._shot_chk_vars = {}   # shot.index → BooleanVar

        # Display toggle vars — proper BooleanVars, toggled correctly
        self._show_traces = tk.BooleanVar(value=True)
        self._show_acp    = tk.BooleanVar(value=True)
        self._show_bbox_s = tk.BooleanVar(value=False)
        self._show_bbox_a = tk.BooleanVar(value=False)
        self._dot_only    = tk.BooleanVar(value=False)
        self._show_group  = tk.BooleanVar(value=False)

        self.title("Series Review — Splatt2")
        self.configure(bg=BG_DARK)
        self.geometry("1200x750")
        self.minsize(900, 600)
        self.resizable(True, True)

        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(150, self._init_renderer)

    # =========================================================================
    # BUILD
    # =========================================================================

    def _build(self):
        # ── Top bar: two-dropdown day / series picker ───────────────────────
        top = tk.Frame(self, bg=BG_MID)
        top.pack(side="top", fill="x")

        tk.Label(top, text="Day:", bg=BG_MID, fg=TEXT_DIM,
                 font=FL).pack(side="left", padx=(12, 4), pady=10)
        self._day_var = tk.StringVar()
        self._day_combo = ttk.Combobox(top, textvariable=self._day_var,
                                        state="readonly", width=18, font=FM)
        self._day_combo.pack(side="left", pady=10)
        self._day_combo.bind("<<ComboboxSelected>>", self._on_day_selected)

        tk.Label(top, text="Series:", bg=BG_MID, fg=TEXT_DIM,
                 font=FL).pack(side="left", padx=(12, 4))
        self._series_var = tk.StringVar()
        self._series_combo = ttk.Combobox(top, textvariable=self._series_var,
                                           state="readonly", width=32, font=FM)
        self._series_combo.pack(side="left", pady=10)
        self._series_combo.bind("<<ComboboxSelected>>", self._on_series_selected)

        tk.Button(top, text="⟳", command=self._populate_series_picker,
                  bg=BG_MID, fg=TEXT_SEC, font=FL, relief="flat",
                  cursor="hand2", padx=6).pack(side="left", padx=4)

        self._view_lbl = tk.Label(top, text="", bg=BG_MID, fg=TEXT_DIM, font=FL)
        self._view_lbl.pack(side="right", padx=12)

        # ── Body ─────────────────────────────────────────────────────────────
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True)

        # Target canvas (left, expands)
        left = tk.Frame(body, bg=BG_DARK)
        left.pack(side="left", fill="both", expand=True, padx=(8, 4), pady=8)
        self._canvas = tk.Canvas(left, bg=BG_DARK, highlightthickness=0, bd=0)
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Configure>", self._on_canvas_resize)

        # Right panel (fixed 300px wide)
        right = tk.Frame(body, bg=BG_PANEL, width=300)
        right.pack(side="right", fill="y", padx=(4, 8), pady=8)
        right.pack_propagate(False)
        self._right = right
        self._build_right(right)

    def _build_right(self, parent):
        # ── Stats ─────────────────────────────────────────────────────────────
        sc = tk.Frame(parent, bg=BG_CARD)
        sc.pack(fill="x", padx=6, pady=(6, 3))
        tk.Label(sc, text="STATISTICS", bg=BG_CARD, fg=TEXT_DIM,
                 font=FL).pack(anchor="nw", padx=8, pady=(4, 2))
        sf = tk.Frame(sc, bg=BG_CARD)
        sf.pack(fill="x", padx=6, pady=(0, 6))
        self._stat_lbls = {}
        # Two-column layout for compact stats
        pairs = [
            ("score","Score"), ("avg","Avg"),
            ("mr","MR"),       ("es","ES"),
            ("fom","FOM"),     ("cep","CEP"),
            ("std_x","Std X"), ("std_y","Std Y"),
            ("mpi_x","MPI X"), ("mpi_y","MPI Y"),
            ("best","Best"),   ("worst","Worst"),
        ]
        for i in range(0, len(pairs), 2):
            row = tk.Frame(sf, bg=BG_CARD); row.pack(fill="x", pady=1)
            for key, lbl in pairs[i:i+2]:
                cell = tk.Frame(row, bg=BG_CARD); cell.pack(side="left", expand=True, fill="x")
                tk.Label(cell, text=lbl+":", bg=BG_CARD, fg=TEXT_DIM,
                         font=FL, anchor="w").pack(side="left")
                v = tk.Label(cell, text="—", bg=BG_CARD, fg=TEXT_SEC,
                             font=FM, anchor="w")
                v.pack(side="left", padx=(2,8))
                self._stat_lbls[key] = v

        # ── Display toggles ────────────────────────────────────────────────────
        tc = tk.Frame(parent, bg=BG_PANEL)
        tc.pack(fill="x", padx=6, pady=(0, 4))
        tk.Label(tc, text="DISPLAY", bg=BG_PANEL, fg=TEXT_DIM,
                 font=FL).pack(anchor="nw", pady=(2,2))

        self._tog_buttons = {}
        # Row 1: Traces, ACP, Dot
        r1 = tk.Frame(tc, bg=BG_PANEL); r1.pack(fill="x", pady=1)
        for key, text, var, col in [
            ("traces",  "Traces",   self._show_traces, ACCENT),
            ("acp",     "ACP",      self._show_acp,    "#4f8fff"),
            ("dot",     "● Dot",    self._dot_only,    ACCENT2),
        ]:
            btn = self._make_tog_btn(r1, text, var, col)
            btn.pack(side="left", padx=(0, 2))
            self._tog_buttons[key] = btn
        # Row 2: Shots Box, ACP Box, Group circle
        r2 = tk.Frame(tc, bg=BG_PANEL); r2.pack(fill="x", pady=1)
        for key, text, var, col in [
            ("bbox_s",  "Shots Box", self._show_bbox_s, "#4f8fff"),
            ("bbox_a",  "ACP Box",   self._show_bbox_a, "#4f8fff"),
            ("group",   "○ Group",   self._show_group,  "#4f8fff"),
        ]:
            btn = self._make_tog_btn(r2, text, var, col)
            btn.pack(side="left", padx=(0, 2))
            self._tog_buttons[key] = btn

        # ── Shot list ──────────────────────────────────────────────────────────
        lc = tk.Frame(parent, bg=BG_CARD)
        lc.pack(fill="both", expand=True, padx=6, pady=(0, 3))

        # Header with All/None
        hdr = tk.Frame(lc, bg=BG_CARD)
        hdr.pack(fill="x", padx=6, pady=(4, 2))
        tk.Label(hdr, text="SHOTS  (✓=show, ✕=delete)",
                 bg=BG_CARD, fg=TEXT_DIM, font=FL).pack(side="left")
        tk.Button(hdr, text="All",
                  command=lambda: self._select_all(True),
                  bg=BG_CARD, fg=TEXT_DIM, font=FL,
                  relief="flat", padx=4, cursor="hand2").pack(side="right")
        tk.Button(hdr, text="None",
                  command=lambda: self._select_all(False),
                  bg=BG_CARD, fg=TEXT_DIM, font=FL,
                  relief="flat", padx=4, cursor="hand2").pack(side="right")

        # Scrollable shot rows
        sf2 = tk.Frame(lc, bg=BG_DARK)
        sf2.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self._list_canvas = tk.Canvas(sf2, bg=BG_DARK,
                                       highlightthickness=0, bd=0)
        vsb = tk.Scrollbar(sf2, orient="vertical",
                            command=self._list_canvas.yview,
                            bg=BG_DARK, troughcolor=BG_DARK)
        self._list_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._list_canvas.pack(side="left", fill="both", expand=True)
        self._list_inner = tk.Frame(self._list_canvas, bg=BG_DARK)
        self._list_win = self._list_canvas.create_window(
            (0, 0), window=self._list_inner, anchor="nw")

        def _cfg_scroll(e):
            self._list_canvas.configure(
                scrollregion=self._list_canvas.bbox("all"))
            self._list_canvas.itemconfig(
                self._list_win, width=self._list_canvas.winfo_width())
        self._list_inner.bind("<Configure>", _cfg_scroll)

        def _wheel(e):
            self._list_canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        self._list_canvas.bind("<MouseWheel>", _wheel)

        # ── Actions ────────────────────────────────────────────────────────────
        bf = tk.Frame(parent, bg=BG_PANEL)
        bf.pack(fill="x", padx=6, pady=(0, 6))
        tk.Button(bf, text="▶  Next Series",
                  command=self._next_series,
                  bg=ACCENT, fg=BG_DARK, font=FB, relief="flat",
                  pady=6, cursor="hand2").pack(fill="x", pady=1)
        tk.Button(bf, text="💾  Save CSV",
                  command=self._save,
                  bg=BG_CARD, fg=TEXT_SEC, font=FB, relief="flat",
                  pady=5, cursor="hand2").pack(fill="x", pady=1)
        tk.Button(bf, text="✕  Close",
                  command=self._on_close,
                  bg=BG_CARD, fg=TEXT_SEC, font=FB, relief="flat",
                  pady=5, cursor="hand2").pack(fill="x", pady=1)

    def _make_tog_btn(self, parent, text, var, col):
        """Create a toggle button that actually works — no closure bug."""
        def toggle():
            var.set(not var.get())
            btn.config(
                bg=col    if var.get() else BG_CARD,
                fg=BG_DARK if var.get() else TEXT_SEC)
            self._redraw()
        btn = tk.Button(parent, text=text, command=toggle,
                        bg=col    if var.get() else BG_CARD,
                        fg=BG_DARK if var.get() else TEXT_SEC,
                        font=FL, relief="flat", bd=0,
                        padx=6, pady=4, cursor="hand2")
        return btn

    # =========================================================================
    # SERIES PICKER
    # =========================================================================

    def _populate_series_picker(self):
        """Build the day + series dropdowns from live session and saved files."""
        from core.session import load_session_history
        save_dir = (self.cfg.get("save_directory") or "").strip() or _default_save_dir()
        save_dir = os.path.abspath(save_dir)

        # Load saved history {day_label: [session_dict, ...]}
        self._history = load_session_history(save_dir)

        # Build day list: "Today (live)" first, then saved days newest-first
        today = __import__("time").strftime("%Y-%m-%d")
        day_labels = ["Today (live)"]
        for day in sorted(self._history.keys(), reverse=True):
            display = day if day != today else f"{day} (today)"
            day_labels.append(display)
        self._day_labels = day_labels  # display label → raw key map
        self._day_keys   = ["__live__"] + list(sorted(self._history.keys(), reverse=True))

        self._day_combo["values"] = day_labels
        self._day_combo.current(0)
        self._day_var.set(day_labels[0])
        self._on_day_selected()

    def _on_day_selected(self, event=None):
        """Populate the series dropdown for the selected day."""
        idx = self._day_combo.current()
        if idx < 0:
            return
        day_key = self._day_keys[idx] if idx < len(self._day_keys) else "__live__"
        self._current_day_key = day_key

        series_entries = []

        if day_key == "__live__":
            # Live session series
            all_series = sorted(set(s.series for s in self.live_session.shots))
            if not all_series:
                all_series = [self.live_session.current_series]
            for sn in all_series:
                shots = [s for s in self.live_session.shots if s.series == sn]
                score = sum(s.score for s in shots if s.match_shot and not s.deleted)
                n = len([s for s in shots if not s.deleted])
                label = f"Series {sn}  ({n} shots,  {score:.1f} pts)"
                series_entries.append({"label": label, "type": "live", "series": sn})
        else:
            # Saved day — list each file as a series entry
            for h in self._history.get(day_key, []):
                shots_data = h["raw"].get("shots", [])
                all_sn = sorted(set(s.get("series", 1) for s in shots_data))
                for sn in all_sn:
                    s_shots = [s for s in shots_data if s.get("series", 1) == sn]
                    scores  = [s["score"] for s in s_shots]
                    label   = (f"{h['date']}  {h['name']}  — Series {sn}"
                               f"  ({len(s_shots)} shots,  {sum(scores):.1f} pts)")
                    series_entries.append({"label": label, "type": "saved",
                                           "series": sn, "history": h,
                                           "raw_shots": s_shots})

        self._series_entries = series_entries
        labels = [e["label"] for e in series_entries]
        self._series_combo["values"] = labels
        if labels:
            # Default to most recent series
            target_sn = self.live_session.current_series if day_key == "__live__" else None
            sel = 0
            if target_sn:
                for i, e in enumerate(series_entries):
                    if e["series"] == target_sn:
                        sel = i; break
            self._series_combo.current(sel)
            self._series_var.set(labels[sel])
        self._load_selected_series()

    def _on_series_selected(self, event=None):
        self._load_selected_series()

    def _load_selected_series(self):
        """Load whichever series is selected into the view."""
        if not hasattr(self, "_series_entries") or not self._series_entries:
            return
        idx = self._series_combo.current()
        if idx < 0 or idx >= len(self._series_entries):
            return
        entry = self._series_entries[idx]

        if entry["type"] == "live":
            self._view_session = self.live_session
            self._view_series  = entry["series"]
            self._is_live      = (entry["series"] == self.live_session.current_series)
            self._view_lbl.config(text="● LIVE" if self._is_live else "○ Past",
                                   fg=ACCENT if self._is_live else TEXT_DIM)
        else:
            from core.session import Session, Shot, reconstruct_shot_traces
            raw_shots = entry["raw_shots"]
            traces    = reconstruct_shot_traces({"shots": raw_shots})
            fake_sess = Session(entry["history"]["name"])
            fake_sess.current_series = entry["series"]
            for i, sd in enumerate(raw_shots):
                acp = sd.get("aim_centrepoint")
                fake_sess.shots.append(Shot(
                    index=sd.get("index", i+1),
                    timestamp=sd.get("timestamp", 0),
                    aim_mm=tuple(sd["aim_mm"]),
                    score=sd["score"],
                    ring_index=sd.get("ring_index", 0),
                    series=sd.get("series", 1),
                    trace=traces[i] if i < len(traces) else None,
                    aim_centrepoint=tuple(acp) if acp else None,
                ))
            self._view_session = fake_sess
            self._view_series  = entry["series"]
            self._is_live      = False
            self._view_lbl.config(text="○ Saved", fg=TEXT_DIM)

        self._rebuild_shot_list()
        self._update_stats()
        self._redraw()

    # =========================================================================
    # SHOT LIST
    # =========================================================================

    def _rebuild_shot_list(self):
        """Rebuild the per-shot checkbox rows for the current view."""
        for w in self._list_inner.winfo_children():
            w.destroy()
        self._shot_chk_vars = {}

        shots = [s for s in self._view_session.shots
                 if s.series == self._view_series]

        if not shots:
            tk.Label(self._list_inner, text="No shots in this series.",
                     bg=BG_DARK, fg=TEXT_DIM, font=FL).pack(pady=8)
            return

        for shot in shots:
            var = tk.BooleanVar(value=not shot.deleted)
            self._shot_chk_vars[shot.index] = var

            row = tk.Frame(self._list_inner, bg=BG_DARK)
            row.pack(fill="x", pady=1, padx=2)

            sc  = shot.score
            sc_s = f"{sc:.1f}" if sc != int(sc) else str(int(sc))
            col = GOLD if sc >= 10 else (ACCENT if sc >= 9 else
                  TEXT_PRI if sc >= 7 else (ACCENT2 if sc == 0 else TEXT_SEC))
            if shot.deleted:
                col = TEXT_DIM

            # Checkbox — compact one-line label
            chk_text = (f"#{shot.index:>2}  {sc_s:>5}pt"
                        f"  ({shot.aim_mm[0]:>+5.1f},{shot.aim_mm[1]:>+5.1f})")
            chk = tk.Checkbutton(
                row, text=chk_text, variable=var,
                bg=BG_DARK, fg=col, selectcolor=BG_CARD,
                activebackground=BG_DARK, font=("Consolas", 9),
                anchor="w",
                command=lambda s=shot, v=var: self._on_shot_toggle(s, v))
            chk.pack(side="left", fill="x", expand=True)

            # Delete button — only for live editable sessions
            if self._is_live:
                tk.Button(row, text="✕",
                          command=lambda s=shot: self._delete_shot(s),
                          bg=BG_DARK, fg=ACCENT2,
                          font=("Consolas", 9), relief="flat", bd=0,
                          padx=3, cursor="hand2").pack(side="right")

    def _on_shot_toggle(self, shot, var):
        shot.deleted = not var.get()
        self._update_stats()
        self._redraw()

    def _delete_shot(self, shot):
        if messagebox.askyesno("Delete",
                                f"Delete shot #{shot.index} ({shot.score} pts)?",
                                parent=self):
            try:
                self._view_session.shots.remove(shot)
            except ValueError:
                pass
            self._rebuild_shot_list()
            self._update_stats()
            self._redraw()

    def _select_all(self, value: bool):
        shots = [s for s in self._view_session.shots
                 if s.series == self._view_series]
        for shot in shots:
            shot.deleted = not value
            if shot.index in self._shot_chk_vars:
                self._shot_chk_vars[shot.index].set(value)
        self._update_stats()
        self._redraw()

    # =========================================================================
    # RENDERER
    # =========================================================================

    def _init_renderer(self):
        from core.target_renderer import TargetRenderer
        w = self._canvas.winfo_width()
        h = self._canvas.winfo_height()
        if w < 50 or h < 50:
            self.after(100, self._init_renderer)
            return
        cal = float(self.cfg.get("shot_circle_calibre_mm",
                                   self.target_cfg.get("calibre_mm", 4.5)))
        self._renderer = TargetRenderer((w, h), self.target_cfg,
                                         display_calibre_mm=cal,
                                         display_cfg=self.cfg)
        self._populate_series_picker()

    def _on_canvas_resize(self, event):
        from core.target_renderer import TargetRenderer
        if event.width < 50 or event.height < 50:
            return
        cal = float(self.cfg.get("shot_circle_calibre_mm",
                                   self.target_cfg.get("calibre_mm", 4.5)))
        self._renderer = TargetRenderer((event.width, event.height),
                                         self.target_cfg,
                                         display_calibre_mm=cal,
                                         display_cfg=self.cfg)
        self._redraw()

    def _redraw(self):
        if self._renderer is None:
            return
        import cv2
        from PIL import Image, ImageTk

        visible = [s for s in self._view_session.shots
                   if s.series == self._view_series and not s.deleted]

        img = self._renderer.render(
            shots=visible,
            show_mpi=self._show_group.get(),
            show_group=self._show_group.get(),
            current_series=self._view_series,
            show_acp=self._show_acp.get(),
            show_traces=self._show_traces.get(),
            show_bbox_shots=self._show_bbox_s.get(),
            show_bbox_acp=self._show_bbox_a.get(),
            show_dot_only=self._dot_only.get(),
            trace_alpha=1.0,  # full brightness in review
        )
        photo = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        if self._tgt_img_id is None:
            self._tgt_img_id = self._canvas.create_image(
                0, 0, anchor="nw", image=photo)
        else:
            self._canvas.itemconfig(self._tgt_img_id, image=photo)
        self._canvas._img = photo

    # =========================================================================
    # STATS
    # =========================================================================

    def _update_stats(self):
        visible = [s for s in self._view_session.shots
                   if s.series == self._view_series
                   and s.match_shot and not s.deleted]
        n = len(visible)

        def _f(v, fmt=".2f"):
            return f"{v:{fmt}}" if v is not None else "—"

        if not visible:
            for v in self._stat_lbls.values():
                v.config(text="—", fg=TEXT_SEC)
            return

        import numpy as np
        coords = np.array([s.aim_mm for s in visible])
        radii  = np.sqrt(coords[:,0]**2 + coords[:,1]**2)
        score  = sum(s.score for s in visible)
        avg    = score / n

        self._stat_lbls["score"].config(
            text=f"{score:.1f} ({n})", fg=GOLD if score > 0 else TEXT_SEC)
        self._stat_lbls["avg"  ].config(text=f"{avg:.2f}")
        self._stat_lbls["mr"   ].config(text=f"{float(np.mean(radii)):.2f}mm")

        if n >= 2:
            es = float(max(
                np.linalg.norm(coords[i]-coords[j])
                for i in range(n) for j in range(i+1,n)))
            fom = (float(np.max(coords[:,0])-np.min(coords[:,0])) +
                   float(np.max(coords[:,1])-np.min(coords[:,1]))) / 2
            cep = float(np.percentile(radii, 50))
            sx  = float(np.std(coords[:,0]))
            sy  = float(np.std(coords[:,1]))
            mpi = (float(np.mean(coords[:,0])), float(np.mean(coords[:,1])))
            self._stat_lbls["es"   ].config(text=f"{es:.2f}mm")
            self._stat_lbls["fom"  ].config(text=f"{fom:.2f}mm")
            self._stat_lbls["cep"  ].config(text=f"{cep:.2f}mm")
            self._stat_lbls["std_x"].config(text=f"{sx:.2f}mm")
            self._stat_lbls["std_y"].config(text=f"{sy:.2f}mm")
            self._stat_lbls["mpi_x"].config(text=f"{mpi[0]:+.2f}mm")
            self._stat_lbls["mpi_y"].config(text=f"{mpi[1]:+.2f}mm")
        else:
            for k in ("es","fom","cep","std_x","std_y","mpi_x","mpi_y"):
                self._stat_lbls[k].config(text="—")

        best  = max(visible, key=lambda s: s.score)
        worst = min(visible, key=lambda s: s.score)
        self._stat_lbls["best" ].config(
            text=f"#{best.index} {best.score:.1f}", fg=GOLD)
        self._stat_lbls["worst"].config(
            text=f"#{worst.index} {worst.score:.1f}", fg=ACCENT2)

    # =========================================================================
    # ACTIONS
    # =========================================================================

    def _next_series(self):
        if self.on_next_series:
            self.on_next_series()
        self._on_close()

    def _save(self):
        from tkinter.filedialog import asksaveasfilename
        out = asksaveasfilename(
            parent=self,
            title="Save series CSV",
            defaultextension=".csv",
            filetypes=[("CSV","*.csv"),("All","*.*")],
            initialfile=f"series_{self._view_series}.csv")
        if out:
            self._view_session.save_csv(out)

    def _on_close(self):
        if self.on_close_refresh:
            self.on_close_refresh()
        self.destroy()
