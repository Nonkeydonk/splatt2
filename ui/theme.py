"""Visual theme: palette, fonts and a button factory.

Centralising these lets dialogs and the main window share a consistent
look without duplicating colour values.
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable, Optional

# Background tones, dark to light.
BG_DARK = "#0f0f13"
BG_MID = "#16161d"
BG_PANEL = "#1c1c25"
BG_CARD = "#22222e"

# Foreground accents and text.
ACCENT = "#00e5a0"
ACCENT2 = "#ff4f6d"
TEXT_PRI = "#f0f0f8"
TEXT_SEC = "#c0c0d8"
TEXT_DIM = "#9090b0"
BORDER = "#2a2a3a"
GOLD = "#ffd060"

# Fonts. Values match the existing UI; rename only inside the UI package.
FM = ("Consolas", 10)
FT = ("Segoe UI", 9, "bold")
FS = ("Consolas", 42, "bold")
FL = ("Segoe UI", 9)
FB = ("Segoe UI", 10)
FH = ("Segoe UI", 11)


def make_button(
    parent: tk.Misc,
    text: str,
    command: Callable[[], None],
    accent: bool = False,
    width: Optional[int] = None,
) -> tk.Button:
    """Themed flat button used throughout the UI."""
    fg = BG_DARK if accent else TEXT_SEC
    bg = ACCENT if accent else BG_CARD
    kwargs = dict(
        bg=bg, fg=fg,
        activebackground=ACCENT if accent else BORDER,
        activeforeground=fg,
        font=FB, relief="flat", bd=0,
        padx=8, pady=5, cursor="hand2",
        text=text, command=command,
    )
    if width:
        kwargs["width"] = width
    return tk.Button(parent, **kwargs)
