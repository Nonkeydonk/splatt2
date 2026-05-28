"""Visual theme: palette, fonts, button factories and ttk styling.

Tk on macOS uses a native button renderer that ignores ``bg``/``fg``,
so all interactive widgets in this app are ``ttk`` instances driven by
the named styles configured in :func:`apply_theme`.
"""

from __future__ import annotations

import tkinter as tk
import tkinter.ttk as ttk
from typing import Callable, Optional

# Background tones, dark to light.
BG_DARK = "#0f0f13"
BG_MID = "#16161d"
BG_PANEL = "#1c1c25"
BG_CARD = "#22222e"

# Foreground accents and text.
ACCENT = "#00e5a0"
ACCENT2 = "#ff4f6d"
BLUE = "#4f8fff"
TEXT_PRI = "#f0f0f8"
TEXT_SEC = "#c0c0d8"
TEXT_DIM = "#9090b0"
BORDER = "#2a2a3a"
GOLD = "#ffd060"

# Fonts.
FM = ("Consolas", 10)
FT = ("Segoe UI", 9, "bold")
FS = ("Consolas", 42, "bold")
FL = ("Segoe UI", 9)
FB = ("Segoe UI", 10)
FH = ("Segoe UI", 11)
FL_BOLD = ("Segoe UI", 9, "bold")


# Style names. Toggle-on styles are created lazily per accent colour
# so a single shared idle style covers the off state.
_STYLE_DEFAULT = "Splatt.TButton"
_STYLE_ACCENT = "Splatt.Accent.TButton"
_STYLE_DANGER = "Splatt.Danger.TButton"
_STYLE_CHIP = "Splatt.Chip.TButton"
_STYLE_TOGGLE_OFF = "Splatt.Toggle.Off.TButton"

_VARIANT_STYLES = {
    "default": _STYLE_DEFAULT,
    "accent": _STYLE_ACCENT,
    "danger": _STYLE_DANGER,
    "chip": _STYLE_CHIP,
    "toggle_off": _STYLE_TOGGLE_OFF,
}

# Maps an accent colour to its lazily-created toggle-on style name.
_TOGGLE_ON_STYLES: dict[str, str] = {}


def _toggle_on_style_name(accent_color: str) -> str:
    return f"Splatt.Toggle.On.{accent_color.lstrip('#')}.TButton"


def make_button(
    parent: tk.Misc,
    text: str,
    command: Callable[[], None],
    *,
    accent: bool = False,
    danger: bool = False,
    width: Optional[int] = None,
) -> ttk.Button:
    """Themed flat button.

    ``accent`` picks the primary action style, ``danger`` the destructive
    one. Without either, the neutral default style is used.
    """
    if danger:
        style = _STYLE_DANGER
    elif accent:
        style = _STYLE_ACCENT
    else:
        style = _STYLE_DEFAULT
    kwargs: dict = {"text": text, "command": command, "style": style,
                    "cursor": "hand2"}
    if width is not None:
        kwargs["width"] = width
    return ttk.Button(parent, **kwargs)


def make_toggle_button(
    parent: tk.Misc,
    text: str,
    state: bool,
    command: Callable[[], None],
    *,
    accent_color: str = ACCENT,
) -> ttk.Button:
    """Themed two-state toggle button.

    The active state fills with ``accent_color`` and uses bold text;
    the inactive state shares the chip-like idle look.
    """
    btn = ttk.Button(parent, text=text, command=command, cursor="hand2")
    set_toggle_state(btn, state, accent_color=accent_color)
    return btn


def make_chip_button(
    parent: tk.Misc,
    text: str,
    command: Callable[[], None],
) -> ttk.Button:
    """Compact preset/value button used inline in dialog rows."""
    return ttk.Button(parent, text=text, command=command,
                      style=_STYLE_CHIP, cursor="hand2")


def set_button_variant(button: ttk.Button, variant: str) -> None:
    """Switch a button between visual variants at runtime.

    ``variant`` is one of ``default``, ``accent``, ``danger``, ``chip``
    or ``toggle_off``. Unknown names fall back to ``default``.
    """
    button.configure(style=_VARIANT_STYLES.get(variant, _STYLE_DEFAULT))


def set_toggle_state(
    button: ttk.Button,
    active: bool,
    *,
    accent_color: str = ACCENT,
) -> None:
    """Toggle a button between its on (filled) and off (idle) styling."""
    if active:
        _ensure_toggle_on_style(accent_color)
        button.configure(style=_TOGGLE_ON_STYLES[accent_color])
    else:
        button.configure(style=_STYLE_TOGGLE_OFF)


def _ensure_toggle_on_style(accent_color: str) -> None:
    if accent_color in _TOGGLE_ON_STYLES:
        return
    name = _toggle_on_style_name(accent_color)
    s = ttk.Style()
    s.configure(
        name,
        background=accent_color, foreground=BG_DARK,
        bordercolor=accent_color,
        lightcolor=accent_color, darkcolor=accent_color,
        focuscolor=accent_color,
        font=FL_BOLD, padding=(8, 3), relief="flat", borderwidth=0,
    )
    s.map(
        name,
        background=[("active", accent_color), ("pressed", accent_color)],
        foreground=[("active", BG_DARK), ("pressed", BG_DARK)],
        bordercolor=[("active", accent_color)],
    )
    _TOGGLE_ON_STYLES[accent_color] = name


def apply_theme(root: tk.Misc) -> None:
    """Install the dark palette on ``root`` and all its toplevels.

    Must run before any widgets are created so option-database entries
    apply to plain ``tk`` widgets in dialogs. ttk style settings are
    global to the interpreter and propagate to every toplevel.
    """
    style = ttk.Style(root)
    # ``clam`` is the only built-in theme that honours custom colours
    # reliably across platforms; the macOS ``aqua`` theme overrides
    # backgrounds for buttons and entries.
    style.theme_use("clam")

    _configure_buttons(style)
    _configure_inputs(style)
    _configure_containers(style)
    _configure_indicators(style)
    _install_option_db(root)


def _configure_buttons(style: ttk.Style) -> None:
    style.configure(
        _STYLE_DEFAULT,
        background=BG_CARD, foreground=TEXT_PRI,
        bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
        focuscolor=BG_CARD,
        font=FB, padding=(10, 6), relief="flat", borderwidth=0,
    )
    style.map(
        _STYLE_DEFAULT,
        background=[("active", BG_MID), ("pressed", BORDER),
                    ("disabled", BG_PANEL)],
        foreground=[("active", TEXT_PRI), ("pressed", ACCENT),
                    ("disabled", TEXT_DIM)],
        bordercolor=[("focus", ACCENT), ("active", BORDER)],
    )

    style.configure(
        _STYLE_ACCENT,
        background=ACCENT, foreground=BG_DARK,
        bordercolor=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT,
        focuscolor=ACCENT,
        font=FB, padding=(10, 6), relief="flat", borderwidth=0,
    )
    style.map(
        _STYLE_ACCENT,
        background=[("active", ACCENT), ("pressed", "#00b87f"),
                    ("disabled", BG_PANEL)],
        foreground=[("active", BG_DARK), ("pressed", BG_DARK),
                    ("disabled", TEXT_DIM)],
        bordercolor=[("focus", ACCENT)],
    )

    style.configure(
        _STYLE_DANGER,
        background=ACCENT2, foreground=BG_DARK,
        bordercolor=ACCENT2, lightcolor=ACCENT2, darkcolor=ACCENT2,
        focuscolor=ACCENT2,
        font=FB, padding=(10, 6), relief="flat", borderwidth=0,
    )
    style.map(
        _STYLE_DANGER,
        background=[("active", ACCENT2), ("pressed", "#cc3a55")],
        foreground=[("active", BG_DARK), ("pressed", BG_DARK)],
    )

    chip_config = dict(
        background=BG_CARD, foreground=TEXT_DIM,
        bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
        focuscolor=BG_CARD,
        font=FL, padding=(8, 3), relief="flat", borderwidth=0,
    )
    chip_map = dict(
        background=[("active", BG_MID), ("pressed", BORDER)],
        foreground=[("active", TEXT_PRI), ("pressed", ACCENT)],
        bordercolor=[("active", ACCENT), ("focus", ACCENT)],
    )
    for name in (_STYLE_CHIP, _STYLE_TOGGLE_OFF):
        style.configure(name, **chip_config)
        style.map(name, **chip_map)

    for accent in (ACCENT, GOLD, BLUE, ACCENT2):
        _ensure_toggle_on_style(accent)


def _configure_inputs(style: ttk.Style) -> None:
    style.configure(
        "TCombobox",
        fieldbackground=BG_CARD, background=BG_CARD,
        foreground=TEXT_PRI, arrowcolor=ACCENT,
        bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
        selectbackground=BG_CARD, selectforeground=TEXT_PRI,
        insertcolor=ACCENT, padding=4,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", BG_CARD), ("disabled", BG_PANEL)],
        foreground=[("disabled", TEXT_DIM)],
        background=[("active", BG_MID), ("readonly", BG_CARD)],
        bordercolor=[("focus", ACCENT), ("hover", ACCENT)],
        arrowcolor=[("hover", TEXT_PRI)],
    )

    for name in ("TEntry", "TSpinbox"):
        style.configure(
            name,
            fieldbackground=BG_CARD, background=BG_CARD,
            foreground=TEXT_PRI,
            bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
            insertcolor=ACCENT,
            selectbackground=ACCENT, selectforeground=BG_DARK,
            padding=4,
        )
        style.map(
            name,
            fieldbackground=[("disabled", BG_PANEL)],
            foreground=[("disabled", TEXT_DIM)],
            bordercolor=[("focus", ACCENT)],
        )

    for name in ("TScale", "Horizontal.TScale"):
        style.configure(
            name,
            background=BG_MID, troughcolor=BG_CARD,
            bordercolor=BG_MID, lightcolor=BG_MID, darkcolor=BG_MID,
        )


def _configure_containers(style: ttk.Style) -> None:
    style.configure("TFrame", background=BG_DARK)
    style.configure("TLabel", background=BG_DARK, foreground=TEXT_SEC)
    style.configure(
        "TLabelframe",
        background=BG_DARK, foreground=TEXT_DIM,
        bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
    )
    style.configure(
        "TLabelframe.Label",
        background=BG_DARK, foreground=TEXT_DIM,
    )

    style.configure("TNotebook", background=BG_DARK, borderwidth=0)
    style.configure(
        "TNotebook.Tab",
        background=BG_CARD, foreground=TEXT_SEC,
        padding=[10, 4], borderwidth=0,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", BG_PANEL), ("active", BG_MID)],
        foreground=[("selected", ACCENT), ("active", TEXT_PRI)],
    )

    for name in ("Vertical.TScrollbar", "Horizontal.TScrollbar"):
        style.configure(
            name,
            background=BG_CARD, troughcolor=BG_DARK,
            bordercolor=BG_DARK, arrowcolor=TEXT_DIM,
            gripcount=0, relief="flat",
            lightcolor=BG_CARD, darkcolor=BG_CARD,
        )
        style.map(
            name,
            background=[("active", BG_MID), ("pressed", BORDER)],
            arrowcolor=[("active", ACCENT), ("pressed", ACCENT)],
        )


def _configure_indicators(style: ttk.Style) -> None:
    for name, col in (("Quality", ACCENT), ("Audio", BLUE)):
        style.configure(
            f"{name}.Horizontal.TProgressbar",
            troughcolor=BG_DARK, background=col,
            bordercolor=BG_DARK, lightcolor=col, darkcolor=col,
        )

    style.configure(
        "TCheckbutton",
        background=BG_DARK, foreground=TEXT_SEC,
        focuscolor=BG_DARK,
        indicatorcolor=BG_CARD, indicatorbackground=BG_CARD,
        bordercolor=BORDER,
    )
    style.map(
        "TCheckbutton",
        background=[("active", BG_DARK)],
        foreground=[("disabled", TEXT_DIM), ("active", TEXT_PRI)],
        indicatorcolor=[("selected", ACCENT), ("pressed", ACCENT)],
    )

    style.configure(
        "TRadiobutton",
        background=BG_DARK, foreground=TEXT_SEC,
        focuscolor=BG_DARK,
        indicatorcolor=BG_CARD, bordercolor=BORDER,
    )
    style.map(
        "TRadiobutton",
        background=[("active", BG_DARK)],
        foreground=[("disabled", TEXT_DIM), ("active", TEXT_PRI)],
        indicatorcolor=[("selected", ACCENT)],
    )


def _install_option_db(root: tk.Misc) -> None:
    """Theme plain ``tk`` widgets that don't go through ``ttk.Style``."""
    options = {
        "*TCombobox*Listbox.background": BG_CARD,
        "*TCombobox*Listbox.foreground": TEXT_PRI,
        "*TCombobox*Listbox.selectBackground": ACCENT,
        "*TCombobox*Listbox.selectForeground": BG_DARK,
        "*TCombobox*Listbox.borderWidth": 0,
        "*TCombobox*Listbox.relief": "flat",
        "*TCombobox*Listbox.font": FM,

        "*Listbox.background": BG_CARD,
        "*Listbox.foreground": TEXT_PRI,
        "*Listbox.selectBackground": ACCENT,
        "*Listbox.selectForeground": BG_DARK,
        "*Listbox.borderWidth": 0,
        "*Listbox.highlightThickness": 0,

        "*Entry.background": BG_CARD,
        "*Entry.foreground": TEXT_PRI,
        "*Entry.insertBackground": ACCENT,
        "*Entry.selectBackground": ACCENT,
        "*Entry.selectForeground": BG_DARK,
        "*Entry.relief": "flat",
        "*Entry.borderWidth": 0,
        "*Entry.highlightThickness": 1,
        "*Entry.highlightBackground": BORDER,
        "*Entry.highlightColor": ACCENT,

        "*Checkbutton.background": BG_DARK,
        "*Checkbutton.foreground": TEXT_SEC,
        "*Checkbutton.activeBackground": BG_DARK,
        "*Checkbutton.activeForeground": TEXT_PRI,
        "*Checkbutton.selectColor": BG_CARD,
        "*Checkbutton.borderWidth": 0,
        "*Checkbutton.highlightThickness": 0,

        "*Menu.background": BG_PANEL,
        "*Menu.foreground": TEXT_PRI,
        "*Menu.activeBackground": ACCENT,
        "*Menu.activeForeground": BG_DARK,
        "*Menu.borderWidth": 0,
        "*Menu.relief": "flat",

        "*Scrollbar.background": BG_CARD,
        "*Scrollbar.troughColor": BG_DARK,
        "*Scrollbar.activeBackground": BG_MID,
        "*Scrollbar.borderWidth": 0,
        "*Scrollbar.elementBorderWidth": 0,
        "*Scrollbar.highlightThickness": 0,
    }
    for pattern, value in options.items():
        root.option_add(pattern, value)
