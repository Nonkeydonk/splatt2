"""Path resolution for source, PyInstaller and Nuitka builds.

`resource_path` returns bundled read-only assets (targets/ etc).
`user_data_dir` and friends return writable per-user paths following
OS conventions. Set SPLATT2_USER_DIR to override the user dir.
"""

import os
import sys
from pathlib import Path

APP_NAME = "Splatt2"


def is_frozen() -> bool:
    """True when running from a PyInstaller or Nuitka bundle."""
    return (getattr(sys, "frozen", False)
            or hasattr(sys, "_MEIPASS")
            or "__compiled__" in globals())


def _bundle_root() -> Path:
    # PyInstaller sets _MEIPASS to the bundle root (extracted temp dir
    # for onefile, _internal/ for onedir). Source and Nuitka layouts
    # both have core/ one level below the project root.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parent.parent


def resource_path(*parts: str) -> Path:
    """Path to a bundled read-only asset, e.g. ``resource_path('targets')``."""
    return _bundle_root().joinpath(*parts)


def _platform_user_dir() -> Path:
    override = os.environ.get("SPLATT2_USER_DIR")
    if override:
        return Path(override).expanduser()

    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME

    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / APP_NAME


def user_data_dir() -> Path:
    """Per-user writable directory, created on first access."""
    p = _platform_user_dir()
    p.mkdir(parents=True, exist_ok=True)
    return p


def config_path() -> Path:
    return user_data_dir() / "splatt2_config.json"


def crash_log_path() -> Path:
    return user_data_dir() / "splatt2_crash.log"


def sessions_dir() -> Path:
    """Default sessions/ folder, created on first access."""
    p = user_data_dir() / "sessions"
    p.mkdir(parents=True, exist_ok=True)
    return p
