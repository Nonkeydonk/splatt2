"""Application entry point with crash logging."""

from __future__ import annotations

import datetime
import os
import shutil
import sys
import traceback

_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)


def _migrate_legacy_crash_log(destination: str) -> None:
    """Copy any pre-1.2 crash log next to ``main.py`` into the user dir."""
    if os.path.exists(destination):
        return
    legacy = os.path.join(_BASE, "splatt2_crash.log")
    if not os.path.isfile(legacy):
        return
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy2(legacy, destination)
    except OSError:
        pass


def _write_crash_log(exc_text: str):
    """Append an exception traceback to the user crash log.

    Returns the log path on success, or ``None`` if the log could not be
    written.
    """
    from core.paths import crash_log_path

    path = str(crash_log_path())
    _migrate_legacy_crash_log(path)
    try:
        with open(path, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            divider = "=" * 60
            f.write(f"\n{divider}\n")
            f.write(f"Crash at {timestamp}\n")
            f.write(f"{divider}\n")
            f.write(exc_text)
            f.write("\n")
        return path
    except OSError:
        return None


def _show_crash_dialog(exc_text: str, log_path) -> None:
    """Best-effort Tk dialog showing the crash message and log location."""
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception:
        return

    try:
        root = tk.Tk()
        root.withdraw()
        message = f"Splatt2 crashed unexpectedly.\n\n{exc_text[:400]}\n\n"
        if log_path:
            message += (
                f"Full details saved to:\n{log_path}\n\n"
                "Please include this file when reporting the issue."
            )
        messagebox.showerror("Splatt2 — Crash", message)
        root.destroy()
    except Exception:
        pass


def main() -> None:
    print("Starting Splatt2...")
    from ui.app import SplattApp
    SplattApp().run()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        exc_text = traceback.format_exc()
        print(exc_text)
        log_path = _write_crash_log(exc_text)
        _show_crash_dialog(exc_text, log_path)
        sys.exit(1)
