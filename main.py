"""
Splatt2 — DIY Target Shooting Trainer
Entry point with crash logging.
"""

import os
import shutil
import sys
import traceback

_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)


def _migrate_legacy_crash_log(dst: str) -> None:
    """Copy a pre-1.2 crash log from next to main.py into the user dir."""
    if os.path.exists(dst):
        return
    legacy = os.path.join(_BASE, "splatt2_crash.log")
    if os.path.isfile(legacy):
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(legacy, dst)
        except OSError:
            pass


def _write_crash_log(exc_text: str):
    import datetime
    from core.paths import crash_log_path

    path = str(crash_log_path())
    _migrate_legacy_crash_log(path)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"Crash at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'=' * 60}\n")
            f.write(exc_text)
            f.write("\n")
        return path
    except OSError:
        return None


def main():
    print("Starting Splatt2...")
    from ui.app import SplattApp
    app = SplattApp()
    app.run()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        exc_text = traceback.format_exc()
        print(exc_text)
        log_path = _write_crash_log(exc_text)

        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            msg = f"Splatt2 crashed unexpectedly.\n\n{exc_text[:400]}\n\n"
            if log_path:
                msg += f"Full details saved to:\n{log_path}\n\nPlease include this file when reporting the issue."
            messagebox.showerror("Splatt2 — Crash", msg)
            root.destroy()
        except Exception:
            pass

        sys.exit(1)
