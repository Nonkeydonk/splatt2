"""
Cross-platform PyInstaller build entry point for Splatt2.

Usage:
    python build/build.py            # build for the current OS
    python build/build.py --clean    # delete dist/ and build cache first

Output:
    dist/splatt2-<os>-<arch>/        # onedir bundle, ready to zip

The spec file (build/splatt2.spec) holds the actual build description.
This script handles cleanup, output renaming, and tagging by OS/arch so
CI and local builds produce identically-named artifacts.
"""

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "build" / "splatt2.spec"
DIST = ROOT / "dist"
WORK = ROOT / "build" / "_pyinstaller_work"


def _os_tag() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def _arch_tag() -> str:
    m = platform.machine().lower()
    return {
        "amd64": "x64", "x86_64": "x64",
        "arm64": "arm64", "aarch64": "arm64",
    }.get(m, m or "unknown")


def _rename_output(target_name: str) -> Path:
    """Move PyInstaller's per-OS output into dist/splatt2-<os>-<arch>/.

    On macOS the BUNDLE step produces ``dist/Splatt2.app`` — the artifact
    end users actually launch. It's moved inside the target folder so
    every platform produces a single zippable directory.
    """
    src = DIST / "splatt2"
    dst = DIST / target_name
    if dst.exists():
        shutil.rmtree(dst)
    if src.exists():
        src.rename(dst)

    if sys.platform == "darwin":
        app = DIST / "Splatt2.app"
        if app.exists():
            dst.mkdir(parents=True, exist_ok=True)
            shutil.move(str(app), str(dst / "Splatt2.app"))
    return dst


def _build(clean: bool) -> int:
    if clean:
        for p in (DIST, WORK):
            if p.exists():
                print(f"[build] Removing {p}")
                shutil.rmtree(p)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        f"--distpath={DIST}",
        f"--workpath={WORK}",
        str(SPEC),
    ]
    print("[build] " + " ".join(cmd))
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        return rc

    target = f"splatt2-{_os_tag()}-{_arch_tag()}"
    out = _rename_output(target)
    print(f"[build] Output: {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Build Splatt2 with PyInstaller")
    p.add_argument("--clean", action="store_true",
                   help="Delete dist/ and build cache before building")
    return _build(p.parse_args().clean)


if __name__ == "__main__":
    sys.exit(main())
