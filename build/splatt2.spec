# PyInstaller spec for Splatt2.
#
# Run via build/build.py rather than directly so the working dir, output
# layout and platform tweaks are consistent.

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).parent
APP_NAME = "splatt2"
ENTRY = str(ROOT / "main.py")

datas = [(str(ROOT / "targets"), "targets")]
binaries = []
hiddenimports = [
    # Pillow's tkinter integration is loaded dynamically.
    "PIL._tkinter_finder",
]

# sounddevice ships its own PortAudio binary in _sounddevice_data/.
sd_datas, sd_binaries, sd_hidden = collect_all("sounddevice")
datas += sd_datas
binaries += sd_binaries
hiddenimports += sd_hidden

# scipy uses lazy submodule imports that PyInstaller's tracer misses.
hiddenimports += collect_submodules("scipy")


a = Analysis(
    [ENTRY],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)


_icon_dir = ROOT / "build" / "icons"
if sys.platform.startswith("win"):
    _icon = _icon_dir / "splatt2.ico"
elif sys.platform == "darwin":
    _icon = _icon_dir / "splatt2.icns"
else:
    _icon = _icon_dir / "splatt2.png"
icon_arg = str(_icon) if _icon.exists() else None


exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    console=False,
    disable_windowed_traceback=False,
    icon=icon_arg,
    contents_directory=".",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name=APP_NAME,
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Splatt2.app",
        bundle_identifier="com.splatt2.app",
        icon=icon_arg,
        info_plist={
            "CFBundleName": "Splatt2",
            "CFBundleDisplayName": "Splatt2",
            "NSHighResolutionCapable": True,
            "NSCameraUsageDescription":
                "Splatt2 uses the camera to track the rifle's aim point.",
            "NSMicrophoneUsageDescription":
                "Splatt2 uses the microphone to detect the shot click.",
        },
    )
