# -*- mode: python ; coding: utf-8 -*-

"""PyInstaller spec for building scorpipe.exe (Windows).

Build (PowerShell):
  pip install -U pyinstaller
  pyinstaller packaging/windows/scorpipe.spec
"""

from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(__file__).resolve().parents[2]

block_cipher = None

datas = []
datas += collect_data_files("scorpio_pipe")  # package data (resources, html templates, etc.)

hiddenimports = []
hiddenimports += collect_submodules("scorpio_pipe")

a = Analysis(
    [str(ROOT / "packaging" / "windows" / "entry_gui.py")],
    pathex=[str(ROOT), str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="scorpipe",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="scorpipe",
)
