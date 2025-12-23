from __future__ import annotations

import os
import sys
from pathlib import Path


def user_data_root(app_name: str = "Scorpipe") -> Path:
    """Return a per-user writable data directory.

    Used for caches, user libraries, and default work roots when running a
    frozen Windows build (installed under Program Files, which is not writable).
    """

    app = (app_name or "Scorpipe").strip() or "Scorpipe"

    if sys.platform.startswith("win"):
        base = (os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or "").strip()
        if base:
            return Path(base) / app
        # fallback
        return Path.home() / "AppData" / "Local" / app

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / app

    # Linux / other
    xdg = (os.environ.get("XDG_DATA_HOME") or "").strip()
    if xdg:
        return Path(xdg) / app
    return Path.home() / ".local" / "share" / app


def user_cache_root(app_name: str = "Scorpipe") -> Path:
    """Return a per-user writable cache directory."""
    root = user_data_root(app_name)
    return root / "cache"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
