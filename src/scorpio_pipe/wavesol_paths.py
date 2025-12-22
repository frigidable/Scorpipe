from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def slugify_disperser(disperser: str | None) -> str:
    """Convert a disperser/grating name into a filesystem-safe slug."""
    s = (disperser or "").strip()
    if not s:
        return "default"

    # keep ASCII-ish characters; replace everything else with underscores
    s = s.replace("@", "_")
    s = re.sub(r"[^0-9A-Za-z._+-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "default"


def get_selected_disperser(cfg: dict[str, Any]) -> str:
    """Best-effort extract the selected disperser from config."""
    # 1) explicit override
    w = cfg.get("wavesol") if isinstance(cfg.get("wavesol"), dict) else {}
    if isinstance(w, dict):
        d = str(w.get("disperser", "") or "").strip()
        if d:
            return d

    # 2) setup saved by autoconfig
    fr = cfg.get("frames") if isinstance(cfg.get("frames"), dict) else {}
    setup = fr.get("__setup__") if isinstance(fr, dict) else None
    if isinstance(setup, dict):
        d = str(setup.get("disperser", "") or "").strip()
        if d:
            return d

    return ""


def resolve_work_dir(cfg: dict[str, Any]) -> Path:
    work_dir = Path(str(cfg.get("work_dir", "."))).expanduser()
    if not work_dir.is_absolute():
        base = Path(str(cfg.get("config_dir", "."))).expanduser().resolve()
        work_dir = (base / work_dir).resolve()
    else:
        work_dir = work_dir.resolve()
    return work_dir


def wavesol_dir(cfg: dict[str, Any]) -> Path:
    """Return a disperser-specific wavesolution directory.

    New convention (v2.1+):
      work_dir/wavesol/<disperser_slug>/...

    Backward-compatible fallback:
      if work_dir/wavesol exists and the subdir doesn't, we still allow reading
      from the legacy flat layout.
    """
    wd = resolve_work_dir(cfg)
    base = wd / "wavesol"
    slug = slugify_disperser(get_selected_disperser(cfg))
    sub = base / slug

    # Legacy compatibility: if legacy files exist directly in wavesol/, keep it usable.
    if (base / "superneon.fits").exists() and not sub.exists():
        return base
    return sub
