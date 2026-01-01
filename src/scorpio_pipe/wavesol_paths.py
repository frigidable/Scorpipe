from __future__ import annotations
from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.workspace_paths import stage_dir

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


def wavesol_dir(cfg: dict[str, Any]) -> Path:
    """Return a disperser-specific wavesolution directory.

    New convention (v5.38+):
      work_dir/products/07_wavesol/<disperser_slug>/...

    Backward-compatible fallback:
      if work_dir/wavesol exists and the subdir doesn't, we still allow reading
      from the legacy flat layout.
    """
    wd = resolve_work_dir(cfg)
    base = stage_dir(wd, "wavesolution")
    slug = slugify_disperser(get_selected_disperser(cfg))
    sub = base / slug

    # Legacy compatibility:
    # - old base: work_dir/wavesol/
    # - old per-disperser: work_dir/wavesol/<slug>/
    legacy_base = wd / "wavesol"
    legacy_sub = legacy_base / slug

    # If new location has files, prefer it.
    if sub.exists() or any((sub / n).exists() for n in ("lambda_map.fits", "rectification_model.json", "superneon.fits")):
        return sub

    # If old files exist directly in wavesol/, keep it usable.
    if (legacy_base / "superneon.fits").exists() and not legacy_sub.exists():
        return legacy_base
    # Prefer per-disperser legacy if present.
    if legacy_sub.exists():
        return legacy_sub
    # Default to new target (even if it doesn't exist yet).
    return sub
