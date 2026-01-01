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
      work_dir/08_wavesol[/<disperser_slug>]/...

    Backward-compatible fallback:
      if work_dir/wavesol exists and the subdir doesn't, we still allow reading
      from the legacy flat layout.
    """
    wd = resolve_work_dir(cfg)
    # Canonical stage directory (v5.40+ layout): 08_wavesol/
    base = stage_dir(wd, "wavesol")
    slug = slugify_disperser(get_selected_disperser(cfg))
    sub = base / slug

    def _has_core_artifacts(p: Path) -> bool:
        names = (
            "lambda_map.fits",
            "rectification_model.json",
            "superneon.fits",
        )
        try:
            return any((p / n).exists() for n in names)
        except Exception:
            return False

    # If artifacts are already stored directly in the stage root, use that.
    # This is the default for most runs and for P2 synthetic tests.
    if _has_core_artifacts(base):
        return base

    # For multiple dispersers, we optionally keep a per-disperser subdir.
    # Only treat it as "selected" for reading if it contains the expected artifacts;
    # empty directories can appear transiently in synthetic runs.
    if slug != "default" and _has_core_artifacts(sub):
        return sub

# Legacy compatibility:
    # - old base: work_dir/wavesol/
    # - old per-disperser: work_dir/wavesol/<slug>/
    legacy_base = wd / "wavesol"
    legacy_sub = legacy_base / slug

    # Legacy: prefer per-disperser legacy if it contains core artifacts.
    if legacy_sub.exists() and _has_core_artifacts(legacy_sub):
        return legacy_sub

    # Legacy: if core artifacts exist directly in work_dir/wavesol/, keep it usable.
    if legacy_base.exists() and _has_core_artifacts(legacy_base):
        return legacy_base

    # Default: keep artifacts directly in the stage root.
    if slug == "default":
        return base
    # Default to new target (even if it doesn't exist yet).
    return sub
