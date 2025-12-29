"""Canonical workspace path helpers.

Layout (v5.38.6)
----------------
Stage outputs live directly under the *run root*:

    run_root/NN_slug/

Per-exposure outputs (when enabled) live under the stage directory:

    run_root/NN_slug/<stem_short>/

There is intentionally **no** "products/" directory in the new layout.

Compatibility
-------------
- If ``run_root/products`` exists, :func:`stage_dir` will transparently use it as
  the stage base (older layouts).
- If a run was produced by an older v5.38.x layout with different stage
  numbering/slugs, :func:`stage_dir` will *read* from an existing legacy
  ``NN_<legacy_slug>`` directory when the canonical one is missing.

Notes
-----
We keep a small compatibility API surface for internal callers (e.g. QC) by
providing :func:`resolve_input_path`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from scorpio_pipe.stage_registry import REGISTRY


_RE_STEM_SHORT = re.compile(r"^(s\d+)")

# Directory-suffix fallbacks for legacy runs (v5.38.5 and earlier).
# These are *directory slugs*, not stage keys.
_LEGACY_DIR_SLUGS: dict[str, list[str]] = {
    # New slug first, then known legacy names.
    "biascorr": ["bias", "biascorr"],
    "flatfield": ["flat", "flatfield"],
    "cosmics": ["cosmics"],
    "superneon": ["superneon"],
    "arclineid": ["lineid", "arclineid"],
    "wavesol": ["wavesol"],
    "linearize": ["linearize"],
    "sky": ["sky"],
    "stack2d": ["stack", "stack2d"],
    "extract1d": ["extract", "extract1d"],
}


def _stage_base(run_root: str | Path) -> Path:
    rr = Path(run_root)
    legacy = rr / "products"
    if legacy.is_dir():
        return legacy
    return rr


def _pick_best_match(dirs: list[Path]) -> Path | None:
    if not dirs:
        return None

    def _prefix(p: Path) -> int:
        try:
            return int(p.name.split("_", 1)[0])
        except Exception:
            return -1

    # Prefer the highest numeric prefix (newer numbering tends to be higher).
    return sorted(dirs, key=_prefix)[-1]


def _find_legacy_stage_dir(run_root: Path, stage_key: str, canonical: Path) -> Path | None:
    """Try to locate a legacy stage directory for reading.

    We only use this fallback when the canonical directory does not exist.
    """

    if canonical.exists():
        return None

    resolved = REGISTRY.resolve_key(stage_key)
    slugs = _LEGACY_DIR_SLUGS.get(resolved)
    if not slugs:
        return None

    # Search both the selected stage base and the plain run_root.
    roots: list[Path] = []
    base = _stage_base(run_root)
    roots.append(base)
    if base != run_root:
        roots.append(run_root)

    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for slug in slugs:
            found.extend([p for p in root.glob(f"??_{slug}") if p.is_dir()])

        # Special case: unified sky stage.
        # Legacy layouts may have multiple sky directories (e.g. two-step sky subtraction).
        if resolved == "sky":
            found.extend([p for p in root.glob("??_sky*") if p.is_dir()])

    return _pick_best_match(found)


def stage_dir(run_root: str | Path, stage_key: str) -> Path:
    """Return stage directory: ``run_root/NN_slug``.

    - New layout: directly under ``run_root``.
    - Legacy layout: under ``run_root/products`` if that directory exists.
    - Legacy numbering/slugs: if the canonical dir is missing but a matching
      legacy ``NN_<legacy_slug>`` exists, return that existing dir.
    """

    rr = Path(run_root)
    base = _stage_base(rr)
    canonical = base / REGISTRY.dir_name(stage_key)

    legacy = _find_legacy_stage_dir(rr, stage_key, canonical)
    return legacy or canonical


def extract_stem_short(raw_path_or_stem: Any) -> str:
    """Extract a compact per-exposure stem.

    Rules
    -----
    - If name matches ``^(s\\d+)`` -> return that group.
      Example: ``s23840510_obj`` -> ``s23840510``
    - Else: return the filename stem.
      Example: ``obj_0001.fits`` -> ``obj_0001``
    """

    if raw_path_or_stem is None:
        return ""

    p = Path(str(raw_path_or_stem))
    stem = p.stem
    m = _RE_STEM_SHORT.match(stem)
    if m:
        return str(m.group(1))
    return stem


def per_exp_dir(
    run_root: str | Path, stage_key: str, raw_path_or_stem: str | Path
) -> Path:
    """Return per-exposure directory: ``run_root/NN_slug/<stem_short>/``."""

    return stage_dir(run_root, stage_key) / extract_stem_short(raw_path_or_stem)


def resolve_input_path(
    product_key: str,
    run_root: str | Path,
    stage_key: str,
    *,
    relpath: str | Path | None = None,
    raw_path: str | Path | None = None,
    raw_stem: str | None = None,
    extra_candidates: list[str | Path] | None = None,
    strict: bool = False,
) -> Path:
    """Resolve an input artifact path.

    Canonical layout uses stage dirs directly under ``run_root``.
    If this ``run_root`` is a legacy workspace (has ``products/``), stage dirs
    are resolved under ``run_root/products`` via :func:`stage_dir`.

    Parameters
    ----------
    product_key
        Debug label for error messages (unused by resolver logic).
    stage_key
        Stage key (canonical or alias).
    relpath
        Relative path inside stage dir / per-exp dir.
    raw_path
        If provided, resolves under per-exp dir.
    """

    _ = product_key  # reserved for future diagnostics

    rel = Path(relpath) if relpath is not None else None
    cands: list[Path] = []

    # Per-exposure location
    if raw_path is not None and rel is not None:
        cands.append(per_exp_dir(run_root, stage_key, raw_path) / rel)
    if raw_stem and rel is not None:
        cands.append(per_exp_dir(run_root, stage_key, raw_stem) / rel)

    # Stage root location
    if rel is not None:
        cands.append(stage_dir(run_root, stage_key) / rel)
    else:
        cands.append(stage_dir(run_root, stage_key))

    # Extra candidates (already relative to run_root or absolute)
    if extra_candidates:
        for x in extra_candidates:
            p = Path(x)
            if not p.is_absolute():
                p = Path(run_root) / p
            cands.append(p)

    # First existing wins
    for p in cands:
        try:
            if p.exists():
                return p
        except Exception:
            continue

    if strict:
        raise FileNotFoundError(
            f"{product_key}: expected input not found; tried: "
            + ", ".join(str(x) for x in cands[:8])
        )

    # Return canonical even if missing
    return cands[0] if cands else stage_dir(run_root, stage_key)
