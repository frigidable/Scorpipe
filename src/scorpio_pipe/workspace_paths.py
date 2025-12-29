"""Canonical workspace path helpers.

Layout (v5.38.5)
----------------
Stage outputs live directly under the *run root*:

    run_root/NN_slug/

Per-exposure outputs (when enabled) live under the stage directory:

    run_root/NN_slug/<stem_short>/

There is intentionally **no** "products/" directory in the new layout.

Compatibility
-------------
For old workspaces that have ``products/``, :func:`stage_dir` will transparently
use it as a stage base.

Notes
-----
We keep a small compatibility API surface for internal callers (e.g. QC) by
providing :func:`resolve_input_path`. Legacy fallbacks are limited to selecting
``run_root/products`` as a stage base when present.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from scorpio_pipe.stage_registry import REGISTRY


_RE_STEM_SHORT = re.compile(r"^(s\d+)")


def _stage_base(run_root: str | Path) -> Path:
    rr = Path(run_root)
    legacy = rr / "products"
    if legacy.is_dir():
        return legacy
    return rr


def stage_dir(run_root: str | Path, stage_key: str) -> Path:
    """Return canonical stage directory: ``run_root/NN_slug``.

    If ``run_root/products`` exists, returns ``run_root/products/NN_slug`` to
    keep legacy workspaces functional.
    """

    base = _stage_base(run_root)
    return base / REGISTRY.dir_name(stage_key)


def extract_stem_short(raw_path_or_stem: Any) -> str:
    """Extract a compact per-exposure stem.

    Rules
    -----
    - If name matches ``^(s\d+)`` -> return that group.
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


def per_exp_dir(run_root: str | Path, stage_key: str, raw_path_or_stem: str | Path) -> Path:
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
            f"{product_key}: expected input not found; tried: " + ", ".join(str(x) for x in cands[:8])
        )

    # Return canonical even if missing
    return cands[0] if cands else stage_dir(run_root, stage_key)
