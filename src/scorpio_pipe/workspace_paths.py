"""Canonical workspace path helpers.

This module implements the *single* canonical layout for stage outputs:

    work_dir/
      raw/
      products/
        NN_slug/
          ...stage outputs...
          <raw_stem>/   # per-exposure directory (when enabled)
      manifest/

Reading uses a new→legacy resolver (see :func:`resolve_input_path`) so old
workspaces remain usable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from scorpio_pipe.stage_registry import REGISTRY


def products_dir(work_dir: str | Path) -> Path:
    """Return canonical products root."""

    return Path(work_dir) / "products"


def stage_dir(work_dir: str | Path, stage_key: str) -> Path:
    """Return canonical stage directory: ``products/NN_slug``."""

    wd = Path(work_dir)
    return products_dir(wd) / REGISTRY.dir_name(stage_key)


def per_exp_dir(work_dir: str | Path, stage_key: str, raw_stem: str) -> Path:
    """Return per-exposure directory under a stage: ``products/NN_slug/<stem>/``."""

    return stage_dir(work_dir, stage_key) / str(raw_stem)


def legacy_candidates(
    work_dir: str | Path,
    stage_key: str,
    *,
    relpath: str | Path | None = None,
    raw_stem: str | None = None,
    extra: Iterable[Path] | None = None,
) -> list[Path]:
    """Return a list of plausible legacy locations for a stage artifact.

    Parameters
    ----------
    relpath
        Relative path inside the stage output directory (e.g. ``"lin_preview.fits"``).
        If omitted, candidates will point at directories.
    raw_stem
        If set, include per-exposure legacy layouts.
    extra
        Additional explicit candidates (appended).
    """

    wd = Path(work_dir)
    r = Path(relpath) if relpath is not None else None

    def _join(base: Path) -> Path:
        if r is None:
            return base
        return base / r

    out: list[Path] = []

    k = (stage_key or "").strip().lower()

    # Common legacy roots.
    prod = wd / "products"

    if k == "linearize":
        # Old canonical: products/lin/[per_exp/*]
        if raw_stem:
            out += [
                _join(prod / "lin" / "per_exp" / raw_stem),
                _join(wd / "lin" / "per_exp" / raw_stem),
                _join(prod / "lin" / "per_exp"),
                _join(wd / "lin" / "per_exp"),
            ]
        out += [
            _join(prod / "lin"),
            _join(wd / "lin"),
        ]
    elif k == "sky":
        if raw_stem:
            out += [
                _join(prod / "sky" / "per_exp" / raw_stem),
                _join(wd / "sky" / "per_exp" / raw_stem),
                _join(prod / "sky" / "per_exp"),
                _join(wd / "sky" / "per_exp"),
            ]
        out += [_join(prod / "sky"), _join(wd / "sky")]
    elif k == "stack2d":
        # Historical locations:
        #   - work_dir/stack/ (very old)
        #   - work_dir/products/stack/ (old)
        #   - work_dir/products/10_stack2d/ (v5.38+ canonical)
        out += [
            _join(prod / "10_stack"),
            _join(prod / "10_stack2d"),
            _join(prod / "stack"),
            _join(prod / "stack2d"),
            _join(wd / "stack"),
        ]
    elif k == "extract1d":
        out += [_join(prod / "spec"), _join(wd / "spec")]
    elif k in {"superbias", "superflat"}:
        out += [
            _join(wd / "calibs"),
            _join(wd / "calib"),
        ]
    elif k == "qc_report" or k == "qc":
        out += [_join(wd / "qc"), _join(wd / "report"), _join(prod / "qc")]
    elif k == "manifest":
        out += [_join(wd / "qc"), _join(wd / "report"), _join(wd / "manifest")]
    elif k in {"wavesolution", "wavesol"}:
        # Wavesolution has a dedicated subtree; fallback handled in wavesol_paths.
        out += [_join(wd / "wavesol")]
    else:
        # Generic: old style (work_dir/<stage_key>/) and products/<slug>/
        try:
            slug = REGISTRY.get(k).slug
        except Exception:
            slug = k
        out += [_join(prod / slug), _join(wd / slug), _join(wd / k)]

    if extra is not None:
        out += [Path(p) for p in extra]

    # Remove obvious duplicates while keeping order.
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)
    return uniq


def resolve_input_path(
    product_key: str,
    work_dir: str | Path,
    stage_key: str,
    *,
    relpath: str | Path | None = None,
    raw_stem: str | None = None,
    extra_candidates: Iterable[Path] | None = None,
) -> Path:
    """Resolve an input artifact path using new→legacy fallback.

    Parameters
    ----------
    product_key
        A short stable key used for debugging/logging. The resolver doesn't need
        to know semantics of the key.
    stage_key
        Canonical stage key.
    relpath
        Relative path under the stage directory.
    raw_stem
        Optional raw stem for per-exposure directories.
    """

    wd = Path(work_dir)
    r = Path(relpath) if relpath is not None else None

    if raw_stem is not None:
        base = per_exp_dir(wd, stage_key, raw_stem)
    else:
        base = stage_dir(wd, stage_key)
    p_new = base if r is None else (base / r)
    if p_new.exists():
        return p_new

    for cand in legacy_candidates(
        wd, stage_key, relpath=r, raw_stem=raw_stem, extra=extra_candidates
    ):
        if cand.exists():
            return cand

    # Nothing exists; return canonical path (useful for error messages).
    return p_new
