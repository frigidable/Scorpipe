from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scorpio_pipe.io.atomic import atomic_write_json
from scorpio_pipe.io.mef import read_sci_var_mask
from scorpio_pipe.io.quicklook import write_quicklook_png


def _stamp_utc() -> str:
    # ISO-ish, filesystem safe
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def compare_cache_root(run_root: Path) -> Path:
    return Path(run_root) / "ui" / "compare_cache"


def _copy_if_exists(src: Path, dst: Path) -> None:
    try:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    except Exception:
        pass


def snapshot_stage(
    *,
    stage_key: str,
    stage_dir: Path,
    label: str,
    patterns: tuple[str, ...],
    stamp: str | None = None,
) -> Path | None:
    """Copy selected artifacts into ui/compare_cache/<stage>/<stamp>/<label>/..."""

    stage_dir = Path(stage_dir)
    run_root = stage_dir.parent
    stamp = stamp or _stamp_utc()

    root = compare_cache_root(run_root) / stage_key / stamp / label
    root.mkdir(parents=True, exist_ok=True)

    any_copied = False
    for pat in patterns:
        for src in sorted(stage_dir.glob(pat)):
            if src.is_file():
                _copy_if_exists(src, root / src.name)
                any_copied = True

    return root if any_copied else None


def _robust_sigma(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=np.float64)
    good = np.isfinite(a)
    if not np.any(good):
        return float("nan")
    vals = a[good]
    med = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - med))
    return float(1.4826 * mad)


def diff_mef_to_png(
    *,
    a_fits: Path,
    b_fits: Path,
    out_png: Path,
    out_json: Path | None = None,
    method: str = "asinh",
) -> dict[str, Any] | None:
    """Compute B-A for MEF FITS and save quicklook + small metrics JSON."""

    try:
        a_sci, _a_var, a_msk, _ = read_sci_var_mask(a_fits)
        b_sci, _b_var, b_msk, _ = read_sci_var_mask(b_fits)
        da = np.asarray(b_sci, dtype=np.float64) - np.asarray(a_sci, dtype=np.float64)

        # For diff, do not mask aggressively; just use finiteness.
        meta = {
            "a": str(a_fits.name),
            "b": str(b_fits.name),
            "sigma_mad": _robust_sigma(da),
            "median": float(np.nanmedian(da[np.isfinite(da)])) if np.any(np.isfinite(da)) else float("nan"),
        }
        write_quicklook_png(da, out_png, mask=None, method=method, meta=meta)

        if out_json is not None:
            atomic_write_json(out_json, meta, indent=2)
        return meta
    except Exception:
        return None


def build_stage_diff(
    *,
    stage_key: str,
    stamp: str,
    run_root: Path,
    a_dir: Path,
    b_dir: Path,
    stems: list[str],
    a_suffix: str,
    b_suffix: str,
) -> dict[str, Any]:
    """Build per-exposure diff PNGs inside ui/compare_cache/.../diff."""

    out_root = compare_cache_root(Path(run_root)) / stage_key / stamp / "diff"
    out_root.mkdir(parents=True, exist_ok=True)

    per: dict[str, Any] = {}
    for stem in stems:
        a_f = Path(a_dir) / f"{stem}{a_suffix}"
        b_f = Path(b_dir) / f"{stem}{b_suffix}"
        if not (a_f.exists() and b_f.exists()):
            continue
        meta = diff_mef_to_png(
            a_fits=a_f,
            b_fits=b_f,
            out_png=out_root / f"{stem}_diff.png",
            out_json=out_root / f"{stem}_diff.json",
        )
        if meta is not None:
            per[stem] = meta

    summary = {
        "stage": stage_key,
        "stamp": stamp,
        "n_pairs": len(per),
        "per_exposure": per,
    }
    atomic_write_json(out_root / "diff_metrics.json", summary, indent=2)
    return summary