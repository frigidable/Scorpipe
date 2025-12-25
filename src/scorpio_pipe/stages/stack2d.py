"""Stack rectified sky-subtracted frames in (λ, y).

This is the first *real* stacking stage for the v5.x long-slit branch.

Algorithm (v5.12 interim):
  - read per-exposure MEF products (SCI + optional VAR/MASK)
  - compute weights = 1/VAR (fallback to 1)
  - exclude masked pixels (MASK != 0)
  - optional iterative sigma-clipping (per pixel) using VAR as noise model
  - output stacked MEF with SCI/VAR/MASK and coverage map (COV)

The implementation is chunked along y to keep memory bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import json
import numpy as np
from astropy.io import fits

from ..plot_style import mpl_style
from ..wavesol_paths import resolve_work_dir


MASK_NO_COVERAGE = np.uint16(1 << 0)
MASK_CLIPPED = np.uint16(1 << 3)


def _open_mef(path: Path) -> tuple[np.ndarray, fits.Header, np.ndarray | None, np.ndarray | None]:
    """Return (sci, hdr, var, mask) from a MEF or simple FITS."""
    with fits.open(path, memmap=True) as hdul:
        hdr = hdul[0].header.copy()
        sci = hdul[0].data
        if sci is None:
            # Try SCI ext
            if "SCI" in hdul:
                sci = hdul["SCI"].data
                hdr = hdul["SCI"].header.copy()
            else:
                raise ValueError(f"No SCI data found in {path}")
        sci = np.asarray(sci, dtype=np.float32)
        var = None
        mask = None
        if "VAR" in hdul:
            try:
                var = np.asarray(hdul["VAR"].data, dtype=np.float32)
            except Exception:
                var = None
        if "MASK" in hdul:
            try:
                mask = np.asarray(hdul["MASK"].data, dtype=np.uint16)
            except Exception:
                mask = None
        return sci, hdr, var, mask


def _write_mef(path: Path, sci: np.ndarray, hdr: fits.Header, *, var: np.ndarray | None, mask: np.ndarray | None, cov: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdus: list[fits.HDUBase] = [fits.PrimaryHDU(np.asarray(sci, dtype=np.float32), header=hdr)]
    if var is not None:
        hdus.append(fits.ImageHDU(np.asarray(var, dtype=np.float32), name="VAR"))
    if mask is not None:
        hdus.append(fits.ImageHDU(np.asarray(mask, dtype=np.uint16), name="MASK"))
    if cov is not None:
        hdus.append(fits.ImageHDU(np.asarray(cov, dtype=np.int16), name="COV"))
    fits.HDUList(hdus).writeto(path, overwrite=True)


def _iter_slices(ny: int, chunk: int) -> Iterable[slice]:
    chunk = int(max(8, chunk))
    for y0 in range(0, int(ny), chunk):
        yield slice(y0, min(ny, y0 + chunk))


def run_stack2d(cfg: dict[str, Any], *, inputs: Iterable[Path], out_dir: Path | None = None) -> dict[str, Any]:
    st_cfg = (cfg.get("stack2d") or {}) if isinstance(cfg.get("stack2d"), dict) else {}
    if not bool(st_cfg.get("enabled", True)):
        return {"skipped": True, "reason": "stack2d.enabled=false"}

    wd = resolve_work_dir(cfg)
    if out_dir is None:
        out_dir = wd / "products" / "stack"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [Path(p) for p in inputs]
    files = [p for p in files if p.exists()]
    if not files:
        raise FileNotFoundError("stack2d: no input FITS files")

    # Read shapes from first file.
    sci0, hdr0, var0, mask0 = _open_mef(files[0])
    ny, nx = sci0.shape

    sigma_clip = float(st_cfg.get("sigma_clip", 4.0))
    maxiter = int(st_cfg.get("maxiter", 3))
    chunk = int(st_cfg.get("chunk_rows", 128))

    out_sci = np.zeros((ny, nx), dtype=np.float32)
    out_var = np.zeros((ny, nx), dtype=np.float32)
    out_mask = np.zeros((ny, nx), dtype=np.uint16)
    out_cov = np.zeros((ny, nx), dtype=np.int16)

    # Keep HDUs open (memmap) for slicing.
    hduls = [fits.open(p, memmap=True) for p in files]
    try:
        for ys in _iter_slices(ny, chunk):
            # Build stacks for this block.
            sci_stack = []
            var_stack = []
            mask_stack = []
            for h in hduls:
                sci = h[0].data
                if sci is None and "SCI" in h:
                    sci = h["SCI"].data
                sci_stack.append(np.asarray(sci[ys, :], dtype=np.float32))
                v = None
                m = None
                if "VAR" in h:
                    v = np.asarray(h["VAR"].data[ys, :], dtype=np.float32)
                if "MASK" in h:
                    m = np.asarray(h["MASK"].data[ys, :], dtype=np.uint16)
                var_stack.append(v)
                mask_stack.append(m)

            S = np.stack(sci_stack, axis=0)  # (nexp, y, x)
            if all(v is None for v in var_stack):
                V = None
                W = np.ones_like(S, dtype=np.float32)
            else:
                # Replace missing VAR with large values -> small weights
                V = np.stack([
                    (np.asarray(v, dtype=np.float32) if v is not None else np.full_like(sci_stack[0], 1e12, dtype=np.float32))
                    for v in var_stack
                ], axis=0)
                W = np.where(np.isfinite(V) & (V > 0), 1.0 / V, 0.0).astype(np.float32)

            if any(m is not None for m in mask_stack):
                M = np.stack([
                    (np.asarray(m, dtype=np.uint16) if m is not None else np.zeros_like(mask_stack[0] if mask_stack[0] is not None else sci_stack[0], dtype=np.uint16))
                    for m in mask_stack
                ], axis=0)
                W = np.where(M != 0, 0.0, W)
            else:
                M = None

            # Mask NaNs
            W = np.where(np.isfinite(S), W, 0.0)

            # Iterative sigma clipping using VAR model (or robust residual if no VAR).
            clipped = np.zeros_like(W, dtype=bool)
            for _ in range(max(0, maxiter)):
                wsum = np.sum(W, axis=0)
                mu = np.where(wsum > 0, np.sum(W * S, axis=0) / wsum, np.nan)
                if V is not None:
                    sigma = np.sqrt(np.maximum(np.sum((W ** 2) * V, axis=0) / np.maximum(wsum ** 2, 1e-20), 0.0))
                else:
                    # robust estimate
                    med = np.nanmedian(S, axis=0)
                    mad = np.nanmedian(np.abs(S - med), axis=0)
                    sigma = 1.4826 * mad
                bad = np.abs(S - mu) > (sigma_clip * np.maximum(sigma, 1e-6))
                bad = bad & (W > 0)
                if not np.any(bad):
                    break
                clipped |= bad
                W = np.where(bad, 0.0, W)

            wsum = np.sum(W, axis=0)
            cov = np.sum(W > 0, axis=0).astype(np.int16)
            mu = np.where(wsum > 0, np.sum(W * S, axis=0) / wsum, 0.0)
            var_out = np.where(wsum > 0, 1.0 / wsum, np.nan)

            m_out = np.zeros(mu.shape, dtype=np.uint16)
            m_out = np.where(wsum <= 0, m_out | MASK_NO_COVERAGE, m_out)
            m_out = np.where(np.any(clipped, axis=0), m_out | MASK_CLIPPED, m_out)
            if M is not None:
                # preserve any remaining flags (OR across exposures that contributed)
                contrib = (W > 0)
                m_or = np.bitwise_or.reduce(np.where(contrib, M, 0), axis=0).astype(np.uint16)
                m_out |= m_or

            out_sci[ys, :] = mu.astype(np.float32)
            out_var[ys, :] = var_out.astype(np.float32)
            out_mask[ys, :] = m_out
            out_cov[ys, :] = cov

    finally:
        for h in hduls:
            try:
                h.close()
            except Exception:
                pass

    out_fits = out_dir / "stacked2d.fits"
    hdr = hdr0.copy()
    hdr["HISTORY"] = "Scorpio Pipe v5.13: stack2d"
    _write_mef(out_fits, out_sci, hdr, var=out_var, mask=out_mask, cov=out_cov)

    # quick QC plot: coverage histogram
    out_png = out_dir / "coverage.png"
    if bool(st_cfg.get("save_png", True)):
        try:
            import matplotlib.pyplot as plt

            with mpl_style():
                fig = plt.figure(figsize=(6.0, 3.6))
                ax = fig.add_subplot(111)
                ax.hist(out_cov.ravel(), bins=range(int(out_cov.max()) + 2))
                ax.set_xlabel("Coverage (N exposures)")
                ax.set_ylabel("Pixels")
                fig.tight_layout()
                fig.savefig(out_png)
                plt.close(fig)
        except Exception:
            pass

    done = out_dir / "stack2d_done.json"
    payload = {
        "ok": True,
        "n_inputs": len(files),
        "output_fits": str(out_fits),
        "output_png": str(out_png) if out_png.exists() else None,
        "sigma_clip": sigma_clip,
        "maxiter": maxiter,
    }
    done.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
