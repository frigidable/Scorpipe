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

from pathlib import Path
from typing import Any, Iterable

import json
import numpy as np
from astropy.io import fits

from scorpio_pipe.fits_utils import open_fits_smart

from scorpio_pipe.io.mef import write_sci_var_mask, try_read_grid
from scorpio_pipe.version import PIPELINE_VERSION
from scorpio_pipe.maskbits import BADPIX, COSMIC, NO_COVERAGE, REJECTED, SATURATED, USER

from ..plot_style import mpl_style
from scorpio_pipe.paths import resolve_work_dir
from ..shift_utils import xcorr_shift_subpix


MASK_NO_COVERAGE = NO_COVERAGE
MASK_CLIPPED = REJECTED

# Bits that make a pixel unusable for stacking.
FATAL_BITS = np.uint16(NO_COVERAGE | BADPIX | COSMIC | SATURATED | USER | REJECTED)


def _xcorr_subpix_shift_1d(ref: np.ndarray, cur: np.ndarray, max_shift: int) -> tuple[float, float]:
    """Return (shift_pix, score) to apply to `cur` to best match `ref`.

    Uses NumPy-only normalized dot-product xcorr with a parabola refinement
    around the best integer shift (see :mod:`scorpio_pipe.shift_utils`).
    """

    est = xcorr_shift_subpix(ref, cur, max_shift=max_shift)
    return float(est.shift_pix), float(est.score)





def _colsel_from_windows(hdr: fits.Header, nx: int, windows_A: Any | None, windows_pix: Any | None = None) -> np.ndarray | None:
    """Return boolean column selector for wavelength windows.

    Supports either Angstrom windows on a linear WCS (CRVAL1/CDELT1[/CD1_1]/CRPIX1)
    or pixel windows. Returns None if no usable selector can be built.
    """

    # Pixel windows take precedence.
    if windows_pix is not None:
        try:
            sel = np.zeros(nx, dtype=bool)
            for w in list(windows_pix):
                if isinstance(w, (list, tuple)) and len(w) >= 2:
                    a, b = int(w[0]), int(w[1])
                elif isinstance(w, dict):
                    a, b = int(w.get("x0")), int(w.get("x1"))
                else:
                    continue
                if a > b:
                    a, b = b, a
                a = max(0, min(nx - 1, a))
                b = max(0, min(nx - 1, b))
                sel[a : b + 1] = True
            if sel.sum() >= 16:
                return sel
        except Exception:
            pass

    if windows_A is None:
        return None

    crval1 = hdr.get("CRVAL1")
    cdelt1 = hdr.get("CDELT1", hdr.get("CD1_1"))
    crpix1 = hdr.get("CRPIX1", 1.0)
    if crval1 is None or cdelt1 is None:
        return None

    try:
        crval1 = float(crval1)
        cdelt1 = float(cdelt1)
        crpix1 = float(crpix1)
    except Exception:
        return None

    if not np.isfinite(cdelt1) or abs(cdelt1) <= 0:
        return None

    def w_to_i(w: float) -> float:
        # WCS: w = CRVAL1 + (i+1 - CRPIX1)*CDELT1
        return (w - crval1) / cdelt1 + crpix1 - 1.0

    sel = np.zeros(nx, dtype=bool)
    try:
        for win in list(windows_A):
            if isinstance(win, dict):
                w0 = float(win.get("w0", win.get("lo")))
                w1 = float(win.get("w1", win.get("hi")))
            elif isinstance(win, (list, tuple)) and len(win) >= 2:
                w0, w1 = float(win[0]), float(win[1])
            else:
                continue
            if w0 > w1:
                w0, w1 = w1, w0
            i0 = int(np.floor(w_to_i(w0)))
            i1 = int(np.ceil(w_to_i(w1)))
            i0 = max(0, min(nx - 1, i0))
            i1 = max(0, min(nx - 1, i1))
            sel[i0 : i1 + 1] = True
    except Exception:
        return None

    if sel.sum() < 16:
        return None
    return sel

def _take_block_yshift(arr: np.ndarray, y0: int, y1: int, shift: int, *, fill: float) -> tuple[np.ndarray, np.ndarray]:
    """Take arr block [y0:y1, :] from a frame shifted by `shift` in y.

    We interpret `shift` as a translation applied to the *input* frame to align
    it into the *output* y-grid:
        out[y + shift] <- in[y]

    Returns (block, filled_mask) where filled_mask marks pixels that were filled.
    """
    arr = np.asarray(arr)
    ny, nx = arr.shape
    shift = int(shift)
    out = np.full((y1 - y0, nx), fill, dtype=np.float32 if arr.dtype.kind == "f" else arr.dtype)
    filled = np.ones((y1 - y0, nx), dtype=bool)

    src0 = y0 - shift
    src1 = y1 - shift

    v0 = max(0, src0)
    v1 = min(ny, src1)
    if v1 <= v0:
        return out, filled

    dst0 = (v0 + shift) - y0
    dst1 = dst0 + (v1 - v0)
    out[dst0:dst1, :] = arr[v0:v1, :]
    filled[dst0:dst1, :] = False
    return out, filled


def _take_block_yshift_subpix(arr: np.ndarray, y0: int, y1: int, shift: float, *, fill: float) -> tuple[np.ndarray, np.ndarray]:
    """Subpixel y-shifted block (SCI-like), using linear interpolation.

    Sign convention matches the integer helper: out[y+shift] <- in[y].
    For fractional shifts we sample input rows at y_in = y_out - shift.

    Returns (block, filled_mask).
    """

    arr = np.asarray(arr, dtype=float)
    ny, nx = arr.shape
    s = float(shift)
    if not np.isfinite(s) or abs(s) < 1e-9:
        blk = arr[y0:y1, :].copy()
        filled = ~np.isfinite(blk)
        return blk, filled

    y_out = np.arange(y0, y1, dtype=float)
    y_in = y_out - s
    i0 = np.floor(y_in).astype(int)
    i1 = i0 + 1
    frac = (y_in - i0).astype(float)
    valid = (i0 >= 0) & (i1 < ny)
    i0c = np.clip(i0, 0, ny - 1)
    i1c = np.clip(i1, 0, ny - 1)

    out = np.full((y1 - y0, nx), fill, dtype=np.float32)
    filled = np.ones((y1 - y0, nx), dtype=bool)
    if np.any(valid):
        a0 = arr[i0c, :]
        a1 = arr[i1c, :]
        w1 = frac[:, None]
        w0 = 1.0 - w1
        vv = valid[:, None]
        out[vv] = (w0[vv] * a0[vv] + w1[vv] * a1[vv])
        filled[vv] = False
    return out, filled


def _take_block_yshift_subpix_var(var: np.ndarray, y0: int, y1: int, shift: float, *, fill: float) -> tuple[np.ndarray, np.ndarray]:
    """Subpixel y-shifted block for VAR with (w0^2, w1^2) propagation."""

    var = np.asarray(var, dtype=float)
    ny, nx = var.shape
    s = float(shift)
    if not np.isfinite(s) or abs(s) < 1e-9:
        blk = var[y0:y1, :].copy()
        filled = ~np.isfinite(blk)
        return np.asarray(blk, dtype=np.float32), filled

    y_out = np.arange(y0, y1, dtype=float)
    y_in = y_out - s
    i0 = np.floor(y_in).astype(int)
    i1 = i0 + 1
    frac = (y_in - i0).astype(float)
    valid = (i0 >= 0) & (i1 < ny)
    i0c = np.clip(i0, 0, ny - 1)
    i1c = np.clip(i1, 0, ny - 1)

    out = np.full((y1 - y0, nx), fill, dtype=np.float32)
    filled = np.ones((y1 - y0, nx), dtype=bool)
    if np.any(valid):
        v0 = var[i0c, :]
        v1 = var[i1c, :]
        w1 = frac[:, None]
        w0 = 1.0 - w1
        vv = valid[:, None]
        out[vv] = (w0[vv] ** 2) * v0[vv] + (w1[vv] ** 2) * v1[vv]
        filled[vv] = False
    return out, filled


def _take_block_yshift_subpix_mask(mask: np.ndarray, y0: int, y1: int, shift: float) -> np.ndarray:
    """Subpixel y-shifted mask block (conservative OR)."""

    mask = np.asarray(mask, dtype=np.uint16)
    ny, nx = mask.shape
    s = float(shift)
    if not np.isfinite(s) or abs(s) < 1e-9:
        return mask[y0:y1, :].copy()

    y_out = np.arange(y0, y1, dtype=float)
    y_in = y_out - s
    i0 = np.floor(y_in).astype(int)
    i1 = i0 + 1
    valid = (i0 >= 0) & (i1 < ny)
    i0c = np.clip(i0, 0, ny - 1)
    i1c = np.clip(i1, 0, ny - 1)

    out = np.full((y1 - y0, nx), MASK_NO_COVERAGE, dtype=np.uint16)
    if np.any(valid):
        m0 = mask[i0c, :]
        m1 = mask[i1c, :]
        vv = valid[:, None]
        out[vv] = (m0[vv] | m1[vv])
    return out


def _take_block_yshift_mask(mask: np.ndarray, y0: int, y1: int, shift: int) -> np.ndarray:
    """Take uint16 mask block with y-shift; filled pixels get MASK_NO_COVERAGE."""
    mask = np.asarray(mask, dtype=np.uint16)
    ny, nx = mask.shape
    shift = int(shift)
    out = np.full((y1 - y0, nx), MASK_NO_COVERAGE, dtype=np.uint16)

    src0 = y0 - shift
    src1 = y1 - shift
    v0 = max(0, src0)
    v1 = min(ny, src1)
    if v1 <= v0:
        return out
    dst0 = (v0 + shift) - y0
    dst1 = dst0 + (v1 - v0)
    out[dst0:dst1, :] = mask[v0:v1, :]
    return out


def _open_mef(path: Path) -> tuple[np.ndarray, fits.Header, np.ndarray | None, np.ndarray | None]:
    """Return (sci, hdr, var, mask) from a MEF or simple FITS."""
    # Use memmap='auto' to avoid Astropy strict_memmap failures when the file
    # declares BZERO/BSCALE/BLANK (common for unsigned MASK extensions).
    with open_fits_smart(path, memmap="auto") as hdul:
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


def _write_mef(
    path: Path,
    sci: np.ndarray,
    hdr: fits.Header,
    *,
    var: np.ndarray | None,
    mask: np.ndarray | None,
    cov: np.ndarray | None,
) -> None:
    """Write stacked 2D product as MEF (SCI/VAR/MASK [+COV])."""
    grid = try_read_grid(hdr)
    extra: list[fits.ImageHDU] = []
    if cov is not None:
        extra.append(fits.ImageHDU(np.asarray(cov, dtype=np.int16), name="COV"))
    write_sci_var_mask(path, sci, var=var, mask=mask, header=hdr, grid=grid, extra_hdus=extra)


def _iter_slices(ny: int, chunk: int) -> Iterable[slice]:
    chunk = int(max(8, chunk))
    for y0 in range(0, int(ny), chunk):
        yield slice(y0, min(ny, y0 + chunk))


def run_stack2d(cfg: dict[str, Any], *, inputs: Iterable[Path], out_dir: Path | None = None) -> dict[str, Any]:
    st_cfg = (cfg.get("stack2d") or {}) if isinstance(cfg.get("stack2d"), dict) else {}
    if not bool(st_cfg.get("enabled", True)):
        # Keep a stable payload shape even when skipped.
        # Downstream code/tests may still expect keys like "stacked2d_fits".
        return {
            "skipped": True,
            "reason": "stack2d.enabled=false",
            "stacked2d_fits": None,
            "stacked2d_png": None,
            "qc_png": None,
        }

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

    # Optional y-alignment (subpixel shifts) before stacking.
    ya_cfg = st_cfg.get("y_align")
    if isinstance(ya_cfg, dict):
        y_align_enabled = bool(ya_cfg.get("enabled", False))
        y_align_max = int(ya_cfg.get("max_shift_pix", 10))
    else:
        y_align_enabled = bool(st_cfg.get("y_align_enabled", False))
        y_align_max = int(st_cfg.get("y_align_max_shift_pix", 10))


    out_sci = np.zeros((ny, nx), dtype=np.float32)
    out_var = np.zeros((ny, nx), dtype=np.float32)
    out_mask = np.zeros((ny, nx), dtype=np.uint16)
    out_cov = np.zeros((ny, nx), dtype=np.int16)

    # Keep HDUs open (memmap) for slicing.
    # Use memmap='auto' to avoid Astropy strict_memmap failures when any HDU
    # declares BZERO/BSCALE/BLANK (typical for unsigned MASK bitmasks).
    hduls = [open_fits_smart(p, memmap="auto") for p in files]
    try:
        # Precompute per-exposure y offsets (subpixel) if requested.
        y_shifts = [0.0 for _ in files]
        y_scores: list[float | None] = [None for _ in files]
        y_offsets: list[dict[str, Any]] = []
        if y_align_enabled and len(files) > 1:
            # Build a crude spatial profile for each exposure.
            y_align_mode = 'full'
            y_align_windows_A = None
            y_align_windows_pix = None
            y_align_windows_unit = 'auto'
            y_align_use_positive = True
            if isinstance(ya_cfg, dict):
                y_align_mode = str(ya_cfg.get('mode', 'full') or 'full').strip().lower()
                y_align_windows_A = ya_cfg.get('windows_A') or ya_cfg.get('windows') or ya_cfg.get('windows_angstrom')
                y_align_windows_pix = ya_cfg.get('windows_pix') or ya_cfg.get('windows_pixels')
                y_align_windows_unit = str(ya_cfg.get('windows_unit', 'auto') or 'auto').strip().lower()
                y_align_use_positive = bool(ya_cfg.get('use_positive_flux', True))

            profiles = []
            for h in hduls:
                sci = h[0].data
                if sci is None and "SCI" in h:
                    sci = h["SCI"].data
                s = np.asarray(sci, dtype=np.float32)
                m = None
                if "MASK" in h:
                    try:
                        m = np.asarray(h["MASK"].data, dtype=np.uint16)
                    except Exception:
                        m = None
                good = np.isfinite(s)
                if m is not None:
                    good &= ((m & FATAL_BITS) == 0)

                sel = None
                if y_align_mode == "windows":
                    try:
                        hdr0 = h[0].header
                        has_wcs = (hdr0.get('CRVAL1') is not None) and (hdr0.get('CDELT1') is not None or hdr0.get('CD1_1') is not None)
                        unit = str(y_align_windows_unit or 'auto').lower()
                        wA = y_align_windows_A
                        wp = y_align_windows_pix
                        if unit in ('a', 'angstrom'):
                            wp = None
                        elif unit in ('pix', 'pixel', 'pixels'):
                            wA = None
                        else:  # auto
                            if has_wcs and wA is not None:
                                wp = None
                            elif (not has_wcs) and wp is not None:
                                wA = None
                        sel = _colsel_from_windows(hdr0, nx, wA, windows_pix=wp)
                    except Exception:
                        sel = None

                if sel is not None:
                    s2 = s[:, sel]
                    good2 = good[:, sel]
                else:
                    s2 = s
                    good2 = good

                if y_align_use_positive:
                    s2 = np.maximum(s2, 0.0)
                prof = np.nansum(np.where(good2, s2, 0.0), axis=1)
                profiles.append(prof)

            ref = profiles[0]
            for i in range(1, len(profiles)):
                sh, sc = _xcorr_subpix_shift_1d(ref, profiles[i], y_align_max)
                y_shifts[i] = float(sh)
                try:
                    y_scores[i] = float(sc)
                except Exception:
                    y_scores[i] = None
            y_offsets = [
                {"file": p.name, "y_shift_pix": float(sh), "score": y_scores[i]}
                for i, (p, sh) in enumerate(zip(files, y_shifts))
            ]
        else:
            y_offsets = [{"file": p.name, "y_shift_pix": 0.0, "score": None} for p in files]

        for ys in _iter_slices(ny, chunk):
            # Build stacks for this block.
            sci_stack = []
            var_stack = []
            mask_stack = []
            y0 = int(ys.start or 0)
            y1 = int(ys.stop or ny)
            for i, h in enumerate(hduls):
                sh = float(y_shifts[i]) if (y_align_enabled and i < len(y_shifts)) else 0.0

                sci = h[0].data
                if sci is None and "SCI" in h:
                    sci = h["SCI"].data
                if y_align_enabled and abs(sh) > 1e-6:
                    block_s, filled_s = _take_block_yshift_subpix(np.asarray(sci), y0, y1, sh, fill=float("nan"))
                else:
                    block_s, filled_s = _take_block_yshift(np.asarray(sci), y0, y1, int(round(sh)), fill=float("nan"))
                sci_stack.append(np.asarray(block_s, dtype=np.float32))

                v = None
                m = None
                if "VAR" in h:
                    if y_align_enabled and abs(sh) > 1e-6:
                        block_v, filled_v = _take_block_yshift_subpix_var(
                            np.asarray(h["VAR"].data), y0, y1, sh, fill=float("inf")
                        )
                    else:
                        block_v, filled_v = _take_block_yshift(
                            np.asarray(h["VAR"].data), y0, y1, int(round(sh)), fill=float("inf")
                        )
                    v = np.asarray(block_v, dtype=np.float32)
                if "MASK" in h:
                    if y_align_enabled and abs(sh) > 1e-6:
                        m = _take_block_yshift_subpix_mask(np.asarray(h["MASK"].data), y0, y1, sh)
                    else:
                        m = _take_block_yshift_mask(np.asarray(h["MASK"].data), y0, y1, int(round(sh)))
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
    hdr["HISTORY"] = f"Scorpio Pipe {PIPELINE_VERSION}: stack2d"
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
        "shape": [int(ny), int(nx)],
        "method": "variance-weighted mean + sigma-clip",
        "n_inputs": len(files),
        # Contract keys used by tests/UI.
        "stacked2d_fits": str(out_fits),
        "coverage_png": str(out_png) if out_png.exists() else None,
        "stack2d_done_json": str(done),
        # Backwards-compat aliases (older UI/tests):
        "output_fits": str(out_fits),
        "output_png": str(out_png) if out_png.exists() else None,
        "sigma_clip": sigma_clip,
        "maxiter": maxiter,
        "chunk_rows": chunk,
        "y_align_enabled": bool(y_align_enabled),
        "y_align_mode": str(ya_cfg.get("mode", "full")).lower() if isinstance(ya_cfg, dict) else "full",
        "y_align_windows_A": (ya_cfg.get("windows_A") or ya_cfg.get("windows") or ya_cfg.get("windows_angstrom")) if isinstance(ya_cfg, dict) else None,
        "y_align_windows_pix": (ya_cfg.get("windows_pix") or ya_cfg.get("windows_pixels")) if isinstance(ya_cfg, dict) else None,
        "y_align_windows_unit": (ya_cfg.get("windows_unit") or 'auto') if isinstance(ya_cfg, dict) else 'auto',
        "y_align_use_positive_flux": bool(ya_cfg.get("use_positive_flux", True)) if isinstance(ya_cfg, dict) else True,
        "y_offsets": y_offsets,
    }
    done.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
