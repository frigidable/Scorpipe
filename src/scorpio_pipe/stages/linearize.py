"""Linearization / rectification using a 2D dispersion solution (lambda_map).

This stage creates a linear-wavelength (λ, y) product on a common wavelength grid.

Core requirement for v5.x:
  - Build a *linearized summed* object frame from already cosmic-cleaned frames.
  - Preserve (first-order) flux by using edge-based cumulative rebinning.
  - Provide variance and mask as first-class citizens where possible.

Current implementation (v5.18):
  - Rectifies each exposure onto a *common* linear λ grid (flux-conserving rebin).
  - Writes per-exposure rectified products: *_rectified.fits (SCI/VAR/MASK).
  - Optionally creates a quick-look stacked preview (lin_preview.fits) for ROI/QC.

Notes:
  - Variance is a first-pass estimate (Poisson + read-noise). It is refined later.
  - If cosmic masks are available as FITS alongside cleaned frames, they are propagated.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from astropy.io import fits

from scorpio_pipe.io.mef import WaveGrid, read_sci_var_mask, write_sci_var_mask
from scorpio_pipe.noise_model import estimate_variance_adu2, resolve_noise_params
from scorpio_pipe.maskbits import (
    BADPIX,
    COSMIC,
    EDGE,
    NO_COVERAGE,
    REJECTED,
    SATURATED,
    USER,
    summarize as summarize_mask,
)

from ..logging_utils import get_logger
from ..provenance import add_provenance
from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.wavesol_paths import wavesol_dir

log = get_logger(__name__)



def _read_lambda_map_meta(hdr: fits.Header, lam_map: np.ndarray) -> tuple[str, str, str]:
    '''Return (unit, waveref, source) for wavelength metadata.

    We strongly prefer explicit metadata written by :mod:`wavesolution`.
    If missing, we fall back to a heuristic *with a loud warning*.

    unit is a FITS-like string (e.g. 'Angstrom', 'nm', or 'pix').
    waveref is 'air' or 'vacuum' (best-effort; defaults to 'air').
    source describes where unit came from.
    '''

    def _norm_unit(s: str) -> str:
        s0 = s.strip()
        if not s0:
            return ""
        s_low = s0.lower().replace("å", "angstrom")
        if s_low in {"a", "aa", "ang", "angs", "angstrom", "ångström", "angstroms", "Å"}:
            return "Angstrom"
        if s_low in {"nm", "nanometer", "nanometers"}:
            return "nm"
        if s_low in {"pix", "pixel", "pixels", "px"}:
            return "pix"
        return s0

    # Preferred explicit keys
    for key in ("WAVEUNIT", "SCORP_LU", "CUNIT1", "BUNIT", "WUNIT"):
        v = hdr.get(key, None)
        if v is None:
            continue
        u = _norm_unit(str(v))
        if u:
            # waveref best-effort
            wr = str(hdr.get("WAVEREF", hdr.get("WAVREF", "air")) or "air").strip().lower()
            if wr not in {"air", "vacuum"}:
                wr = "air"
            return u, wr, f"{key}"

    # Back-compat: some historic products stored WAT1_001 like IRAF
    for key in ("WAT1_001",):
        v = hdr.get(key, None)
        if v is None:
            continue
        u = _norm_unit(str(v))
        if u:
            wr = str(hdr.get("WAVEREF", hdr.get("WAVREF", "air")) or "air").strip().lower()
            if wr not in {"air", "vacuum"}:
                wr = "air"
            return u, wr, f"{key}"

    # Fallback (heuristic) — keep working for synthetic tests / legacy lambda_map
    u = _infer_lambda_unit(hdr, lam_map)
    wr = str(hdr.get("WAVEREF", hdr.get("WAVREF", "air")) or "air").strip().lower()
    if wr not in {"air", "vacuum"}:
        wr = "air"
    return u, wr, "heuristic"

def _infer_lambda_unit(hdr: fits.Header, lam_map: np.ndarray) -> str:
    """Infer wavelength unit string for output WCS.

    We prefer explicit FITS WCS metadata if present. Otherwise, we fall back
    to a lightweight heuristic based on typical long-slit spectroscopic
    ranges.

    Returns FITS-like unit strings (e.g. 'Angstrom', 'nm').
    """

    # 1) Explicit header values
    for key in ("SCORP_LU", "CUNIT1", "WUNIT", "WAT1_001"):
        try:
            v = hdr.get(key, None)
            if v is None:
                continue
            s = str(v).strip()
            if not s:
                continue
            # Normalize a few common spellings
            s_low = s.lower()
            if s_low in {"a", "aa", "angstrom", "ang", "angs", "Å"}:
                return "Angstrom"
            if s_low in {"nm", "nanometer", "nanometers"}:
                return "nm"
            return s
        except Exception:
            continue

    # 2) Heuristic by scale of values
    try:
        med = float(np.nanmedian(lam_map[np.isfinite(lam_map)]))
        # Typical optical spectra: 3500-10000 Angstrom ~ 350-1000 nm
        if 50.0 <= med <= 2000.0:
            return "nm"
    except Exception:
        pass
    return "Angstrom"


def _set_linear_wcs(
    hdr: fits.Header, wave0: float, dw: float, *, unit: str = "Angstrom"
) -> fits.Header:
    """Return a copy of header with a simple linear wavelength WCS.

    Note: we intentionally do not try to set NAXIS-related keywords here.
    """
    ohdr = fits.Header(hdr)
    ohdr["CRVAL1"] = float(wave0)
    ohdr["CDELT1"] = float(dw)
    ohdr["CRPIX1"] = 1.0
    ohdr["CTYPE1"] = "WAVE"
    ohdr["CUNIT1"] = str(unit)
    return ohdr


def _write_mef(
    path: Path,
    sci: np.ndarray,
    hdr: fits.Header,
    var: np.ndarray | None,
    mask: np.ndarray | None,
    cov: np.ndarray | None,
    *,
    wave0: float | None = None,
    dw: float | None = None,
    unit: str = "Angstrom",
) -> None:
    """Write a MEF product with SCI/VAR/MASK and optional COV.

    Uses the common writer from :mod:`scorpio_pipe.io.mef` and adds a COV
    extension for coverage / weights bookkeeping.
    """
    extra = []
    if cov is not None:
        try:
            extra.append(
                fits.ImageHDU(data=np.asarray(cov, dtype=np.float32), name="COV")
            )
        except Exception:
            pass

    grid = None
    if wave0 is not None and dw is not None:
        try:
            grid = WaveGrid(
                lambda0=float(wave0),
                dlambda=float(dw),
                nlam=int(np.asarray(sci).shape[1]),
                unit=str(unit),
            )
        except Exception:
            grid = None

    # Provide primary data for legacy tooling (fits.getdata reads primary).
    write_sci_var_mask(
        path,
        np.asarray(sci, dtype=np.float32),
        var=(None if var is None else np.asarray(var, dtype=np.float32)),
        mask=(None if mask is None else np.asarray(mask, dtype=np.uint16)),
        header=fits.Header(hdr),
        grid=grid,
        extra_hdus=extra if extra else None,
        primary_data=np.asarray(sci, dtype=np.float32),
        overwrite=True,
    )


@dataclass
class LinearizeResult:
    out_sum_fits: Path
    out_sum_json: Path
    per_exp_dir: Optional[Path]
    wave0: float
    dw: float
    nlam: int


def _open_fits_resilient(path: Path) -> Tuple[np.ndarray, fits.Header]:
    """Open FITS robustly (similar policy as in Inspect)."""
    for memmap in (True, False):
        try:
            with fits.open(
                path,
                memmap=memmap,
                ignore_missing_simple=True,
                ignore_missing_end=True,
            ) as hdul:
                data = hdul[0].data
                hdr = hdul[0].header
                if data is None:
                    raise ValueError("Primary HDU has no data")
                return np.asarray(data, dtype=np.float64), hdr
        except Exception:
            continue
    # final, let exception propagate with context
    with fits.open(path) as hdul:
        data = hdul[0].data
        if data is None:
            raise ValueError("Primary HDU has no data")
        return np.asarray(data, dtype=np.float64), hdul[0].header


def _open_science_with_optional_var_mask(
    path: Path,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, fits.Header]:
    """Open science frame.

    Supports both plain 2D primary-HDU FITS and pipeline MEF products with
    SCI/VAR/MASK extensions.
    """
    try:
        with fits.open(path, memmap=False) as hdul:
            if "SCI" in hdul:
                sci, var, mask, hdr = read_sci_var_mask(path)
                sci64 = np.asarray(sci, dtype=np.float64)
                var64 = None if var is None else np.asarray(var, dtype=np.float64)
                return sci64, var64, mask, hdr
    except Exception:
        # fall back to primary-only
        pass
    data, hdr = _open_fits_resilient(path)
    return data, None, None, fits.Header(hdr)


### Noise / variance helpers live in scorpio_pipe.noise_model


def _guess_saturation_level_adu(
    hdr: fits.Header, lcfg: Dict[str, Any]
) -> Optional[float]:
    """Estimate detector saturation level in ADU.

    Order of preference:
      - config override: linearize.saturation_adu (or saturation_adu, sat_adu)
      - header keywords: SATURATE, SATLEVEL, SAT_...
      - common unsigned-16bit convention (BITPIX=16 with BZERO=32768 and BSCALE=1): 65535

    Returns None if nothing sensible can be inferred.
    """

    # config override
    for ck in ("saturation_adu", "sat_adu", "saturation_level_adu"):
        if ck in lcfg and lcfg.get(ck) is not None:
            try:
                v = float(lcfg.get(ck))
                if math.isfinite(v) and v > 0:
                    return v
            except Exception:
                pass

    # header keywords
    for k in ("SATURATE", "SATLEVEL", "SATUR", "SATLEVEL", "SATURATION"):
        if k in hdr:
            try:
                v = float(hdr[k])
                if math.isfinite(v) and v > 0:
                    return v
            except Exception:
                pass

    # heuristic: unsigned 16-bit storage
    try:
        bitpix = int(hdr.get("BITPIX")) if hdr.get("BITPIX") is not None else None
    except Exception:
        bitpix = None
    if bitpix == 16:
        try:
            bscale = float(hdr.get("BSCALE", 1.0))
            bzero = float(hdr.get("BZERO", 0.0))
        except Exception:
            bscale, bzero = 1.0, 0.0
        if abs(bscale - 1.0) < 1e-9 and (
            abs(bzero - 32768.0) < 1e-3 or abs(bzero) < 1e-3
        ):
            return 65535.0

    # fallback: sometimes DATAMAX is close to saturation
    try:
        dmax = float(hdr.get("DATAMAX")) if hdr.get("DATAMAX") is not None else None
        if dmax is not None and math.isfinite(dmax) and dmax > 60000:
            return 65535.0
    except Exception:
        pass

    return None


def _lambda_edges(lam: np.ndarray) -> np.ndarray:
    """Pixel edges for a monotonic wavelength coordinate (len N -> N+1)."""
    lam = np.asarray(lam, dtype=np.float64)
    n = lam.size
    if n < 2:
        raise ValueError("Need at least 2 pixels to compute edges")
    edges = np.empty(n + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (lam[:-1] + lam[1:])
    # extrapolate boundaries
    edges[0] = lam[0] - (edges[1] - lam[0])
    edges[-1] = lam[-1] + (lam[-1] - edges[-2])
    return edges


def _monotonicize_lambda_centers(lam_centers: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return strictly monotonic *increasing* wavelength centers.

    The lambda_map should be monotonic along the dispersion axis. Small numerical
    imperfections (and occasional reversed dispersion) do occur; this helper
    keeps the pixel order (except a full reverse when dispersion is decreasing)
    and enforces strict monotonicity with a tiny step.
    """
    lam = np.asarray(lam_centers, dtype=float)
    if lam.size < 2:
        return lam, False

    d = np.diff(lam)
    d = d[np.isfinite(d)]
    reversed_disp = bool(d.size > 0 and np.nanmedian(d) < 0)
    if reversed_disp:
        lam = lam[::-1]
        d = -d

    dpos = d[d > 0]
    step = float(np.nanmedian(dpos)) if dpos.size else 1.0
    eps = max(abs(step) * 1e-3, 1e-6)

    out = lam.copy()
    for i in range(1, out.size):
        if not np.isfinite(out[i]):
            out[i] = out[i - 1] + eps
        elif out[i] <= out[i - 1]:
            out[i] = out[i - 1] + eps

    return out, reversed_disp


def _rebin_row_var_weightsquared(
    var_in: np.ndarray, lam_centers: np.ndarray, edges_out: np.ndarray
) -> np.ndarray:
    """Flux-conserving variance propagation for the rebinning transform.

    For output y = Σ w_j x_j (independent input pixels), Var(y) = Σ w_j^2 Var(x_j),
    where w_j is the overlap fraction of the output bin with input pixel j.
    """
    v = np.asarray(var_in, dtype=float)
    lam = np.asarray(lam_centers, dtype=float)
    edges_out = np.asarray(edges_out, dtype=float)

    if v.size == 0 or edges_out.size < 2:
        return np.zeros(max(edges_out.size - 1, 0), dtype=float)

    lam_m, rev = _monotonicize_lambda_centers(lam)
    if rev:
        v = v[::-1]

    edges_in = _lambda_edges(lam_m)
    widths = np.diff(edges_in)
    widths = np.where(widths > 0, widths, np.nan)

    n_in = v.size
    n_out = edges_out.size - 1
    out = np.zeros(n_out, dtype=float)

    i = 0
    k = 0
    while i < n_in and k < n_out:
        a0 = edges_in[i]
        a1 = edges_in[i + 1]
        b0 = edges_out[k]
        b1 = edges_out[k + 1]

        lo = a0 if a0 > b0 else b0
        hi = a1 if a1 < b1 else b1
        if hi > lo and np.isfinite(widths[i]) and widths[i] > 0 and np.isfinite(v[i]):
            frac = (hi - lo) / widths[i]
            out[k] += (frac * frac) * v[i]

        if a1 <= b1:
            i += 1
        else:
            k += 1

    return out


def _rebin_row_cumulative(
    values: np.ndarray, lam_centers: np.ndarray, edges_out: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Flux-conserving rebin of a 1D row onto a new wavelength grid.

    Implements an edge-based cumulative integral approach:
    - input pixel intervals are estimated from wavelength centers,
    - flux density is assumed constant within each interval,
    - the cumulative integral is interpolated to output bin edges.

    Returns the integrated flux per output bin and the coverage fraction (0..1).
    """
    v = np.asarray(values, dtype=float)
    lam = np.asarray(lam_centers, dtype=float)
    edges_out = np.asarray(edges_out, dtype=float)

    if v.size == 0 or edges_out.size < 2:
        n = max(edges_out.size - 1, 0)
        return np.zeros(n, dtype=float), np.zeros(n, dtype=float)

    lam_m, rev = _monotonicize_lambda_centers(lam)
    if rev:
        v = v[::-1]

    edges_in = _lambda_edges(lam_m)
    c = np.zeros(edges_in.size, dtype=float)
    c[1:] = np.cumsum(np.where(np.isfinite(v), v, 0.0))

    c_out = np.interp(edges_out, edges_in, c, left=c[0], right=c[-1])
    out = np.diff(c_out)

    lo_in = edges_in[0]
    hi_in = edges_in[-1]
    lo = edges_out[:-1]
    hi = edges_out[1:]
    overlap = np.maximum(0.0, np.minimum(hi, hi_in) - np.maximum(lo, lo_in))
    cov = overlap / np.maximum(hi - lo, 1e-12)

    return out.astype(float), cov.astype(float)


def _compute_output_grid(
    lam_map: np.ndarray,
    *,
    dw_hint: float,
    mode: str,
    lo_pct: float,
    hi_pct: float,
    imin_pct: float,
    imax_pct: float,
) -> Tuple[float, float, float, int]:
    """Pick a common λ grid based on the lambda_map.

    mode:
      - intersection: robust intersection across Y (recommended)
      - percentile: robust global min/max percentiles
      - union: robust union across Y
    """
    lam = np.asarray(lam_map, dtype=np.float64)
    finite = np.isfinite(lam)
    if not np.any(finite):
        raise ValueError("lambda_map contains no finite values")

    # per-row ranges for robust policies
    row_mins = []
    row_maxs = []
    for r in lam:
        m = r[np.isfinite(r)]
        if m.size < 8:
            continue
        row_mins.append(float(np.min(m)))
        row_maxs.append(float(np.max(m)))
    if not row_mins:
        lo = float(np.nanpercentile(lam[finite], lo_pct))
        hi = float(np.nanpercentile(lam[finite], hi_pct))
    else:
        rm = np.asarray(row_mins, dtype=float)
        rx = np.asarray(row_maxs, dtype=float)
        mmode = str(mode or "intersection").strip().lower()
        if mmode == "intersection":
            lo = float(np.percentile(rm, float(imin_pct)))
            hi = float(np.percentile(rx, float(imax_pct)))
            if not (hi > lo):
                # fallback to global percentiles
                lo = float(np.nanpercentile(lam[finite], lo_pct))
                hi = float(np.nanpercentile(lam[finite], hi_pct))
        elif mmode == "union":
            lo = float(np.percentile(rm, float(lo_pct)))
            hi = float(np.percentile(rx, float(hi_pct)))
        else:
            lo = float(np.nanpercentile(lam[finite], lo_pct))
            hi = float(np.nanpercentile(lam[finite], hi_pct))

    dw = float(dw_hint)
    if not (math.isfinite(dw) and dw > 0):
        # median spacing across a few central rows
        mids = []
        for yy in np.linspace(0, lam.shape[0] - 1, num=min(7, lam.shape[0]), dtype=int):
            row = lam[yy]
            rr = row[np.isfinite(row)]
            if rr.size < 16:
                continue
            d = np.diff(rr)
            d = d[np.isfinite(d) & (d > 0)]
            if d.size:
                mids.append(float(np.median(d)))
        dw = float(np.median(mids)) if mids else 1.0
        dw = max(dw, 0.1)

    n = int(max(16, math.ceil((hi - lo) / dw)))
    return lo, hi, dw, n



def _get_exptime_seconds(hdr: fits.Header) -> float:
    """Extract exposure time in seconds from a FITS header.

    We accept a variety of common keywords. If none is present, returns 1.0
    and the caller should warn if normalization is requested.
    """
    for key in ("EXPTIME", "EXPOSURE", "ITIME", "TEXPTIME", "ELAPTIME"):
        v = hdr.get(key, None)
        if v is None:
            continue
        try:
            vv = float(v)
            if math.isfinite(vv) and vv > 0:
                return vv
        except Exception:
            continue
    return 1.0


def _open_rectified_handles(paths: list[Path]):
    """Open rectified MEF files (SCI/VAR/MASK) with memmap for block-wise stacking."""
    handles = []
    for p in paths:
        hdul = fits.open(p, memmap=True)
        # Find by EXTNAME; fall back by index
        def _get(ext):
            try:
                return hdul[ext].data
            except Exception:
                # try name lookup
                try:
                    return hdul[ext.upper()].data
                except Exception:
                    return None

        sci = _get("SCI")
        if sci is None:
            sci = hdul[1].data
        var = _get("VAR")
        mask = _get("MASK")
        handles.append({"path": p, "hdul": hdul, "sci": sci, "var": var, "mask": mask})
    return handles


def _close_rectified_handles(handles):
    for h in handles:
        try:
            h["hdul"].close()
        except Exception:
            pass


def _robust_stack_sigma_clip(
    rect_paths: list[Path],
    *,
    exclude_bits: int,
    sigma: float = 4.0,
    maxiters: int = 2,
    block_rows: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Robust stack of rectified frames using sigma-clipping across exposures.

    Works block-wise in Y to avoid holding the full (nexp, ny, nlam) cube in RAM.
    Returns (sci, var, mask, coverage, stats).
    """
    if not rect_paths:
        raise ValueError("No rectified frames to stack")

    handles = _open_rectified_handles(rect_paths)
    try:
        sci0 = handles[0]["sci"]
        if sci0 is None:
            raise ValueError("Rectified SCI data not found")
        ny, nlam = sci0.shape

        out_sci = np.zeros((ny, nlam), dtype=np.float32)
        out_var = np.zeros((ny, nlam), dtype=np.float32)
        out_mask = np.zeros((ny, nlam), dtype=np.uint16)
        coverage = np.zeros((ny, nlam), dtype=np.int16)

        rejected_count = np.zeros((ny, nlam), dtype=np.int16)

        total_valid_samples = 0
        total_rejected_samples = 0

        for y0 in range(0, ny, int(max(1, block_rows))):
            y1 = min(ny, y0 + int(max(1, block_rows)))
            by = y1 - y0
            nexp = len(handles)

            vals = np.empty((nexp, by, nlam), dtype=np.float64)
            vars_ = np.empty((nexp, by, nlam), dtype=np.float64)
            masks = np.empty((nexp, by, nlam), dtype=np.uint16)

            for j, h in enumerate(handles):
                s = np.asarray(h["sci"][y0:y1, :], dtype=np.float64)
                if h["var"] is not None:
                    v = np.asarray(h["var"][y0:y1, :], dtype=np.float64)
                else:
                    v = np.full_like(s, np.nan, dtype=np.float64)
                if h["mask"] is not None:
                    m = np.asarray(h["mask"][y0:y1, :], dtype=np.uint16)
                else:
                    m = np.zeros((by, nlam), dtype=np.uint16)

                # valid sample mask
                valid = (v > 0) & np.isfinite(s) & ((m & exclude_bits) == 0)
                total_valid_samples += int(np.sum(valid))

                s[~valid] = np.nan
                v[~valid] = np.nan

                vals[j] = s
                vars_[j] = v
                masks[j] = m

            # Iterative sigma clipping
            cur_vals = vals
            cur_vars = vars_

            for _it in range(int(maxiters)):
                center = np.nanmedian(cur_vals, axis=0)
                # MAD-based scatter
                resid = np.abs(cur_vals - center[None, :, :])
                mad = np.nanmedian(resid, axis=0)
                scatter = 1.4826 * mad
                scatter = np.where(np.isfinite(scatter) & (scatter > 0), scatter, np.nan)

                # Combined expected sigma per sample
                combined = np.sqrt(np.nan_to_num(scatter, nan=0.0) ** 2 + np.nan_to_num(cur_vars, nan=0.0))
                combined = np.where(combined > 0, combined, np.nan)

                reject = np.isfinite(cur_vals) & (resid > float(sigma) * combined)
                if not np.any(reject):
                    break
                total_rejected_samples += int(np.sum(reject))
                cur_vals = cur_vals.copy()
                cur_vars = cur_vars.copy()
                cur_vals[reject] = np.nan
                cur_vars[reject] = np.nan

                rejected_count[y0:y1, :] += np.sum(reject, axis=0).astype(np.int16)

            # Final weighted mean
            w = np.zeros_like(cur_vars, dtype=np.float64)
            good = np.isfinite(cur_vals) & np.isfinite(cur_vars) & (cur_vars > 0)
            w[good] = 1.0 / cur_vars[good]
            sumw = np.sum(w, axis=0)
            sumwv = np.sum(w * np.nan_to_num(cur_vals, nan=0.0), axis=0)

            mean = np.zeros((by, nlam), dtype=np.float32)
            varo = np.zeros((by, nlam), dtype=np.float32)
            ok = sumw > 0
            mean[ok] = (sumwv[ok] / sumw[ok]).astype(np.float32)
            varo[ok] = (1.0 / np.maximum(sumw[ok], 1e-30)).astype(np.float32)

            cov = np.sum(good, axis=0).astype(np.int16)
            coverage[y0:y1, :] = cov

            # Mask: OR of accepted masks + REJECTED where something was rejected
            mout = np.zeros((by, nlam), dtype=np.uint16)
            for j in range(nexp):
                # accepted where good[j] True
                mj = masks[j]
                mout |= np.where(good[j], mj, 0).astype(np.uint16)
            rej_pix = rejected_count[y0:y1, :] > 0
            mout[rej_pix] |= REJECTED
            mout[cov == 0] |= NO_COVERAGE

            out_sci[y0:y1, :] = mean
            out_var[y0:y1, :] = varo
            out_mask[y0:y1, :] = mout

        stats = {
            "total_valid_samples": int(total_valid_samples),
            "total_rejected_samples": int(total_rejected_samples),
            "rejected_fraction": float(total_rejected_samples / total_valid_samples) if total_valid_samples else 0.0,
            "rejected_pixels_fraction": float(np.mean(rejected_count > 0)) if rejected_count.size else 0.0,
        }
        return out_sci, out_var, out_mask, coverage, stats
    finally:
        _close_rectified_handles(handles)


def run_linearize(
    cfg: Dict[str, Any],
    out_dir: Optional[Path] = None,
    *,
    cancel_token: Any | None = None,
) -> Dict[str, Any]:
    """Run linearization.

    Inputs:
      - lambda_map: from wavesolution stage
      - cleaned object frames (prefer cosmics outputs if present)

    Outputs (v5.14+):
      - products/lin/lin_preview.fits (MEF: SCI [+VAR,+MASK,+COV])  # quick-look stack for ROI/QC
      - products/lin/linearize_done.json
      - per-exposure rectified frames under products/lin/per_exp/

    Backward compatibility:
      - mirrors preview to work_dir/lin/obj_sum_lin.fits and work_dir/lin/obj_sum_lin.png
    """
    cfg = dict(cfg)
    work_dir = resolve_work_dir(cfg)
    if out_dir is None:
        out_dir = work_dir / "products" / "lin"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lcfg = cfg.get("linearize", {}) if isinstance(cfg.get("linearize"), dict) else {}
    # Canonical keys in v5.x schema:
    #   dlambda_A, lambda_min_A, lambda_max_A, per_exposure
    # Backward-compatible aliases:
    #   dw, wmin/wmax, save_per_exposure
    dw_raw = lcfg.get("dlambda_A", lcfg.get("dw", 1.0))
    if dw_raw is None:
        dw = float("nan")
    elif isinstance(dw_raw, str) and dw_raw.strip().lower() in {
        "auto",
        "data",
        "from_data",
    }:
        dw = float("nan")
    else:
        dw = float(dw_raw)
    y_crop_top = int(lcfg.get("y_crop_top", 0))
    y_crop_bottom = int(lcfg.get("y_crop_bottom", 0))
    save_per_exp = bool(
        lcfg.get(
            "per_exposure",
            lcfg.get("save_per_frame", lcfg.get("save_per_exposure", True)),
        )
    )

    # Locate lambda_map (prefer per-disperser layout)
    lam_path = None
    if (
        isinstance(lcfg.get("lambda_map_path"), str)
        and str(lcfg.get("lambda_map_path")).strip()
    ):
        lam_path = Path(str(lcfg.get("lambda_map_path"))).expanduser()
        if not lam_path.is_absolute():
            lam_path = (work_dir / lam_path).resolve()
    if lam_path is None:
        wsol = wavesol_dir(cfg)
        cand = wsol / "lambda_map.fits"
        if cand.exists():
            lam_path = cand
    if lam_path is None:
        # last resort: search anywhere under work_dir/wavesol
        base = work_dir / "wavesol"
        cand = list(base.rglob("lambda_map.fits")) + list(
            base.rglob("lambda_map*.fits")
        )
        lam_path = cand[0] if cand else None
    if lam_path is None or not lam_path.exists():
        raise FileNotFoundError(
            "lambda_map.fits not found (expected under work_dir/wavesol/<disperser>/)"
        )

    lam_map, lam_hdr = _open_fits_resilient(lam_path)
    if lam_map.ndim != 2:
        raise ValueError("lambda_map must be a 2D image")

    # Get science frames list (prefer cosmics cleaned)
    frames = cfg.get("frames", {}) if isinstance(cfg.get("frames"), dict) else {}
    obj_frames = frames.get("obj", [])
    if not obj_frames:
        raise ValueError("No object frames configured (frames.obj is empty)")

    # Prefer cosmics products if present.
    # NOTE: cosmics stage historically used different layouts. Try a few.
    cosm_cfg = cfg.get("cosmics", {}) if isinstance(cfg.get("cosmics"), dict) else {}
    method = str(cosm_cfg.get("method", "stack_mad"))
    kind = "obj"
    cosm_root = work_dir / "cosmics"
    cand_clean = [
        cosm_root / kind / "clean",  # current
        cosm_root / method / kind / "clean",  # legacy (method/kind)
        cosm_root / method / "clean",  # legacy (method)
    ]
    cand_mask = [
        cosm_root / kind / "masks_fits",
        cosm_root / method / kind / "masks_fits",
        cosm_root / method / "masks_fits",
        cosm_root / kind / "mask_fits",  # very old typo
    ]
    clean_dir = next((p for p in cand_clean if p.exists()), cand_clean[0])
    mask_dir = next((p for p in cand_mask if p.exists()), cand_mask[0])

    resolved = []
    for fp in obj_frames:
        p = Path(fp)
        # config paths are expected to be absolute or relative to data_dir
        if not p.is_absolute():
            data_dir = Path(cfg.get("data_dir", "")).expanduser()
            if data_dir:
                p = (data_dir / p).expanduser()
        base = p.stem
        clean_p = clean_dir / f"{base}_clean.fits"
        resolved.append(clean_p if clean_p.exists() else p)
    sci_paths = [Path(p) for p in resolved]


    # Output wavelength grid (common for the whole series)
    unit, waveref, unit_src = _read_lambda_map_meta(lam_hdr, lam_map)
    if unit_src == "heuristic":
        log.warning(
            "lambda_map is missing explicit wavelength metadata (WAVEUNIT/WAVEREF); "
            "falling back to heuristic unit=%s waveref=%s. "
            "Please re-run Wavesolution to regenerate lambda_map.fits.",
            unit,
            waveref,
        )

    wmin_cfg = lcfg.get("lambda_min_A", lcfg.get("wmin"))
    wmax_cfg = lcfg.get("lambda_max_A", lcfg.get("wmax"))

    bunit_mode = str(
        lcfg.get("bunit_mode", lcfg.get("output_bunit_mode", "adu_bin"))
    ).strip().lower()
    if bunit_mode in {
        "adu/angstrom",
        "adu_per_angstrom",
        "adu_per_a",
        "adu/a",
        "per_angstrom",
        "density",
        "adu/nm",
        "adu_per_nm",
        "per_nm",
        "adu_per_unit",
    }:
        bunit_mode = "adu_per_unit"
    elif bunit_mode in {"adu/bin", "adu_bin", "bin", "integrated"}:
        bunit_mode = "adu_bin"
    else:
        # unknown -> keep backwards-compatible behavior
        bunit_mode = "adu_bin"

    # grid policy
    grid_mode = str(lcfg.get("grid_mode", "intersection"))
    lo_pct = float(lcfg.get("grid_lo_pct", 1.0))
    hi_pct = float(lcfg.get("grid_hi_pct", 99.0))
    imin_pct = float(lcfg.get("grid_intersection_min_pct", 95.0))
    imax_pct = float(lcfg.get("grid_intersection_max_pct", 5.0))

    # dw: if in pixel space and user did not provide, default to 1
    if unit == "pix" and not (math.isfinite(dw) and dw > 0):
        dw = 1.0

    if wmin_cfg is not None and wmax_cfg is not None:
        wave0 = float(wmin_cfg)
        wmax = float(wmax_cfg)
        if not (math.isfinite(dw) and dw > 0):
            # compute from data
            _, _, dw, _ = _compute_output_grid(
                lam_map,
                dw_hint=dw,
                mode=grid_mode,
                lo_pct=lo_pct,
                hi_pct=hi_pct,
                imin_pct=imin_pct,
                imax_pct=imax_pct,
            )
        if wmax <= wave0:
            raise ValueError("linearize.lambda_max_A must be > lambda_min_A")
        nlam = int(max(16, np.ceil((wmax - wave0) / dw)))
    else:
        wave0_auto, wmax_auto, dw, nlam = _compute_output_grid(
            lam_map,
            dw_hint=dw,
            mode=grid_mode,
            lo_pct=lo_pct,
            hi_pct=hi_pct,
            imin_pct=imin_pct,
            imax_pct=imax_pct,
        )
        # allow partial overrides
        wave0 = float(wmin_cfg) if wmin_cfg is not None else float(wave0_auto)
        wmax = float(wmax_cfg) if wmax_cfg is not None else float(wmax_auto)
        nlam = int(max(16, np.ceil((wmax - wave0) / dw)))

    wave_edges = wave0 + dw * np.arange(nlam + 1, dtype=np.float64)

    # Output science units (integrated per-bin vs per-wavelength-unit)
    scale_to_density = bunit_mode == "adu_per_unit" and unit != "pix"
    if scale_to_density:
        if not (math.isfinite(dw) and dw > 0):
            raise ValueError("dw must be positive for bunit_mode=adu_per_unit")
        bunit = f"ADU/{unit}"
    else:
        bunit = "ADU/bin"

    # Crop by Y if requested
    ny, nx = lam_map.shape
    y0 = max(0, y_crop_top)
    y1 = ny - max(0, y_crop_bottom)
    if y1 <= y0 + 1:
        raise ValueError("Invalid y-crop; resulting height is <= 1")
    lam_map = lam_map[y0:y1, :]
    ny = lam_map.shape[0]

    per_exp_dir = out_dir / "per_exp" if save_per_exp else None
    if per_exp_dir is not None:
        per_exp_dir.mkdir(parents=True, exist_ok=True)

    # Stacking configuration
    normalize_exptime = bool(lcfg.get("normalize_exptime", True))
    robust_stack = bool(lcfg.get("robust_stack", True))
    stack_sigma = float(lcfg.get("stack_sigma", 4.0) or 4.0)
    stack_maxiters = int(lcfg.get("stack_maxiters", 2) or 2)
    # Exclude these mask bits from stacking weights
    exclude_bits = BADPIX | COSMIC | SATURATED | NO_COVERAGE | USER

    # Ensure per-exposure products exist when robust stacking is enabled
    if robust_stack and not save_per_exp:
        log.warning("robust_stack=True requires per-exposure rectified frames; enabling save_per_exposure.")
        save_per_exp = True
        per_exp_dir = out_dir / "per_exp"
        per_exp_dir.mkdir(parents=True, exist_ok=True)

    rect_paths: list[Path] = []
    exp_times_s: list[float] = []

    # Apply EXPTIME normalization policy to declared output units
    if normalize_exptime:
        if isinstance(bunit, str) and bunit.startswith("ADU/"):
            bunit = "ADU/s/" + bunit.split("ADU/", 1)[1]
        elif bunit == "ADU/bin":
            bunit = "ADU/s/bin"
        else:
            bunit = "ADU/s"

    # Legacy (non-robust) accumulator is kept for fallback only
    stack_num = np.zeros((ny, nlam), dtype=np.float64)
    stack_den = np.zeros((ny, nlam), dtype=np.float64)
    stack_var_den = np.zeros((ny, nlam), dtype=np.float64)
    coverage = np.zeros((ny, nlam), dtype=np.int16)
    mask_sum = np.zeros((ny, nlam), dtype=np.uint16)

    sat_levels_adu: list[float] = []

    # Noise model overrides (optional).
    gain_override = lcfg.get("gain_e_per_adu")
    rdnoise_override = lcfg.get("read_noise_e")
    gain_override_f = float(gain_override) if gain_override is not None else None
    rdnoise_override_f = (
        float(rdnoise_override) if rdnoise_override is not None else None
    )
    noise_meta: list[dict[str, Any]] = []

    for i, p in enumerate(sci_paths, 1):
        if cancel_token is not None and getattr(cancel_token, "cancelled", False):
            raise RuntimeError("Cancelled")
        log.info("Linearize %d/%d: %s", i, len(sci_paths), p.name)
        data, var_in, mask_in, hdr = _open_science_with_optional_var_mask(p)
        if data.ndim != 2:
            raise ValueError(f"Science frame must be 2D: {p}")
        data = data[y0:y1, :]
        if data.shape != lam_map.shape:
            raise ValueError(
                f"Science shape {data.shape} != lambda_map shape {lam_map.shape} for {p.name}"
            )

        mask = None
        if mask_in is not None:
            mask = np.asarray(mask_in, dtype=np.uint16)[y0:y1, :]
        # Resolve (gain, read-noise) in a single, centralized way.
        npar = resolve_noise_params(
            hdr,
            gain_override=gain_override_f,
            rdnoise_override=rdnoise_override_f,
        )
        hdr.setdefault("GAIN", float(npar.gain_e_per_adu))
        hdr.setdefault("RDNOISE", float(npar.rdnoise_e))
        hdr.setdefault("NOISRC", str(npar.source))

        if var_in is not None:
            var = np.asarray(var_in, dtype=np.float64)[y0:y1, :]
            var_source = "input"
        else:
            var_est, _npar2 = estimate_variance_adu2(
                data,
                hdr,
                gain_override=gain_override_f,
                rdnoise_override=rdnoise_override_f,
            )
            var = np.asarray(var_est, dtype=np.float64)
            var_source = "estimated"

        # Sanity: variance must be finite and >= 0. Flag bad pixels via mask.
        bad_var = ~np.isfinite(var) | (var < 0)
        if np.any(bad_var):
            var = np.where(bad_var, 0.0, var)
            if mask is None:
                mask = np.zeros_like(var, dtype=np.uint16)
            mask[bad_var] |= BADPIX

        noise_meta.append(
            {
                "path": str(p),
                "var_source": var_source,
                "gain_e_per_adu": float(npar.gain_e_per_adu),
                "rdnoise_e": float(npar.rdnoise_e),
                "noise_source": str(npar.source),
            }
        )

        # Saturation masking (optional): flag detector pixels near/above full well
        mask_saturation = bool(lcfg.get("mask_saturation", True))
        sat_level = _guess_saturation_level_adu(hdr, lcfg) if mask_saturation else None
        if sat_level is not None:
            try:
                sat_levels_adu.append(float(sat_level))
            except Exception:
                pass
        sat_margin = float(lcfg.get("saturation_margin_adu", 0.0) or 0.0)
        satmask = None
        if sat_level is not None and mask_saturation:
            try:
                thr = float(sat_level) - float(sat_margin)
                if np.isfinite(thr):
                    satmask = np.isfinite(data) & (data >= thr)
            except Exception:
                satmask = None

        # Merge legacy cosmics masks (if present) with any input MASK plane.
        base_for_mask = Path(p).stem
        if base_for_mask.endswith("_clean"):
            base_for_mask = base_for_mask[:-6]
        mp = mask_dir / f"{base_for_mask}_mask.fits"
        if mp.exists():
            try:
                m, _ = _open_fits_resilient(mp)
                cm = ((m.astype(np.uint16) > 0).astype(np.uint16) * COSMIC).astype(
                    np.uint16
                )
                cm = cm[y0:y1, :]
                if mask is None:
                    mask = cm
                else:
                    mask = np.asarray((mask | cm), dtype=np.uint16)
            except Exception as e:
                log.warning("Failed to load cosmic mask %s: %s", mp.name, e)
                # keep existing mask if any

        rect = np.zeros((ny, nlam), dtype=np.float32)
        rect_var = np.zeros((ny, nlam), dtype=np.float32)
        rect_mask = np.zeros((ny, nlam), dtype=np.uint16)
        rect_cov = np.zeros((ny, nlam), dtype=np.float32)

        for yy in range(ny):
            lam_row = lam_map[yy]
            # Bad rows: mark no coverage
            if not np.any(np.isfinite(lam_row)):
                rect_mask[yy, :] |= NO_COVERAGE
                continue
            # Use only finite subset for rebin; if too few finite points, skip
            finite = np.isfinite(lam_row)
            if np.sum(finite) < 8:
                rect_mask[yy, :] |= NO_COVERAGE
                continue
            v_row, cov = _rebin_row_cumulative(
                data[yy, finite], lam_row[finite], wave_edges
            )
            vv_row = _rebin_row_var_weightsquared(
                var[yy, finite], lam_row[finite], wave_edges
            )
            rect[yy] = v_row
            rect_var[yy] = np.maximum(vv_row, 0.0)
            rect_cov[yy] = cov
            # mark incomplete coverage
            # Coverage is a *fraction* per output bin; mark incomplete bins as EDGE.
            rect_mask[yy, (cov > 0) & (cov < 0.999)] |= EDGE
            rect_mask[yy, cov <= 0] |= NO_COVERAGE
            if mask is not None:
                # Bit-preserving mask propagation: propagate each known bit separately.
                for bit in (BADPIX, COSMIC, SATURATED, USER, REJECTED):
                    m_row, _ = _rebin_row_cumulative(
                        ((mask[yy, finite] & bit) > 0).astype(np.float64),
                        lam_row[finite],
                        wave_edges,
                    )
                    rect_mask[yy, m_row > 0] |= bit

            if satmask is not None:
                # propagate saturation flags through the same rebin operator
                sm_row, _ = _rebin_row_cumulative(
                    satmask[yy, finite].astype(np.float64), lam_row[finite], wave_edges
                )
                rect_mask[yy, sm_row > 0] |= SATURATED
        # Optional: express data as a *density* per wavelength unit instead of per-bin counts.
        if scale_to_density:
            rect /= float(dw)
            rect_var /= float(dw) ** 2
        # Record exposure time and optionally normalize to ADU/s (or ADU/s/<unit>)
        exptime_s = _get_exptime_seconds(hdr)
        exp_times_s.append(float(exptime_s))
        if normalize_exptime:
            if not (math.isfinite(exptime_s) and exptime_s > 0):
                log.warning("EXPTIME missing/invalid in %s; using 1.0 s for normalization", p.name)
                exptime_s = 1.0
            rect /= float(exptime_s)
            rect_var /= float(exptime_s) ** 2

        # Per-exposure output
        if per_exp_dir is not None:
            ohdr = _set_linear_wcs(hdr, wave0, dw, unit=unit)
            ohdr["WAVEUNIT"] = (str(unit), "Wavelength unit (explicit)")
            ohdr["WAVEREF"] = (str(waveref), "Wavelength reference (air/vacuum)")
            ohdr["BUNIT"] = (str(bunit), "Data unit")
            ohdr["NORMEXP"] = (bool(normalize_exptime), "Normalize to per-second units (ADU/s)")
            ohdr["TEXPS"] = (float(exptime_s), "Exposure time used for normalization (s)")
            # Saturation metadata (if used)
            if sat_level is not None and mask_saturation:
                try:
                    ohdr["SATLEV"] = (
                        float(sat_level),
                        "Saturation level (ADU) used to flag MASK",
                    )
                    ohdr["SATMARG"] = (float(sat_margin), "Saturation margin (ADU)")
                except Exception:
                    pass
            ohdr = add_provenance(ohdr, cfg, stage="linearize")

            # Canonical naming: *_rectified.fits
            #
            # Important (v5.38+): do NOT write compatibility aliases (e.g. *_lin.fits)
            # into canonical stage folders. Legacy filenames are supported via
            # resolve_input_path(...) when reading old workspaces.
            out_rect = per_exp_dir / f"{base_for_mask}_rectified.fits"
            _write_mef(
                out_rect,
                rect,
                ohdr,
                rect_var,
                rect_mask,
                rect_cov,
                wave0=wave0,
                dw=dw,
                unit=unit,
            )
            rect_paths.append(out_rect)

        if not robust_stack:
                    # Stack: inverse-variance weighted mean
                    w = np.zeros_like(rect_var, dtype=np.float64)
                    fatal_bits = NO_COVERAGE | BADPIX | COSMIC | SATURATED | USER | REJECTED
                    good = (rect_var > 0) & ((rect_mask & fatal_bits) == 0)
                    w[good] = 1.0 / rect_var[good]
                    stack_num += rect.astype(np.float64) * w
                    stack_den += w
                    # For the output variance (of the weighted mean): var = 1/sum(w)
                    stack_var_den += w
                    coverage += good.astype(np.int16)
                    mask_sum |= rect_mask

    # Final stack
    stack_stats: dict[str, Any] = {}
    if robust_stack:
        if not rect_paths:
            raise RuntimeError("robust_stack=True but no per-exposure rectified frames are available")
        out_sci, out_var, out_mask, coverage, stack_stats = _robust_stack_sigma_clip(
            rect_paths,
            exclude_bits=exclude_bits,
            sigma=stack_sigma,
            maxiters=stack_maxiters,
            block_rows=int(lcfg.get("stack_block_rows", 64) or 64),
        )
    else:
        out_sci = np.zeros((ny, nlam), dtype=np.float32)
        out_var = np.zeros((ny, nlam), dtype=np.float32)
        ok = stack_den > 0
        out_sci[ok] = (stack_num[ok] / stack_den[ok]).astype(np.float32)
        out_var[ok] = (1.0 / np.maximum(stack_var_den[ok], 1e-20)).astype(np.float32)
        out_mask = mask_sum.copy()
        out_mask[~ok] |= NO_COVERAGE

    # Output products
    preview_fits = out_dir / "lin_preview.fits"
    done_json = out_dir / "linearize_done.json"

    hdr0 = _set_linear_wcs(lam_hdr, wave0, dw, unit=unit)
    hdr0["WAVEUNIT"] = (str(unit), "Wavelength unit (explicit)")
    hdr0["WAVEREF"] = (str(waveref), "Wavelength reference (air/vacuum)")
    hdr0["BUNIT"] = (str(bunit), "Data unit")
    hdr0["NORMEXP"] = (bool(normalize_exptime), "Normalize stack to per-second units (ADU/s)")
    hdr0["NEXP"] = (int(len(exp_times_s)), "Number of exposures stacked")
    try:
        hdr0["TEXPTOT"] = (float(np.sum(exp_times_s)), "Total exposure time (s)")
        hdr0["TEXPMED"] = (float(np.median(exp_times_s)), "Median exposure time (s)")
    except Exception:
        pass
    hdr0["STKMD"] = ("sigclip" if robust_stack else "wmean", "Stacking method")
    if robust_stack:
        hdr0["STKSIG"] = (float(stack_sigma), "Sigma clip threshold")
        hdr0["STKITR"] = (int(stack_maxiters), "Sigma clip iterations")
    hdr0 = add_provenance(hdr0, cfg, stage="linearize")
    _write_mef(
        preview_fits,
        out_sci,
        hdr0,
        out_var,
        out_mask,
        coverage,
        wave0=wave0,
        dw=dw,
        unit=unit,
    )

    # Quicklook PNG
    preview_png = out_dir / "lin_preview.png"
    if bool(lcfg.get("save_png", True)):
        try:
            import matplotlib.pyplot as plt

            from scorpio_pipe.plot_style import mpl_style

            with mpl_style():
                fig = plt.figure(figsize=(8.0, 3.6))
                ax = fig.add_subplot(111)
                im = ax.imshow(out_sci, origin="lower", aspect="auto")
                ax.set_xlabel("λ pixel")
                ax.set_ylabel("y")
                ax.set_title("Linearized preview (stacked)")
                fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05)
                fig.tight_layout()
                fig.savefig(preview_png)
                plt.close(fig)
        except Exception:
            pass

    payload = {
        "stage": "linearize",
        "lambda_map": str(lam_path),
        "wave_unit": str(unit),
        "wave_ref": str(waveref),
        "unit_source": str(unit_src),
        "bunit": str(bunit),
        "bunit_mode": str(bunit_mode),
        "normalize_exptime": bool(normalize_exptime),
        "exptime_total_s": float(np.sum(exp_times_s)) if exp_times_s else None,
        "exptime_median_s": float(np.median(exp_times_s)) if exp_times_s else None,
        "stacking": {
            "method": "sigma_clip" if robust_stack else "wmean",
            "sigma": float(stack_sigma) if robust_stack else None,
            "maxiters": int(stack_maxiters) if robust_stack else None,
            **(stack_stats or {}),
        },
        "inputs": [str(p) for p in sci_paths],
        "wave0": wave0,
        "dw": dw,
        "nlam": nlam,
        "per_exposure": str(per_exp_dir) if per_exp_dir is not None else None,
        "saturation": {
            "mask_saturation": bool(lcfg.get("mask_saturation", True)),
            "saturation_margin_adu": float(
                lcfg.get("saturation_margin_adu", 0.0) or 0.0
            ),
            "saturation_level_adu": float(np.nanmedian(sat_levels_adu))
            if sat_levels_adu
            else None,
            "saturation_level_min_adu": float(np.nanmin(sat_levels_adu))
            if sat_levels_adu
            else None,
            "saturation_level_max_adu": float(np.nanmax(sat_levels_adu))
            if sat_levels_adu
            else None,
        },
        "noise": {
            "gain_override_e_per_adu": gain_override_f,
            "read_noise_override_e": rdnoise_override_f,
            "gain_median_e_per_adu": float(np.median([m["gain_e_per_adu"] for m in noise_meta]))
            if noise_meta
            else None,
            "rdnoise_median_e": float(np.median([m["rdnoise_e"] for m in noise_meta]))
            if noise_meta
            else None,
            "per_exposure": noise_meta,
        },
        "products": {
            "preview_fits": str(preview_fits),
            "preview_png": str(preview_png) if preview_png.exists() else None,
        },
    }

    # Backward-compatibility: older code/tests expect preview paths at top-level.
    payload["preview_fits"] = str(preview_fits)
    if preview_png.exists():
        payload["preview_png"] = str(preview_png)

    # Wave grid metadata (used by downstream stages & external tooling)
    wave_grid_json = out_dir / "wave_grid.json"
    try:
        wave_grid_json.write_text(
            json.dumps(
                {
                    "unit": str(unit),
                    "wave0": float(wave0),
                    "dw": float(dw),
                    "nlam": int(nlam),
                    "grid_mode": str(lcfg.get("grid_mode", "intersection")),
                    "y_crop": {"y0": int(y0), "y1": int(y1)},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        payload.setdefault("products", {})["wave_grid_json"] = str(wave_grid_json)
    except Exception:
        pass

    # QC metrics for quick inspection
    try:
        from scorpio_pipe.work_layout import ensure_work_layout

        qc_dir = ensure_work_layout(work_dir).manifest
        qc_dir.mkdir(parents=True, exist_ok=True)
        qc_path = qc_dir / "linearize_qc.json"

        fatal_bits = NO_COVERAGE | BADPIX | COSMIC | SATURATED | USER | REJECTED
        good = (out_var > 0) & ((out_mask & fatal_bits) == 0)
        snr = np.zeros_like(out_sci, dtype=np.float64)
        snr[good] = np.abs(out_sci[good].astype(np.float64)) / np.sqrt(
            np.maximum(out_var[good].astype(np.float64), 1e-20)
        )
        snr_vals = snr[good]
        cov_nonzero = float(np.count_nonzero(coverage > 0) / float(coverage.size)) if coverage.size else 0.0
        qc = {
            "stage": "linearize",
            "preview": str(preview_fits),
            "wave0": float(wave0),
            "dw": float(dw),
            "nlam": int(nlam),
            "wave_unit": str(unit),
            "wave_ref": str(waveref),
            "bunit": str(bunit),
            "exptime_policy": {
                "normalize_exptime": bool(normalize_exptime),
                "n_exp": int(len(exp_times_s)),
                "total_s": float(np.sum(exp_times_s)) if exp_times_s else None,
                "median_s": float(np.median(exp_times_s)) if exp_times_s else None,
            },
            "stacking": {
                "method": "sigma_clip" if robust_stack else "wmean",
                "sigma": float(stack_sigma) if robust_stack else None,
                "maxiters": int(stack_maxiters) if robust_stack else None,
                "exclude_bits": int(exclude_bits),
                **(stack_stats or {}),
            },
            "coverage": {
                "min": int(np.min(coverage)),
                "median": float(np.median(coverage)),
                "max": int(np.max(coverage)),
                "nonzero_frac": cov_nonzero,
            },
            "mask_summary": summarize_mask(out_mask),
            "noise": {
                "gain_median_e_per_adu": float(np.median([m["gain_e_per_adu"] for m in noise_meta]))
                if noise_meta
                else None,
                "rdnoise_median_e": float(np.median([m["rdnoise_e"] for m in noise_meta]))
                if noise_meta
                else None,
                "gain_override_e_per_adu": gain_override_f,
                "read_noise_override_e": rdnoise_override_f,
            },
            "snr_abs": {
                "median": float(np.nanmedian(snr_vals)) if snr_vals.size else None,
                "p10": float(np.nanpercentile(snr_vals, 10)) if snr_vals.size else None,
                "p90": float(np.nanpercentile(snr_vals, 90)) if snr_vals.size else None,
            },
        }
        qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
        payload.setdefault("products", {})["qc_json"] = str(qc_path)
    except Exception:
        pass
    done_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Legacy mirroring (disabled by default).
    #
    # v5.38+ keeps legacy paths for *reading* via resolve_input_path(...), but
    # does not write duplicated outputs unless explicitly requested.
    try:
        compat = cfg.get("compat") if isinstance(cfg.get("compat"), dict) else {}
        if bool(compat.get("write_legacy_outputs", False)):
            legacy_dir = work_dir / "lin"
            if legacy_dir.resolve() != out_dir.resolve() and legacy_dir.is_dir():
                import shutil

                shutil.copy2(preview_fits, legacy_dir / "obj_sum_lin.fits")
                if preview_png.exists():
                    shutil.copy2(preview_png, legacy_dir / "obj_sum_lin.png")
                shutil.copy2(done_json, legacy_dir / "linearize_done.json")
    except Exception:
        pass

    log.info("Linearize done: %s", preview_fits)
    return payload