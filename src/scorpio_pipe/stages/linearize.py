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
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from astropy.io import fits

from scorpio_pipe.io.mef import WaveGrid, read_sci_var_mask, write_sci_var_mask
from scorpio_pipe.noise_model import estimate_variance_adu2, resolve_noise_params
from scorpio_pipe.units_model import ensure_electron_units
from scorpio_pipe.maskbits import (
    BADPIX,
    COSMIC,
    EDGE,
    INVALID_WAVELENGTH,
    NO_COVERAGE,
    OUTSIDE_SLIT,
    REJECTED,
    SATURATED,
    SKYMODEL_FAIL,
    USER,
    summarize as summarize_mask,
)

from ..logging_utils import get_logger
from ..provenance import add_provenance
from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.wavesol_paths import wavesol_dir
from scorpio_pipe.workspace_paths import stage_dir

from scorpio_pipe.compare_cache import build_stage_diff, snapshot_stage
from scorpio_pipe.io.atomic import atomic_write_json
from scorpio_pipe.io.quicklook import quicklook_from_mef
from scorpio_pipe.adapters.parallel_map import parallel_map

log = get_logger(__name__)


@dataclass(frozen=True)
class _LinearizeRowPlan:
    finite_idx: np.ndarray
    src_idx: np.ndarray
    weights: np.ndarray
    weights_sq: np.ndarray
    starts: np.ndarray
    counts: np.ndarray


@dataclass(frozen=True)
class _LinearizeGridPlan:
    rows: tuple[_LinearizeRowPlan | None, ...]
    nout: int


_LINEARIZE_GRID_CACHE_LOCK = threading.RLock()
_LINEARIZE_GRID_CACHE: OrderedDict[tuple[Any, ...], _LinearizeGridPlan] = OrderedDict()


def _linearize_grid_cache_enabled() -> bool:
    raw = str(os.getenv("SCORPIPE_LINEARIZE_GRID_CACHE", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _linearize_grid_cache_max_items() -> int:
    raw = os.getenv("SCORPIPE_LINEARIZE_GRID_CACHE_ITEMS", "2")
    try:
        return max(1, int(raw))
    except Exception:
        return 2


def _reduce_bins(flat: np.ndarray, starts: np.ndarray, counts: np.ndarray, nout: int) -> np.ndarray:
    out = np.zeros(int(nout), dtype=float)
    if flat.size == 0 or starts.size == 0 or counts.size == 0:
        return out
    nz = counts > 0
    if not np.any(nz):
        return out
    out[nz] = np.add.reduceat(np.asarray(flat, dtype=float), starts[nz])
    return out


def _build_linearize_row_plan(lam_row: np.ndarray, edges_out: np.ndarray) -> _LinearizeRowPlan | None:
    finite_idx = np.flatnonzero(np.isfinite(lam_row))
    if finite_idx.size < 8 or edges_out.size < 2:
        return None

    lam_use = np.asarray(lam_row[finite_idx], dtype=float)
    lam_m, rev = _monotonicize_lambda_centers(lam_use)
    if rev:
        finite_idx = finite_idx[::-1]

    edges_in = _lambda_edges(lam_m)
    widths = np.diff(edges_in)
    nb = int(edges_out.size - 1)

    starts = np.zeros(nb, dtype=np.int32)
    counts = np.zeros(nb, dtype=np.int32)
    src_idx_list: list[int] = []
    weights_list: list[float] = []

    i = 0
    n = int(finite_idx.size)
    flat_pos = 0
    for j in range(nb):
        starts[j] = int(flat_pos)
        lo = float(edges_out[j])
        hi = float(edges_out[j + 1])
        while i < n and float(edges_in[i + 1]) <= lo:
            i += 1
        ii = i
        bin_w = hi - lo
        if not np.isfinite(bin_w) or bin_w <= 0:
            continue
        while ii < n and float(edges_in[ii]) < hi:
            w_i = float(widths[ii]) if ii < widths.size else 0.0
            if w_i > 0 and np.isfinite(w_i):
                a = float(edges_in[ii])
                b = float(edges_in[ii + 1])
                overlap = min(hi, b) - max(lo, a)
                if overlap > 0:
                    src_idx_list.append(int(ii))
                    weights_list.append(float(overlap / bin_w))
                    counts[j] += 1
                    flat_pos += 1
            ii += 1

    src_idx = np.asarray(src_idx_list, dtype=np.int32)
    weights = np.asarray(weights_list, dtype=np.float64)
    return _LinearizeRowPlan(
        finite_idx=np.asarray(finite_idx, dtype=np.int32),
        src_idx=src_idx,
        weights=weights,
        weights_sq=np.square(weights, dtype=np.float64),
        starts=starts,
        counts=counts,
    )


def _build_linearize_grid_plan(lam_map: np.ndarray, edges_out: np.ndarray) -> _LinearizeGridPlan:
    rows = tuple(_build_linearize_row_plan(lam_map[yy], edges_out) for yy in range(int(lam_map.shape[0])))
    return _LinearizeGridPlan(rows=rows, nout=max(int(edges_out.size - 1), 0))


def _get_linearize_grid_plan(
    lam_map: np.ndarray,
    *,
    edges_out: np.ndarray,
    frame_signature_key: str,
    lambda_map_key: str,
    wave0: float,
    dw: float,
    nlam: int,
) -> tuple[_LinearizeGridPlan, bool]:
    cache_key = (
        str(frame_signature_key),
        str(lambda_map_key),
        int(lam_map.shape[0]),
        int(lam_map.shape[1]),
        float(wave0),
        float(dw),
        int(nlam),
    )
    if _linearize_grid_cache_enabled():
        with _LINEARIZE_GRID_CACHE_LOCK:
            cached = _LINEARIZE_GRID_CACHE.get(cache_key)
            if cached is not None:
                _LINEARIZE_GRID_CACHE.move_to_end(cache_key)
                return cached, True

    plan = _build_linearize_grid_plan(lam_map, edges_out)
    if _linearize_grid_cache_enabled():
        with _LINEARIZE_GRID_CACHE_LOCK:
            _LINEARIZE_GRID_CACHE[cache_key] = plan
            _LINEARIZE_GRID_CACHE.move_to_end(cache_key)
            while len(_LINEARIZE_GRID_CACHE) > _linearize_grid_cache_max_items():
                _LINEARIZE_GRID_CACHE.popitem(last=False)
    return plan, False


def clear_linearize_grid_cache() -> None:
    with _LINEARIZE_GRID_CACHE_LOCK:
        _LINEARIZE_GRID_CACHE.clear()


def _apply_linearize_row_plan(
    values: np.ndarray,
    row_plan: _LinearizeRowPlan,
    *,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(values, dtype=float)[row_plan.finite_idx]
    if valid_mask is None:
        valid = np.ones(vals.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)[row_plan.finite_idx]
        if valid.shape != vals.shape:
            raise ValueError('valid_mask must have same shape as values')

    src_idx = row_plan.src_idx
    src_valid = valid[src_idx] & np.isfinite(vals[src_idx])
    contrib = np.where(src_valid, vals[src_idx] * row_plan.weights, 0.0)
    out = _reduce_bins(contrib, row_plan.starts, row_plan.counts, row_plan.counts.size)

    cov_contrib = np.where(valid[src_idx], row_plan.weights, 0.0)
    cov = _reduce_bins(cov_contrib, row_plan.starts, row_plan.counts, row_plan.counts.size)
    cov = np.clip(cov, 0.0, 1.0)
    return out.astype(float), cov.astype(float)


def _apply_linearize_row_plan_var(
    var_in: np.ndarray,
    row_plan: _LinearizeRowPlan,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    vals = np.asarray(var_in, dtype=float)[row_plan.finite_idx]
    if valid_mask is None:
        valid = np.ones(vals.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)[row_plan.finite_idx]
        if valid.shape != vals.shape:
            raise ValueError('valid_mask must have same shape as var_in')

    src_idx = row_plan.src_idx
    src_valid = valid[src_idx] & np.isfinite(vals[src_idx]) & (vals[src_idx] >= 0)
    contrib = np.where(src_valid, vals[src_idx] * row_plan.weights_sq, 0.0)
    out = _reduce_bins(contrib, row_plan.starts, row_plan.counts, row_plan.counts.size)
    return out.astype(float)


def _apply_linearize_row_plan_maskbit(mask_row: np.ndarray, row_plan: _LinearizeRowPlan, bit: int) -> np.ndarray:
    vals = ((np.asarray(mask_row)[row_plan.finite_idx].astype(np.uint16) & np.uint16(bit)) > 0)
    src_idx = row_plan.src_idx
    contrib = np.where(vals[src_idx], row_plan.weights, 0.0)
    return _reduce_bins(contrib, row_plan.starts, row_plan.counts, row_plan.counts.size) > 0


def _apply_linearize_row_plan_bool(mask_row: np.ndarray, row_plan: _LinearizeRowPlan) -> np.ndarray:
    vals = np.asarray(mask_row, dtype=bool)[row_plan.finite_idx]
    src_idx = row_plan.src_idx
    contrib = np.where(vals[src_idx], row_plan.weights, 0.0)
    return _reduce_bins(contrib, row_plan.starts, row_plan.counts, row_plan.counts.size) > 0


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
    wave_ref: str = "UNKNOWN",
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

    # Spectral frame bookkeeping (external-friendly defaults).
    hdr.setdefault("SPECSYS", str(hdr.get("SPECSYS", "TOPOCENT") or "TOPOCENT"))
    hdr.setdefault("SSYSOBS", str(hdr.get("SPECSYS", "TOPOCENT") or "TOPOCENT"))
    grid = None
    if wave0 is not None and dw is not None:
        try:
            grid = WaveGrid(
                lambda0=float(wave0),
                dlambda=float(dw),
                nlam=int(np.asarray(sci).shape[1]),
                unit=str(unit),
                wave_ref=str(wave_ref),
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
    # Inspection and contract validation are deliberately separate:
    # an invalid SCI/VAR/MASK MEF must NEVER turn into a primary-only image.
    # Preserve relaxed header acceptance for legitimate legacy raw frames.
    with fits.open(
        path, memmap=False, ignore_missing_simple=True, ignore_missing_end=True
    ) as hdul:
        has_sci = "SCI" in hdul
    if has_sci:
        try:
            sci, var, mask, hdr = read_sci_var_mask(path)
        except Exception as exc:
            raise ValueError(
                f"Invalid SCI/VAR/MASK MEF at {path}; refusing "
                "primary-HDU fallback or noise re-estimation"
            ) from exc
        sci64 = np.asarray(sci, dtype=np.float64)
        var64 = None if var is None else np.asarray(var, dtype=np.float64)
        return sci64, var64, mask, hdr

    data, hdr = _open_fits_resilient(path)
    return data, None, None, fits.Header(hdr)


def _raw_stem_from_path(p: Path) -> str:
    """Derive original raw stem from any pipeline product path."""

    stem = p.stem
    # common suffixes used in intermediate products
    for suf in (
        "_clean",
        "_skysub_raw",
        "_skymodel_raw",
        "_rectified",
        "_skysub",
        "_skymodel",
        "_lin",
    ):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem


def _sky_rows_from_cfg(
    cfg: Dict[str, Any],
    rect: np.ndarray,
    rect_var: np.ndarray | None,
    rect_mask: np.ndarray | None,
) -> np.ndarray:
    """Return y-rows used as sky in the *rectified* grid.

    P1-C unifies slit ROI logic via :mod:`scorpio_pipe.sky_geometry`.
    We use the same ROI policy here for residual cleanup.

    Fallback: top/bottom 15% bands.
    """

    ny = int(rect.shape[0])

    try:
        from scorpio_pipe.sky_geometry import compute_sky_geometry, roi_from_cfg

        roi = roi_from_cfg(cfg)
        sky = cfg.get("sky") if isinstance(cfg.get("sky"), dict) else {}
        gcfg = sky.get("geometry") if isinstance(sky.get("geometry"), dict) else {}

        fatal_bits = int(NO_COVERAGE | BADPIX | COSMIC | SATURATED | USER | REJECTED)
        geom = compute_sky_geometry(
            rect,
            rect_var,
            rect_mask,
            roi=roi,
            roi_policy=str(gcfg.get("roi_policy", "prefer_user")),
            fatal_bits=fatal_bits,
            edge_margin_px=int(gcfg.get("edge_margin_px", 16) or 16),
            profile_x_percentile=float(gcfg.get("profile_x_percentile", 50.0) or 50.0),
            thresh_sigma=float(gcfg.get("thresh_sigma", 3.0) or 3.0),
            dilation_px=int(gcfg.get("dilation_px", 3) or 3),
            min_obj_width_px=int(gcfg.get("min_obj_width_px", 6) or 6),
            min_sky_width_px=int(gcfg.get("min_sky_width_px", 12) or 12),
        )
        rows = np.where(np.asarray(geom.mask_sky_y, dtype=bool))[0]
        if rows.size >= 8:
            return rows.astype(int)
    except Exception:
        pass

    band = max(8, int(0.15 * ny))
    return np.asarray(list(range(0, band)) + list(range(max(0, ny - band), ny)), dtype=int)


def _robust_sigma_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float('nan')
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return float(1.4826 * mad)


def _weighted_ls_beta(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least squares solve with basic guards."""
    # Solve (X^T W X) b = X^T W y.
    W = w[:, None]
    A = X.T @ (W * X)
    b = X.T @ (W[:, 0] * y)
    try:
        return np.linalg.solve(A, b)
    except Exception:
        # Fall back to lstsq for near-singular cases.
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _robust_fit_y_linear(
    y_vals: np.ndarray,
    yhat: np.ndarray,
    w_base: np.ndarray,
    *,
    y_order: int,
    n_iter: int,
    huber_k: float,
    clip_sigma: float,
) -> tuple[float, float, int, float]:
    """Robust (Huber-like) WLS fit of y-order 0/1: a0 + a1*yhat."""

    y = np.asarray(y_vals, dtype=float)
    w0 = np.asarray(w_base, dtype=float)
    ok = np.isfinite(y) & np.isfinite(w0) & (w0 > 0)
    if y_order >= 1:
        ok = ok & np.isfinite(yhat)
    if ok.sum() < (2 if y_order >= 1 else 1) + 2:
        return float('nan'), float('nan'), int(ok.sum()), float('nan')

    yy = y[ok]
    ww = w0[ok]
    yh = yhat[ok] if y_order >= 1 else None

    if y_order >= 1:
        X = np.column_stack([np.ones_like(yy), yh])
    else:
        X = np.ones((yy.size, 1), dtype=float)

    w = ww.copy()
    beta = _weighted_ls_beta(X, yy, w)

    sigma = float('nan')
    for _ in range(max(1, int(n_iter))):
        pred = X @ beta
        r = yy - pred
        sigma = _robust_sigma_mad(r)
        if not (sigma > 0 and np.isfinite(sigma)):
            sigma = float(np.nanstd(r)) if np.isfinite(np.nanstd(r)) else 1.0
            sigma = max(sigma, 1e-6)
        # Huber weights
        t = np.abs(r) / (huber_k * sigma + 1e-12)
        wh = np.ones_like(t)
        m = t > 1
        wh[m] = 1.0 / t[m]
        # Hard clip
        if clip_sigma and clip_sigma > 0:
            wh[np.abs(r) > (clip_sigma * sigma)] = 0.0
        w = ww * wh
        if np.count_nonzero(w > 0) < X.shape[1] + 2:
            break
        beta = _weighted_ls_beta(X, yy, w)

    a0 = float(beta[0])
    a1 = float(beta[1]) if (y_order >= 1 and beta.size > 1) else 0.0
    n_used = int(np.count_nonzero(w > 0))
    return a0, a1, n_used, float(sigma)


def _smooth_coeff_spline(x: np.ndarray, y: np.ndarray, w: np.ndarray, *, knot_step: float) -> np.ndarray:
    """Coarse B-spline smoothing via LSQUnivariateSpline.

    If SciPy spline fit fails, falls back to a moving median.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if ok.sum() < 8:
        return y.copy()

    xo = x[ok]
    yo = y[ok]
    wo = w[ok]

    # Build interior knots (exclude boundaries).
    if not (knot_step > 0 and np.isfinite(knot_step)):
        knot_step = float(np.nanmedian(np.diff(xo))) * 25
    lo = float(xo.min())
    hi = float(xo.max())
    n_int = int(max(0, math.floor((hi - lo) / knot_step) - 1))
    if n_int < 1:
        return y.copy()

    knots = [lo + knot_step * (i + 1) for i in range(n_int)]
    knots = [t for t in knots if (t > lo and t < hi)]

    try:
        from scipy.interpolate import LSQUnivariateSpline

        spl = LSQUnivariateSpline(xo, yo, knots, w=wo, k=3)
        return spl(x)
    except Exception:
        # Fallback: moving median with window ~ knot_step.
        dx = float(np.nanmedian(np.diff(xo))) if xo.size > 2 else 1.0
        win = int(max(5, round(knot_step / max(dx, 1e-6))))
        if win % 2 == 0:
            win += 1
        out = y.copy()
        half = win // 2
        for i in range(y.size):
            a = max(0, i - half)
            b = min(y.size, i + half + 1)
            vv = y[a:b]
            vv = vv[np.isfinite(vv)]
            out[i] = float(np.median(vv)) if vv.size else float('nan')
        return out


def _post_rectified_sky_cleanup(
    rect: np.ndarray,
    rect_var: np.ndarray,
    rect_mask: np.ndarray,
    cfg: Dict[str, Any],
    mode: str,
    *,
    lam_centers_A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Residual sky cleanup on rectified grid.

    This is a *post*-Kelson residual model intended to remove low-frequency
    residuals left after RAW sky subtraction and rectification.

    The model is fit **only** on sky pixels (ROI-aware) and is smooth along λ
    via coarse B-splines.

    Returns (cleaned_rect, residual_model, diag).
    """

    mode_n = str(mode or "auto").strip().lower()
    requested = "auto"
    if mode_n in {"off", "0", "false", "no"}:
        requested = "off"
    elif mode_n in {"on", "1", "true", "yes"}:
        requested = "on"

    if requested == "off":
        return rect, np.zeros_like(rect, dtype=np.float32), {
            "cleanup_requested": "off",
            "cleanup_executed": False,
            "cleanup_applied": False,
            "cleanup_decision": "off",
            "decision_reason": "off",
            "model_params": {},
            "roi": {"roi_source": "none", "roi_valid": False},
            "metrics_before": {"sky_residual_metric": None, "object_risk_metric": None},
            "metrics_after": {"sky_residual_metric": None, "object_risk_metric": None},
        }

    lcfg = cfg.get('linearize') if isinstance(cfg.get('linearize'), dict) else {}
    ccfg = lcfg.get('cleanup') if isinstance(lcfg.get('cleanup'), dict) else {}

    knot_step_A = float(ccfg.get('knot_step_A', ccfg.get('knot_step', 25.0)) or 25.0)
    y_order = int(ccfg.get('y_order', 0) or 0)
    n_iter = int(ccfg.get('n_iter', 3) or 3)
    robust_clip_sigma = float(ccfg.get('robust_clip_sigma', 4.0) or 4.0)
    huber_k = float(ccfg.get('huber_k', 1.5) or 1.5)
    auto_delta_min = float(ccfg.get('auto_delta_min', 0.10) or 0.10)
    auto_delta_min_frac = float(ccfg.get('auto_delta_min_frac', 0.0) or 0.0)
    object_tol_sigma = float(ccfg.get('object_tol_sigma', 0.25) or 0.25)
    quiet_sky_percentile = float(ccfg.get('quiet_sky_percentile', 40.0) or 40.0)

    ny, nx = rect.shape

    # ROI-aware sky/object masks.
    roi_used: Dict[str, Any] = {"roi_source": "fallback", "roi_valid": False}
    mask_sky_y = None
    mask_obj_y = None
    try:
        from scorpio_pipe.sky_geometry import compute_sky_geometry, roi_from_cfg

        roi = roi_from_cfg(cfg)
        sky = cfg.get('sky') if isinstance(cfg.get('sky'), dict) else {}
        gcfg = sky.get('geometry') if isinstance(sky.get('geometry'), dict) else {}

        fatal_bits = int(NO_COVERAGE | BADPIX | COSMIC | SATURATED | USER | REJECTED)
        geom = compute_sky_geometry(
            rect,
            rect_var,
            rect_mask,
            roi=roi,
            roi_policy=str(gcfg.get('roi_policy', 'prefer_user')),
            fatal_bits=fatal_bits,
            edge_margin_px=int(gcfg.get('edge_margin_px', 16) or 16),
            profile_x_percentile=float(gcfg.get('profile_x_percentile', 50.0) or 50.0),
            thresh_sigma=float(gcfg.get('thresh_sigma', 3.0) or 3.0),
            dilation_px=int(gcfg.get('dilation_px', 3) or 3),
            min_obj_width_px=int(gcfg.get('min_obj_width_px', 6) or 6),
            min_sky_width_px=int(gcfg.get('min_sky_width_px', 12) or 12),
        )
        mask_sky_y = np.asarray(geom.mask_sky_y, dtype=bool)
        mask_obj_y = np.asarray(geom.mask_obj_y, dtype=bool)
        roi_used = dict(geom.roi_used)
        # Keep full metrics for the report (including contamination and flags).
        roi_used["metrics"] = dict(geom.metrics)
    except Exception:
        # fallback bands: top/bottom 15%, object = middle 25%
        band = max(8, int(0.15 * ny))
        mask_sky_y = np.zeros(ny, dtype=bool)
        mask_sky_y[:band] = True
        mask_sky_y[max(0, ny - band):] = True
        mid = ny // 2
        obj_half = max(3, int(0.125 * ny))
        mask_obj_y = np.zeros(ny, dtype=bool)
        mask_obj_y[max(0, mid - obj_half): min(ny, mid + obj_half + 1)] = True

    sky_rows = np.where(mask_sky_y)[0]
    obj_rows = np.where(mask_obj_y)[0]

    fatal = NO_COVERAGE | BADPIX | COSMIC | SATURATED | USER | REJECTED
    good = (rect_mask & fatal) == 0
    okv = np.isfinite(rect_var) & (rect_var > 0)

    model_params = {
        "knot_step_A": float(knot_step_A),
        "y_order": int(y_order),
        "n_iter": int(n_iter),
        "robust_clip_sigma": float(robust_clip_sigma),
        "huber_k": float(huber_k),
        "auto_delta_min": float(auto_delta_min),
        "auto_delta_min_frac": float(auto_delta_min_frac),
        "object_tol_sigma": float(object_tol_sigma),
        "quiet_sky_percentile": float(quiet_sky_percentile),
    }

    # --- metrics BEFORE ---
    sky_norm = rect[sky_rows, :]
    sky_w = np.sqrt(rect_var[sky_rows, :])
    m_sky = good[sky_rows, :] & okv[sky_rows, :] & np.isfinite(sky_norm) & np.isfinite(sky_w)
    sky_metric_before = _robust_sigma_mad((sky_norm[m_sky] / sky_w[m_sky]) if m_sky.any() else np.array([], dtype=float))

    # quiet λ region selection based on sky median amplitude
    quiet_cols = np.ones(nx, dtype=bool)
    try:
        med_sky_col = np.nanmedian(np.where(m_sky, sky_norm, np.nan), axis=0)
        a = np.abs(med_sky_col)
        thr = float(np.nanpercentile(a[np.isfinite(a)], quiet_sky_percentile)) if np.isfinite(a).sum() else float('nan')
        if np.isfinite(thr):
            quiet_cols = a <= thr
        if quiet_cols.sum() < max(8, int(0.05 * nx)):
            quiet_cols = np.ones(nx, dtype=bool)
    except Exception:
        quiet_cols = np.ones(nx, dtype=bool)

    obj_metric_before = float('nan')
    obj_noise_scale = float('nan')
    try:
        obj = rect[obj_rows, :]
        obj_w = np.sqrt(rect_var[obj_rows, :])
        m_obj = good[obj_rows, :] & okv[obj_rows, :] & np.isfinite(obj) & np.isfinite(obj_w)
        m_obj = m_obj & quiet_cols[None, :]
        if m_obj.any():
            obj_metric_before = float(np.nanmedian(obj[m_obj]))
            # noise scale in SCI units
            obj_noise_scale = float(np.nanmedian(obj_w[m_obj]))
    except Exception:
        pass

    # --- fit residual model: per-λ robust WLS across y ---
    y = np.arange(ny, dtype=float)
    yhat = (y - (ny - 1) / 2.0) / (max(ny - 1, 1) / 2.0)

    a0 = np.full(nx, np.nan, dtype=float)
    a1 = np.full(nx, 0.0, dtype=float)
    n_used = np.zeros(nx, dtype=int)

    # Precompute weights: 1/VAR on sky pixels.
    for x in range(nx):
        sky_pix = rect[sky_rows, x]
        sky_var = rect_var[sky_rows, x]
        sky_good = good[sky_rows, x] & np.isfinite(sky_pix) & np.isfinite(sky_var) & (sky_var > 0)
        if sky_good.sum() < (3 if y_order >= 1 else 2):
            continue
        w_base = 1.0 / sky_var[sky_good]
        a0x, a1x, nx_used, _ = _robust_fit_y_linear(
            sky_pix[sky_good],
            yhat[sky_rows][sky_good],
            w_base,
            y_order=y_order,
            n_iter=n_iter,
            huber_k=huber_k,
            clip_sigma=robust_clip_sigma,
        )
        a0[x] = a0x
        a1[x] = a1x
        n_used[x] = nx_used

    # If the sky is essentially unavailable, do not attempt to smooth/apply.
    n_finite = int(np.isfinite(a0).sum())
    if n_finite < max(8, int(0.05 * nx)):
        # For "on" we still report the failure, but do not apply a junk model.
        decision = "auto_rejected" if requested == "auto" else "on"
        return rect, np.zeros_like(rect, dtype=np.float32), {
            "cleanup_requested": requested,
            "cleanup_executed": False,
            "cleanup_applied": False,
            "cleanup_decision": decision,
            "decision_reason": "insufficient_sky_pixels",
            "model_params": model_params,
            "roi": roi_used,
            "metrics_before": {
                "sky_residual_metric": float(sky_metric_before)
                if np.isfinite(sky_metric_before)
                else None,
                "object_risk_metric": float(obj_metric_before)
                if np.isfinite(obj_metric_before)
                else None,
            },
            # output is unchanged
            "metrics_after": {
                "sky_residual_metric": float(sky_metric_before)
                if np.isfinite(sky_metric_before)
                else None,
                "object_risk_metric": float(obj_metric_before)
                if np.isfinite(obj_metric_before)
                else None,
            },
        }

    # Smooth along λ (in Angstrom).
    w_smooth = np.sqrt(np.maximum(n_used.astype(float), 0.0))
    a0_s = _smooth_coeff_spline(lam_centers_A, a0, w_smooth, knot_step=knot_step_A)
    a1_s = _smooth_coeff_spline(lam_centers_A, a1, w_smooth, knot_step=knot_step_A) if y_order >= 1 else np.zeros_like(a0_s)

    model = (a0_s[None, :] + a1_s[None, :] * yhat[:, None]).astype(np.float32)
    cleaned = (rect - model).astype(np.float32)

    # --- metrics AFTER ---
    sky_norm2 = cleaned[sky_rows, :]
    m_sky2 = good[sky_rows, :] & okv[sky_rows, :] & np.isfinite(sky_norm2) & np.isfinite(sky_w)
    sky_metric_after = _robust_sigma_mad((sky_norm2[m_sky2] / sky_w[m_sky2]) if m_sky2.any() else np.array([], dtype=float))

    obj_metric_after = float('nan')
    try:
        obj2 = cleaned[obj_rows, :]
        m_obj2 = good[obj_rows, :] & okv[obj_rows, :] & np.isfinite(obj2) & np.isfinite(obj_w)
        m_obj2 = m_obj2 & quiet_cols[None, :]
        if m_obj2.any():
            obj_metric_after = float(np.nanmedian(obj2[m_obj2]))
    except Exception:
        pass

    # --- AUTO gating decision ---
    applied = True
    decision_reason = "on"
    decision = "on" if requested == "on" else "auto_applied"
    improve_abs = float("nan")
    improve_frac = float("nan")
    obj_delta = float("nan")
    obj_tol = float("nan")

    if requested == "auto":
        applied = False
        decision = "auto_rejected"
        decision_reason = "auto_reject"

        ok_improve = False
        if np.isfinite(sky_metric_before) and np.isfinite(sky_metric_after):
            improve_abs = float(sky_metric_before - sky_metric_after)
            improve_frac = float(improve_abs / max(float(sky_metric_before), 1e-12))
            ok_improve = (improve_abs >= auto_delta_min) or (
                auto_delta_min_frac > 0 and improve_frac >= auto_delta_min_frac
            )

        ok_object = True
        if np.isfinite(obj_metric_before) and np.isfinite(obj_metric_after) and np.isfinite(obj_noise_scale):
            obj_delta = float(obj_metric_after - obj_metric_before)
            obj_tol = float(object_tol_sigma * max(float(obj_noise_scale), 1e-12))
            ok_object = obj_delta >= -obj_tol

        if ok_improve and ok_object:
            applied = True
            decision = "auto_applied"
            decision_reason = "auto_accept"

    if not applied:
        # Do NOT modify science, but keep the model for diagnostics.
        return rect, model, {
            "cleanup_requested": requested,
            "cleanup_executed": True,
            "cleanup_applied": False,
            "cleanup_decision": decision,
            "decision_reason": decision_reason,
            "model_params": model_params,
            "roi": roi_used,
            "improvement": {
                "sky_improve_abs": float(improve_abs) if np.isfinite(improve_abs) else None,
                "sky_improve_frac": float(improve_frac) if np.isfinite(improve_frac) else None,
                "obj_delta": float(obj_delta) if np.isfinite(obj_delta) else None,
                "obj_tol": float(obj_tol) if np.isfinite(obj_tol) else None,
            },
            "metrics_before": {
                "sky_residual_metric": float(sky_metric_before)
                if np.isfinite(sky_metric_before)
                else None,
                "object_risk_metric": float(obj_metric_before)
                if np.isfinite(obj_metric_before)
                else None,
            },
            # output is unchanged; keep candidate metrics separately
            "metrics_after": {
                "sky_residual_metric": float(sky_metric_before)
                if np.isfinite(sky_metric_before)
                else None,
                "object_risk_metric": float(obj_metric_before)
                if np.isfinite(obj_metric_before)
                else None,
            },
            "candidate_metrics_after": {
                "sky_residual_metric": float(sky_metric_after)
                if np.isfinite(sky_metric_after)
                else None,
                "object_risk_metric": float(obj_metric_after)
                if np.isfinite(obj_metric_after)
                else None,
            },
        }

    return cleaned, model, {
        "cleanup_requested": requested,
        "cleanup_executed": True,
        "cleanup_applied": True,
        "cleanup_decision": decision,
        "decision_reason": decision_reason,
        "model_params": model_params,
        "roi": roi_used,
        "improvement": {
            "sky_improve_abs": float(improve_abs) if np.isfinite(improve_abs) else None,
            "sky_improve_frac": float(improve_frac) if np.isfinite(improve_frac) else None,
            "obj_delta": float(obj_delta) if np.isfinite(obj_delta) else None,
            "obj_tol": float(obj_tol) if np.isfinite(obj_tol) else None,
        },
        "metrics_before": {
            "sky_residual_metric": float(sky_metric_before)
            if np.isfinite(sky_metric_before)
            else None,
            "object_risk_metric": float(obj_metric_before)
            if np.isfinite(obj_metric_before)
            else None,
        },
        "metrics_after": {
            "sky_residual_metric": float(sky_metric_after)
            if np.isfinite(sky_metric_after)
            else None,
            "object_risk_metric": float(obj_metric_after)
            if np.isfinite(obj_metric_after)
            else None,
        },
        "candidate_metrics_after": {
            "sky_residual_metric": float(sky_metric_after)
            if np.isfinite(sky_metric_after)
            else None,
            "object_risk_metric": float(obj_metric_after)
            if np.isfinite(obj_metric_after)
            else None,
        },
    }


### Noise / variance helpers live in scorpio_pipe.noise_model


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
    var_in: np.ndarray,
    lam_centers: np.ndarray,
    edges_out: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Variance propagation matched to :func:`_rebin_row_cumulative`.

    The rebinning operator treats input samples ``x_j`` as *piecewise-constant
    flux density* over the corresponding wavelength interval. The output is
    the *bin-averaged* value on ``edges_out``:

        y_k = (1/Δλ_k) * Σ_j (overlap_{k,j} * x_j)

    assuming independent input pixels:

        Var(y_k) = Σ_j (a_{k,j}^2 * Var(x_j)),  a_{k,j} = overlap_{k,j}/Δλ_k.

    We implement an efficient two-pointer sweep over monotonically increasing
    input/output edges (O(N+M)). Invalid input pixels (valid_mask False or
    non-finite / negative variance) contribute zero.
    """

    v = np.asarray(var_in, dtype=float)
    lam = np.asarray(lam_centers, dtype=float)
    edges_out = np.asarray(edges_out, dtype=float)

    if v.size == 0 or edges_out.size < 2:
        return np.zeros(max(edges_out.size - 1, 0), dtype=float)

    if valid_mask is None:
        valid = np.ones_like(v, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != v.shape:
            raise ValueError('valid_mask must have same shape as var_in')

    lam_m, rev = _monotonicize_lambda_centers(lam)
    if rev:
        v = v[::-1]
        valid = valid[::-1]

    edges_in = _lambda_edges(lam_m)
    widths = np.diff(edges_in)

    nb = int(edges_out.size - 1)
    out = np.zeros(nb, dtype=float)

    # Sweep indices
    i = 0
    n = v.size
    for j in range(nb):
        lo = float(edges_out[j])
        hi = float(edges_out[j + 1])
        # advance i to first pixel that can overlap (edges_in[i+1] > lo)
        while i < n and float(edges_in[i + 1]) <= lo:
            i += 1
        ii = i
        bin_w = hi - lo
        if not np.isfinite(bin_w) or bin_w <= 0:
            continue

        while ii < n and float(edges_in[ii]) < hi:
            if valid[ii]:
                vi = v[ii]
                if np.isfinite(vi) and vi >= 0:
                    w_i = float(widths[ii]) if ii < widths.size else 0.0
                    if w_i > 0 and np.isfinite(w_i):
                        a = float(edges_in[ii])
                        b = float(edges_in[ii + 1])
                        overlap = min(hi, b) - max(lo, a)
                        if overlap > 0:
                            w = overlap / bin_w
                            out[j] += (w * w) * vi
            ii += 1

    return out.astype(float)


def _rebin_row_cumulative(
    values: np.ndarray,
    lam_centers: np.ndarray,
    edges_out: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Flux-conserving rebin of a 1D row onto a new wavelength grid.

    Edge-based cumulative integral approach:
      - input pixel intervals are estimated from wavelength centers,
      - flux density is assumed constant within each interval,
      - the cumulative integral is interpolated to output bin edges.

    Coverage definition
    -------------------
    Coverage is the *fraction of output bin wavelength span* that is supported by
    **valid** input pixels. This is important for correct mask semantics when
    SCI/VAR contain NaN/Inf or when mask marks pixels unusable.

    Parameters
    ----------
    values
        Per-pixel *integrated* signal (e.g. ADU per detector pixel).
    lam_centers
        Wavelength centers per detector pixel (must be same length as values).
    edges_out
        Output bin edges (monotonic increasing).
    valid_mask
        Boolean mask of input pixels that are allowed to contribute. Invalid
        pixels contribute zero and reduce coverage.

    Returns
    -------
    out
        Mean value per output bin (i.e. value density per unit of the
        coordinate encoded by ``lam_centers``). For a constant input background,
        this stays approximately constant after resampling.
    cov
        Coverage fraction per output bin in [0, 1].
    """

    v = np.asarray(values, dtype=float)
    lam = np.asarray(lam_centers, dtype=float)
    edges_out = np.asarray(edges_out, dtype=float)

    if v.size == 0 or edges_out.size < 2:
        n = max(edges_out.size - 1, 0)
        return np.zeros(n, dtype=float), np.zeros(n, dtype=float)

    if valid_mask is None:
        valid = np.ones_like(v, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != v.shape:
            raise ValueError('valid_mask must have same shape as values')

    # Enforce strictly increasing lambda (and reverse values accordingly).
    lam_m, rev = _monotonicize_lambda_centers(lam)
    if rev:
        v = v[::-1]
        valid = valid[::-1]

    edges_in = _lambda_edges(lam_m)
    widths = np.diff(edges_in)

    # Treat ``v`` as a piecewise-constant *density* on input intervals.
    # Integrate it over λ to resample, then divide by output bin width to get
    # the mean density in each output bin.
    v_use = np.where(valid & np.isfinite(v), v, 0.0)
    c = np.zeros(edges_in.size, dtype=float)
    c[1:] = np.cumsum(v_use * widths)
    c_out = np.interp(edges_out, edges_in, c, left=c[0], right=c[-1])
    integ = np.diff(c_out)

    # Coverage length uses the same integral machinery.
    w_use = np.where(valid, widths, 0.0)
    cc = np.zeros(edges_in.size, dtype=float)
    cc[1:] = np.cumsum(w_use)
    cc_out = np.interp(edges_out, edges_in, cc, left=cc[0], right=cc[-1])
    cov_len = np.diff(cc_out)
    bin_w = np.maximum(np.diff(edges_out), 1e-12)
    out = integ / bin_w
    cov = cov_len / bin_w
    cov = np.clip(cov, 0.0, 1.0)

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




def _process_linearize_frame(task: Dict[str, Any]) -> Dict[str, Any]:
    p = Path(str(task["path"]))
    lam_map = np.asarray(task["lam_map"], dtype=np.float64)
    linearize_grid_plan: _LinearizeGridPlan = task["linearize_grid_plan"]
    ny = int(task["ny"])
    nlam = int(task["nlam"])
    y0 = int(task["y0"])
    y1 = int(task["y1"])
    mask_dir = Path(str(task["mask_dir"]))
    lcfg = dict(task.get("lcfg") or {})
    cfg = dict(task.get("cfg") or {})
    gain_override_f = task.get("gain_override_f")
    rdnoise_override_f = task.get("rdnoise_override_f")
    scale_to_bin = bool(task.get("scale_to_bin", False))
    normalize_exptime = bool(task.get("normalize_exptime", False))
    wave0 = float(task["wave0"])
    dw = float(task["dw"])
    unit = str(task.get("unit", "Angstrom"))

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
            cfg=cfg,
            gain_override=gain_override_f,
            rdnoise_override=rdnoise_override_f,
            require_gain=True,
        )
        var = np.asarray(var_est, dtype=np.float64)
        var_source = "estimated"

    data_e, var_e, hdr_e, prov, npar_e = ensure_electron_units(
        data,
        var,
        hdr,
        cfg=cfg,
        gain_override=gain_override_f,
        rdnoise_override=rdnoise_override_f,
        require_gain=True,
    )
    data = np.asarray(data_e, dtype=np.float64)
    var = np.asarray(var_e, dtype=np.float64)
    hdr = hdr_e

    bad_var = ~np.isfinite(var) | (var < 0)
    if np.any(bad_var):
        var = np.where(bad_var, 0.0, var)
        if mask is None:
            mask = np.zeros_like(var, dtype=np.uint16)
        mask[bad_var] |= BADPIX

    bad_sci = ~np.isfinite(data)
    if np.any(bad_sci):
        data = np.where(bad_sci, 0.0, data)
        var = np.where(bad_sci, 0.0, var)
        if mask is None:
            mask = np.zeros_like(var, dtype=np.uint16)
        mask[bad_sci] |= BADPIX

    noise_meta = {
        "path": str(p),
        "var_source": var_source,
        "gain_e_per_adu": float(npar.gain_e_per_adu),
        "rdnoise_e": float(npar.rdnoise_e),
        "noise_source": str(npar.source),
    }

    # Only the upstream detector DQ plane can establish saturation at this
    # point: sky subtraction has changed SCI and invalidated ADC thresholds.
    if str(hdr.get("SATQ") or "").strip().upper() != "RAW_EVAL":
        hdr["SATQ"] = (
            "PROP" if mask is not None and np.any((mask & SATURATED) != 0) else "UNKNOWN",
            "Saturation evidence from upstream, not sky-sub SCI",
        )

    base_for_mask = _raw_stem_from_path(Path(p))
    mp = mask_dir / f"{base_for_mask}_mask.fits"
    if mp.exists():
        try:
            m, _ = _open_fits_resilient(mp)
            mm = np.asarray(m)
            try:
                mx = int(np.nanmax(mm)) if np.isfinite(mm).any() else 0
            except Exception:
                mx = 0
            if mx <= 1:
                cm = ((mm.astype(np.uint16) > 0).astype(np.uint16) * COSMIC).astype(np.uint16)
            else:
                cm = mm.astype(np.uint16)
            cm = cm[y0:y1, :]
            if mask is None:
                mask = cm
            else:
                mask = np.asarray((mask | cm), dtype=np.uint16)
        except Exception as e:
            log.warning("Failed to load cosmic mask %s: %s", mp.name, e)

    rect = np.zeros((ny, nlam), dtype=np.float32)
    rect_var = np.zeros((ny, nlam), dtype=np.float32)
    rect_mask = np.zeros((ny, nlam), dtype=np.uint16)
    rect_cov = np.zeros((ny, nlam), dtype=np.float32)

    for yy in range(ny):
        row_plan = linearize_grid_plan.rows[yy]
        if row_plan is None:
            rect_mask[yy, :] |= NO_COVERAGE
            continue
        valid = np.isfinite(data[yy]) & np.isfinite(var[yy]) & (var[yy] >= 0)
        if mask is not None:
            # NO_COVERAGE and INVALID_WAVELENGTH pixels are not valid data.
            # Their missing flux reduces effective output-bin coverage.
            valid &= ((mask[yy] & (BADPIX | NO_COVERAGE | INVALID_WAVELENGTH | SATURATED)) == 0)

        v_row, cov = _apply_linearize_row_plan(
            data[yy],
            row_plan,
            valid_mask=valid,
        )
        vv_row = _apply_linearize_row_plan_var(
            var[yy],
            row_plan,
            valid_mask=valid,
        )
        rect[yy] = v_row
        rect_var[yy] = np.maximum(vv_row, 0.0)
        rect_cov[yy] = cov
        rect_mask[yy, (cov > 0) & (cov < 0.999)] |= EDGE
        rect_mask[yy, cov <= 0] |= NO_COVERAGE
        if mask is not None:
            # Propagate all other canonical quality bits by overlap, even if
            # they were set upstream by a degraded sky pass-through. Output
            # NO_COVERAGE is derived from *valid* geometric coverage above:
            # forwarding it on any overlap would incorrectly discard partial
            # but otherwise usable bins. EDGE records partial coverage.
            for bit in (
                BADPIX, COSMIC, SATURATED, USER, REJECTED, EDGE,
                OUTSIDE_SLIT, INVALID_WAVELENGTH, SKYMODEL_FAIL,
            ):
                rect_mask[yy, _apply_linearize_row_plan_maskbit(mask[yy], row_plan, bit)] |= bit

    if scale_to_bin:
        rect *= float(dw)
        rect_var *= float(dw) ** 2

    exptime_s = _get_exptime_seconds(hdr)
    if normalize_exptime:
        if not (math.isfinite(exptime_s) and exptime_s > 0):
            log.warning("EXPTIME missing/invalid in %s; using 1.0 s for normalization", p.name)
            exptime_s = 1.0
        rect /= float(exptime_s)
        rect_var /= float(exptime_s) ** 2

    roi_report: Dict[str, Any] = {"roi_source": "none", "roi_valid": False}
    roi_flags: list[dict[str, Any]] = []
    mask_sky_y = None
    mask_obj_y = None
    try:
        from scorpio_pipe.sky_geometry import compute_sky_geometry, roi_from_cfg

        roi = roi_from_cfg(cfg)
        sky_cfg = cfg.get("sky", {}) if isinstance(cfg.get("sky"), dict) else {}
        if not sky_cfg and isinstance(cfg.get("sky_sub"), dict):
            sky_cfg = cfg.get("sky_sub") or {}
        gcfg = sky_cfg.get("geometry") if isinstance(sky_cfg.get("geometry"), dict) else {}

        fatal_bits = int(NO_COVERAGE | BADPIX | COSMIC | SATURATED | USER | REJECTED)
        geom = compute_sky_geometry(
            rect,
            rect_var,
            rect_mask,
            roi=roi,
            roi_policy=str(gcfg.get("roi_policy", "prefer_user")),
            fatal_bits=fatal_bits,
            edge_margin_px=int(gcfg.get("edge_margin_px", 16) or 16),
            profile_x_percentile=float(gcfg.get("profile_x_percentile", 50.0) or 50.0),
            thresh_sigma=float(gcfg.get("thresh_sigma", 3.0) or 3.0),
            dilation_px=int(gcfg.get("dilation_px", 3) or 3),
            min_obj_width_px=int(gcfg.get("min_obj_width_px", 6) or 6),
            min_sky_width_px=int(gcfg.get("min_sky_width_px", 12) or 12),
        )
        mask_sky_y = np.asarray(geom.mask_sky_y, dtype=bool)
        mask_obj_y = np.asarray(geom.mask_obj_y, dtype=bool)
        roi_report = dict(geom.roi_used)
        roi_metrics = dict(geom.metrics)
        roi_report["sky_contamination_metric"] = roi_metrics.get("sky_contamination_metric")
        roi_flags = list(roi_metrics.get("flags") or [])
    except Exception:
        band = max(8, int(0.15 * rect.shape[0]))
        mask_sky_y = np.zeros(rect.shape[0], dtype=bool)
        mask_sky_y[:band] = True
        mask_sky_y[max(0, rect.shape[0] - band):] = True
        mid = rect.shape[0] // 2
        obj_half = max(3, int(0.125 * rect.shape[0]))
        mask_obj_y = np.zeros(rect.shape[0], dtype=bool)
        mask_obj_y[max(0, mid - obj_half): min(rect.shape[0], mid + obj_half + 1)] = True

    ccfg = lcfg.get("cleanup") if isinstance(lcfg.get("cleanup"), dict) else {}
    quiet_sky_percentile = float(ccfg.get("quiet_sky_percentile", 40.0) or 40.0)

    sky_rows = np.where(np.asarray(mask_sky_y, dtype=bool))[0]
    obj_rows = np.where(np.asarray(mask_obj_y, dtype=bool))[0]
    fatal = NO_COVERAGE | BADPIX | COSMIC | SATURATED | USER | REJECTED
    good = (rect_mask & fatal) == 0
    okv = np.isfinite(rect_var) & (rect_var > 0)

    sky_metric_before = float("nan")
    obj_metric_before = float("nan")
    try:
        sky_norm = rect[sky_rows, :]
        sky_w = np.sqrt(rect_var[sky_rows, :])
        m_sky = good[sky_rows, :] & okv[sky_rows, :] & np.isfinite(sky_norm) & np.isfinite(sky_w)
        sky_metric_before = _robust_sigma_mad(
            (sky_norm[m_sky] / sky_w[m_sky]) if m_sky.any() else np.array([], dtype=float)
        )

        quiet_cols = np.ones(rect.shape[1], dtype=bool)
        try:
            med_sky_col = np.nanmedian(np.where(m_sky, sky_norm, np.nan), axis=0)
            a = np.abs(med_sky_col)
            thr = float(
                np.nanpercentile(a[np.isfinite(a)], quiet_sky_percentile)
            ) if np.isfinite(a).sum() else float("nan")
            if np.isfinite(thr):
                quiet_cols = a <= thr
            if quiet_cols.sum() < max(8, int(0.05 * rect.shape[1])):
                quiet_cols = np.ones(rect.shape[1], dtype=bool)
        except Exception:
            quiet_cols = np.ones(rect.shape[1], dtype=bool)

        obj = rect[obj_rows, :]
        obj_w = np.sqrt(rect_var[obj_rows, :])
        m_obj = good[obj_rows, :] & okv[obj_rows, :] & np.isfinite(obj) & np.isfinite(obj_w)
        m_obj = m_obj & quiet_cols[None, :]
        if m_obj.any():
            obj_metric_before = float(np.nanmedian(obj[m_obj]))
    except Exception:
        pass

    metrics_before = {
        "sky_residual_metric": float(sky_metric_before) if np.isfinite(sky_metric_before) else None,
        "object_risk_metric": float(obj_metric_before) if np.isfinite(obj_metric_before) else None,
    }

    sky_cfg = cfg.get("sky", {}) if isinstance(cfg.get("sky"), dict) else {}
    if not sky_cfg and isinstance(cfg.get("sky_sub"), dict):
        sky_cfg = cfg.get("sky_sub") or {}
    cleanup_raw = lcfg.get(
        "post_sky_cleanup", lcfg.get("sky_cleanup", sky_cfg.get("post_cleanup", "auto"))
    )
    post_cleanup = str(cleanup_raw).strip().lower()
    if post_cleanup not in {"off", "auto", "on"}:
        post_cleanup = "auto"

    residual_model = None
    cleanup_diag: Dict[str, Any] = {
        "cleanup_requested": post_cleanup,
        "cleanup_executed": False,
        "cleanup_applied": False,
        "cleanup_decision": "off" if post_cleanup == "off" else "auto_rejected",
        "decision_reason": "off" if post_cleanup == "off" else "not_run",
        "model_params": {},
        "roi": roi_report,
        "metrics_before": metrics_before,
        "metrics_after": metrics_before,
    }
    if post_cleanup in {"on", "auto"}:
        x = np.arange(rect.shape[1], dtype=np.float64)
        lam_cent = wave0 + dw * (x + 0.5)
        lam_cent_A = lam_cent * 10.0 if str(unit).lower() in {"nm", "nanometer", "nanometers"} else lam_cent
        rect2, model2, diag2 = _post_rectified_sky_cleanup(
            rect,
            rect_var,
            rect_mask,
            cfg,
            mode=post_cleanup,
            lam_centers_A=np.asarray(lam_cent_A, dtype=np.float64),
        )
        cleanup_diag = dict(diag2)
        cleanup_diag.setdefault("roi", roi_report)
        rect = rect2
        residual_model = model2

    return {
        "path": str(p),
        "base_for_mask": base_for_mask,
        "hdr": hdr,
        "rect": rect,
        "rect_var": rect_var,
        "rect_mask": rect_mask,
        "rect_cov": rect_cov,
        "mask_sky_y": mask_sky_y,
        "roi_report": roi_report,
        "roi_flags": roi_flags,
        "metrics_before": metrics_before,
        "cleanup_diag": cleanup_diag,
        "residual_model": residual_model,
        "noise_meta": noise_meta,
        "exptime_s": float(exptime_s),
        "sat_level_adu": float(sat_level) if sat_level is not None else None,
        "no_coverage_frac": float(np.mean((rect_mask & NO_COVERAGE) != 0)),
    }

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
      - 10_linearize/lin_preview.fits (MEF: SCI [+VAR,+MASK,+COV])  # quick-look stack for ROI/QC
      - 10_linearize/linearize_done.json
      - per-exposure rectified frames under 10_linearize/per_exp/

    """
    cfg = dict(cfg)
    work_dir = resolve_work_dir(cfg)
    if out_dir is None:
        out_dir = stage_dir(work_dir, "linearize")
    out_dir = Path(out_dir)
    # Canonical + legacy done markers
    done_json = out_dir / "done.json"
    done_json_legacy = out_dir / "linearize_done.json"
    payload: dict[str, Any] = {"stage": "linearize", "status": "fail"}
    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- Compare A/B cache (before overwrite) ---
        compare_stamp: str | None = None
        compare_a: Path | None = None
        stems_for_compare: list[str] = []
        if done_json.exists():
            try:
                compare_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                compare_a = snapshot_stage(
                    stage_key="linearize",
                    stage_dir=out_dir,
                    label="A",
                    patterns=(
                        "done.json",
                        "linearize_done.json",
                        "wave_grid.json",
                        "*_skysub.fits",
                        "*_skymodel.fits",
                        "*_rectified.fits",
                        "*_skysub.png*",
                        "*_skysub_skywin.png*",
                        "*_skymodel.png*",
                    ),
                    stamp=compare_stamp,
                )
            except Exception:
                compare_stamp = None
                compare_a = None

        lcfg = cfg.get("linearize", {}) if isinstance(cfg.get("linearize"), dict) else {}
        wave_ref = str((cfg.get("project", {}) or {}).get("wave_reference", "UNKNOWN") or "UNKNOWN")
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
        # Locate rectification model / lambda_map (prefer formal artifact from wavesolution).
        rect_model_path: Path | None = None
        rect_model_sha256: str | None = None
        lambda_map_sha256: str | None = None
        lam_path: Path | None = None
        rect_model: dict[str, Any] | None = None
        legacy_discovery_used = False

        # P0-B11: if a validated dataset_manifest.json exists, forbid any
        # directory search for calibration artifacts. The manifest must be the
        # single source of truth.
        from scorpio_pipe.dataset.manifest_io import load_dataset_manifest

        # If the manifest exists but is invalid, this raises immediately.
        _, manifest_path = load_dataset_manifest(cfg, work_dir, require=False, require_v3=True)

        def _sha256(p: Path) -> str:
            import hashlib

            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()

        # 1) Explicit override (debug / power users)
        if (
            isinstance(lcfg.get("lambda_map_path"), str)
            and str(lcfg.get("lambda_map_path")).strip()
        ):
            lam_path = Path(str(lcfg.get("lambda_map_path"))).expanduser()
            if not lam_path.is_absolute():
                lam_path = (work_dir / lam_path).resolve()

        # 2) Canonical contract: 08_wavesol/rectification_model.json -> points to lambda_map
        if lam_path is None:
            wsol = wavesol_dir(cfg)
            model_cand = wsol / "rectification_model.json"
            if model_cand.exists():
                rect_model_path = model_cand
                rect_model_sha256 = _sha256(model_cand)
                try:
                    model = json.loads(model_cand.read_text(encoding="utf-8"))
                    rect_model = model
                    lm_rel = (model.get("lambda_map", {}) or {}).get("path", "lambda_map.fits")
                    lm_p = Path(str(lm_rel))
                    if not lm_p.is_absolute():
                        lm_p = (wsol / lm_p).resolve()
                    lam_path = lm_p
                    if not lam_path.exists():
                        raise FileNotFoundError(f"lambda_map not found: {lam_path}")
                    expected = (model.get("lambda_map", {}) or {}).get("sha256", None)
                    if expected:
                        got = _sha256(lam_path)
                        lambda_map_sha256 = got
                        if str(got).lower() != str(expected).lower():
                            raise RuntimeError(
                                "lambda_map sha256 mismatch vs rectification_model.json "
                                f"(expected {expected}, got {got})"
                            )
                except Exception as e:
                    raise RuntimeError(f"Invalid rectification_model.json: {e}") from e

            # Fallback: look for lambda_map.fits in the resolved wavesol directory
            if lam_path is None:
                cand = wsol / "lambda_map.fits"
                if cand.exists():
                    lam_path = cand

            # Strict for new layouts: require rectification_model.json (no guessing).
            legacy_base = work_dir / 'wavesol'
            try:
                is_legacy = legacy_base in [wsol, *wsol.parents]
            except Exception:
                is_legacy = str(wsol).startswith(str(legacy_base))
            if rect_model_path is None and not is_legacy and lam_path is not None:
                raise FileNotFoundError(
                    f"Missing rectification_model.json in {wsol}. Re-run stage 08_wavesol to regenerate formal artifacts."
                )

        # 3) Last resort: legacy discovery (kept ONLY for old workspaces)
        if lam_path is None:
            if manifest_path is not None:
                raise FileNotFoundError(
                    "lambda_map could not be resolved from canonical artifacts, and legacy directory search is disabled "
                    f"because dataset_manifest.json is present: {manifest_path}. "
                    "Re-run the formal wavesolution stage (08_wavesol) to regenerate rectification_model.json/lambda_map, "
                    "or fix the workspace layout."
                )

            # Legacy-only fallback: directory search (explicitly warned + surfaced in QC).
            base = work_dir / "wavesol"
            cand = list(base.rglob("lambda_map.fits")) + list(base.rglob("lambda_map*.fits"))
            lam_path = cand[0] if cand else None
            if lam_path is not None:
                legacy_discovery_used = True
                try:
                    from loguru import logger

                    logger.warning(
                        "Using legacy directory search for lambda_map (no dataset_manifest.json). "
                        "This is deprecated; prefer rectification_model.json in 08_wavesol. "
                        f"Selected: {lam_path}"
                    )
                except Exception:
                    pass

        if lam_path is None or not lam_path.exists():
            raise FileNotFoundError(
                "lambda_map.fits not found (expected under work_dir/08_wavesol/.../)"
            )

        if lambda_map_sha256 is None:
            try:
                lambda_map_sha256 = _sha256(lam_path)
            except Exception:
                lambda_map_sha256 = None

        lam_map, lam_hdr = _open_fits_resilient(lam_path)

        if lam_map.ndim != 2:
            raise ValueError("lambda_map must be a 2D image")

        # Strict lambda_map validation (P1-B)
        from scorpio_pipe.qc.lambda_map import validate_lambda_map

        exp_shape = tuple(int(x) for x in lam_map.shape)
        exp_unit = None
        exp_ref = None
        if rect_model is not None:
            try:
                ish = rect_model.get('input_shape')
                if isinstance(ish, list) and len(ish) == 2:
                    exp_shape = (int(ish[0]), int(ish[1]))
                lm = rect_model.get('lambda_map') if isinstance(rect_model.get('lambda_map'), dict) else {}
                exp_unit = lm.get('unit') if isinstance(lm.get('unit'), str) else None
                exp_ref = lm.get('waveref') if isinstance(lm.get('waveref'), str) and str(lm.get('waveref')).strip() else None
            except Exception:
                pass

        # NOTE: lambda_map is validated *after* the stage input contract checks (sky products),
        # so that contract errors are not masked by synthetic placeholder wavelength maps.

        # Get science frames list (prefer cosmics cleaned)
        frames = cfg.get("frames", {}) if isinstance(cfg.get("frames"), dict) else {}
        obj_frames = frames.get("obj", [])
        if not obj_frames:
            raise ValueError("No object frames configured (frames.obj is empty)")

        # v5.39+ prefers Sky Subtraction outputs (RAW detector geometry):
        #   09_sky/<stem>_skysub_raw.fits
        # Backward compatibility:
        #   - legacy runs can still linearize cosmics-cleaned frames directly.

        # Locate cosmics products for legacy fallback and for additional masks.
        cosm_cfg = cfg.get("cosmics", {}) if isinstance(cfg.get("cosmics"), dict) else {}
        method = str(cosm_cfg.get("method", "stack_mad"))
        kind = "obj"
        cosm_root = stage_dir(work_dir, "cosmics")
        # legacy layouts (pre-stage_dir)
        legacy_cosm_root = work_dir / "cosmics"
        cand_clean = [
            cosm_root / kind / "clean",  # current
            legacy_cosm_root / kind / "clean",
            legacy_cosm_root / method / kind / "clean",
            legacy_cosm_root / method / "clean",
        ]
        cand_mask = [
            cosm_root / kind / "masks_fits",
            legacy_cosm_root / kind / "masks_fits",
            legacy_cosm_root / method / kind / "masks_fits",
            legacy_cosm_root / method / "masks_fits",
            legacy_cosm_root / kind / "mask_fits",  # very old typo
        ]
        clean_dir = next((p for p in cand_clean if p.exists()), cand_clean[0])
        mask_dir = next((p for p in cand_mask if p.exists()), cand_mask[0])

        # Resolve raw input frame paths (config-relative).
        config_dir = Path(str(cfg.get("config_dir", "."))).expanduser().resolve()
        data_dir = Path(str(cfg.get("data_dir", ""))).expanduser()
        if str(data_dir).strip():
            try:
                data_dir = data_dir.resolve()
            except Exception:
                pass

        def _resolve_input(fp: Any) -> Path:
            p = Path(str(fp)).expanduser()
            if p.is_absolute():
                return p
            # Prefer config_dir (used throughout the pipeline), but also allow data_dir.
            cand = (config_dir / p)
            if cand.exists():
                return cand.resolve()
            if str(data_dir).strip():
                return (data_dir / p).expanduser().resolve()
            return cand.resolve()

        # Validate that the wavelength solution applies to these frames (frame_signature).
        from scorpio_pipe.frame_signature import FrameSignature, format_signature_mismatch

        frame_signature_key = f"shape={int(exp_shape[0])}x{int(exp_shape[1])}, bin=?, roi=?, readout=?"
        if rect_model is not None:
            fs = rect_model.get('frame_signature') if isinstance(rect_model, dict) else None
            if isinstance(fs, dict) and fs:
                exp_sig = FrameSignature.from_setup(fs)
                frame_signature_key = exp_sig.describe()
                got_sig = FrameSignature.from_path(_resolve_input(obj_frames[0]))
                if not got_sig.is_compatible_with(exp_sig):
                    raise ValueError(format_signature_mismatch(expected=exp_sig, got=got_sig, path=_resolve_input(obj_frames[0])))
                # Shape must also match lambda_map
                if tuple(got_sig.shape) != tuple(exp_shape):
                    raise ValueError(
                        f"Frame shape {got_sig.shape} does not match lambda_map shape {exp_shape}."
                    )
                frame_signature_key = got_sig.describe()
        elif obj_frames:
            try:
                frame_signature_key = FrameSignature.from_path(_resolve_input(obj_frames[0])).describe()
            except Exception:
                pass


        # Determine whether we are in the new (09_sky -> 10_linearize) layout.
        # Do NOT infer this solely from the stage_dir name (it always starts
        # with NN); instead, require that the run_root actually contains staged
        # directories.
        # Layout detection: legacy workspaces store wavesol products under work_dir/wavesol/
        # and do NOT require a prior sky stage. New workspaces (v5.39+) require 09_sky products.
        is_new_layout = (not is_legacy)
        sky_root = stage_dir(work_dir, "sky")
        sky_refine_root = stage_dir(work_dir, "sky_refine")
        if is_new_layout and not ((sky_refine_root.exists() and any(sky_refine_root.glob("*_skysub_refined_raw.fits"))) or (sky_root.exists() and any(sky_root.glob("*_skysub_raw.fits")))):
            raise FileNotFoundError(
                "Sky outputs not found for v5.39+ layout. "
                f"Run stage {sky_root.name} first (expected {sky_root.name}/*_skysub_raw.fits or sky_refine/*_skysub_refined_raw.fits)."
            )

        # Strict lambda_map validation (P1-B)
        validate_lambda_map(
            lam_path,
            expected_shape=exp_shape,
            expected_unit=exp_unit,
            expected_waveref=exp_ref,
        )

        sci_paths: list[Path] = []
        model_paths: list[Path | None] = []

        for fp in obj_frames:
            raw_p = _resolve_input(fp)
            stem = raw_p.stem
            sky_refined_p = sky_refine_root / f"{stem}_skysub_refined_raw.fits"
            sky_corr_p = sky_refine_root / f"{stem}_skycorr_raw.fits"
            sky_p = sky_root / f"{stem}_skysub_raw.fits"
            sky_model_p = sky_root / f"{stem}_skymodel_raw.fits"

            if sky_refined_p.exists():
                sci_paths.append(sky_refined_p)
                model_paths.append(sky_corr_p if sky_corr_p.exists() else (sky_model_p if sky_model_p.exists() else None))
                continue

            if sky_p.exists():
                sci_paths.append(sky_p)
                model_paths.append(sky_model_p if sky_model_p.exists() else None)
                continue

            if is_new_layout:
                raise FileNotFoundError(
                    f"Missing sky product for {stem}: expected {sky_p.name} in {sky_root}" 
                )

            # Legacy fallback: use cosmics-cleaned frame if present, else raw.
            clean_p = clean_dir / f"{stem}_clean.fits"
            sci_paths.append(clean_p if clean_p.exists() else raw_p)
            model_paths.append(None)


        # Output wavelength grid (common for the whole series)
        unit, waveref, unit_src = _read_lambda_map_meta(lam_hdr, lam_map)
        # For strict science runs we *prefer* explicit metadata, but for
        # synthetic tests and legacy products we allow heuristics.
        # _read_lambda_map_meta returns the *key* used for explicit metadata
        # (e.g. "WAVEUNIT") and "heuristic" when it had to infer.
        unit_meta_inferred = (str(unit_src).strip().lower() == "heuristic")

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

        # If a rectification_model.json is present, it is the single source of truth
        # for the output wavelength grid.
        if rect_model is not None:
            grid = rect_model.get('wavelength_grid') if isinstance(rect_model, dict) else None
            if isinstance(grid, dict) and grid.get('type', 'linear') == 'linear':
                try:
                    m_unit = grid.get('unit')
                    if isinstance(m_unit, str) and m_unit and str(m_unit) != str(unit):
                        raise ValueError(f"rectification_model grid unit {m_unit} != lambda_map unit {unit}")
                    m_wave0 = float(grid['lam_start'])
                    m_dw = float(grid['lam_step'])
                    m_nlam = int(grid['nlam'])
                    m_wmax = float(grid.get('lam_end', m_wave0 + m_dw * m_nlam))
                    if not (math.isfinite(m_dw) and m_dw > 0 and math.isfinite(m_wave0) and math.isfinite(m_wmax) and m_wmax > m_wave0):
                        raise ValueError('invalid wavelength_grid values')
                    # Override config-driven heuristics
                    wmin_cfg = m_wave0
                    wmax_cfg = m_wmax
                    dw = m_dw
                    grid_mode = 'explicit'
                    # Validate y-crop reproducibility if recorded
                    ycrop = rect_model.get('y_crop') if isinstance(rect_model, dict) else None
                    if isinstance(ycrop, dict):
                        mt = int(ycrop.get('top', y_crop_top) or 0)
                        mb = int(ycrop.get('bottom', y_crop_bottom) or 0)
                        if (mt, mb) != (y_crop_top, y_crop_bottom):
                            raise ValueError(f"y-crop mismatch: config ({y_crop_top},{y_crop_bottom}) != model ({mt},{mb})")
                except Exception as e:
                    raise RuntimeError(f"Invalid wavelength_grid in rectification_model.json: {e}") from e

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


        # ------------------------------------------------------------
        # Preserve the nominal (pre-flexure) wavelength grid for provenance/QC.
        wave0_nominal = float(wave0)
        wmax_nominal = float(wmax)

        # Apply delta_lambda (flexure correction) from Sky Subtraction
        # ------------------------------------------------------------
        delta_lambda_A = None
        delta_lambda_applied = False
        flexure_flag = "MISSING"
        flexure_score = None

        # Read sky_done.json if present (new layout) to obtain the median flexure shift.
        sky_done_path = sky_root / "sky_done.json"
        if not sky_done_path.exists():
            sky_done_path = sky_root / "done.json"
        sky_done = None
        if sky_done_path.exists():
            try:
                sky_done = json.loads(sky_done_path.read_text(encoding="utf-8"))
            except Exception:
                sky_done = None

        if isinstance(sky_done, dict):
            flex = sky_done.get("flexure") if isinstance(sky_done.get("flexure"), dict) else None
            if isinstance(flex, dict):
                try:
                    delta_lambda_A = flex.get("delta_A_median")
                    if delta_lambda_A is not None:
                        delta_lambda_A = float(delta_lambda_A)
                except Exception:
                    delta_lambda_A = None
                try:
                    flexure_score = flex.get("flexure_score_median")
                    if flexure_score is not None:
                        flexure_score = float(flexure_score)
                except Exception:
                    flexure_score = None
                try:
                    sigma_delta_A = flex.get("sigma_delta_A_median")
                    sigma_delta_A = (None if sigma_delta_A is None else float(sigma_delta_A))
                except Exception:
                    sigma_delta_A = None

                # Mirror Sky stage warning logic: uncertain if sigma is large or score low.
                kel = (cfg.get("sky", {}).get("kelson_raw") if isinstance(cfg.get("sky"), dict) else {})
                try:
                    delta_uncertain_A = float(kel.get("delta_uncertain_A", 0.4))
                except Exception:
                    delta_uncertain_A = 0.4
                try:
                    delta_score_warn = float(kel.get("delta_score_warn", 0.35))
                except Exception:
                    delta_score_warn = 0.35

                if delta_lambda_A is None or not (math.isfinite(delta_lambda_A)):
                    flexure_flag = "MISSING"
                else:
                    flexure_flag = "OK"
                    if sigma_delta_A is not None and math.isfinite(sigma_delta_A) and sigma_delta_A > delta_uncertain_A:
                        flexure_flag = "UNCERTAIN"
                    if flexure_score is not None and math.isfinite(flexure_score) and flexure_score < delta_score_warn:
                        flexure_flag = "UNCERTAIN"

        # Policy: apply only if flexure_flag == OK (default) unless user overrides.
        delta_policy = str(lcfg.get("delta_lambda_policy", "apply_if_ok")).strip().lower()
        if delta_policy in {"off", "false", "no", "0"}:
            delta_policy = "off"
        elif delta_policy in {"always", "force"}:
            delta_policy = "always"
        else:
            delta_policy = "apply_if_ok"

        if delta_policy != "off" and delta_lambda_A is not None and unit != "pix":
            try:
                if str(unit).lower() in {"nm", "nanometer", "nanometers"}:
                    delta_u = delta_lambda_A * 0.1
                else:
                    delta_u = delta_lambda_A

                if delta_policy == "always" or flexure_flag == "OK":
                    wave0 = float(wave0) + float(delta_u)
                    wmax = float(wmax) + float(delta_u)
                    # IMPORTANT: the lambda-map is the mapping pixel->wavelength.
                    # If we adjust the output grid due to flexure, we must shift
                    # the lambda-map as well; otherwise binning is inconsistent and
                    # features can move in the *wrong* direction.
                    try:
                        lam_map = (np.asarray(lam_map, dtype=np.float64) + float(delta_u)).astype(np.float32)
                    except Exception:
                        # If this fails, we still keep the shift for provenance.
                        pass
                    delta_lambda_applied = True
            except Exception:
                delta_lambda_applied = False


        wave_edges = wave0 + dw * np.arange(nlam + 1, dtype=np.float64)

        # Output science units. The rebinning primitive returns a *density*
        # (ADU per coordinate unit). If requested, we convert it back to
        # integrated ADU per output bin.
        scale_to_bin = bunit_mode == "adu_bin" and unit != "pix"
        if scale_to_bin and not (math.isfinite(dw) and dw > 0):
            raise ValueError("dw must be positive for bunit_mode=adu_bin")

        if bunit_mode == "adu_per_unit" and unit != "pix":
            bunit = f"ADU/{unit}"
        elif bunit_mode == "adu_per_unit" and unit == "pix":
            bunit = "ADU/pix"
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

        lambda_map_cache_key = str(lambda_map_sha256 or lam_path.resolve())
        linearize_grid_plan, linearize_grid_cache_hit = _get_linearize_grid_plan(
            lam_map,
            edges_out=wave_edges,
            frame_signature_key=frame_signature_key,
            lambda_map_key=lambda_map_cache_key,
            wave0=float(wave0),
            dw=float(dw),
            nlam=int(nlam),
        )

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

        per_exposure: list[dict[str, Any]] = []
        out_skysub_paths: list[str] = []
        out_skymodel_paths: list[str] = []
        no_coverage_fracs: list[float] = []

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

        perf_cfg = lcfg.get("perf") if isinstance(lcfg.get("perf"), dict) else {}
        frame_parallel_backend = str(perf_cfg.get("backend", "thread") or "thread").strip().lower()
        frame_parallel_n_jobs = perf_cfg.get("n_jobs")
        frame_tasks: list[dict[str, Any]] = [
            {
                "path": str(p),
                "lam_map": lam_map,
                "linearize_grid_plan": linearize_grid_plan,
                "ny": int(ny),
                "nlam": int(nlam),
                "y0": int(y0),
                "y1": int(y1),
                "mask_dir": str(mask_dir),
                "lcfg": lcfg,
                "cfg": cfg,
                "gain_override_f": gain_override_f,
                "rdnoise_override_f": rdnoise_override_f,
                "scale_to_bin": bool(scale_to_bin),
                "normalize_exptime": bool(normalize_exptime),
                "wave0": float(wave0),
                "dw": float(dw),
                "unit": str(unit),
            }
            for p in sci_paths
        ]

        for i, frame_res in enumerate(
            parallel_map(
                frame_tasks,
                _process_linearize_frame,
                n_jobs=frame_parallel_n_jobs,
                backend=frame_parallel_backend,
                chunksize=1,
            ),
            1,
        ):
            if cancel_token is not None and getattr(cancel_token, "cancelled", False):
                raise RuntimeError("Cancelled")

            p = Path(str(frame_res["path"]))
            base_for_mask = str(frame_res["base_for_mask"])
            stem = base_for_mask
            log.info("Linearize %d/%d: %s", i, len(sci_paths), p.name)

            hdr = frame_res["hdr"]
            rect = np.asarray(frame_res["rect"], dtype=np.float32)
            rect_var = np.asarray(frame_res["rect_var"], dtype=np.float32)
            rect_mask = np.asarray(frame_res["rect_mask"], dtype=np.uint16)
            rect_cov = np.asarray(frame_res["rect_cov"], dtype=np.float32)
            mask_sky_y = frame_res.get("mask_sky_y")
            roi_report = dict(frame_res.get("roi_report") or {"roi_source": "none", "roi_valid": False})
            roi_flags = list(frame_res.get("roi_flags") or [])
            metrics_before = dict(frame_res.get("metrics_before") or {})
            cleanup_diag = dict(frame_res.get("cleanup_diag") or {})
            residual_model = frame_res.get("residual_model")

            if isinstance(frame_res.get("noise_meta"), dict):
                noise_meta.append(dict(frame_res["noise_meta"]))

            exptime_s = float(frame_res.get("exptime_s") or 0.0)
            exp_times_s.append(float(exptime_s))
            sat_level = frame_res.get("sat_level_adu")
            if sat_level is not None:
                try:
                    sat_levels_adu.append(float(sat_level))
                except Exception:
                    pass
            no_coverage_frac = float(frame_res.get("no_coverage_frac") or 0.0)

            # Output header (shared by all outputs for this exposure)
            ohdr = _set_linear_wcs(hdr, wave0, dw, unit=unit)
            ohdr["WAVEUNIT"] = (str(unit), "Wavelength unit (explicit)")
            ohdr["WAVEREF"] = (str(waveref), "Wavelength reference (air/vacuum)")
            ohdr["BUNIT"] = (str(bunit), "Data unit")
            ohdr["NORMEXP"] = (
                bool(normalize_exptime),
                "Normalize to per-second units (ADU/s)",
            )
            ohdr["TEXPS"] = (float(exptime_s), "Exposure time used for normalization (s)")
            if delta_lambda_A is not None:
                try:
                    ohdr["DLAM_A"] = (float(delta_lambda_A), "Wavelength shift from sky flexure (Angstrom)")
                    ohdr["DLAMAP"] = (bool(delta_lambda_applied), "Applied delta-lambda shift to the grid")
                    if flexure_score is not None:
                        ohdr["FLXSCR"] = (float(flexure_score), "Flexure score (higher=better)")
                    if flexure_flag is not None:
                        ohdr["FLXFLAG"] = (str(flexure_flag), "Flexure flag")
                    ohdr["DLAMPOL"] = (str(delta_policy), "Delta-lambda apply policy")
                except Exception:
                    pass

            if sat_level is not None and bool(lcfg.get("mask_saturation", True)):
                try:
                    ohdr["SATLEV"] = (
                        float(sat_level),
                        "Saturation level (ADU) used to flag MASK",
                    )
                    ohdr["SATMARG"] = (float(lcfg.get("saturation_margin_adu", 0.0) or 0.0), "Saturation margin (ADU)")
                except Exception:
                    pass
            ohdr = add_provenance(ohdr, cfg, stage="linearize")

            out_skysub = out_dir / f"{base_for_mask}_skysub.fits"
            out_skymodel_path: Path | None = None
            _write_mef(
                out_skysub,
                rect,
                ohdr,
                rect_var,
                rect_mask,
                rect_cov,
                wave0=wave0,
                dw=dw,
                unit=unit,
                wave_ref=wave_ref,
            )
            rect_paths.append(out_skysub)

            if bool(cleanup_diag.get("cleanup_executed")) and residual_model is not None:
                model_out = np.asarray(residual_model, dtype=np.float32)
                out_model = out_dir / f"{base_for_mask}_skymodel.fits"
                out_skymodel_path = out_model
                _write_mef(
                    out_model,
                    model_out,
                    ohdr,
                    np.zeros_like(rect_var, dtype=np.float32),
                    rect_mask,
                    rect_cov,
                    wave0=wave0,
                    dw=dw,
                    unit=unit,
                )

            if per_exp_dir is not None:
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

            skysub_png = out_dir / f"{base_for_mask}_skysub.png"
            skysub_skywin_png = out_dir / f"{base_for_mask}_skysub_skywin.png"
            skymodel_png: Path | None = None
            try:
                quicklook_from_mef(out_skysub, skysub_png, method="linear", k=4.0)
                quicklook_from_mef(
                    out_skysub,
                    skysub_skywin_png,
                    method="asinh",
                    k=4.0,
                    row_mask=(None if mask_sky_y is None else np.asarray(mask_sky_y, dtype=bool)),
                )
                if out_skymodel_path is not None:
                    skymodel_png = out_dir / f"{base_for_mask}_skymodel.png"
                    quicklook_from_mef(out_skymodel_path, skymodel_png, method="linear", k=4.0)
            except Exception:
                pass

            stems_for_compare.append(stem)

            delta_u = None
            if delta_lambda_A is not None:
                try:
                    delta_u = float(delta_lambda_A) * 0.1 if str(unit).lower() in {"nm", "nanometer", "nanometers"} else float(delta_lambda_A)
                except Exception:
                    delta_u = None

            delta_policy_report = "off" if delta_policy == "off" else "missing"
            if delta_lambda_A is not None:
                if delta_lambda_applied:
                    delta_policy_report = "shift_grid"
                else:
                    delta_policy_report = "not_applied_uncertain" if str(flexure_flag).upper() != "OK" else "missing"

            exp_flags: list[dict[str, Any]] = []
            try:
                exp_flags.extend(list(roi_flags or []))
            except Exception:
                pass
            if delta_lambda_A is not None and (not delta_lambda_applied) and delta_policy != "off":
                exp_flags.append(
                    {
                        "code": "FLEXURE_SHIFT_NOT_APPLIED",
                        "severity": "WARN",
                        "message": "Delta-lambda shift was estimated but not applied to the linearization grid.",
                        "hint": "If flexure_flag is UNCERTAIN, this is expected with policy=apply_if_ok. Use policy=always only if you understand the risk.",
                        "value": {
                            "delta_lambda_A": float(delta_lambda_A) if delta_lambda_A is not None else None,
                            "flexure_score": float(flexure_score) if flexure_score is not None else None,
                            "flexure_flag": str(flexure_flag) if flexure_flag is not None else None,
                        },
                    }
                )

            per_exposure.append(
                {
                    "stem": stem,
                    "inputs": {
                        "skysub_raw_path": str(p),
                    },
                    "grid": {
                        "lam_start": float(wave0),
                        "lam_step": float(dw),
                        "nlam": int(nlam),
                        "unit": str(unit),
                        "ref": str(waveref),
                    },
                    "outputs": {
                        "skysub_path": str(out_skysub),
                        "skymodel_path": str(out_skymodel_path) if out_skymodel_path is not None else None,
                        "skysub_png": str(skysub_png) if skysub_png is not None else None,
                        "skysub_skywin_png": str(skysub_skywin_png) if skysub_skywin_png is not None else None,
                        "skymodel_png": str(skymodel_png) if skymodel_png is not None else None,
                    },
                    "resampling": {
                        "no_coverage_frac": no_coverage_frac,
                    },
                    "delta_lambda": {
                        "value": float(delta_u) if delta_u is not None else None,
                        "value_A": float(delta_lambda_A) if delta_lambda_A is not None else None,
                        "unit": str(unit),
                        "flexure_score": float(flexure_score) if flexure_score is not None else None,
                        "flexure_flag": str(flexure_flag) if flexure_flag is not None else None,
                        "applied": bool(delta_lambda_applied),
                        "delta_lambda_policy": str(delta_policy_report),
                        "policy": str(delta_policy),
                    },
                    "roi": {
                        "roi_source": roi_report.get("roi_source"),
                        "roi_valid": bool(roi_report.get("roi_valid")) if roi_report.get("roi_valid") is not None else False,
                        "roi_bands": roi_report.get("roi_bands"),
                        "sky_contamination_metric": roi_report.get("sky_contamination_metric"),
                    },
                    "cleanup": cleanup_diag,
                    "metrics_before": dict(metrics_before),
                    "metrics_after": dict(cleanup_diag.get("metrics_after") or metrics_before),
                    "flags": exp_flags,
                    "residual_sky_cleanup": cleanup_diag,
                }
            )
            out_skysub_paths.append(str(out_skysub))
            if out_skymodel_path is not None:
                out_skymodel_paths.append(str(out_skymodel_path))
            no_coverage_fracs.append(no_coverage_frac)

            if not robust_stack:
                w = np.zeros_like(rect_var, dtype=np.float64)
                fatal_bits = NO_COVERAGE | BADPIX | COSMIC | SATURATED | USER | REJECTED
                good = (rect_var > 0) & ((rect_mask & fatal_bits) == 0)
                w[good] = 1.0 / rect_var[good]
                stack_num += rect.astype(np.float64) * w
                stack_den += w
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

        # Aggregate per-exposure flags into a stage-level list (deduplicated by code/message)
        stage_flags: list[dict[str, Any]] = []
        seen = set()
        for pe in per_exposure:
            for fl in (pe.get("flags") or []):
                try:
                    key = (str(fl.get("code")), str(fl.get("severity")), str(fl.get("message")), str(fl.get("hint")))
                except Exception:
                    key = None
                if key is not None and key in seen:
                    continue
                if key is not None:
                    seen.add(key)
                stage_flags.append(dict(fl))

        def _median_from_per_exposure(which: str, key: str) -> float | None:
            vals: list[float] = []
            for pe in per_exposure:
                block = pe.get(which) if isinstance(pe.get(which), dict) else None
                v = block.get(key) if isinstance(block, dict) else None
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except Exception:
                    pass
            return float(np.median(vals)) if vals else None

        # Stage-level ROI / cleanup summary: use the first exposure (they share the same config)
        roi_stage = per_exposure[0].get("roi") if per_exposure else {"roi_source": "none", "roi_valid": False}
        cleanup_stage = per_exposure[0].get("cleanup") if per_exposure else {}

        # Delta-lambda stage report
        delta_u = None
        if delta_lambda_A is not None:
            try:
                delta_u = float(delta_lambda_A) * 0.1 if str(unit).lower() in {"nm", "nanometer", "nanometers"} else float(delta_lambda_A)
            except Exception:
                delta_u = None
        delta_policy_report = "off" if delta_policy == "off" else "missing"
        if delta_lambda_A is not None:
            if delta_lambda_applied:
                delta_policy_report = "shift_grid"
            elif str(flexure_flag).upper() != "OK":
                delta_policy_report = "not_applied_uncertain"

        payload = {
            "stage": "linearize",
            "status": "ok",

            # Inputs
            "inputs": {
                "skysub_raw_path": str(sci_paths[0]) if len(sci_paths) == 1 else None,
                "skysub_raw_paths": [str(p) for p in sci_paths],
                "lambda_map_path": str(lam_path),
                "rectification_model_path": str(rect_model_path) if rect_model_path else None,
                "sky_done_path": str(sky_done_path) if (sky_done_path is not None and sky_done_path.exists()) else None,
            },

            # Grid
            "grid": {
                "lam_start": float(wave0),
                "lam_start_nominal": float(wave0_nominal) if (wave0_nominal is not None) else None,
                "lam_step": float(dw),
                "nlam": int(nlam),
                "unit": str(unit),
                "ref": str(waveref),
                "grid_mode": str(lcfg.get("grid_mode", "intersection")),
            },

            # Resampling contract
            "resampling": {
                "mapping_type": "cumulative_rebin",
                "interpolation": "bin_overlap_flux_conserving",
                "var_policy": "VAR_out = Σ_k (a_k^2 * VAR_k)  (uncorrelated assumption)",
                "mask_policy": "MASK_out = OR_k(MASK_k over fatal bits); NO_COVERAGE if no contributing pixels",
                "no_coverage_frac_median": float(np.median(no_coverage_fracs)) if no_coverage_fracs else None,
                "no_coverage_frac_max": float(np.max(no_coverage_fracs)) if no_coverage_fracs else None,
                "grid_cache": {
                    "enabled": bool(_linearize_grid_cache_enabled()),
                    "hit": bool(linearize_grid_cache_hit),
                    "frame_signature": str(frame_signature_key),
                    "lambda_map_key": str(lambda_map_cache_key),
                },
                "parallel_map": {
                    "backend": str(frame_parallel_backend),
                    "n_jobs_requested": frame_parallel_n_jobs,
                    "n_items": int(len(frame_tasks)),
                },
            },

            # Flexure / delta-lambda
            "delta_lambda": {
                "value": float(delta_u) if delta_u is not None else None,
                "value_A": float(delta_lambda_A) if delta_lambda_A is not None else None,
                "unit": str(unit),
                "flexure_score": float(flexure_score) if flexure_score is not None else None,
                "flexure_flag": str(flexure_flag) if flexure_flag is not None else None,
                "applied": bool(delta_lambda_applied),
                "delta_lambda_policy": str(delta_policy_report),
                "policy": str(delta_policy),
            },

            # ROI summary (for reporting; per-exposure is stored below)
            "roi": dict(roi_stage) if isinstance(roi_stage, dict) else {"roi_source": "none", "roi_valid": False},

            # Cleanup summary
            "cleanup": dict(cleanup_stage) if isinstance(cleanup_stage, dict) else {},

            # Metrics (stage-level medians)
            "metrics_before": {
                "sky_residual_metric": _median_from_per_exposure("metrics_before", "sky_residual_metric"),
                "object_risk_metric": _median_from_per_exposure("metrics_before", "object_risk_metric"),
            },
            "metrics_after": {
                "sky_residual_metric": _median_from_per_exposure("metrics_after", "sky_residual_metric"),
                "object_risk_metric": _median_from_per_exposure("metrics_after", "object_risk_metric"),
            },

            # Outputs
            "outputs": {
                "skysub_path": str(out_skysub_paths[0]) if len(out_skysub_paths) == 1 else None,
                "skysub_paths": list(out_skysub_paths),
                "skymodel_path": str(out_skymodel_paths[0]) if len(out_skymodel_paths) == 1 else None,
                "skymodel_paths": list(out_skymodel_paths),
                "preview_fits": str(preview_fits),
                "preview_png": str(preview_png) if preview_png.exists() else None,
            },

            # Flags
            "flags": stage_flags,

            # Detailed per-exposure diagnostics
            "per_exposure": per_exposure,

            # Legacy / compatibility keys
            "lambda_map": str(lam_path),
            "lambda_map_sha256": lambda_map_sha256,
            "rectification_model": str(rect_model_path) if rect_model_path else None,
            "rectification_model_sha256": rect_model_sha256,
            "wave_unit": str(unit),
            "wave_ref": str(waveref),
            "unit_source": str(unit_src),
            "bunit": str(bunit),
            "bunit_mode": str(bunit_mode),
            "normalize_exptime": bool(normalize_exptime),
            "exptime_total_s": float(np.sum(exp_times_s)) if exp_times_s else None,
            "exptime_median_s": float(np.median(exp_times_s)) if exp_times_s else None,
            "wave0": float(wave0),
            "dw": float(dw),
            "nlam": int(nlam),
            "stacking": {
                "method": "sigma_clip" if robust_stack else "wmean",
                "sigma": float(stack_sigma) if robust_stack else None,
                "maxiters": int(stack_maxiters) if robust_stack else None,
                **(stack_stats or {}),
            },
            "saturation": {
                "mask_saturation": bool(lcfg.get("mask_saturation", True)),
                "saturation_margin_adu": float(lcfg.get("saturation_margin_adu", 0.0) or 0.0),
                "saturation_level_adu": float(np.nanmedian(sat_levels_adu)) if sat_levels_adu else None,
                "saturation_level_min_adu": float(np.nanmin(sat_levels_adu)) if sat_levels_adu else None,
                "saturation_level_max_adu": float(np.nanmax(sat_levels_adu)) if sat_levels_adu else None,
            },
            "noise": {
                "gain_override_e_per_adu": gain_override_f,
                "read_noise_override_e": rdnoise_override_f,
                "gain_median_e_per_adu": float(np.median([m["gain_e_per_adu"] for m in noise_meta])) if noise_meta else None,
                "rdnoise_median_e": float(np.median([m["rdnoise_e"] for m in noise_meta])) if noise_meta else None,
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

        # Scalars used by QC report + QC gate
        cov_nonzero = (
            float(np.count_nonzero(coverage > 0) / float(coverage.size)) if coverage.size else 0.0
        )
        rejected_fraction = None
        try:
            rejected_fraction = float((stack_stats or {}).get("rejected_fraction"))
        except Exception:
            rejected_fraction = None

        # Stage-level QC flags (used by the QC gate)
        try:
            from scorpio_pipe.qc.flags import make_flag, max_severity
            from scorpio_pipe.qc_thresholds import compute_thresholds

            thr, thr_meta = compute_thresholds(cfg)
            qc_flags: list[dict[str, Any]] = []

            if legacy_discovery_used:
                qc_flags.append(
                    make_flag(
                        "LEGACY_LAMBDA_MAP_DISCOVERY",
                        "WARN",
                        "lambda_map was located via legacy directory search (no dataset_manifest.json)",
                        hint="Re-run 08_wavesol to regenerate rectification_model.json and avoid guessing.",
                    )
                )

            # Lambda-map metadata: prefer explicit WAVEUNIT/WAVEREF, but allow
            # heuristic inference for legacy/synthetic data.
            if unit_meta_inferred:
                qc_flags.append(
                    make_flag(
                        "QC_LAMBDA_MAP_META",
                        "WARN",
                        "lambda_map is missing explicit WAVEUNIT/WAVEREF; inferred heuristically",
                        unit=unit,
                        waveref=waveref,
                    )
                )

            # Coverage: lower is worse
            if cov_nonzero <= float(thr.linearize_cov_nonzero_bad):
                qc_flags.append(
                    make_flag(
                        "QC_LINEARIZE_COVERAGE",
                        "ERROR",
                        "Linearize coverage nonzero fraction is critically low",
                        value=cov_nonzero,
                        warn_le=float(thr.linearize_cov_nonzero_warn),
                        bad_le=float(thr.linearize_cov_nonzero_bad),
                    )
                )
            elif cov_nonzero <= float(thr.linearize_cov_nonzero_warn):
                qc_flags.append(
                    make_flag(
                        "QC_LINEARIZE_COVERAGE",
                        "WARN",
                        "Linearize coverage nonzero fraction is low",
                        value=cov_nonzero,
                        warn_le=float(thr.linearize_cov_nonzero_warn),
                        bad_le=float(thr.linearize_cov_nonzero_bad),
                    )
                )

            # Rejection: higher is worse
            if rejected_fraction is not None:
                rjf = float(rejected_fraction)
                if rjf >= float(thr.linearize_rejected_frac_bad):
                    qc_flags.append(
                        make_flag(
                            "QC_LINEARIZE_REJECTED",
                            "ERROR",
                            "Linearize robust stack rejected fraction is too high",
                            value=rjf,
                            warn_ge=float(thr.linearize_rejected_frac_warn),
                            bad_ge=float(thr.linearize_rejected_frac_bad),
                        )
                    )
                elif rjf >= float(thr.linearize_rejected_frac_warn):
                    qc_flags.append(
                        make_flag(
                            "QC_LINEARIZE_REJECTED",
                            "WARN",
                            "Linearize robust stack rejected fraction is high",
                            value=rjf,
                            warn_ge=float(thr.linearize_rejected_frac_warn),
                            bad_ge=float(thr.linearize_rejected_frac_bad),
                        )
                    )

            # Skyline residual (after optional residual cleanup)
            try:
                sky_after = metrics_after.get("sky_residual_metric") if isinstance(metrics_after, dict) else None
                if sky_after is not None:
                    v = float(sky_after)
                    if np.isfinite(v):
                        if v >= float(thr.sky_resid_mad_snr_bad):
                            qc_flags.append(
                                make_flag(
                                    "SKYLINE_RESIDUAL_HIGH",
                                    "ERROR",
                                    "Skyline residuals after Linearize are high",
                                    value=v,
                                    warn_ge=float(thr.sky_resid_mad_snr_warn),
                                    bad_ge=float(thr.sky_resid_mad_snr_bad),
                                )
                            )
                        elif v >= float(thr.sky_resid_mad_snr_warn):
                            qc_flags.append(
                                make_flag(
                                    "SKYLINE_RESIDUAL_HIGH",
                                    "WARN",
                                    "Skyline residuals after Linearize are elevated",
                                    value=v,
                                    warn_ge=float(thr.sky_resid_mad_snr_warn),
                                    bad_ge=float(thr.sky_resid_mad_snr_bad),
                                )
                            )
            except Exception:
                pass

            # Object eating risk (only meaningful if cleanup was forced)
            try:
                if bool(cleanup_diag.get("cleanup_applied")):
                    od = cleanup_diag.get("obj_delta_median")
                    tol = cleanup_diag.get("obj_tol")
                    if od is not None and tol is not None:
                        odf = float(od)
                        t = float(tol)
                        if np.isfinite(odf) and np.isfinite(t) and odf < -t:
                            qc_flags.append(
                                make_flag(
                                    "OBJECT_EATING_RISK",
                                    "WARN",
                                    "Residual sky cleanup may oversubtract the object rows",
                                    value=odf,
                                    bad_le=-t,
                                )
                            )
            except Exception:
                pass

            payload["qc"] = {
                "flags": qc_flags,
                "max_severity": max_severity(qc_flags),
                "thresholds": thr.to_dict(),
                "thresholds_meta": thr_meta,
            }

            # Merge QC flags into the main flags list (deduplicated)
            try:
                merged = list(payload.get("flags") or []) + list(qc_flags or [])
                seen = set()
                deduped: list[dict[str, Any]] = []
                for fl in merged:
                    try:
                        key = (
                            str(fl.get("code")),
                            str(fl.get("severity")),
                            str(fl.get("message")),
                            str(fl.get("hint")),
                        )
                    except Exception:
                        key = None
                    if key is not None and key in seen:
                        continue
                    if key is not None:
                        seen.add(key)
                    deduped.append(dict(fl))
                payload["flags"] = deduped
            except Exception:
                pass

            try:
                _sev = str(payload.get("qc", {}).get("max_severity", "OK") or "OK").upper()
                payload["status"] = (
                    "ok" if _sev in {"OK", "INFO"} else ("warn" if _sev == "WARN" else "fail")
                )
            except Exception:
                payload["status"] = "ok"

            # If any non-QC flags are present, elevate status accordingly.
            try:
                sev = "OK"
                for fl in (payload.get("flags") or []):
                    s = str(fl.get("severity") or "OK").upper()
                    if s == "ERROR":
                        sev = "ERROR"
                        break
                    if s == "WARN" and sev != "ERROR":
                        sev = "WARN"
                if payload.get("status") != "fail":
                    if sev == "ERROR":
                        payload["status"] = "fail"
                    elif sev == "WARN" and payload.get("status") == "ok":
                        payload["status"] = "warn"
            except Exception:
                pass

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

        # Write canonical + legacy done markers
        done_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        try:
            done_json_legacy.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

        # --- Compare A/B cache (after write) ---
        # Build diffs *only* when we had a pre-run snapshot.
        try:
            if compare_stamp and compare_a and stems_for_compare:
                compare_b = snapshot_stage(
                    stage_key="linearize",
                    stage_dir=out_dir,
                    label="B",
                    patterns=(
                        "done.json",
                        "linearize_done.json",
                        "qc_linearize.json",
                        "wave_grid.json",
                        "*_skysub.fits",
                        "*_skymodel.fits",
                        "*_rectified.fits",
                        "*_skysub.png*",
                        "*_skysub_skywin.png*",
                        "*_skymodel.png*",
                    ),
                    stamp=compare_stamp,
                )
                build_stage_diff(
                    stage_key="linearize",
                    stamp=compare_stamp,
                    run_root=work_dir,
                    a_dir=compare_a,
                    b_dir=compare_b,
                    stems=stems_for_compare,
                    a_suffix="_skysub.fits",
                    b_suffix="_skysub.fits",
                )
        except Exception:
            pass


        log.info("Linearize done: %s", preview_fits)
        return payload
    except Exception as e:
        raise
