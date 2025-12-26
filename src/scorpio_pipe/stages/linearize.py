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

from scorpio_pipe.io.mef import (
    WaveGrid,
    write_sci_var_mask,
    try_read_grid,
    read_sci_var_mask,
)
from scorpio_pipe.maskbits import (
    BADPIX,
    COSMIC,
    EDGE,
    NO_COVERAGE,
    REJECTED,
    SATURATED,
    USER,
    header_cards as mask_header_cards,
    summarize as summarize_mask,
)

from ..logging_utils import get_logger
from ..provenance import add_provenance
from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.wavesol_paths import wavesol_dir

log = get_logger(__name__)


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
                return (
                    sci.astype(np.float64, copy=False),
                    (None if var is None else var.astype(np.float64, copy=False)),
                    mask,
                    hdr,
                )
    except Exception:
        # fall back to primary-only
        pass
    data, hdr = _open_fits_resilient(path)
    return data, None, None, fits.Header(hdr)


def _get_gain_e_per_adu(hdr: fits.Header, default: float = 1.0) -> float:
    for key in ("GAIN", "EGAIN", "CCDGAIN"):
        if key in hdr:
            try:
                v = float(hdr[key])
                if math.isfinite(v) and v > 0:
                    return v
            except Exception:
                pass
    return float(default)


def _get_read_noise_e(hdr: fits.Header, default: float = 3.0) -> float:
    for key in ("RDNOISE", "READNOIS", "RON", "RNOISE"):
        if key in hdr:
            try:
                v = float(hdr[key])
                if math.isfinite(v) and v >= 0:
                    return v
            except Exception:
                pass
    return float(default)


def _estimate_variance_adu(data_adu: np.ndarray, hdr: fits.Header) -> np.ndarray:
    """First-pass variance in ADU^2 from Poisson + read-noise.

    Assumes data_adu is already bias/flat corrected to a reasonable extent.
    Negative values are clipped for the Poisson term.
    """
    g = _get_gain_e_per_adu(hdr)
    rn = _get_read_noise_e(hdr)
    # electrons
    data_e = np.clip(data_adu * g, 0.0, None)
    var_e = data_e + rn * rn
    var_adu = var_e / (g * g)
    return var_adu.astype(np.float32)


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


def _rebin_row_cumulative(
    values: np.ndarray,
    lam_row: np.ndarray,
    edges_out: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flux-conserving rebin of integrated-per-pixel values.

    Uses cumulative integral on *pixel edges* and linear interpolation.
    Returns (rebinned_values, coverage_fraction).
    """
    lam_row = np.asarray(lam_row, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if lam_row.shape != values.shape:
        raise ValueError("lam_row and values must have the same shape")

    # Ensure monotonic increasing wavelength.
    if not np.all(np.diff(lam_row) > 0):
        order = np.argsort(lam_row)
        lam_row = lam_row[order]
        values = values[order]

    edges_in = _lambda_edges(lam_row)
    c = np.empty(values.size + 1, dtype=np.float64)
    c[0] = 0.0
    np.cumsum(values, out=c[1:])
    # Interpolate cumulative at output edges.
    c_out = np.interp(edges_out, edges_in, c, left=0.0, right=float(c[-1]))
    out = np.diff(c_out)

    # Coverage fraction: how much of each output bin is within the input λ-range.
    lo = float(edges_in[0])
    hi = float(edges_in[-1])
    bin_lo = edges_out[:-1]
    bin_hi = edges_out[1:]
    overlap = np.clip(np.minimum(bin_hi, hi) - np.maximum(bin_lo, lo), 0.0, None)
    cov = overlap / np.maximum(bin_hi - bin_lo, 1e-12)
    return out.astype(np.float32), cov.astype(np.float32)


def _write_mef(
    path: Path,
    sci: np.ndarray,
    hdr: fits.Header,
    var: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    cov: Optional[np.ndarray] = None,
    grid: WaveGrid | None = None,
) -> None:
    """Write 2D MEF product with canonical SCI/VAR/MASK and optional COV.

    Primary HDU stores SCI data for legacy readers; SCI extension is canonical.
    """
    if grid is None:
        grid = try_read_grid(hdr)
    extra: list[fits.ImageHDU] = []
    if cov is not None:
        extra.append(fits.ImageHDU(data=np.asarray(cov, dtype=np.float32), name="COV"))
    write_sci_var_mask(
        path,
        sci,
        var=var,
        mask=mask,
        header=hdr,
        grid=grid,
        extra_hdus=extra,
        primary_data=sci,
    )


def _set_linear_wcs(hdr: fits.Header, wave0: float, dw: float) -> fits.Header:
    hdr = hdr.copy()
    # 1-indexed FITS WCS
    hdr["CTYPE1"] = ("WAVE", "Linear wavelength")
    hdr["CUNIT1"] = ("Angstrom", "Wavelength unit")
    hdr["CRPIX1"] = (1.0, "Reference pixel")
    hdr["CRVAL1"] = (float(wave0), "Wavelength at CRPIX1")
    hdr["CDELT1"] = (float(dw), "Angstrom per pixel")
    return hdr


def _infer_lambda_unit(lam_hdr: fits.Header, lam_map: np.ndarray) -> str:
    """Best-effort guess for lambda_map unit.

    SCORPIO wavesolution products usually store wavelengths in Angstrom,
    but some intermediate workflows may store pixel coordinates.
    """
    for k in ("CUNIT1", "BUNIT", "UNIT", "CUNIT"):
        if k in lam_hdr:
            try:
                u = str(lam_hdr[k]).strip().lower()
                if "ang" in u or "\u00c5" in u or u == "a":
                    return "Angstrom"
                if "pix" in u or "px" in u:
                    return "pix"
            except Exception:
                pass
    # heuristic fallback
    lam = np.asarray(lam_map, dtype=float)
    finite = np.isfinite(lam)
    if not np.any(finite):
        return "Angstrom"
    v = float(np.nanmedian(lam[finite]))
    return "pix" if v < 2000.0 else "Angstrom"


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
    stack_preview = bool(lcfg.get("stack_preview", True))

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
    unit = _infer_lambda_unit(lam_hdr, lam_map)
    wmin_cfg = lcfg.get("lambda_min_A", lcfg.get("wmin"))
    wmax_cfg = lcfg.get("lambda_max_A", lcfg.get("wmax"))

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

    stack_num = np.zeros((ny, nlam), dtype=np.float64)
    stack_den = np.zeros((ny, nlam), dtype=np.float64)
    stack_var_den = np.zeros((ny, nlam), dtype=np.float64)
    coverage = np.zeros((ny, nlam), dtype=np.int16)
    mask_sum = np.zeros((ny, nlam), dtype=np.uint16)

    sat_levels_adu: list[float] = []

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
        if var_in is not None:
            var = np.asarray(var_in, dtype=np.float64)[y0:y1, :]
        else:
            var = _estimate_variance_adu(data, hdr)
        mask = None
        if mask_in is not None:
            mask = np.asarray(mask_in, dtype=np.uint16)[y0:y1, :]

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
                    mask = (mask | cm).astype(np.uint16, copy=False)
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
            vv_row, _ = _rebin_row_cumulative(
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

        # Per-exposure output
        if per_exp_dir is not None:
            ohdr = _set_linear_wcs(hdr, wave0, dw)
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

            # Canonical v5.18+ naming: *_rectified.fits (keep *_lin.fits as a compatibility alias)
            out_rect = per_exp_dir / f"{base_for_mask}_rectified.fits"
            out_lin_alias = per_exp_dir / f"{base_for_mask}_lin.fits"
            _write_mef(out_rect, rect, ohdr, rect_var, rect_mask, rect_cov)
            try:
                if out_lin_alias.resolve() != out_rect.resolve():
                    import shutil

                    shutil.copy2(out_rect, out_lin_alias)
            except Exception:
                pass

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

    hdr0 = _set_linear_wcs(lam_hdr, wave0, dw)
    hdr0["BUNIT"] = ("ADU", "Data unit")
    hdr0 = add_provenance(hdr0, cfg, stage="linearize")
    _write_mef(preview_fits, out_sci, hdr0, out_var, out_mask, coverage)

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
        "products": {
            "preview_fits": str(preview_fits),
            "preview_png": str(preview_png) if preview_png.exists() else None,
        },
    }

    # Wave grid metadata (used by downstream stages & external tooling)
    wave_grid_json = out_dir / "wave_grid.json"
    try:
        wave_grid_json.write_text(
            json.dumps(
                {
                    "unit": "Angstrom",
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
        qc_dir = work_dir / "qc"
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
            "coverage": {
                "min": int(np.min(coverage)),
                "median": float(np.median(coverage)),
                "max": int(np.max(coverage)),
            },
            "mask_summary": summarize_mask(out_mask),
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

    # Legacy mirroring (UI / older workflows expect work_dir/lin/obj_sum_lin.*)
    try:
        legacy_dir = work_dir / "lin"
        if legacy_dir.resolve() != out_dir.resolve():
            legacy_dir.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copy2(preview_fits, legacy_dir / "obj_sum_lin.fits")
            if preview_png.exists():
                shutil.copy2(preview_png, legacy_dir / "obj_sum_lin.png")
            shutil.copy2(done_json, legacy_dir / "linearize_done.json")
    except Exception:
        pass

    log.info("Linearize done: %s", preview_fits)
    return payload
