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


def _set_linear_wcs(hdr: fits.Header, wave0: float, dw: float, *, unit: str = "Angstrom") -> fits.Header:
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
            extra.append(fits.ImageHDU(data=np.asarray(cov, dtype=np.float32), name="COV"))
        except Exception:
            pass

    grid = None
    if wave0 is not None and dw is not None:
        try:
            grid = WaveGrid(lambda0=float(wave0), dlambda=float(dw), nlam=int(np.asarray(sci).shape[1]), unit=str(unit))
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


def _open_science_with_optional_var_mask(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, fits.Header]:
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



def _guess_saturation_level_adu(hdr: fits.Header, lcfg: Dict[str, Any]) -> Optional[float]:
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
        if abs(bscale - 1.0) < 1e-9 and (abs(bzero - 32768.0) < 1e-3 or abs(bzero) < 1e-3):
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


def _rebin_row_var_weightsquared(var_in: np.ndarray, lam_centers: np.ndarray, edges_out: np.ndarray) -> np.ndarray:
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


def _rebin_row_cumulative(values: np.ndarray, lam_centers: np.ndarray, edges_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def run_linearize(cfg: Dict[str, Any], out_dir: Optional[Path] = None, *, cancel_token: Any | None = None) -> Dict[str, Any]:
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
    elif isinstance(dw_raw, str) and dw_raw.strip().lower() in {"auto", "data", "from_data"}:
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
    if isinstance(lcfg.get("lambda_map_path"), str) and str(lcfg.get("lambda_map_path")).strip():
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
        cand = list(base.rglob("lambda_map.fits")) + list(base.rglob("lambda_map*.fits"))
        lam_path = cand[0] if cand else None
    if lam_path is None or not lam_path.exists():
        raise FileNotFoundError("lambda_map.fits not found (expected under work_dir/wavesol/<disperser>/)")

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
        cosm_root / kind / "clean",                  # current
        cosm_root / method / kind / "clean",         # legacy (method/kind)
        cosm_root / method / "clean",                # legacy (method)
    ]
    cand_mask = [
        cosm_root / kind / "masks_fits",
        cosm_root / method / kind / "masks_fits",
        cosm_root / method / "masks_fits",
        cosm_root / kind / "mask_fits",              # very old typo
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
            _, _, dw, _ = _compute_output_grid(lam_map, dw_hint=dw, mode=grid_mode, lo_pct=lo_pct, hi_pct=hi_pct, imin_pct=imin_pct, imax_pct=imax_pct)
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
        mask_saturation = bool(lcfg.get('mask_saturation', True))
        sat_level = _guess_saturation_level_adu(hdr, lcfg) if mask_saturation else None
        if sat_level is not None:
            try:
                sat_levels_adu.append(float(sat_level))
            except Exception:
                pass
        sat_margin = float(lcfg.get('saturation_margin_adu', 0.0) or 0.0)
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
                cm = ((m.astype(np.uint16) > 0).astype(np.uint16) * COSMIC).astype(np.uint16)
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
            v_row, cov = _rebin_row_cumulative(data[yy, finite], lam_row[finite], wave_edges)
            vv_row = _rebin_row_var_weightsquared(var[yy, finite], lam_row[finite], wave_edges)
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
                        ((mask[yy, finite] & bit) > 0).astype(np.float64), lam_row[finite], wave_edges
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
            ohdr = _set_linear_wcs(hdr, wave0, dw, unit=unit)
            # Saturation metadata (if used)
            if sat_level is not None and mask_saturation:
                try:
                    ohdr["SATLEV"] = (float(sat_level), "Saturation level (ADU) used to flag MASK")
                    ohdr["SATMARG"] = (float(sat_margin), "Saturation margin (ADU)")
                except Exception:
                    pass
            ohdr = add_provenance(ohdr, cfg, stage="linearize")

            # Canonical v5.18+ naming: *_rectified.fits (keep *_lin.fits as a compatibility alias)
            out_rect = per_exp_dir / f"{base_for_mask}_rectified.fits"
            out_lin_alias = per_exp_dir / f"{base_for_mask}_lin.fits"
            _write_mef(out_rect, rect, ohdr, rect_var, rect_mask, rect_cov, wave0=wave0, dw=dw, unit=unit)
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

    hdr0 = _set_linear_wcs(lam_hdr, wave0, dw, unit=unit)
    hdr0["BUNIT"] = ("ADU", "Data unit")
    hdr0 = add_provenance(hdr0, cfg, stage="linearize")
    _write_mef(preview_fits, out_sci, hdr0, out_var, out_mask, coverage, wave0=wave0, dw=dw, unit=unit)

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
            "saturation_margin_adu": float(lcfg.get("saturation_margin_adu", 0.0) or 0.0),
            "saturation_level_adu": float(np.nanmedian(sat_levels_adu)) if sat_levels_adu else None,
            "saturation_level_min_adu": float(np.nanmin(sat_levels_adu)) if sat_levels_adu else None,
            "saturation_level_max_adu": float(np.nanmax(sat_levels_adu)) if sat_levels_adu else None,
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
        qc_dir = work_dir / "qc"
        qc_dir.mkdir(parents=True, exist_ok=True)
        qc_path = qc_dir / "linearize_qc.json"

        fatal_bits = NO_COVERAGE | BADPIX | COSMIC | SATURATED | USER | REJECTED
        good = (out_var > 0) & ((out_mask & fatal_bits) == 0)
        snr = np.zeros_like(out_sci, dtype=np.float64)
        snr[good] = np.abs(out_sci[good].astype(np.float64)) / np.sqrt(np.maximum(out_var[good].astype(np.float64), 1e-20))
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
