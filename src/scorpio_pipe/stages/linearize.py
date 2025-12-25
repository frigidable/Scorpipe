"""Linearization / rectification using a 2D dispersion solution (lambda_map).

This stage creates a linear-wavelength (λ, y) product on a common wavelength grid.

Core requirement for v5.x:
  - Build a *linearized summed* object frame from already cosmic-cleaned frames.
  - Preserve (first-order) flux by using edge-based cumulative rebinning.
  - Provide variance and mask as first-class citizens where possible.

Current implementation (v5.3):
  - Rectifies each exposure onto a common λ grid (optional output per exposure).
  - Stacks rectified exposures into obj_sum_lin.fits using inverse-variance weights.

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

from ..logging_utils import get_logger
from ..provenance import add_provenance
from ..wavesol_paths import resolve_work_dir

log = get_logger(__name__)


MASK_NO_COVERAGE = np.uint16(1 << 0)
MASK_BAD = np.uint16(1 << 1)
MASK_COSMIC = np.uint16(1 << 2)


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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ph = fits.PrimaryHDU(data=sci.astype(np.float32), header=hdr)
    hdus = [ph]
    if var is not None:
        h = fits.ImageHDU(data=var.astype(np.float32), name="VAR")
        hdus.append(h)
    if mask is not None:
        h = fits.ImageHDU(data=mask.astype(np.uint16), name="MASK")
        hdus.append(h)
    if cov is not None:
        h = fits.ImageHDU(data=cov.astype(np.int16), name="COV")
        hdus.append(h)
    fits.HDUList(hdus).writeto(path, overwrite=True)


def _set_linear_wcs(hdr: fits.Header, wave0: float, dw: float) -> fits.Header:
    hdr = hdr.copy()
    # 1-indexed FITS WCS
    hdr["CTYPE1"] = ("WAVE", "Linear wavelength")
    hdr["CUNIT1"] = ("Angstrom", "Wavelength unit")
    hdr["CRPIX1"] = (1.0, "Reference pixel")
    hdr["CRVAL1"] = (float(wave0), "Wavelength at CRPIX1")
    hdr["CDELT1"] = (float(dw), "Angstrom per pixel")
    return hdr


def _guess_output_grid(lam_map: np.ndarray, dw_hint: float) -> Tuple[float, float, int]:
    """Pick a common λ grid based on the lambda_map range."""
    lam = np.asarray(lam_map, dtype=np.float64)
    finite = np.isfinite(lam)
    if not np.any(finite):
        raise ValueError("lambda_map contains no finite values")
    lo = float(np.nanpercentile(lam[finite], 0.2))
    hi = float(np.nanpercentile(lam[finite], 99.8))

    dw = float(dw_hint)
    if not (math.isfinite(dw) and dw > 0):
        # rough fallback from median spacing in the center
        mid = lam[lam.shape[0] // 2]
        d = np.diff(mid[np.isfinite(mid)])
        dw = float(np.nanmedian(d)) if d.size else 1.0
        dw = max(dw, 0.1)

    n = int(max(16, math.ceil((hi - lo) / dw)))
    return lo, dw, n


def run_linearize(cfg: Dict[str, Any], out_dir: Optional[Path] = None, *, cancel_token: Any | None = None) -> Dict[str, Any]:
    """Run linearization.

    Inputs:
      - lambda_map: from wavesolution stage
      - cleaned object frames (prefer cosmics outputs if present)

    Outputs:
      - work_dir/lin/obj_sum_lin.fits  (MEF: SCI [+VAR,+MASK,+COV])
      - work_dir/lin/linearize_done.json
      - optional per-exposure rectified frames under work_dir/lin/per_exp/
    """
    cfg = dict(cfg)
    work_dir = resolve_work_dir(cfg)
    if out_dir is None:
        out_dir = work_dir / "lin"
    out_dir.mkdir(parents=True, exist_ok=True)

    lcfg = cfg.get("linearize", {}) if isinstance(cfg.get("linearize"), dict) else {}
    # Canonical keys in v5.x schema:
    #   dlambda_A, lambda_min_A, lambda_max_A, per_exposure
    # Backward-compatible aliases:
    #   dw, wmin/wmax, save_per_exposure
    dw = float(lcfg.get("dlambda_A", lcfg.get("dw", 1.0)))
    y_crop_top = int(lcfg.get("y_crop_top", 0))
    y_crop_bottom = int(lcfg.get("y_crop_bottom", 0))
    save_per_exp = bool(lcfg.get("per_exposure", lcfg.get("save_per_exposure", True)))

    # Locate lambda_map
    wavesol_dir = work_dir / "wavesol"
    lam_path = None
    # most common location in this project:
    cand = list(wavesol_dir.rglob("lambda_map.fits"))
    if cand:
        lam_path = cand[0]
    else:
        # fallback: legacy name
        cand = list(wavesol_dir.rglob("lambda_map*.fits"))
        lam_path = cand[0] if cand else None
    if lam_path is None or not lam_path.exists():
        raise FileNotFoundError("lambda_map.fits not found under work_dir/wavesol")

    lam_map, lam_hdr = _open_fits_resilient(lam_path)
    if lam_map.ndim != 2:
        raise ValueError("lambda_map must be a 2D image")

    # Get science frames list (prefer cosmics cleaned)
    frames = cfg.get("frames", {}) if isinstance(cfg.get("frames"), dict) else {}
    obj_frames = frames.get("obj", [])
    if not obj_frames:
        raise ValueError("No object frames configured (frames.obj is empty)")

    # Prefer cosmics products if present
    cosm_cfg = cfg.get("cosmics", {}) if isinstance(cfg.get("cosmics"), dict) else {}
    method = str(cosm_cfg.get("method", "stack_mad"))
    cosm_dir = work_dir / "cosmics" / method
    clean_dir = cosm_dir / "clean"
    mask_dir = cosm_dir / "mask_fits"

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

    # Output λ grid
    wmin_cfg = lcfg.get("lambda_min_A", lcfg.get("wmin"))
    wmax_cfg = lcfg.get("lambda_max_A", lcfg.get("wmax"))
    if wmin_cfg is not None and wmax_cfg is not None:
        wave0 = float(wmin_cfg)
        wmax = float(wmax_cfg)
        if wmax <= wave0:
            raise ValueError("linearize.lambda_max_A must be > lambda_min_A")
        # include the right edge
        nlam = int(np.ceil((wmax - wave0) / dw))
    else:
        wave0, dw, nlam = _guess_output_grid(lam_map, dw)
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

    for i, p in enumerate(sci_paths, 1):
        if cancel_token is not None and getattr(cancel_token, "cancelled", False):
            raise RuntimeError("Cancelled")
        log.info("Linearize %d/%d: %s", i, len(sci_paths), p.name)
        data, hdr = _open_fits_resilient(p)
        if data.ndim != 2:
            raise ValueError(f"Science frame must be 2D: {p}")
        data = data[y0:y1, :]
        if data.shape != lam_map.shape:
            raise ValueError(
                f"Science shape {data.shape} != lambda_map shape {lam_map.shape} for {p.name}"
            )

        var = _estimate_variance_adu(data, hdr)
        mask = None
        base_for_mask = Path(p).stem
        if base_for_mask.endswith("_clean"):
            base_for_mask = base_for_mask[:-6]
        mp = mask_dir / f"{base_for_mask}_mask.fits"
        if mp.exists():
            try:
                m, _ = _open_fits_resilient(mp)
                mask = (m.astype(np.uint16) > 0).astype(np.uint16) * MASK_COSMIC
                mask = mask[y0:y1, :]
            except Exception as e:
                log.warning("Failed to load cosmic mask %s: %s", mp.name, e)
                mask = None

        rect = np.zeros((ny, nlam), dtype=np.float32)
        rect_var = np.zeros((ny, nlam), dtype=np.float32)
        rect_mask = np.zeros((ny, nlam), dtype=np.uint16)

        for yy in range(ny):
            lam_row = lam_map[yy]
            # Bad rows: mark no coverage
            if not np.any(np.isfinite(lam_row)):
                rect_mask[yy, :] |= MASK_NO_COVERAGE
                continue
            # Use only finite subset for rebin; if too few finite points, skip
            finite = np.isfinite(lam_row)
            if np.sum(finite) < 8:
                rect_mask[yy, :] |= MASK_NO_COVERAGE
                continue
            v_row, cov = _rebin_row_cumulative(data[yy, finite], lam_row[finite], wave_edges)
            vv_row, _ = _rebin_row_cumulative(var[yy, finite], lam_row[finite], wave_edges)
            rect[yy] = v_row
            rect_var[yy] = np.maximum(vv_row, 0.0)
            # mark incomplete coverage
            rect_mask[yy, cov < 0.999] |= MASK_NO_COVERAGE
            if mask is not None:
                # crude mask propagation: if any masked detector pixel falls into a λ bin,
                # mark the same bin as masked. We approximate using rebinning of a 0/1 mask.
                m_row, _ = _rebin_row_cumulative(
                    (mask[yy, finite] > 0).astype(np.float64), lam_row[finite], wave_edges
                )
                rect_mask[yy, m_row > 0] |= MASK_COSMIC

        # Per-exposure output
        if per_exp_dir is not None:
            ohdr = _set_linear_wcs(hdr, wave0, dw)
            ohdr = add_provenance(ohdr, cfg, stage="linearize")
            _write_mef(per_exp_dir / f"{p.stem}_lin.fits", rect, ohdr, rect_var, rect_mask)

        # Stack: inverse-variance weighted mean
        w = np.zeros_like(rect_var, dtype=np.float64)
        good = (rect_var > 0) & (rect_mask == 0)
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
    out_mask[~ok] |= MASK_NO_COVERAGE

    # Output products
    sum_fits = out_dir / "obj_sum_lin.fits"
    sum_json = out_dir / "linearize_done.json"

    hdr0 = _set_linear_wcs(lam_hdr, wave0, dw)
    hdr0["BUNIT"] = ("ADU", "Data unit")
    hdr0 = add_provenance(hdr0, cfg, stage="linearize")
    _write_mef(sum_fits, out_sci, hdr0, out_var, out_mask, coverage)

    payload = {
        "stage": "linearize",
        "lambda_map": str(lam_path),
        "inputs": [str(p) for p in sci_paths],
        "wave0": wave0,
        "dw": dw,
        "nlam": nlam,
        "per_exposure": str(per_exp_dir) if per_exp_dir is not None else None,
        "products": {
            "sum_fits": str(sum_fits),
        },
    }
    sum_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("Linearize done: %s", sum_fits)
    return payload
