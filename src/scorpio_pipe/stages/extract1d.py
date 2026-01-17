"""Object Extraction: 2D → 1D spectra (TRACE + FIXED).

This stage extracts 1D spectra from a long-slit 2D product on a linear
wavelength grid (axis-1 = λ, axis-0 = spatial y).

Inputs (default)
----------------
Prefer the stacked 2D product:
  11_stack/stack2d.fits

Explicit expert mode
--------------------
If the user explicitly enables ``extract1d.input_mode=single_frame``, the stage
can extract from a chosen single rectified sky-subtracted exposure:
  10_linearize/<stem>_skysub.fits

Outputs
-------
12_extract/spec1d.fits
  - Image extensions for GUI preview (2×Nλ): FLUX2, VAR2, MASK2, NPIX2, SKY2
  - Table HDUs with full column structure for both products:
      * TRACE  (LAMBDA, FLUX_TRACE, VAR_TRACE, MASK_TRACE, SKY_TRACE, NPIX_TRACE)
      * FIXED  (LAMBDA, FLUX_FIXED, VAR_FIXED, MASK_FIXED, SKY_FIXED, NPIX_FIXED)
12_extract/trace.json
12_extract/extract_done.json
12_extract/spec1d.png (optional quicklook)
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from astropy.io import fits
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d

from scorpio_pipe import maskbits
from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.provenance import add_provenance
from scorpio_pipe.version import PIPELINE_VERSION

log = logging.getLogger(__name__)


# ------------------------------- utilities -------------------------------


def _roi_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """ROI is owned by the Sky stage; extraction reuses it for consistency."""
    sky = cfg.get("sky") if isinstance(cfg.get("sky"), dict) else {}
    roi = sky.get("roi") if isinstance(sky.get("roi"), dict) else {}
    return dict(roi)


def _read_mef(
    path: Path,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], fits.Header]:
    """Read pipeline MEF products (PRIMARY or SCI for image; optional VAR/MASK)."""
    with fits.open(path, memmap=False) as hdul:
        hdr = hdul[0].header.copy()

        sci = hdul[0].data
        if sci is None or np.asarray(sci).ndim < 2:
            if "SCI" in hdul and hdul["SCI"].data is not None:
                sci = hdul["SCI"].data
                shdr = hdul["SCI"].header
                for k in ("CRVAL1", "CDELT1", "CD1_1", "CRPIX1", "CTYPE1", "CUNIT1"):
                    if k not in hdr and k in shdr:
                        hdr[k] = shdr[k]
            else:
                raise ValueError(f"No SCI data found in {path}")

        sci = np.asarray(sci, dtype=float)

        var = None
        mask = None
        if "VAR" in hdul:
            try:
                var = np.asarray(hdul["VAR"].data, dtype=float)
            except Exception:
                var = None
        if "MASK" in hdul:
            try:
                mask = np.asarray(hdul["MASK"].data, dtype=np.uint16)
            except Exception:
                mask = None

    return sci, var, mask, hdr


def _linear_wave_axis(hdr: fits.Header, nlam: int) -> np.ndarray:
    crval = hdr.get("CRVAL1")
    cdelt = hdr.get("CDELT1", hdr.get("CD1_1"))
    crpix = hdr.get("CRPIX1", 1.0)
    if crval is None or cdelt is None:
        return np.arange(nlam, dtype=float)
    i = np.arange(nlam, dtype=float)
    return float(crval) + (i + 1.0 - float(crpix)) * float(cdelt)


def _mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    med = float(np.median(x))
    return float(1.4826 * np.median(np.abs(x - med)))


def _clamp_int(v: float, lo: int, hi: int) -> int:
    return int(np.clip(int(round(float(v))), lo, hi))


def _safe_json(obj: Any) -> Any:
    """Make numpy-ish payloads JSON-serializable."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    return obj


def _is_stack2d_input(path: Path, hdr: fits.Header) -> bool:
    name = path.name.lower()
    if name in {"stack2d.fits", "stacked2d.fits"}:
        return True
    parts = {p.lower() for p in path.parts}
    if "11_stack" in parts:
        return True
    # heuristic: stack2d writes STKMETH
    if "STKMETH" in hdr:
        return True
    return False


# ------------------------------ trace model ------------------------------


@dataclass
class TraceModel:
    y_trace: np.ndarray  # (nlam,)
    y_fixed: np.ndarray  # (nlam,)
    aperture_half_width: int
    meta: dict[str, Any]


def _estimate_aperture_half_width(
    sci: np.ndarray,
    mask: Optional[np.ndarray],
    *,
    roi: dict[str, Any],
    y_center: float,
    min_hw: int = 2,
    max_hw: Optional[int] = None,
) -> int:
    """Estimate aperture half-width using ROI if available, else from profile FWHM."""
    ny, nlam = sci.shape
    max_hw = int(max_hw or max(6, ny // 4))

    # 1) ROI (object band) if available.
    y0 = roi.get("obj_y0")
    y1 = roi.get("obj_y1")
    if isinstance(y0, (int, float)) and isinstance(y1, (int, float)):
        y0i = int(np.clip(int(round(y0)), 0, ny - 1))
        y1i = int(np.clip(int(round(y1)), y0i + 1, ny))
        hw = int(max(min_hw, round(0.5 * (y1i - y0i))))
        return int(np.clip(hw, min_hw, max_hw))

    # 2) Empirical from collapsed profile.
    fatal = None
    if mask is not None and mask.shape == sci.shape:
        # treat any non-zero mask as invalid for this rough estimate
        fatal = mask != 0

    prof = np.nanmedian(sci, axis=1)
    if fatal is not None:
        tmp = sci.copy()
        tmp[fatal] = np.nan
        prof = np.nanmedian(tmp, axis=1)

    if not np.isfinite(prof).any():
        return int(np.clip(6, min_hw, max_hw))

    # Remove a baseline using the outer quartiles.
    yy = np.arange(ny, dtype=float)
    q = np.nanpercentile(prof, [10.0, 50.0, 90.0])
    baseline = float(q[1])
    prof0 = prof - baseline
    # find peak near y_center
    ypk = int(np.clip(int(round(y_center)), 0, ny - 1))
    # allow peak to slide within +/- 20 px
    w = int(min(20, ny // 3))
    a0 = max(0, ypk - w)
    a1 = min(ny, ypk + w + 1)
    sl = prof0[a0:a1]
    if not np.isfinite(sl).any():
        return int(np.clip(6, min_hw, max_hw))
    ypk = int(np.nanargmax(sl) + a0)
    peak = float(prof0[ypk])
    if not np.isfinite(peak) or peak <= 0:
        return int(np.clip(6, min_hw, max_hw))

    half = 0.5 * peak
    above = np.where(np.isfinite(prof0) & (prof0 >= half))[0]
    if above.size < 2:
        return int(np.clip(6, min_hw, max_hw))
    # choose the contiguous segment containing the peak
    # (simple approach: nearest indices around peak)
    left = above[above <= ypk]
    right = above[above >= ypk]
    if left.size == 0 or right.size == 0:
        return int(np.clip(6, min_hw, max_hw))
    yL = int(left.min())
    yR = int(right.max())
    fwhm = max(1.0, float(yR - yL))
    # Box aperture: ~1.5×(FWHM/2) is a reasonable default for long-slit
    hw = int(math.ceil(1.5 * 0.5 * fwhm))
    return int(np.clip(hw, min_hw, max_hw))


def _centroid_from_profile(
    prof: np.ndarray,
    *,
    y_mask: Optional[np.ndarray] = None,
    sky_windows: Optional[list[tuple[int, int]]] = None,
    min_snr: float = 3.0,
) -> tuple[float, dict[str, Any]]:
    """Robust centroid from a 1D spatial profile."""
    ny = prof.size
    yy = np.arange(ny, dtype=float)
    p = np.asarray(prof, dtype=float)

    # Baseline from sky windows when present.
    baseline = float(np.nanmedian(p))
    sky_sigma = float("nan")
    used_sky = False
    if sky_windows:
        vals = []
        for (a, b) in sky_windows:
            a = int(np.clip(a, 0, ny - 1))
            b = int(np.clip(b, a, ny - 1))
            vv = p[a : b + 1]
            vv = vv[np.isfinite(vv)]
            if vv.size:
                vals.append(float(np.median(vv)))
        if vals:
            baseline = float(np.median(vals))
            used_sky = True
            # estimate noise from sky zones as robust scatter of those medians
            sky_sigma = _mad_sigma(np.asarray(vals, dtype=float))

    s = p - baseline
    if y_mask is not None and y_mask.size == ny:
        s = np.where(y_mask, s, np.nan)

    if not np.isfinite(s).any():
        return float("nan"), {
            "used_sky": used_sky,
            "baseline": baseline,
            "snr": float("nan"),
        }

    # Focus around the peak; use only positive weights.
    ypk = int(np.nanargmax(s))
    win = int(min(20, ny // 3))
    y0 = max(0, ypk - win)
    y1 = min(ny, ypk + win + 1)
    ss = s[y0:y1]
    ss = np.clip(np.nan_to_num(ss, nan=0.0), 0.0, np.inf)
    if float(np.sum(ss)) <= 0:
        return float("nan"), {
            "used_sky": used_sky,
            "baseline": baseline,
            "snr": float("nan"),
        }
    ywin = yy[y0:y1]
    yc = float(np.sum(ywin * ss) / np.sum(ss))

    # SNR estimate
    peak = float(np.nanmax(ss))
    if not np.isfinite(sky_sigma) or sky_sigma <= 0:
        # fallback: robust scatter of full profile residuals
        sky_sigma = _mad_sigma(s)
    snr = float(peak / sky_sigma) if np.isfinite(sky_sigma) and sky_sigma > 0 else float("nan")

    if np.isfinite(snr) and snr < float(min_snr):
        return float("nan"), {
            "used_sky": used_sky,
            "baseline": baseline,
            "snr": snr,
        }

    return yc, {
        "used_sky": used_sky,
        "baseline": baseline,
        "snr": snr,
    }


def _estimate_trace_blocks(
    sci: np.ndarray,
    mask: Optional[np.ndarray],
    wave: np.ndarray,
    *,
    roi: dict[str, Any],
    sky_windows: Optional[list[tuple[int, int]]],
    trace_bin_A: float,
    trace_min_snr: float,
    fatal_bits: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Estimate y(λ) from centroids in wavelength blocks."""
    ny, nlam = sci.shape
    dl = float(np.nanmedian(np.abs(np.diff(wave)))) if wave.size > 1 else 1.0
    bin_pix = max(1, int(round(float(trace_bin_A) / max(dl, 1e-6))))

    # Object corridor mask (fallback if ROI missing)
    y0 = roi.get("obj_y0")
    y1 = roi.get("obj_y1")
    if isinstance(y0, (int, float)) and isinstance(y1, (int, float)):
        y0i = int(np.clip(int(round(y0)), 0, ny - 1))
        y1i = int(np.clip(int(round(y1)), y0i + 1, ny))
        y_mask = np.zeros(ny, dtype=bool)
        y_mask[y0i:y1i] = True
    else:
        y_mask = np.zeros(ny, dtype=bool)
        y_mask[int(0.35 * ny) : int(0.65 * ny)] = True

    have_mask = mask is not None and mask.shape == sci.shape

    w_c = []
    y_c = []
    blocks_meta: list[dict[str, Any]] = []

    for x0 in range(0, nlam, bin_pix):
        x1 = min(nlam, x0 + bin_pix)
        block = sci[:, x0:x1]
        if have_mask:
            mb = mask[:, x0:x1]
            block = block.copy()
            block[(mb & fatal_bits) != 0] = np.nan
        prof = np.nanmedian(block, axis=1)
        yc, m = _centroid_from_profile(
            prof, y_mask=y_mask, sky_windows=sky_windows, min_snr=float(trace_min_snr)
        )
        wsl = wave[x0:x1]
        wc = float(np.nanmean(wsl)) if wsl.size else float("nan")
        w_c.append(wc)
        y_c.append(yc)
        blocks_meta.append(
            {
                "x0": int(x0),
                "x1": int(x1),
                "wave_c": wc,
                "y": yc,
                **m,
            }
        )

    return np.asarray(w_c, dtype=float), np.asarray(y_c, dtype=float), blocks_meta


def _smooth_trace_spline(
    wave_c: np.ndarray,
    y_c: np.ndarray,
    wave_full: np.ndarray,
    *,
    sigma_clip: float,
    spline_k: int,
    spline_sigma_pix: float,
    min_points: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Spline smoothing with robust outlier rejection."""
    good = np.isfinite(wave_c) & np.isfinite(y_c)
    w = wave_c[good]
    y = y_c[good]
    meta: dict[str, Any] = {"initial_points": int(w.size)}
    if w.size < int(min_points):
        return np.full_like(wave_full, float(np.nanmedian(y_c)) if np.any(np.isfinite(y_c)) else 0.0), {
            **meta,
            "fallback": True,
            "reason": "too_few_points",
        }

    # sort by wavelength
    srt = np.argsort(w)
    w = w[srt]
    y = y[srt]

    # smoothing factor in spline units
    s = float(w.size) * float(spline_sigma_pix) ** 2
    s = max(0.0, s)

    clip = float(max(2.5, sigma_clip))
    kept = np.ones_like(y, dtype=bool)
    for _ in range(2):
        try:
            spl = UnivariateSpline(w[kept], y[kept], k=int(spline_k), s=s)
        except Exception:
            break
        resid = y - spl(w)
        sig = _mad_sigma(resid)
        if not np.isfinite(sig) or sig <= 0:
            break
        kept = np.abs(resid) <= (clip * sig)
        if kept.sum() < int(min_points):
            kept[:] = True
            break

    try:
        spl = UnivariateSpline(w[kept], y[kept], k=int(spline_k), s=s)
        y_full = spl(wave_full)
        resid = y[kept] - spl(w[kept])
        meta.update(
            {
                "fallback": False,
                "kept_points": int(kept.sum()),
                "rms_pix": float(np.sqrt(np.nanmean(resid**2))) if resid.size else float("nan"),
                "mad_pix": float(_mad_sigma(resid)),
                "k": int(spline_k),
                "s": float(s),
                "sigma_clip": float(clip),
            }
        )
        return np.asarray(y_full, dtype=float), meta
    except Exception as e:
        return np.full_like(wave_full, float(np.nanmedian(y_c)) if np.any(np.isfinite(y_c)) else 0.0), {
            **meta,
            "fallback": True,
            "reason": f"spline_failed: {e}",
        }


def _build_trace_model(
    sci: np.ndarray,
    var: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    wave: np.ndarray,
    *,
    roi: dict[str, Any],
    sky_windows: list[tuple[int, int]] | None,
    cfg: dict[str, Any],
    fatal_bits: int,
) -> TraceModel:
    ny, nlam = sci.shape

    # Defaults / knobs
    trace_bin_A = float(cfg.get("trace_bin_A", 60.0))
    trace_min_snr = float(cfg.get("trace_min_snr", 3.0))
    trace_sigma_clip = float(cfg.get("trace_sigma_clip", cfg.get("trace_outlier_sigma", 4.0)))
    spline_k = int(np.clip(int(cfg.get("trace_spline_k", 3)), 1, 5))
    spline_sigma_pix = float(cfg.get("trace_spline_sigma_pix", 0.7))
    min_blocks = int(max(5, int(cfg.get("trace_min_blocks", 6))))
    min_frac = float(cfg.get("trace_min_valid_frac", 0.25))

    wave_c, y_c, blocks_meta = _estimate_trace_blocks(
        sci,
        mask,
        wave,
        roi=roi,
        sky_windows=sky_windows,
        trace_bin_A=trace_bin_A,
        trace_min_snr=trace_min_snr,
        fatal_bits=int(fatal_bits),
    )

    good = np.isfinite(y_c)
    n_good = int(np.sum(good))
    n_all = int(y_c.size)

    meta: dict[str, Any] = {
        "trace_bin_A": trace_bin_A,
        "trace_min_snr": trace_min_snr,
        "blocks": blocks_meta,
        "blocks_total": n_all,
        "blocks_valid": n_good,
    }

    # Choose fixed center from ROI if available; else from the block median.
    y_fixed_center = None
    if isinstance(cfg.get("fixed_center_y"), (int, float)):
        y_fixed_center = float(cfg["fixed_center_y"])
    else:
        y0 = roi.get("obj_y0")
        y1 = roi.get("obj_y1")
        if isinstance(y0, (int, float)) and isinstance(y1, (int, float)):
            y_fixed_center = 0.5 * (float(y0) + float(y1))
        else:
            y_fixed_center = float(np.nanmedian(y_c)) if np.any(np.isfinite(y_c)) else (0.5 * float(ny - 1))
    y_fixed_center = float(np.clip(y_fixed_center, 0.0, float(ny - 1)))

    # Decide trace: fallback if too few valid blocks
    fallback = (n_good < min_blocks) or (n_all > 0 and (n_good / max(n_all, 1)) < min_frac)
    if fallback:
        y_trace = np.full(nlam, y_fixed_center, dtype=float)
        meta["trace_model"] = {
            "type": "fallback_constant",
            "y0": y_fixed_center,
            "reason": "insufficient_valid_blocks",
        }
    else:
        y_trace, sm = _smooth_trace_spline(
            wave_c,
            y_c,
            wave,
            sigma_clip=trace_sigma_clip,
            spline_k=spline_k,
            spline_sigma_pix=spline_sigma_pix,
            min_points=min_blocks,
        )
        y_trace = np.asarray(y_trace, dtype=float)
        y_trace = np.clip(y_trace, 0.0, float(ny - 1))
        meta["trace_model"] = {"type": "spline", **sm}

    y_fixed = np.full(nlam, y_fixed_center, dtype=float)

    # Aperture half-width
    if isinstance(cfg.get("aperture_half_width"), (int, float)) and float(cfg.get("aperture_half_width")) > 0:
        ap_hw = int(max(1, int(round(float(cfg["aperture_half_width"])))))
        meta["aperture"] = {"mode": "user", "half_width_pix": ap_hw}
    else:
        ap_hw = _estimate_aperture_half_width(
            sci,
            mask,
            roi=roi,
            y_center=y_fixed_center,
            min_hw=2,
            max_hw=max(6, ny // 4),
        )
        meta["aperture"] = {"mode": "auto", "half_width_pix": int(ap_hw)}

    return TraceModel(
        y_trace=y_trace.astype(float),
        y_fixed=y_fixed.astype(float),
        aperture_half_width=int(ap_hw),
        meta=meta,
    )


# ------------------------------- extraction -------------------------------


def _estimate_sky_residual(
    sci: np.ndarray,
    var: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    *,
    sky_windows: list[tuple[int, int]] | None,
    fatal_bits: int,
) -> np.ndarray:
    """Per-λ robust sky residual level from configured sky windows."""
    ny, nlam = sci.shape
    out = np.full(nlam, np.nan, dtype=float)
    if not sky_windows:
        return out
    have_mask = mask is not None and mask.shape == sci.shape
    have_var = var is not None and var.shape == sci.shape
    for j in range(nlam):
        vals = []
        for (a, b) in sky_windows:
            a = int(np.clip(a, 0, ny - 1))
            b = int(np.clip(b, a, ny - 1))
            sl = sci[a : b + 1, j].astype(float)
            good = np.isfinite(sl)
            if have_var:
                vv = var[a : b + 1, j]
                good &= np.isfinite(vv)
            if have_mask:
                mm = mask[a : b + 1, j]
                good &= (mm & fatal_bits) == 0
            if not np.any(good):
                continue
            vals.append(float(np.median(sl[good])))
        if vals:
            out[j] = float(np.median(vals))
    return out


def _boxcar_extract(
    sci: np.ndarray,
    var: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    y_center: np.ndarray,
    *,
    ap_hw: int,
    fatal_bits: int,
    min_good_frac: float,
    sky_resid: Optional[np.ndarray],
    apply_local_sky: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple aperture sum (TRACE or FIXED)."""
    ny, nlam = sci.shape
    ap_hw = int(max(1, ap_hw))
    min_good_frac = float(np.clip(min_good_frac, 0.0, 1.0))

    flux = np.full(nlam, np.nan, dtype=float)
    out_var = np.full(nlam, np.nan, dtype=float)
    out_mask = np.zeros(nlam, dtype=np.uint16)
    out_npix = np.zeros(nlam, dtype=np.int16)

    have_mask = mask is not None and mask.shape == sci.shape
    have_var = var is not None and var.shape == sci.shape

    from scorpio_pipe.maskbits import NO_COVERAGE

    for j in range(nlam):
        yc = float(y_center[j])
        if not np.isfinite(yc):
            out_mask[j] |= np.uint16(NO_COVERAGE)
            continue
        y0 = int(math.floor(yc)) - ap_hw
        y1 = int(math.floor(yc)) + ap_hw + 1
        y0 = max(0, y0)
        y1 = min(ny, y1)
        if y1 <= y0:
            out_mask[j] |= np.uint16(NO_COVERAGE)
            continue

        sl = sci[y0:y1, j].astype(float)
        good = np.isfinite(sl)

        if have_var:
            vv = var[y0:y1, j].astype(float)
            good &= np.isfinite(vv) & (vv > 0)
        else:
            vv = None

        if have_mask:
            mm = mask[y0:y1, j]
            # OR all bits for output (keeps provenance of masked contributors)
            try:
                out_mask[j] |= np.uint16(np.bitwise_or.reduce(mm))
            except Exception:
                pass
            good &= (mm & fatal_bits) == 0

        n_tot = int(y1 - y0)
        n_good = int(np.sum(good))
        out_npix[j] = np.int16(n_good)

        if n_tot <= 0 or n_good <= 0:
            out_mask[j] |= np.uint16(NO_COVERAGE)
            continue
        if n_good < int(math.ceil(min_good_frac * n_tot)):
            out_mask[j] |= np.uint16(NO_COVERAGE)

        f = float(np.sum(sl[good]))
        if apply_local_sky and sky_resid is not None and np.isfinite(sky_resid[j]):
            f = f - float(sky_resid[j]) * float(n_good)
        flux[j] = f
        if vv is not None:
            out_var[j] = float(np.sum(vv[good]))
        else:
            # fallback: empirical
            out_var[j] = float(np.nanvar(sl[good])) if np.isfinite(np.nanvar(sl[good])) else float("nan")

    sky_out = np.asarray(sky_resid, dtype=float) if sky_resid is not None else np.full(nlam, np.nan, dtype=float)
    return flux, out_var, out_mask, sky_out, out_npix


# ------------------------------- IO helpers -------------------------------



def _optimal_extract(
    sci: np.ndarray,
    var: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    y_center: np.ndarray,
    *,
    ap_hw: int,
    fatal_bits: int,
    min_good_frac: float,
    sky_resid: Optional[np.ndarray],
    apply_local_sky: bool,
    smooth_lambda_sigma: float = 3.0,
    clip_negative_profile: bool = True,
    min_profile_sum: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Horne-like optimal extraction (weights by spatial profile + honest VAR).

    Notes
    -----
    We implement the classical weighted estimator:

      F(λ)   = Σ_y P(y|λ) * D(y,λ) / V(y,λ)  /  Σ_y P(y|λ)^2 / V(y,λ)
      Var(λ) = 1 / Σ_y P(y|λ)^2 / V(y,λ)

    where D is the input 2D data (background/sky already subtracted) and P is the
    normalized spatial profile within the extraction aperture.

    The profile is estimated from the data in a moving aperture around `y_center`
    and smoothed along λ with a Gaussian filter to reduce 'learning on noise'.
    """
    ny, nlam = sci.shape
    ap_hw = int(max(1, ap_hw))
    nwin = int(2 * ap_hw + 1)
    min_good_frac = float(np.clip(min_good_frac, 0.0, 1.0))

    have_var = var is not None and var.shape == sci.shape
    have_mask = mask is not None and mask.shape == sci.shape
    if not have_var:
        # Without VAR optimal extraction is undefined; fall back to VAR=1.
        var = np.ones_like(sci, dtype=float)

    # Build a profile cube in aperture coordinates (k=0..nwin-1, λ).
    prof = np.zeros((nwin, nlam), dtype=float)
    good_prof = np.zeros((nwin, nlam), dtype=bool)

    for j in range(nlam):
        yc = float(y_center[j]) if j < len(y_center) else float(np.nanmedian(y_center))
        y0 = _clamp_int(yc - ap_hw, 0, ny - 1)
        y1 = _clamp_int(yc + ap_hw, 0, ny - 1)
        sl = sci[y0 : y1 + 1, j].astype(float, copy=False)
        vv = var[y0 : y1 + 1, j].astype(float, copy=False)
        good = np.isfinite(sl) & np.isfinite(vv) & (vv > 0)
        if have_mask:
            mm = mask[y0 : y1 + 1, j]
            good &= (mm & fatal_bits) == 0

        if sl.size == 0:
            continue

        p = sl.copy()
        if clip_negative_profile:
            p = np.where(np.isfinite(p), np.maximum(p, 0.0), 0.0)
        else:
            p = np.where(np.isfinite(p), p, 0.0)

        # Map into fixed aperture grid.
        k0 = int(y0 - (int(round(yc)) - ap_hw))
        k0 = int(np.clip(k0, 0, nwin - 1))
        k1 = min(nwin, k0 + sl.size)
        prof[k0:k1, j] = p[: (k1 - k0)]
        good_prof[k0:k1, j] = good[: (k1 - k0)]

    # Smooth along λ (each spatial offset independently).
    try:
        sig = float(max(0.0, smooth_lambda_sigma))
    except Exception:
        sig = 3.0
    if sig > 0:
        for k in range(nwin):
            prof[k, :] = gaussian_filter1d(prof[k, :], sigma=sig, mode="nearest")

    # Normalize profile to Σ P = 1 per λ on good pixels.
    P = np.zeros_like(prof, dtype=float)
    n_fallback = 0
    for j in range(nlam):
        g = good_prof[:, j]
        s = float(np.sum(prof[g, j])) if np.any(g) else 0.0
        if not (s > min_profile_sum):
            # Fallback: uniform on good pixels (or full aperture if nothing is good).
            n_fallback += 1
            if np.any(g):
                P[g, j] = 1.0 / float(np.sum(g))
            else:
                P[:, j] = 1.0 / float(nwin)
        else:
            P[g, j] = prof[g, j] / s

    # Now compute the optimal estimator per λ.
    flux = np.full(nlam, np.nan, dtype=float)
    out_var = np.full(nlam, np.nan, dtype=float)
    out_mask = np.zeros(nlam, dtype=np.uint16)
    out_npix = np.zeros(nlam, dtype=np.int16)

    from scorpio_pipe.maskbits import NO_COVERAGE

    meta: dict[str, Any] = {
        "profile_smooth_lambda_sigma": float(sig),
        "profile_clip_negative": bool(clip_negative_profile),
        "profile_fallback_frac": float(n_fallback) / float(max(1, nlam)),
    }

    for j in range(nlam):
        yc = float(y_center[j]) if j < len(y_center) else float(np.nanmedian(y_center))
        y0 = _clamp_int(yc - ap_hw, 0, ny - 1)
        y1 = _clamp_int(yc + ap_hw, 0, ny - 1)
        sl = sci[y0 : y1 + 1, j].astype(float, copy=False)
        vv = var[y0 : y1 + 1, j].astype(float, copy=False)
        good = np.isfinite(sl) & np.isfinite(vv) & (vv > 0)
        if have_mask:
            mm = mask[y0 : y1 + 1, j]
            good &= (mm & fatal_bits) == 0

        n_tot = int(y1 - y0 + 1)
        n_good = int(np.sum(good))
        out_npix[j] = np.int16(n_good)

        if n_tot <= 0 or n_good <= 0:
            out_mask[j] |= np.uint16(NO_COVERAGE)
            continue
        if n_good < int(math.ceil(min_good_frac * n_tot)):
            out_mask[j] |= np.uint16(NO_COVERAGE)

        # Map P (aperture coords) to this slice.
        k0 = int(y0 - (int(round(yc)) - ap_hw))
        k0 = int(np.clip(k0, 0, nwin - 1))
        k1 = min(nwin, k0 + sl.size)
        Pj = np.zeros_like(sl, dtype=float)
        Pj[: (k1 - k0)] = P[k0:k1, j][: (k1 - k0)]

        # Classical estimator.
        denom = float(np.sum((Pj[good] ** 2) / vv[good])) if np.any(good) else 0.0
        if not (denom > 0):
            out_mask[j] |= np.uint16(NO_COVERAGE)
            continue
        num = float(np.sum(Pj[good] * sl[good] / vv[good]))

        f = num / denom

        if apply_local_sky and sky_resid is not None and np.isfinite(sky_resid[j]):
            # Subtract additive residual sky offset properly for the weighted estimator.
            corr = float(np.sum(Pj[good] / vv[good])) / denom
            f = f - float(sky_resid[j]) * corr

        flux[j] = f
        out_var[j] = 1.0 / denom

    sky_out = np.asarray(sky_resid, dtype=float) if sky_resid is not None else np.full(nlam, np.nan, dtype=float)
    return flux, out_var, out_mask, sky_out, out_npix, meta



def _write_spec1d_fits(
    out_path: Path,
    *,
    cfg: Dict[str, Any],
    hdr0: fits.Header,
    wave: np.ndarray,
    trace: TraceModel,
    flux_trace: np.ndarray,
    var_trace: np.ndarray,
    mask_trace: np.ndarray,
    sky_trace: np.ndarray,
    npix_trace: np.ndarray,
    flux_fixed: np.ndarray,
    var_fixed: np.ndarray,
    mask_fixed: np.ndarray,
    sky_fixed: np.ndarray,
    npix_fixed: np.ndarray,
) -> None:
    """Write spec1d.fits (tables + lightweight preview images).

    Contract (P1-F)
    --------------
    - Primary HDU: linear WCS + provenance
    - HDU1: SPEC_TRACE table
    - HDU2: SPEC_FIXED table
    - Additional 2×N preview images for quick GUI inspection
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------- primary header ---------------------------
    ph = fits.Header()

    # Prefer existing WCS keywords if present.
    for k in ("CRVAL1", "CDELT1", "CD1_1", "CRPIX1", "CTYPE1", "CUNIT1"):
        if k in hdr0:
            ph[k] = hdr0[k]
    if "CRVAL1" not in ph and wave.size >= 2 and np.isfinite(wave[:2]).all():
        ph["CRVAL1"] = float(wave[0])
        ph["CDELT1"] = float(wave[1] - wave[0])
        ph["CRPIX1"] = 1.0
        ph["CTYPE1"] = "WAVE"
        ph["CUNIT1"] = "Angstrom"

    # Required stage-level provenance.
    ph["PIPEVER"] = (str(PIPELINE_VERSION), "Pipeline version")
    ph["STAGE"] = ("12_extract", "Pipeline stage (directory name)")

    if "INPUT2D" in hdr0:
        ph["INPUT2D"] = (str(hdr0["INPUT2D"]), "2D input FITS used for extraction")

    # Units.
    ph["BUNIT"] = hdr0.get("BUNIT", "ADU")
    ph["FLUXUNIT"] = hdr0.get("BUNIT", "ADU")
    ph["WAVEUNIT"] = hdr0.get("CUNIT1", "Angstrom")
    # Wavelength medium must be declared once wavelength exists.
    _wr = str(hdr0.get("WAVEREF") or (cfg.get("project", {}) or {}).get("wave_reference") or "air").strip().lower()
    if _wr in {"vac", "vacuo"}:
        _wr = "vacuum"
    if _wr not in {"air", "vacuum"}:
        _wr = "air"
    ph["WAVEREF"] = (_wr, "Wavelength medium (air/vacuum)")

    # ETA stamp from upstream stacking (or missing/assumed).
    if "ETAAPPL" in hdr0:
        ph["ETAAPPL"] = hdr0["ETAAPPL"]
    if "ETAPATH" in hdr0:
        ph["ETAPATH"] = hdr0["ETAPATH"]

    # Upstream degradation flags (P0-L)
    if "QADEGRD" in hdr0:
        ph["QADEGRD"] = hdr0["QADEGRD"]
    for _k in ("SKYOK", "SKYMD"):
        if _k in hdr0:
            ph[_k] = hdr0[_k]

    # Optional provenance links (done.json of upstream stages)
    for k in list(hdr0.keys()):
        if str(k).startswith("PROV"):
            ph[k] = hdr0[k]

    # Explain the 2×N preview images.
    ph["ROW0"] = ("TRACE", "Row 0 in FLUX2/VAR2/MASK2/NPIX2/SKY2")
    ph["ROW1"] = ("FIXED", "Row 1 in FLUX2/VAR2/MASK2/NPIX2/SKY2")

    ph = add_provenance(ph, cfg, stage="12_extract")

    # ---------------------------- table HDUs ----------------------------
    # Sanitize non-finite values: spec1d contract requires finite FLUX/VAR.
    # We map NaN/Inf to 0 and mark NO_COVERAGE in the corresponding MASK.
    def _sanitize_1d(flux, var, mask):
        flux = np.asarray(flux)
        var = np.asarray(var)
        mask = np.asarray(mask)
        bad = (~np.isfinite(flux)) | (~np.isfinite(var))
        if np.any(bad):
            flux = flux.copy(); var = var.copy(); mask = mask.copy()
            flux[bad] = 0.0
            var[bad] = 0.0
            mask[bad] = mask[bad] | int(maskbits.NO_COVERAGE)
        return flux, var, mask, int(np.count_nonzero(bad))

    flux_trace, var_trace, mask_trace, _nb1 = _sanitize_1d(flux_trace, var_trace, mask_trace)
    flux_fixed, var_fixed, mask_fixed, _nb2 = _sanitize_1d(flux_fixed, var_fixed, mask_fixed)
    if (_nb1 + _nb2) > 0:
        ph["SCORPNAN"] = (_nb1 + _nb2, "Non-finite FLUX/VAR sanitized to 0 (NO_COVERAGE)")
    cols_trace = [
        fits.Column(name="LAMBDA", format="D", array=np.asarray(wave, dtype=np.float64)),
        fits.Column(name="FLUX_TRACE", format="E", array=np.asarray(flux_trace, dtype=np.float32)),
        fits.Column(name="VAR_TRACE", format="E", array=np.asarray(var_trace, dtype=np.float32)),
        fits.Column(name="MASK_TRACE", format="J", array=np.asarray(mask_trace, dtype=np.int32)),
        fits.Column(name="SKY_TRACE", format="E", array=np.asarray(sky_trace, dtype=np.float32)),
        fits.Column(name="NPIX_TRACE", format="I", array=np.asarray(npix_trace, dtype=np.int16)),
    ]
    ht = fits.BinTableHDU.from_columns(cols_trace, name="SPEC_TRACE")
    ht.header["WUNIT"] = (hdr0.get("CUNIT1", "Angstrom"), "Wavelength unit")
    ht.header["FUNIT"] = (hdr0.get("BUNIT", "ADU"), "Flux unit")
    ht.header["APHW"] = (int(trace.aperture_half_width), "Aperture half-width (pix)")

    cols_fixed = [
        fits.Column(name="LAMBDA", format="D", array=np.asarray(wave, dtype=np.float64)),
        fits.Column(name="FLUX_FIXED", format="E", array=np.asarray(flux_fixed, dtype=np.float32)),
        fits.Column(name="VAR_FIXED", format="E", array=np.asarray(var_fixed, dtype=np.float32)),
        fits.Column(name="MASK_FIXED", format="J", array=np.asarray(mask_fixed, dtype=np.int32)),
        fits.Column(name="SKY_FIXED", format="E", array=np.asarray(sky_fixed, dtype=np.float32)),
        fits.Column(name="NPIX_FIXED", format="I", array=np.asarray(npix_fixed, dtype=np.int16)),
    ]
    hf = fits.BinTableHDU.from_columns(cols_fixed, name="SPEC_FIXED")
    hf.header["WUNIT"] = (hdr0.get("CUNIT1", "Angstrom"), "Wavelength unit")
    hf.header["FUNIT"] = (hdr0.get("BUNIT", "ADU"), "Flux unit")
    # FITS headers do not allow NaN/Inf keyword values. When the fixed center
    # is unavailable, store the keyword as undefined rather than crashing.
    if trace.y_fixed.size:
        try:
            yfix = float(trace.y_fixed[0])
        except Exception:
            yfix = None
        if yfix is not None and np.isfinite(yfix):
            hf.header["YFIX"] = (yfix, "Fixed center y (pix)")
        else:
            hf.header["YFIX"] = (None, "Fixed center y (pix)")
    else:
        hf.header["YFIX"] = (None, "Fixed center y (pix)")
    hf.header["APHW"] = (int(trace.aperture_half_width), "Aperture half-width (pix)")

    # --------------------------- preview images --------------------------
    def _img(name: str, arr: np.ndarray, dtype) -> fits.ImageHDU:
        return fits.ImageHDU(data=np.asarray(arr, dtype=dtype), name=name)

    flux2 = np.vstack([flux_trace, flux_fixed])
    var2 = np.vstack([var_trace, var_fixed])
    mask2 = np.vstack([mask_trace, mask_fixed]).astype(np.int32)
    npix2 = np.vstack([npix_trace, npix_fixed]).astype(np.int16)
    sky2 = np.vstack([sky_trace, sky_fixed])

    hdus: list[fits.HDUBase] = [
        fits.PrimaryHDU(header=ph),
        ht,
        hf,
        _img("FLUX2", flux2, np.float32),
        _img("VAR2", var2, np.float32),
        _img("MASK2", mask2, np.int32),
        _img("NPIX2", npix2, np.int16),
        _img("SKY2", sky2, np.float32),
    ]

    fits.HDUList(hdus).writeto(out_path, overwrite=True)


def _write_quicklook_png(
    out_png: Path,
    *,
    wave: np.ndarray,
    flux_trace: np.ndarray,
    flux_fixed: np.ndarray,
    y_trace: np.ndarray,
    y_fixed: np.ndarray,
    ap_hw: int,
    roi: dict[str, Any],
    sky_windows: list[tuple[int, int]] | None,
    title: str,
) -> None:
    """Write a quicklook PNG with spectra + geometry/trace diagnostics."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_png.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(10, 6), dpi=150)
        ax1 = fig.add_subplot(211)
        ax1.plot(wave, flux_trace, label="TRACE")
        ax1.plot(wave, flux_fixed, label="FIXED", alpha=0.85)
        ax1.set_xlabel("Wavelength")
        ax1.set_ylabel("Flux")
        ax1.set_title(title)
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="best")

        ax2 = fig.add_subplot(212)
        ax2.plot(wave, y_trace, label="y_trace")
        ax2.plot(wave, y_fixed, label="y_fixed", alpha=0.85)
        if ap_hw is not None and int(ap_hw) > 0:
            ap = float(int(ap_hw))
            ax2.fill_between(wave, y_trace - ap, y_trace + ap, alpha=0.15, label="TRACE ± ap")

        oy0, oy1 = roi.get("obj_y0"), roi.get("obj_y1")
        if oy0 is not None and oy1 is not None:
            ax2.axhspan(float(oy0), float(oy1), alpha=0.08, label="obj ROI")
        if sky_windows:
            for i, (a, b) in enumerate(sky_windows):
                ax2.axhspan(float(a), float(b), alpha=0.05, label="sky" if i == 0 else None)

        ax2.set_xlabel("Wavelength")
        ax2.set_ylabel("Y (pix)")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="best", ncol=3, fontsize=8)

        fig.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)
    except Exception as e:
        log.warning("Failed to write spec1d.png: %s", e)


# ---------------------------------- API ----------------------------------


def _run_extract1d_impl(
    cfg: Dict[str, Any],
    *,
    in_fits: Optional[Path] = None,
    stacked_fits: Optional[Path] = None,  # legacy alias
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    ecfg = cfg.get("extract1d", {}) if isinstance(cfg.get("extract1d"), dict) else {}
    if not bool(ecfg.get("enabled", True)):
        return {"skipped": True, "reason": "extract1d.enabled=false"}

    # Backward-compatible alias used in older code/tests.
    if in_fits is None and stacked_fits is not None:
        in_fits = stacked_fits

    from scorpio_pipe.workspace_paths import stage_dir

    wd = resolve_work_dir(cfg)
    out_dir = Path(out_dir) if out_dir is not None else stage_dir(wd, "extract")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------- resolve input (explicit) -----------------------
    mode = str(ecfg.get("input_mode", "stack2d")).strip().lower()
    # Deprecated compatibility knob.
    if bool(ecfg.get("allow_sky_fallback", False)) and mode in {"stack2d", "stack"}:
        mode = "single_frame"

    if in_fits is None:
        if mode in {"stack2d", "stack"}:
            cand = stage_dir(wd, "stack2d") / "stack2d.fits"
            if not cand.exists():
                cand = stage_dir(wd, "stack2d") / "stacked2d.fits"
            if not cand.exists():
                raise FileNotFoundError(
                    "Stack2D products not found. Run the Stack2D stage first, or set "
                    "extract1d.input_mode=single_frame and choose a linearized skysub frame. "
                    f"Tried: {cand}"
                )
            in_fits = cand
        elif mode in {"single_frame", "single"}:
            stem = ecfg.get("single_frame_stem") or ecfg.get("stem")
            if isinstance(ecfg.get("single_frame_path"), str) and ecfg.get("single_frame_path"):
                in_fits = Path(str(ecfg["single_frame_path"]))
            elif isinstance(stem, str) and stem.strip():
                in_fits = stage_dir(wd, "linearize") / f"{stem.strip()}_skysub.fits"
            else:
                raise ValueError(
                    "extract1d.input_mode=single_frame requires extract1d.single_frame_stem (or single_frame_path)."
                )
            if not Path(in_fits).exists():
                raise FileNotFoundError(f"Single-frame input not found: {in_fits}")
        else:
            raise ValueError(f"Unknown extract1d.input_mode: {mode!r}")

    in_fits = Path(in_fits)
    if not in_fits.exists():
        raise FileNotFoundError(f"No 2D input for extraction: {in_fits}")

    sci, var, mask, hdr0 = _read_mef(in_fits)
    if sci.ndim != 2:
        raise ValueError(f"extract1d expects a 2D (y,λ) frame; got {sci.shape} from {in_fits}")
    ny, nlam = sci.shape
    wave = _linear_wave_axis(hdr0, nlam)

    # ---------------- upstream sky degradation propagation (P0-L) ----------------
    from scorpio_pipe.maskbits import SKYMODEL_FAIL

    in_qadegrd = int(hdr0.get("QADEGRD", 0) or 0) == 1
    try:
        has_skyfail = bool((((np.asarray(mask, dtype=np.uint16)) & np.uint16(SKYMODEL_FAIL)) != 0).any())
    except Exception:
        has_skyfail = False
    upstream_sky_passthrough = bool(in_qadegrd or has_skyfail)
    if upstream_sky_passthrough and not in_qadegrd:
        # Raise downstream degradation even if upstream header lacked QADEGRD.
        hdr0["QADEGRD"] = (1, "Upstream degraded (SKYMODEL_FAIL present)")
        try:
            hdr0.add_history("Downstream: QADEGRD raised due to SKYMODEL_FAIL in input MASK.")
        except Exception:
            pass

    # ------------------------------ ETA stamp ------------------------------
    eta_appl = hdr0.get("ETAAPPL", None)
    is_stack = _is_stack2d_input(in_fits, hdr0)
    eta_warn = None
    if is_stack and eta_appl is None:
        # This is intentionally strict: stack2d must stamp ETAAPPL.
        raise ValueError(
            "Input looks like Stack2D but header lacks ETAAPPL. "
            "Update/regen Stack2D products so downstream stages can interpret VAR correctly."
        )
    if eta_appl is None:
        eta_appl = False
        eta_warn = "ETAAPPL missing in input header; assuming False."
    eta_appl = bool(eta_appl)
    # ------------------------------- geometry ------------------------------
    roi = _roi_from_cfg(cfg)
    # Allow optional per-stage override (advanced).
    if isinstance(ecfg.get("roi"), dict) and ecfg.get("roi"):
        roi.update({k: v for k, v in ecfg.get("roi", {}).items() if v is not None})

    from scorpio_pipe.maskbits import BADPIX, COSMIC, NO_COVERAGE, REJECTED, SATURATED, USER
    fatal_bits = int(NO_COVERAGE | BADPIX | COSMIC | SATURATED | USER | REJECTED)

    sky_windows: list[tuple[int, int]] | None = None
    geo_meta: dict[str, Any] = {}
    try:
        from scorpio_pipe.sky_geometry import compute_sky_geometry, roi_from_cfg

        # Build a temporary cfg view with the merged ROI so roi_from_cfg sees it.
        cfg_tmp = dict(cfg)
        sky_tmp = cfg.get("sky") if isinstance(cfg.get("sky"), dict) else {}
        sky_tmp = dict(sky_tmp)
        sky_tmp["roi"] = dict(roi)
        cfg_tmp["sky"] = sky_tmp

        roi_sel = roi_from_cfg(cfg_tmp)
        g = compute_sky_geometry(
            sci,
            var,
            mask,
            roi=roi_sel,
            roi_policy="prefer_user" if roi_sel is not None else "auto",
            fatal_bits=fatal_bits,
        )

        sky_windows = list(g.sky_windows) if g.sky_windows else []
        geo_meta = {
            "roi_used": g.roi_used,
            "object_spans": g.object_spans,
            "sky_windows": g.sky_windows,
            "metrics": g.metrics,
        }

        # If ROI object band missing, adopt the auto estimate for tracing/aperture.
        if (roi.get("obj_y0") is None or roi.get("obj_y1") is None) and g.object_spans:
            y0, y1 = g.object_spans[0]
            roi["obj_y0"], roi["obj_y1"] = int(y0), int(y1)
            roi["_auto_obj_roi"] = True

    except Exception as e:
        geo_meta = {"warning": f"sky_geometry_failed: {e}"}
        sky_windows = []

    # ------------------------------ trace model ----------------------------
    trace = _build_trace_model(
        sci,
        var,
        mask,
        wave,
        roi=roi,
        sky_windows=sky_windows,
        cfg=ecfg,
        fatal_bits=fatal_bits,
    )

    # ----------------------------- sky residual ----------------------------
    sky_resid = _estimate_sky_residual(
        sci, var, mask, sky_windows=sky_windows, fatal_bits=fatal_bits
    )
    apply_local_sky = bool(ecfg.get("local_sky_correction", False))

    # ------------------------------ extraction -----------------------------
    # Accept both modern 'method' and legacy 'mode' keys (UI historically used 'mode').
    method = str(ecfg.get("method", ecfg.get("mode", "boxcar"))).strip().lower()
    if method == "sum":
        method = "boxcar"
    if method in {"horne", "optimal_extraction"}:
        method = "optimal"
    if method not in {"boxcar", "mean", "optimal"}:
        method = "boxcar"

    min_good_frac = float(ecfg.get("min_good_frac", 0.6))

    opt_meta: dict[str, Any] = {}
    if method == "optimal":
        ocfg = ecfg.get("optimal") if isinstance(ecfg.get("optimal"), dict) else {}
        smooth_sig = float(ocfg.get("profile_smooth_lambda_sigma", 3.0))
        clip_neg = bool(ocfg.get("clip_negative_profile", True))
        min_ps = float(ocfg.get("min_profile_sum", 1e-6))
        flux_tr, var_tr, mask_tr, sky_tr, npix_tr, m_tr = _optimal_extract(
            sci,
            var,
            mask,
            trace.y_trace,
            ap_hw=trace.aperture_half_width,
            fatal_bits=fatal_bits,
            min_good_frac=min_good_frac,
            sky_resid=sky_resid,
            apply_local_sky=apply_local_sky,
            smooth_lambda_sigma=smooth_sig,
            clip_negative_profile=clip_neg,
            min_profile_sum=min_ps,
        )
        flux_fx, var_fx, mask_fx, sky_fx, npix_fx, m_fx = _optimal_extract(
            sci,
            var,
            mask,
            trace.y_fixed,
            ap_hw=trace.aperture_half_width,
            fatal_bits=fatal_bits,
            min_good_frac=min_good_frac,
            sky_resid=sky_resid,
            apply_local_sky=apply_local_sky,
            smooth_lambda_sigma=smooth_sig,
            clip_negative_profile=clip_neg,
            min_profile_sum=min_ps,
        )
        opt_meta = {"trace": m_tr, "fixed": m_fx}
    else:
        flux_tr, var_tr, mask_tr, sky_tr, npix_tr = _boxcar_extract(
            sci,
            var,
            mask,
            trace.y_trace,
            ap_hw=trace.aperture_half_width,
            fatal_bits=fatal_bits,
            min_good_frac=min_good_frac,
            sky_resid=sky_resid,
            apply_local_sky=apply_local_sky,
        )
        flux_fx, var_fx, mask_fx, sky_fx, npix_fx = _boxcar_extract(
            sci,
            var,
            mask,
            trace.y_fixed,
            ap_hw=trace.aperture_half_width,
            fatal_bits=fatal_bits,
            min_good_frac=min_good_frac,
            sky_resid=sky_resid,
            apply_local_sky=apply_local_sky,
        )
    if method == "mean":
        # Convert sum → mean (and propagate VAR scaling).
        n_eff_tr = np.maximum(np.asarray(npix_tr, dtype=float), 1.0)
        n_eff_fx = np.maximum(np.asarray(npix_fx, dtype=float), 1.0)
        flux_tr = flux_tr / n_eff_tr
        flux_fx = flux_fx / n_eff_fx
        var_tr = var_tr / (n_eff_tr**2)
        var_fx = var_fx / (n_eff_fx**2)

    # ------------------------------- outputs ------------------------------
    out_fits = out_dir / "spec1d.fits"
    out_png = out_dir / "spec1d.png"
    trace_json = out_dir / "trace.json"
    done_json = out_dir / "extract_done.json"
    done_legacy = out_dir / "extract1d_done.json"  # legacy alias

    # Header: preserve flux units; never apply eta here.
    ohdr = fits.Header()
    for k in ("CRVAL1", "CDELT1", "CD1_1", "CRPIX1", "CTYPE1", "CUNIT1", "BUNIT"):
        if k in hdr0:
            ohdr[k] = hdr0[k]

    # Upstream degradation propagation (P0-L): keep these in the 1D primary header.
    for k in ("QADEGRD", "SKYOK", "SKYMD"):
        if k in hdr0:
            ohdr[k] = hdr0[k]

    # Carry any upstream provenance links if present.
    for k in list(hdr0.keys()):
        if str(k).startswith("PROV"):
            ohdr[k] = hdr0[k]
    ohdr["ETAAPPL"] = (bool(eta_appl), "eta(lambda) applied to VAR in upstream stacking")
    if "ETAPATH" in hdr0:
        ohdr["ETAPATH"] = hdr0["ETAPATH"]
    ohdr = add_provenance(ohdr, cfg, stage="12_extract")

    # Extra self-describing header cards for the 1D product (per P1-F contract).
    ohdr["PIPEVER"] = (str(PIPELINE_VERSION), "Pipeline version")
    ohdr["STAGE"] = ("12_extract", "Pipeline stage (directory name)")
    try:
        ohdr["INPUT2D"] = str(in_fits.relative_to(wd))
    except Exception:
        ohdr["INPUT2D"] = str(in_fits)

    # Link upstream done.json files when available (for quick provenance browsing).
    prov = []
    try:
        from scorpio_pipe.workspace_paths import stage_dir
        for st, fn in [("stack2d", "stack2d_done.json"), ("sky", "sky_done.json"), ("linearize", "linearize_done.json")]:
            p = stage_dir(wd, st) / fn
            if p.exists():
                try:
                    prov.append(str(p.relative_to(wd)))
                except Exception:
                    prov.append(str(p))
    except Exception:
        prov = []
    for i, pp in enumerate(prov[:9]):
        ohdr[f"PROV{i}"] = (pp, "Upstream done.json")

    _write_spec1d_fits(
        out_fits,
        cfg=cfg,
        hdr0=ohdr,
        wave=wave,
        trace=trace,
        flux_trace=flux_tr,
        var_trace=var_tr,
        mask_trace=mask_tr,
        sky_trace=sky_tr,
        npix_trace=npix_tr,
        flux_fixed=flux_fx,
        var_fixed=var_fx,
        mask_fixed=mask_fx,
        sky_fixed=sky_fx,
        npix_fixed=npix_fx,
    )
    # trace.json
    # Provide stable, explicit metadata for reproducibility and QC.
    trace_model_meta = trace.meta.get("trace_model", {}) if isinstance(trace.meta, dict) else {}
    aperture_meta = trace.meta.get("aperture", {}) if isinstance(trace.meta, dict) else {}

    roi_used_meta = {}
    if isinstance(geo_meta.get("roi_used"), dict):
        roi_used_meta = geo_meta.get("roi_used")

    trace_method = "fallback_fixed" if str(trace_model_meta.get("type")) in {"fallback_constant", "fallback"} else "centroid_spline"

    trace_payload = {
        "stage": "extract1d",
        "pipeline_version": str(PIPELINE_VERSION),
        "input": {
            "fits": str(in_fits),
            "mode": mode,
            "shape": [int(ny), int(nlam)],
        },
        "roi_used": _safe_json(roi_used_meta) if roi_used_meta else _safe_json(roi),
        "sky_windows_used": _safe_json(sky_windows),
        "eta": {
            "eta_applied_in_input": bool(eta_appl),
            "warning": eta_warn,
        },
        "trace_model": {
            "method": trace_method,
            "meta": _safe_json(trace_model_meta),
            "blocks_total": int(trace.meta.get("blocks_total", 0)),
            "blocks_valid": int(trace.meta.get("blocks_valid", 0)),
        },
        "aperture": _safe_json(aperture_meta),
        "y_trace": trace.y_trace.tolist(),
        "y_fixed": trace.y_fixed.tolist(),
        "geometry": _safe_json(geo_meta),
    }
    trace_json.write_text(json.dumps(trace_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # extract_done.json
    # Minimal but informative metrics for QC and downstream automation.
    def _frac_finite(x: np.ndarray) -> float:
        x = np.asarray(x)
        return float(np.isfinite(x).sum() / max(1, x.size))

    def _robust_sigma(x: np.ndarray) -> float | None:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 8:
            return None
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        if not (mad > 0):
            return None
        return float(1.4826 * mad)

    # Stage-level QC flags (P2 schema)
    from scorpio_pipe.qc.flags import make_flag, max_severity

    flag_codes: list[str] = []
    # geometry flags from compute_sky_geometry (if any)
    try:
        gf = geo_meta.get("metrics", {}).get("flags", []) if isinstance(geo_meta.get("metrics"), dict) else []
        for f in gf:
            if isinstance(f, dict) and f.get("code"):
                flag_codes.append(str(f.get("code")))
    except Exception:
        pass

    if not sky_windows:
        flag_codes.append("NO_SKY_WINDOWS")
    if trace_method == "fallback_fixed":
        flag_codes.append("TRACE_FALLBACK_USED")

    if upstream_sky_passthrough:
        flag_codes.append("UPSTREAM_SKY_PASSTHROUGH")

    # 1D mask coverage diagnostics
    from scorpio_pipe.maskbits import NO_COVERAGE
    mask_no_cov_tr = float(((np.asarray(mask_tr, dtype=int) & int(NO_COVERAGE)) != 0).mean())
    mask_no_cov_fx = float(((np.asarray(mask_fx, dtype=int) & int(NO_COVERAGE)) != 0).mean())
    if max(mask_no_cov_tr, mask_no_cov_fx) > 0.05:
        flag_codes.append("LOW_COVERAGE_1D")

    # Materialize P2 flags.
    _flag_meta = {
        "NO_SKY_WINDOWS": (
            "WARN",
            "No sky windows were available; sky residual QC is unreliable.",
            "Define sky bands (ROI) away from the object, or widen them.",
        ),
        "TRACE_FALLBACK_USED": (
            "INFO",
            "Trace model fell back to a fixed aperture center.",
            "Check object visibility and the trace window; consider a wider ROI.",
        ),
        "LOW_COVERAGE_1D": (
            "WARN",
            "Large fraction of 1D spectrum has NO_COVERAGE mask.",
            "Verify slit illumination/coverage and earlier stages (linearize/stack).",
        ),
        "UPSTREAM_SKY_PASSTHROUGH": (
            "WARN",
            "Sky subtraction was skipped upstream (pass-through product).",
            "Inspect SKYMODEL_FAIL/QADEGRD; consider excluding affected exposures in stack2d.",
        ),
    }
    stage_flags = []
    for c in sorted(set(flag_codes)):
        sev, msg, hint = _flag_meta.get(c, ("WARN", c, ""))
        stage_flags.append(make_flag(c, sev, msg, hint=hint))

    qc = {"flags": stage_flags, "max_severity": max_severity(stage_flags)}

    from scorpio_pipe.io.done_json import write_done_json

    done = write_done_json(
        stage="extract",
        stage_dir=out_dir,
        status="ok",
        inputs={
            "fits": str(in_fits),
            "mode": mode,
            "shape": [int(ny), int(nlam)],
        },
        params={
            "eta_applied_in_input": bool(eta_appl),
            "eta_warning": eta_warn,
            "trace_method": trace_method,
            "aperture_half_width": int(trace.aperture_half_width),
        },
        outputs={
            "spec1d_fits": str(out_fits),
            "trace_json": str(trace_json),
            "spec1d_png": str(out_png) if bool(ecfg.get("save_png", True)) else None,
        },
        metrics={
            "trace_frac_finite": _frac_finite(flux_tr),
            "fixed_frac_finite": _frac_finite(flux_fx),
            "trace_npix_median": float(np.nanmedian(npix_tr)),
            "fixed_npix_median": float(np.nanmedian(npix_fx)),
            "trace_no_coverage_frac": mask_no_cov_tr,
            "fixed_no_coverage_frac": mask_no_cov_fx,
            "sky_residual_robust_sigma": _robust_sigma(sky_resid),
        },
        flags=stage_flags,
        qc=qc,
        extra={
            "ok": True,
            "pipeline_version": str(PIPELINE_VERSION),
            "trace": {
                "method": trace_method,
                "aperture_half_width": int(trace.aperture_half_width),
                "y_fixed": float(trace.y_fixed[0]) if trace.y_fixed.size else None,
                "model_meta": _safe_json(trace_model_meta),
                "blocks_total": int(trace.meta.get("blocks_total", 0)),
                "blocks_valid": int(trace.meta.get("blocks_valid", 0)),
                "valid_frac": float(trace.meta.get("blocks_valid", 0))
                / max(1.0, float(trace.meta.get("blocks_total", 0))),
            },
            "extract": {
                "method": method,
                "input_mode": mode,
                "snr_trace_median": (
                    float(np.nanmedian(np.where(np.isfinite(var_tr) & (var_tr > 0), flux_tr / np.sqrt(var_tr), np.nan)))
                    if flux_tr.size else None
                ),
                "snr_fixed_median": (
                    float(np.nanmedian(np.where(np.isfinite(var_fx) & (var_fx > 0), flux_fx / np.sqrt(var_fx), np.nan)))
                    if flux_fx.size else None
                ),
                "optimal_meta": _safe_json(opt_meta) if opt_meta else None,
            },
            "warnings": [w for w in [eta_warn, geo_meta.get("warning")] if w],
            # Backward-compatible code list
            "flag_codes": sorted(set(flag_codes)),
        },
        legacy_paths=[done_json, done_legacy],
    )

    # Optional quicklook PNG
    if bool(ecfg.get("save_png", True)):
        _write_quicklook_png(
            out_png,
            wave=wave,
            flux_trace=flux_tr,
            flux_fixed=flux_fx,
            y_trace=trace.y_trace,
            y_fixed=trace.y_fixed,
            ap_hw=int(trace.aperture_half_width),
            roi=roi,
            sky_windows=sky_windows,
            title=(
                f"Spec1D — {Path(in_fits).name} | ap=±{int(trace.aperture_half_width)}px | {trace_method}"
            ),
        )

    # Legacy products directory copy (older UI/scripts).
    try:
        legacy = wd / "products" / "spec"
        legacy.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_fits, legacy / "spec1d.fits")
        if trace_json.exists():
            shutil.copy2(trace_json, legacy / "trace.json")
        shutil.copy2(done_json, legacy / "extract_done.json")
        shutil.copy2(done_json, legacy / "extract1d_done.json")
        if out_png.exists():
            shutil.copy2(out_png, legacy / "spec1d.png")
    except Exception:
        pass

    return {
        "ok": True,
        "input": str(in_fits),
        "spec1d_fits": str(out_fits),
        "trace_json": str(trace_json),
        "extract_done": str(done_json),
        "spec1d_png": str(out_png) if out_png.exists() else None,
        "etaappl": bool(eta_appl),
        "products": ["TRACE", "FIXED"],
    }


def run_extract1d(
    cfg: Dict[str, Any],
    *,
    in_fits: Optional[Path] = None,
    stacked_fits: Optional[Path] = None,  # legacy alias
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Extraction stage with P2 reliability guard.

    Guarantees that ``done.json`` is written even if the stage fails.
    """

    try:
        return _run_extract1d_impl(
            cfg,
            in_fits=in_fits,
            stacked_fits=stacked_fits,
            out_dir=out_dir,
        )
    except Exception as e:
        from datetime import datetime, timezone

        from scorpio_pipe.io.done_json import write_done_json
        from scorpio_pipe.qc.flags import make_flag, max_severity
        from scorpio_pipe.workspace_paths import stage_dir

        wd = resolve_work_dir(cfg)
        od = out_dir or stage_dir(wd, "extract")
        od.mkdir(parents=True, exist_ok=True)

        in_path = in_fits or stacked_fits
        flags = [
            make_flag(
                "STAGE_FAILED",
                "ERROR",
                f"{type(e).__name__}: {e}",
                hint="See traceback/log for the exact failure location.",
            )
        ]
        qc = {"flags": flags, "max_severity": max_severity(flags)}

        write_done_json(
            stage="extract",
            stage_dir=od,
            status="fail",
            inputs={"fits": str(in_path) if in_path else None},
            params={"extract1d": cfg.get("extract1d", {})},
            outputs={},
            metrics={},
            flags=flags,
            qc=qc,
            error={
                "type": type(e).__name__,
                "message": str(e),
                "repr": repr(e),
            },
            extra={
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "ok": False,
            },
            legacy_paths=[od / "extract_done.json", od / "extract1d_done.json"],
        )

        raise
