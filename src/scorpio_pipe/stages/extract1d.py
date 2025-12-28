"""1D extraction for long-slit products in (λ, y).

v5.14: this stage is no longer a placeholder. It supports:
  - boxcar extraction around an estimated spatial trace
  - (basic) Horne-style optimal extraction using a single profile template

Inputs
------
Prefer stacked 2D product:
  work_dir/products/stack/stacked2d.fits

Fallback:
  work_dir/products/sky/obj_sky_sub.fits (or legacy work_dir/sky/obj_sky_sub.fits)

The input is expected to be an MEF with extensions:
  - PRIMARY (SCI)
  - VAR (optional)
  - MASK (optional, uint16)

Outputs
-------
products/spec/spec1d.fits (PRIMARY=FLUX, EXT=VAR, MASK)
products/spec/spec1d.png
products/spec/trace.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from astropy.io import fits

from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.provenance import add_provenance
from scorpio_pipe.io.mef import try_read_grid, write_sci_var_mask

log = logging.getLogger(__name__)


def _roi_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    sky = cfg.get("sky") if isinstance(cfg.get("sky"), dict) else {}
    roi = sky.get("roi") if isinstance(sky.get("roi"), dict) else {}
    return dict(roi)


def _read_mef(
    path: Path,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], fits.Header]:
    """Read a MEF product.

    Historically some stages wrote the science image into the PRIMARY HDU,
    while newer stages use EXTNAME=SCI (with PRIMARY header carrying
    provenance/WCS). We support both.
    """

    with fits.open(path, memmap=False) as hdul:
        hdr = hdul[0].header.copy()

        sci = hdul[0].data
        # If PRIMARY has no data (common for our MEF writer), NumPy would turn
        # it into a 0-d array (shape=()) which breaks downstream expectations.
        if sci is None or np.asarray(sci).ndim < 2:
            if "SCI" in hdul and hdul["SCI"].data is not None:
                sci = hdul["SCI"].data
                # Ensure wavelength WCS keywords are available in hdr for
                # _linear_wave_axis(). Prefer existing primary cards.
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
    # Common WCS keywords for linear 1D axis.
    crval = hdr.get("CRVAL1")
    cdelt = hdr.get("CDELT1", hdr.get("CD1_1"))
    crpix = hdr.get("CRPIX1", 1.0)
    if crval is None or cdelt is None:
        return np.arange(nlam, dtype=float)
    i = np.arange(nlam, dtype=float)
    return float(crval) + (i + 1.0 - float(crpix)) * float(cdelt)


def _polyfit_trace(
    wbin: np.ndarray, ycen: np.ndarray, w_full: np.ndarray, deg: int
) -> np.ndarray:
    good = np.isfinite(wbin) & np.isfinite(ycen)
    if good.sum() < max(5, deg + 2):
        # fallback: constant trace at median
        y0 = (
            float(np.nanmedian(ycen))
            if np.isfinite(np.nanmedian(ycen))
            else (w_full * 0 + 0)
        )
        return np.full_like(w_full, y0, dtype=float)
    w = wbin[good]
    y = ycen[good]
    # normalize λ for numerical stability
    w0 = float(np.nanmin(w))
    w1 = float(np.nanmax(w))
    x = (w - w0) / max(w1 - w0, 1e-6)
    p = np.polyfit(x, y, deg=deg)
    xfull = (w_full - w0) / max(w1 - w0, 1e-6)
    return np.polyval(p, xfull)


def _estimate_trace(
    sci: np.ndarray,
    var: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    wave: np.ndarray,
    roi: dict[str, Any],
    *,
    trace_bin_A: float,
    trace_smooth_deg: int,
) -> Tuple[np.ndarray, dict[str, Any]]:
    ny, nlam = sci.shape
    # object window
    yobj0 = roi.get("obj_y0")
    yobj1 = roi.get("obj_y1")
    if yobj0 is None or yobj1 is None:
        yobj0, yobj1 = int(0.35 * ny), int(0.65 * ny)
    yobj0 = int(np.clip(yobj0, 0, ny - 1))
    yobj1 = int(np.clip(yobj1, yobj0 + 1, ny))

    # Bin width in pixels
    dl = float(np.nanmedian(np.abs(np.diff(wave)))) if len(wave) > 1 else 1.0
    bin_pix = max(1, int(round(trace_bin_A / max(dl, 1e-6))))

    wbin = []
    ycen = []
    for x0 in range(0, nlam, bin_pix):
        x1 = min(nlam, x0 + bin_pix)
        block = sci[:, x0:x1]
        if mask is not None:
            m = mask[:, x0:x1] > 0
            block = block.copy()
            block[m] = np.nan
        # collapse in λ
        prof = np.nanmean(block, axis=1)
        # focus on object window
        sl = prof[yobj0:yobj1]
        if not np.isfinite(sl).any():
            wbin.append(float(np.nanmean(wave[x0:x1])))
            ycen.append(np.nan)
            continue
        # use positive weights to avoid centroid flips in noisy sky-sub data
        w = np.nan_to_num(sl, nan=0.0)
        w = np.clip(
            w,
            0.0,
            np.nanpercentile(w, 95) if np.isfinite(np.nanpercentile(w, 95)) else np.inf,
        )
        if np.sum(w) <= 0:
            # fallback: argmax
            yc = float(np.nanargmax(sl) + yobj0)
        else:
            yy = np.arange(yobj0, yobj1, dtype=float)
            yc = float(np.sum(yy * w) / np.sum(w))
        wbin.append(float(np.nanmean(wave[x0:x1])))
        ycen.append(yc)

    wbin = np.asarray(wbin, dtype=float)
    ycen = np.asarray(ycen, dtype=float)
    trace = _polyfit_trace(wbin, ycen, wave, deg=int(np.clip(trace_smooth_deg, 0, 8)))

    meta = {
        "ny": int(ny),
        "nlam": int(nlam),
        "yobj0": int(yobj0),
        "yobj1": int(yobj1),
        "bin_pix": int(bin_pix),
        "trace_smooth_deg": int(trace_smooth_deg),
        "centroids": {
            "wave": wbin.tolist(),
            "y": ycen.tolist(),
        },
    }
    return trace.astype(float), meta


def _boxcar_extract(
    sci: np.ndarray,
    var: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    trace: np.ndarray,
    *,
    ap_hw: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ny, nlam = sci.shape
    ap_hw = int(max(1, ap_hw))
    flux = np.zeros(nlam, dtype=float)
    out_var = np.zeros(nlam, dtype=float)
    out_mask = np.zeros(nlam, dtype=np.uint16)
    have_var = var is not None and var.shape == sci.shape
    have_mask = mask is not None and mask.shape == sci.shape
    for j in range(nlam):
        yc = float(trace[j])
        y0 = int(np.floor(yc)) - ap_hw
        y1 = int(np.floor(yc)) + ap_hw + 1
        y0 = max(0, y0)
        y1 = min(ny, y1)
        sl = sci[y0:y1, j]
        if have_mask:
            m = mask[y0:y1, j]
            good = m == 0
            if not np.any(good):
                out_mask[j] = np.uint16(1)
                flux[j] = np.nan
                out_var[j] = np.inf
                continue
            sl = sl[good]
            if have_var:
                vv = var[y0:y1, j][good]
            else:
                vv = None
        else:
            vv = var[y0:y1, j] if have_var else None

        flux[j] = float(np.nansum(sl))
        if vv is not None:
            out_var[j] = float(np.nansum(vv))
        else:
            out_var[j] = float(np.nanvar(sl)) if np.isfinite(np.nanvar(sl)) else np.inf
    return flux, out_var, out_mask


def _build_profile_template(
    sci: np.ndarray,
    mask: Optional[np.ndarray],
    trace: np.ndarray,
    *,
    profile_hw: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a single spatial profile template P0(dy).

    We sample data at y = trace(λ) + dy using linear interpolation and
    average over λ. This avoids explicit resampling of the 2D frame.
    """

    ny, nlam = sci.shape
    profile_hw = int(max(3, profile_hw))
    dys = np.arange(-profile_hw, profile_hw + 1, dtype=float)
    acc = np.zeros_like(dys, dtype=float)
    cnt = np.zeros_like(dys, dtype=float)
    have_mask = mask is not None and mask.shape == sci.shape
    for j in range(nlam):
        yc = float(trace[j])
        if not np.isfinite(yc):
            continue
        for k, dy in enumerate(dys):
            y = yc + float(dy)
            if y < 0 or y > ny - 1:
                continue
            y0 = int(np.floor(y))
            y1 = min(ny - 1, y0 + 1)
            t = y - y0
            if have_mask:
                if mask[y0, j] != 0 or mask[y1, j] != 0:
                    continue
            v = (1 - t) * sci[y0, j] + t * sci[y1, j]
            if np.isfinite(v):
                acc[k] += float(v)
                cnt[k] += 1.0
    prof = np.zeros_like(acc)
    good = cnt > 0
    prof[good] = acc[good] / cnt[good]
    # keep only non-negative template and normalize
    prof = np.clip(
        prof, 0.0, np.nanmax(prof) if np.isfinite(np.nanmax(prof)) else np.inf
    )
    s = float(np.nansum(prof))
    if s <= 0 or not np.isfinite(s):
        # fallback: Gaussian-ish profile
        sigma = max(1.0, 0.35 * profile_hw)
        prof = np.exp(-(dys**2) / (2 * sigma**2))
        s = float(np.nansum(prof))
    prof /= s
    return dys, prof


def _optimal_extract(
    sci: np.ndarray,
    var: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    trace: np.ndarray,
    *,
    ap_hw: int,
    profile_hw: int,
    sigma_clip: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    ny, nlam = sci.shape
    have_var = var is not None and var.shape == sci.shape
    have_mask = mask is not None and mask.shape == sci.shape

    # Template profile
    dys, p0 = _build_profile_template(sci, mask, trace, profile_hw=profile_hw)

    flux = np.full(nlam, np.nan, dtype=float)
    out_var = np.full(nlam, np.inf, dtype=float)
    out_mask = np.zeros(nlam, dtype=np.uint16)

    ap_hw = int(max(3, ap_hw))
    sigma_clip = float(max(2.0, sigma_clip))

    # Precompute profile on integer dy grid used by aperture
    dy_int = np.arange(-ap_hw, ap_hw + 1, dtype=float)
    p_int = np.interp(dy_int, dys, p0, left=0.0, right=0.0)
    # ensure normalized within aperture
    s = float(np.sum(p_int))
    if s > 0:
        p_int = p_int / s

    for j in range(nlam):
        yc = float(trace[j])
        if not np.isfinite(yc):
            out_mask[j] = np.uint16(1)
            continue
        y0 = int(np.round(yc)) - ap_hw
        y1 = int(np.round(yc)) + ap_hw + 1
        if y0 < 0 or y1 > ny:
            out_mask[j] = np.uint16(1)
            continue
        d = sci[y0:y1, j].astype(float)
        if have_var:
            vv = var[y0:y1, j].astype(float)
        else:
            # approximate
            vv = np.maximum(np.nanvar(d), 1.0) * np.ones_like(d)
        good = np.isfinite(d) & np.isfinite(vv) & (vv > 0)
        if have_mask:
            good &= mask[y0:y1, j] == 0
        if not np.any(good):
            out_mask[j] = np.uint16(1)
            continue
        p = p_int.copy()
        # Weighted least squares amplitude for model f*p
        w = np.zeros_like(vv)
        w[good] = 1.0 / vv[good]
        num = float(np.sum(p[good] * d[good] * w[good]))
        den = float(np.sum((p[good] ** 2) * w[good]))
        if den <= 0:
            out_mask[j] = np.uint16(1)
            continue
        fhat = num / den
        # One-pass sigma clipping
        resid = d - fhat * p
        sig = np.sqrt(np.maximum(vv, 1e-12))
        clip = np.abs(resid) / sig > sigma_clip
        good2 = good & (~clip)
        if good2.sum() >= max(3, good.sum() // 2):
            num = float(np.sum(p[good2] * d[good2] * w[good2]))
            den = float(np.sum((p[good2] ** 2) * w[good2]))
            if den > 0:
                fhat = num / den
        flux[j] = float(fhat)
        out_var[j] = float(1.0 / max(den, 1e-20))

    meta = {
        "profile_hw": int(profile_hw),
        "aperture_half_width": int(ap_hw),
        "sigma_clip": float(sigma_clip),
        "profile": {
            "dy": dys.tolist(),
            "p": p0.tolist(),
        },
    }
    return flux, out_var, out_mask, meta


def _write_mef_1d(
    path: Path, flux: np.ndarray, hdr0: fits.Header, var: np.ndarray, mask: np.ndarray
) -> None:
    """Write 1D spectrum as MEF (Primary holds flux for legacy; SCI/VAR/MASK extensions are canonical)."""
    grid = try_read_grid(hdr0)
    write_sci_var_mask(
        path, flux, var=var, mask=mask, header=hdr0, grid=grid, primary_data=flux
    )


def run_extract1d(
    cfg: Dict[str, Any],
    *,
    in_fits: Optional[Path] = None,
    stacked_fits: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    ecfg = cfg.get("extract1d", {}) if isinstance(cfg.get("extract1d"), dict) else {}
    if not bool(ecfg.get("enabled", True)):
        return {"skipped": True, "reason": "extract1d.enabled=false"}
    # Backward-compatible alias used in older code/tests.
    if in_fits is None and stacked_fits is not None:
        in_fits = stacked_fits

    from scorpio_pipe.workspace_paths import resolve_input_path
    from scorpio_pipe.workspace_paths import stage_dir

    wd = resolve_work_dir(cfg)
    out_dir = Path(out_dir) if out_dir is not None else stage_dir(wd, "extract1d")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine input
    if in_fits is None:
        # Prefer Stack2D products (new→legacy resolver).
        p = resolve_input_path(
            "stacked2d",
            wd,
            "stack2d",
            relpath="stacked2d.fits",
        )
        if p.exists():
            in_fits = p
        else:
            # Optional manual-mode fallback: allow extraction directly from a sky-subtracted frame.
            #
            # Important: the GUI pipeline enforces Stack2D → Extract1D. This fallback is only
            # intended for explicit/manual workflows (CLI, notebooks) when users understand
            # they are extracting from a single (non-stacked) frame.
            if not bool(ecfg.get("allow_sky_fallback", False)):
                raise FileNotFoundError(
                    "Stack2D products not found. Run the Stack2D stage first (or pass in_fits=...). "
                    "For manual extraction without Stack2D, set extract1d.allow_sky_fallback=true. Tried: "
                    + str(p)
                )

            from scorpio_pipe.product_naming import (
                legacy_sky_sub_fits_names,
                sky_sub_fits_name,
            )

            def _resolve_skysub(tag: str, *, raw_stem: str | None = None) -> Path:
                # Canonical filename for this tag
                canon_name = sky_sub_fits_name(tag)
                extra: list[Path] = []
                # Try legacy names in a few likely legacy layouts
                for name in legacy_sky_sub_fits_names(tag):
                    extra.extend(
                        [
                            stage_dir(wd, "sky") / tag / name,
                            stage_dir(wd, "sky") / "per_exp" / name,
                            wd / "products" / "sky" / "per_exp" / name,
                            wd / "sky" / "per_exp" / name,
                            stage_dir(wd, "sky") / name,
                            wd / "products" / "sky" / name,
                            wd / "sky" / name,
                        ]
                    )
                return resolve_input_path(
                    "skysub",
                    wd,
                    "sky",
                    raw_stem=raw_stem,
                    relpath=canon_name,
                    extra_candidates=extra,
                )

            # 1) Prefer a stacked/combined sky-sub frame (sky stage in stack mode): obj_skysub.fits
            p_obj = _resolve_skysub("obj")
            if p_obj.exists():
                in_fits = p_obj
            else:
                # 2) If exactly one science exposure is defined, use its skysub frame.
                frames = cfg.get("frames") if isinstance(cfg.get("frames"), dict) else {}
                obj_list = frames.get("obj") if isinstance(frames.get("obj"), list) else []
                stems = [Path(x).stem for x in obj_list if isinstance(x, str) and x.strip()]
                if len(stems) == 1:
                    stem = stems[0]
                    p_one = _resolve_skysub(stem, raw_stem=stem)
                    if p_one.exists():
                        in_fits = p_one
                    else:
                        raise FileNotFoundError(
                            "Manual sky fallback requested but skysub frame not found for stem: "
                            + stem
                            + ". Tried: "
                            + str(p_one)
                        )
                else:
                    # 3) Last-resort: scan the sky stage folder.
                    sky_stage = stage_dir(wd, "sky")
                    found = sorted({p for p in sky_stage.rglob("*_skysub.fits")})
                    if len(found) == 1:
                        in_fits = found[0]
                    else:
                        raise FileNotFoundError(
                            "Manual sky fallback requested but could not unambiguously choose a sky-subtracted frame. "
                            f"Found {len(found)} candidates in {sky_stage}. "
                            "Pass in_fits=... or run Stack2D first."
                        )
    in_fits = Path(in_fits)
    if not in_fits.exists():
        raise FileNotFoundError(
            "No 2D input for extraction. Expected stacked2d.fits or obj_skysub.fits, got: "
            + str(in_fits)
        )

    sci, var, mask, hdr = _read_mef(in_fits)
    if sci.ndim != 2:
        raise ValueError(
            f"extract1d expects a 2D (λ,y) frame; got {sci.shape} from {in_fits}"
        )

    wave = _linear_wave_axis(hdr, sci.shape[1])
    roi = _roi_from_cfg(cfg)

    method = str(ecfg.get("method", "boxcar")).lower().strip()
    # legacy aliases
    if method == "sum":
        method = "boxcar"
    if method not in {"boxcar", "mean", "optimal"}:
        method = "boxcar"

    ap_hw = int(ecfg.get("aperture_half_width", 6))
    # If ROI is defined, initialize aperture from it unless user provided explicit.
    if ("obj_y0" in roi and "obj_y1" in roi) and "aperture_half_width" not in ecfg:
        ap_hw = max(1, int(round(0.5 * (int(roi["obj_y1"]) - int(roi["obj_y0"])))))

    trace_bin_A = float(ecfg.get("trace_bin_A", 60.0))
    trace_smooth_deg = int(ecfg.get("trace_smooth_deg", 3))
    trace, trace_meta = _estimate_trace(
        sci,
        var,
        mask,
        wave,
        roi,
        trace_bin_A=trace_bin_A,
        trace_smooth_deg=trace_smooth_deg,
    )

    if method in {"boxcar", "mean"}:
        flux, v1, m1 = _boxcar_extract(sci, var, mask, trace, ap_hw=ap_hw)
        if method == "mean":
            # convert to mean by dividing by N contributing pixels
            n = 2 * ap_hw + 1
            flux = flux / float(n)
            v1 = v1 / float(n * n)
        opt_meta: dict[str, Any] = {}
    else:
        prof_hw = int(ecfg.get("optimal_profile_half_width", 12))
        sig_clip = float(ecfg.get("optimal_sigma_clip", 5.0))
        flux, v1, m1, opt_meta = _optimal_extract(
            sci,
            var,
            mask,
            trace,
            ap_hw=ap_hw,
            profile_hw=prof_hw,
            sigma_clip=sig_clip,
        )

    # Write outputs
    out_fits = out_dir / "spec1d.fits"
    out_png = out_dir / "spec1d.png"
    trace_json = out_dir / "trace.json"

    ohdr = fits.Header()
    # keep simple linear WCS
    if np.isfinite(wave).all() and len(wave) > 1:
        ohdr["CRVAL1"] = float(wave[0])
        ohdr["CDELT1"] = float(wave[1] - wave[0])
        ohdr["CRPIX1"] = 1.0
        ohdr["CTYPE1"] = "WAVE"
        ohdr["CUNIT1"] = "Angstrom"
    ohdr["BUNIT"] = hdr.get("BUNIT", "ADU")
    ohdr = add_provenance(ohdr, cfg, stage="extract1d")
    _write_mef_1d(out_fits, flux, ohdr, v1, m1)

    trace_payload = {
        "input_fits": str(in_fits),
        "method": method,
        "aperture_half_width": int(ap_hw),
        "trace": {
            "wave": wave.tolist(),
            "y": trace.tolist(),
        },
        "trace_meta": trace_meta,
        "optimal_meta": opt_meta,
    }
    trace_json.write_text(
        json.dumps(trace_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if bool(ecfg.get("save_png", True)):
        try:
            import matplotlib.pyplot as plt

            from scorpio_pipe.plot_style import mpl_style

            snr = np.zeros_like(flux)
            good = np.isfinite(flux) & np.isfinite(v1) & (v1 > 0)
            snr[good] = flux[good] / np.sqrt(v1[good])

            with mpl_style():
                fig = plt.figure(figsize=(8.0, 3.6))
                ax = fig.add_subplot(111)
                ax.plot(wave, flux, lw=1.0)
                ax.set_xlabel("Wavelength (Å)")
                ax.set_ylabel("Flux")
                ax.set_title(
                    f"1D extraction: {method} (median S/N={np.nanmedian(snr):.1f})"
                )
                fig.tight_layout()
                fig.savefig(out_png)
                plt.close(fig)
        except Exception:
            pass

    done = out_dir / "extract1d_done.json"
    payload = {
        "ok": True,
        "input_fits": str(in_fits),
        # Backward-compatible keys used by tests/older callers.
        "spec1d_fits": str(out_fits),
        "spec1d_png": str(out_png) if out_png.exists() else None,
        # Canonical keys (v5.3x+)
        "output_fits": str(out_fits),
        "output_png": str(out_png) if out_png.exists() else None,
        "trace_json": str(trace_json),
        "method": method,
    }
    done.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Legacy mirror (disabled by default).
    #
    # v5.38+ supports legacy paths for *reading* via resolve_input_path(...), but
    # does not write duplicated outputs unless explicitly requested.
    try:
        compat = cfg.get("compat") if isinstance(cfg.get("compat"), dict) else {}
        if bool(compat.get("write_legacy_outputs", False)):
            legacy = wd / "spec"
            if legacy.is_dir() and legacy.resolve() != out_dir.resolve():
                import shutil

                shutil.copy2(out_fits, legacy / "spec1d.fits")
                if out_png.exists():
                    shutil.copy2(out_png, legacy / "spec1d.png")
                shutil.copy2(trace_json, legacy / "trace.json")
                shutil.copy2(done, legacy / "extract1d_done.json")
    except Exception:
        pass

    log.info("Extract1D done: %s", out_fits)
    return payload
