from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from scorpio_pipe.io.atomic import atomic_write_json
from scorpio_pipe.io.mef import read_sci_var_mask, write_sci_var_mask, WaveGrid
from scorpio_pipe.io.atomic_fits import atomic_write_fits
from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.provenance import add_provenance
from scorpio_pipe.version import PIPELINE_VERSION
from scorpio_pipe.wavesol_paths import wavesol_dir


# Manual-derived first-order spectral resolutions for 1'' slit.
# SCORPIO-2 values come from the 2022 observer manual table of grism parameters,
# SCORPIO legacy names are kept for backwards-compatible fallbacks.
_MANUAL_GRISM_PRIORS: dict[str, dict[str, float]] = {
    "VPHG940@600": {"dispersion_A_per_px": 1.16, "resolution_fwhm_A_1arcsec": 7.0},
    "VPHG1026@735": {"dispersion_A_per_px": 1.02, "resolution_fwhm_A_1arcsec": 6.1},
    "VPHG1200@540": {"dispersion_A_per_px": 0.87, "resolution_fwhm_A_1arcsec": 5.2},
    "VPHG1800@590": {"dispersion_A_per_px": 0.50, "resolution_fwhm_A_1arcsec": 3.0},
    "VPHG1800@660": {"dispersion_A_per_px": 0.51, "resolution_fwhm_A_1arcsec": 3.0},
    "VPHG2400@415": {"dispersion_A_per_px": 0.36, "resolution_fwhm_A_1arcsec": 2.2},
    "VPHG3000@400": {"dispersion_A_per_px": 0.32, "resolution_fwhm_A_1arcsec": 1.9},
    "VPHG1200@860": {"dispersion_A_per_px": 0.85, "resolution_fwhm_A_1arcsec": 5.1},
    "VPHG400": {"dispersion_A_per_px": 3.0, "resolution_fwhm_A_1arcsec": 15.0},
    "VPHG550G": {"dispersion_A_per_px": 2.1, "resolution_fwhm_A_1arcsec": 10.0},
    "VPHG550R": {"dispersion_A_per_px": 2.1, "resolution_fwhm_A_1arcsec": 10.0},
    "VPHG1200B": {"dispersion_A_per_px": 0.88, "resolution_fwhm_A_1arcsec": 5.5},
    "VPHG1200G": {"dispersion_A_per_px": 0.88, "resolution_fwhm_A_1arcsec": 5.0},
    "VPHG1200R": {"dispersion_A_per_px": 0.86, "resolution_fwhm_A_1arcsec": 5.0},
    "VPHG1800R": {"dispersion_A_per_px": 0.52, "resolution_fwhm_A_1arcsec": 2.5},
    "VPHG2300G": {"dispersion_A_per_px": 0.38, "resolution_fwhm_A_1arcsec": 2.2},
    "VPHG3000B": {"dispersion_A_per_px": 0.20, "resolution_fwhm_A_1arcsec": 1.5},
}


def _load_lambda_map(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        if data is None and "SCI" in hdul:
            data = hdul["SCI"].data
        if data is None:
            raise ValueError(f"No image data found in {path}")
        hdr = hdul[0].header.copy()
    return np.asarray(data, dtype=float), hdr


def _read_arc_image(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        if data is None and "SCI" in hdul:
            data = hdul["SCI"].data
        if data is None:
            raise ValueError(f"No image data found in {path}")
        hdr = hdul[0].header.copy()
    return np.asarray(data, dtype=float), hdr


def _load_line_centers(wdir: Path) -> np.ndarray:
    hp = wdir / "hand_pairs.txt"
    vals: list[float] = []
    if hp.exists():
        from scorpio_pipe.stages.wavesolution import _read_hand_pairs

        _x, lam, _blend, disabled = _read_hand_pairs(hp)
        lam = np.asarray(lam, dtype=float)
        disabled = np.asarray(disabled, dtype=bool)
        vals = [float(v) for v in lam[(~disabled) & np.isfinite(lam)]]
    if not vals:
        auto_csv = wdir / "manual_pairs_auto.csv"
        if auto_csv.exists():
            try:
                with auto_csv.open("r", encoding="utf-8", newline="") as f:
                    for row in csv.DictReader(f):
                        try:
                            use = str(row.get("use", "1") or "1").strip().lower()
                            if use in {"0", "false", "no"}:
                                continue
                            vals.append(float(row["lambda_A"]))
                        except Exception:
                            continue
            except Exception:
                pass
    arr = np.asarray(sorted({round(float(v), 6) for v in vals if np.isfinite(v)}), dtype=float)
    return arr


def _robust_sigma(arr: np.ndarray) -> float | None:
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return None
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if not (mad > 0):
        return None
    return float(1.4826 * mad)


def _estimate_sigma_px(profile: np.ndarray, x0: int, half_window: int, snr_thresh: float) -> tuple[float | None, dict[str, float]]:
    nx = int(profile.size)
    lo = max(0, int(x0) - int(half_window))
    hi = min(nx, int(x0) + int(half_window) + 1)
    if hi - lo < 5:
        return None, {"reason": "window_too_small"}
    xx = np.arange(lo, hi, dtype=float)
    yy = np.asarray(profile[lo:hi], dtype=float)
    if not np.isfinite(yy).any():
        return None, {"reason": "nonfinite"}
    edge = max(1, min(3, yy.size // 4))
    baseline = float(np.nanmedian(np.r_[yy[:edge], yy[-edge:]]))
    resid = yy - baseline
    noise = _robust_sigma(np.r_[yy[:edge], yy[-edge:]])
    if noise is None or not np.isfinite(noise) or noise <= 0:
        noise = float(np.nanstd(np.r_[yy[:edge], yy[-edge:]])) if np.isfinite(yy).any() else 0.0
    amp = float(np.nanmax(resid)) if np.isfinite(resid).any() else 0.0
    if not np.isfinite(amp) or amp <= 0 or (noise > 0 and amp / max(noise, 1e-6) < float(snr_thresh)):
        return None, {"reason": "low_snr", "amp": float(amp), "noise": float(noise)}
    resid = np.clip(resid, 0.0, None)
    s = float(np.sum(resid))
    if not (s > 0):
        return None, {"reason": "zero_signal"}
    xc = float(np.sum(xx * resid) / s)
    var = float(np.sum(((xx - xc) ** 2) * resid) / s)
    if not np.isfinite(var) or var <= 0:
        return None, {"reason": "bad_moment", "xc": float(xc)}
    sigma_px = float(np.sqrt(var))
    if not np.isfinite(sigma_px) or sigma_px <= 0:
        return None, {"reason": "bad_sigma", "xc": float(xc)}
    return sigma_px, {"xc": float(xc), "amp": float(amp), "noise": float(noise)}


def _manual_fallback_map(lam_map: np.ndarray, hdr: fits.Header) -> tuple[np.ndarray, dict[str, Any]]:
    from scorpio_pipe.long_setup_priors import get_long_setup_prior

    disperse = str(hdr.get("DISPERSE", "") or "").strip().upper()
    slit_arcsec = float(hdr.get("SLITWID", 1.0) or 1.0)
    prior = _MANUAL_GRISM_PRIORS.get(disperse)
    long_prior = get_long_setup_prior(disperse)
    if prior is None and long_prior is None:
        # Conservative generic fallback if disperser is unknown.
        sigma_a = float(2.5 * max(slit_arcsec, 0.75) / 1.0)
        src = "generic"
    elif prior is None and long_prior is not None:
        sigma_a = float(long_prior.empirical_resolution_fwhm_A_1arcsec) / 2.355 * max(slit_arcsec, 0.35)
        src = "long_empirical"
    else:
        sigma_a = float(prior["resolution_fwhm_A_1arcsec"]) / 2.355 * max(slit_arcsec, 0.35)
        src = "manual"
    out = np.full_like(lam_map, np.nan, dtype=float)
    out[np.isfinite(lam_map)] = float(sigma_a)
    meta = {
        "source": src,
        "disperser": disperse,
        "slit_arcsec": float(slit_arcsec),
        "sigma_A": float(sigma_a),
        "fwhm_A": float(2.355 * sigma_a),
    }
    if prior is not None:
        meta["manual_prior"] = dict(prior)
    if long_prior is not None:
        meta["long_setup_prior"] = long_prior.as_dict()
    return out, meta


def _interp_poly(y_nodes: np.ndarray, values: np.ndarray, y_all: np.ndarray, order: int) -> tuple[np.ndarray, list[float] | None]:
    y_nodes = np.asarray(y_nodes, dtype=float)
    values = np.asarray(values, dtype=float)
    m = np.isfinite(y_nodes) & np.isfinite(values)
    if int(np.sum(m)) < 2:
        fill = float(np.nanmedian(values[m])) if np.any(m) else np.nan
        return np.full_like(y_all, fill, dtype=float), None
    deg = min(int(order), int(np.sum(m)) - 1)
    coeffs = np.polyfit(y_nodes[m], values[m], deg=deg)
    return np.polyval(coeffs, y_all).astype(float), [float(c) for c in coeffs]


def _build_measured_lsf_map(
    arc: np.ndarray,
    lam_map: np.ndarray,
    line_centers: np.ndarray,
    *,
    fit_half_window_px: int,
    min_snr: float,
    poly_order_y: int,
    min_line_measures: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    ny, nx = arc.shape
    y_all = np.arange(ny, dtype=float)
    measurements: list[dict[str, Any]] = []
    per_line: list[dict[str, Any]] = []

    for lam0 in np.asarray(line_centers, dtype=float):
        rows = []
        vals = []
        amps = []
        for iy in range(ny):
            row_lam = lam_map[iy]
            if not np.isfinite(row_lam).any():
                continue
            ix = int(np.nanargmin(np.abs(row_lam - lam0)))
            if not np.isfinite(row_lam[ix]) or abs(float(row_lam[ix]) - float(lam0)) > 4.0:
                continue
            sigma_px, meta = _estimate_sigma_px(arc[iy], ix, fit_half_window_px, min_snr)
            if sigma_px is None:
                continue
            left = max(0, ix - 1)
            right = min(nx - 1, ix + 1)
            disp = float(np.nanmedian(np.abs(np.diff(row_lam[left:right + 1]))))
            if not np.isfinite(disp) or disp <= 0:
                continue
            sigma_a = float(sigma_px * disp)
            if not np.isfinite(sigma_a) or sigma_a <= 0:
                continue
            rows.append(float(iy))
            vals.append(float(sigma_a))
            amps.append(float(meta.get("amp", np.nan)))
            measurements.append(
                {
                    "y": int(iy),
                    "lambda_A": float(lam0),
                    "x_pix": int(ix),
                    "sigma_px": float(sigma_px),
                    "sigma_A": float(sigma_a),
                    "amp": float(meta.get("amp", np.nan)),
                    "noise": float(meta.get("noise", np.nan)),
                }
            )
        if len(rows) < int(min_line_measures):
            continue
        sigma_row, coeffs = _interp_poly(np.asarray(rows), np.asarray(vals), y_all, poly_order_y)
        sigma_row = np.clip(sigma_row, 0.05, None)
        per_line.append(
            {
                "lambda_A": float(lam0),
                "n_rows": int(len(rows)),
                "sigma_A_median": float(np.nanmedian(vals)),
                "sigma_A_p16": float(np.nanpercentile(vals, 16)),
                "sigma_A_p84": float(np.nanpercentile(vals, 84)),
                "amp_median": float(np.nanmedian(amps)) if amps else None,
                "poly_coeffs": coeffs,
                "sigma_A_by_row": sigma_row,
            }
        )

    if len(per_line) < 2:
        return None, {
            "status": "insufficient_lines",
            "n_lines_modeled": int(len(per_line)),
            "n_measurements": int(len(measurements)),
            "measurements_preview": measurements[:64],
        }

    line_lams = np.asarray([x["lambda_A"] for x in per_line], dtype=float)
    order = np.argsort(line_lams)
    line_lams = line_lams[order]
    sigma_y_lambda = np.asarray([per_line[i]["sigma_A_by_row"] for i in order], dtype=float).T

    out = np.full_like(lam_map, np.nan, dtype=float)
    for iy in range(ny):
        row_lam = np.asarray(lam_map[iy], dtype=float)
        m = np.isfinite(row_lam)
        if not np.any(m):
            continue
        row_sig = np.interp(row_lam[m], line_lams, sigma_y_lambda[iy], left=sigma_y_lambda[iy, 0], right=sigma_y_lambda[iy, -1])
        out[iy, m] = row_sig

    meta = {
        "status": "measured",
        "n_lines_modeled": int(len(per_line)),
        "n_measurements": int(len(measurements)),
        "line_models": [
            {
                k: (v if k != "sigma_A_by_row" else None)
                for k, v in item.items()
            }
            for item in per_line
        ],
        "measurement_preview": measurements[:256],
        "line_lambdas_A": [float(v) for v in line_lams],
    }
    return out, meta


def _approx_linear_grid_from_lambda_map(lam_map: np.ndarray, hdr: fits.Header) -> WaveGrid | None:
    ny, nx = lam_map.shape
    y0 = int(np.clip(int(round(float(hdr.get("SLITPOS", ny / 2.0) or ny / 2.0))), 0, ny - 1))
    row = np.asarray(lam_map[y0], dtype=float)
    m = np.isfinite(row)
    if int(np.sum(m)) < 8:
        m = np.isfinite(np.nanmedian(lam_map, axis=0))
        row = np.asarray(np.nanmedian(lam_map, axis=0), dtype=float)
    if int(np.sum(m)) < 8:
        return None
    x = np.flatnonzero(m).astype(float)
    y = row[m]
    if y.size < 8:
        return None
    coeffs = np.polyfit(x, y, deg=1)
    return WaveGrid(lambda0=float(np.polyval(coeffs, 0.0)), dlambda=float(coeffs[0]), nlam=int(nx), unit=str(hdr.get("WAVEUNIT", "Angstrom") or "Angstrom"), wave_ref=str(hdr.get("WAVEREF", "air") or "air"))


def run_lsf_model(cfg: dict[str, Any], *, out_dir: Path | None = None) -> dict[str, Any]:
    work_dir = resolve_work_dir(cfg)
    wdir = wavesol_dir(cfg, run_root=work_dir)
    out_dir = Path(out_dir) if out_dir is not None else (Path(work_dir) / "lsf_model")
    out_dir.mkdir(parents=True, exist_ok=True)

    lam_map_path = wdir / "lambda_map.fits"
    arc_path = wdir / "superneon.fits"
    if not lam_map_path.exists():
        raise FileNotFoundError(f"Missing wavelength map: {lam_map_path}")
    if not arc_path.exists():
        raise FileNotFoundError(f"Missing superneon frame: {arc_path}")

    lam_map, lam_hdr = _load_lambda_map(lam_map_path)
    arc, arc_hdr = _read_arc_image(arc_path)
    if arc.shape != lam_map.shape:
        raise ValueError(f"superneon shape {arc.shape} != lambda_map shape {lam_map.shape}")

    lcfg = cfg.get("lsf_model") if isinstance(cfg.get("lsf_model"), dict) else {}
    line_centers = _load_line_centers(wdir)
    measured_map, fit_meta = _build_measured_lsf_map(
        arc,
        lam_map,
        line_centers,
        fit_half_window_px=int(lcfg.get("fit_half_window_px", 6) or 6),
        min_snr=float(lcfg.get("min_line_snr", 4.0) or 4.0),
        poly_order_y=int(lcfg.get("poly_order_y", 2) or 2),
        min_line_measures=int(lcfg.get("min_line_measures", 8) or 8),
    )

    if measured_map is None:
        lsf_map, source_meta = _manual_fallback_map(lam_map, arc_hdr)
        fit_source = "manual_fallback"
    else:
        lsf_map = measured_map
        source_meta = fit_meta
        fit_source = "arc_measured"

    lsf_map = np.asarray(lsf_map, dtype=float)
    lsf_map[~np.isfinite(lam_map)] = np.nan
    grid = _approx_linear_grid_from_lambda_map(lam_map, lam_hdr)

    primary_hdr = fits.Header(lam_hdr)
    primary_hdr["BUNIT"] = ("Angstrom", "Instrumental sigma unit")
    primary_hdr["LSFUNIT"] = ("sigma_A", "LSF map quantity")
    primary_hdr["LSFSRC"] = (str(fit_source), "LSF source")
    primary_hdr["PIPEVER"] = (str(PIPELINE_VERSION), "Pipeline version")
    primary_hdr = add_provenance(primary_hdr, cfg, stage="lsf_model")

    lsf_path = out_dir / "lsf_map.fits"
    write_sci_var_mask(lsf_path, lsf_map.astype(np.float32), header=primary_hdr, grid=grid, primary_data=lsf_map.astype(np.float32))

    finite = lsf_map[np.isfinite(lsf_map)]
    qpath = out_dir / "lsf_qc.json"
    mpath = out_dir / "lsf_model.json"
    summary_1d: dict[str, Any] = {}
    if np.isfinite(lam_map).any() and np.isfinite(lsf_map).any():
        lam1d = np.nanmedian(lam_map, axis=0)
        sig1d = np.nanmedian(lsf_map, axis=0)
        good = np.isfinite(lam1d) & np.isfinite(sig1d)
        summary_1d = {
            "lambda_A": [float(v) for v in lam1d[good]],
            "sigma_A": [float(v) for v in sig1d[good]],
            "fwhm_A": [float(2.355 * v) for v in sig1d[good]],
        }

    model_payload = {
        "stage": "lsf_model",
        "pipeline_version": str(PIPELINE_VERSION),
        "source": fit_source,
        "inputs": {
            "lambda_map": str(lam_map_path),
            "superneon": str(arc_path),
            "hand_pairs": str(wdir / "hand_pairs.txt"),
            "manual_pairs_auto": str(wdir / "manual_pairs_auto.csv"),
        },
        "instrument": {
            "disperser": str(arc_hdr.get("DISPERSE", "") or ""),
            "slit_arcsec": float(arc_hdr.get("SLITWID", np.nan)) if arc_hdr.get("SLITWID") is not None else None,
            "wave_ref": str(lam_hdr.get("WAVEREF", "unknown") or "unknown"),
            "wave_unit": str(lam_hdr.get("WAVEUNIT", "Angstrom") or "Angstrom"),
        },
        "model": source_meta,
        "summary_1d": summary_1d,
        "outputs": {
            "lsf_map": str(lsf_path),
            "lsf_qc": str(qpath),
        },
    }
    atomic_write_json(mpath, model_payload)

    qc_payload = {
        "ok": bool(finite.size > 0),
        "source": fit_source,
        "n_finite": int(finite.size),
        "sigma_A_median": float(np.nanmedian(finite)) if finite.size else None,
        "sigma_A_p16": float(np.nanpercentile(finite, 16)) if finite.size else None,
        "sigma_A_p84": float(np.nanpercentile(finite, 84)) if finite.size else None,
        "fwhm_A_median": float(2.355 * np.nanmedian(finite)) if finite.size else None,
        "shape": [int(v) for v in lsf_map.shape],
        "coverage_frac": float(np.isfinite(lsf_map).sum() / max(1, lsf_map.size)),
        "model": source_meta,
        "warnings": [] if finite.size else ["empty_lsf_map"],
    }
    atomic_write_json(qpath, qc_payload)

    done = {
        "ok": bool(finite.size > 0),
        "source": fit_source,
        "lsf_map": str(lsf_path),
        "lsf_model_json": str(mpath),
        "lsf_qc_json": str(qpath),
    }
    atomic_write_json(out_dir / "done.json", done)
    return done
