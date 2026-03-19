from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from scorpio_pipe.io.atomic import atomic_write_json
from scorpio_pipe.io.done_json import write_done_json
from scorpio_pipe.io.mef import write_sci_var_mask
from scorpio_pipe.fits_utils import open_fits_smart
from scorpio_pipe.provenance import add_provenance
from scorpio_pipe.qc.flags import make_flag, max_severity
from scorpio_pipe.sky_geometry import compute_sky_geometry, expand_object_protection_mask, roi_from_cfg
from scorpio_pipe.workspace_paths import stage_dir
from scorpio_pipe.version import PIPELINE_VERSION


@dataclass(frozen=True)
class WindowRecord:
    stem: str
    x0: int
    x1: int
    lam0_A: float | None
    lam1_A: float | None
    lam_c_A: float | None
    width_px: int
    peak_abs: float
    peak_snr: float
    robust_mean: float
    lower_mean: float
    upper_mean: float
    asymmetry: float
    sign_flip: bool
    classification: str
    action: str
    alpha: float | None
    beta: float | None
    gamma: float | None
    gain_metric: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stem": self.stem,
            "x0": int(self.x0),
            "x1": int(self.x1),
            "lam0_A": self.lam0_A,
            "lam1_A": self.lam1_A,
            "lam_c_A": self.lam_c_A,
            "width_px": int(self.width_px),
            "peak_abs": float(self.peak_abs),
            "peak_snr": float(self.peak_snr),
            "robust_mean": float(self.robust_mean),
            "lower_mean": float(self.lower_mean),
            "upper_mean": float(self.upper_mean),
            "asymmetry": float(self.asymmetry),
            "sign_flip": bool(self.sign_flip),
            "classification": str(self.classification),
            "action": str(self.action),
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "gain_metric": self.gain_metric,
        }


def _read_mef(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, fits.Header]:
    with open_fits_smart(
        path,
        memmap="auto",
        ignore_missing_end=True,
        ignore_missing_simple=True,
        do_not_scale_image_data=False,
    ) as hdul:
        sci = None
        var = None
        mask = None
        hdr = fits.Header()
        for hdu in hdul:
            ext = str(hdu.header.get("EXTNAME", "")).strip().upper()
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            arr = np.asarray(data)
            if arr.ndim < 2:
                continue
            if sci is None and ext in {"", "SCI"}:
                sci = np.asarray(arr, dtype=np.float32)
                hdr = hdu.header.copy()
                if ext == "SCI":
                    continue
            if ext == "VAR":
                var = np.asarray(arr, dtype=np.float32)
            elif ext == "MASK":
                mask = np.asarray(arr, dtype=np.uint16)
            elif sci is None:
                sci = np.asarray(arr, dtype=np.float32)
                hdr = hdu.header.copy()
        if sci is None:
            raise ValueError(f"No SCI image found in {path}")
    return sci, var, mask, hdr


def _robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return float(1.4826 * mad)


def _windowed_stats(arr: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    med = float(np.median(arr))
    return med, float(np.nanmean(arr)), _robust_sigma(arr)


def _merge_boolean_regions(mask: np.ndarray, *, min_width: int = 1, bridge: int = 1) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    n = int(mask.size)
    if n == 0:
        return []
    if bridge > 0:
        m = mask.copy()
        for i in range(1, n - 1):
            if not m[i] and m[max(0, i - bridge) : i].any() and m[i + 1 : min(n, i + bridge + 1)].any():
                m[i] = True
        mask = m
    out: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        if (j - i + 1) >= int(max(1, min_width)):
            out.append((i, j))
        i = j + 1
    return out


def _classify_window(
    *,
    width_px: int,
    peak_snr: float,
    asymmetry: float,
    robust_mean: float,
    lower_mean: float,
    upper_mean: float,
    alpha: float | None,
    beta: float | None,
) -> tuple[str, str]:
    sign_flip = (np.sign(lower_mean) != np.sign(upper_mean)) and abs(lower_mean) > 0 and abs(upper_mean) > 0
    if width_px <= 2 and peak_snr >= 8.0 and asymmetry >= 0.9:
        return "cosmic_leakage", "skip"
    if asymmetry >= 1.2:
        return "sky_window_contamination", "skip"
    if sign_flip and beta is not None and abs(beta) >= max(0.15, 0.75 * abs(alpha or 0.0)):
        return "wavelength_mismatch", "apply"
    if sign_flip:
        return "lsf_mismatch", "apply"
    if abs(robust_mean) > 0:
        return "amplitude_mismatch", "apply"
    return "unknown", "skip"


def _solve_window_coeffs(
    residual: np.ndarray,
    model: np.ndarray,
    variance: np.ndarray | None,
    *,
    max_abs_alpha: float,
    max_abs_beta: float,
    max_abs_gamma: float,
) -> tuple[float | None, float | None, float | None]:
    r = np.asarray(residual, dtype=float)
    m = np.asarray(model, dtype=float)
    g = np.gradient(m, axis=1) if m.ndim >= 2 and m.shape[1] >= 2 else np.zeros_like(m, dtype=float)
    ok = np.isfinite(r) & np.isfinite(m) & np.isfinite(g)
    if variance is not None:
        v = np.asarray(variance, dtype=float)
        ok &= np.isfinite(v) & (v > 0)
        w = np.zeros_like(r, dtype=float)
        w[ok] = 1.0 / np.sqrt(np.maximum(v[ok], 1e-12))
    else:
        w = np.zeros_like(r, dtype=float)
        w[ok] = 1.0
    if int(np.count_nonzero(ok)) < 8:
        return None, None, None
    y = (r[ok] * w[ok]).reshape(-1)
    A = np.column_stack([
        (m[ok] * w[ok]).reshape(-1),
        (g[ok] * w[ok]).reshape(-1),
        w[ok].reshape(-1),
    ])
    try:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    except Exception:
        return None, None, None
    alpha = float(np.clip(coef[0], -max_abs_alpha, max_abs_alpha))
    beta = float(np.clip(coef[1], -max_abs_beta, max_abs_beta))
    gamma = float(np.clip(coef[2], -max_abs_gamma, max_abs_gamma))
    return alpha, beta, gamma


def _taper(width: int) -> np.ndarray:
    n = int(max(1, width))
    if n <= 5:
        return np.ones(n, dtype=float)
    win = np.hanning(n)
    if not np.isfinite(win).all() or np.max(win) <= 0:
        return np.ones(n, dtype=float)
    win = win / np.max(win)
    return win.astype(float)


def _find_windows(
    residual: np.ndarray,
    lower_rows: np.ndarray,
    upper_rows: np.ndarray,
    lam_mid: np.ndarray,
    *,
    detect_sigma: float,
    detect_snr: float,
    min_width_px: int,
) -> tuple[list[tuple[int, int]], dict[str, np.ndarray]]:
    res = np.asarray(residual, dtype=float)
    med_all = np.nanmedian(res, axis=0)
    med_low = np.nanmedian(res[lower_rows, :], axis=0) if lower_rows.size else np.full(res.shape[1], np.nan)
    med_high = np.nanmedian(res[upper_rows, :], axis=0) if upper_rows.size else np.full(res.shape[1], np.nan)
    sigma_cols = np.array([_robust_sigma(res[:, max(0, i - 1) : min(res.shape[1], i + 2)].ravel()) for i in range(res.shape[1])], dtype=float)
    sigma_global = _robust_sigma(med_all)
    if not np.isfinite(sigma_global) or sigma_global <= 0:
        sigma_global = max(float(np.nanstd(med_all)), 1e-6)
    row_factor = max(math.sqrt(max(int(res.shape[0]), 1)), 1.0)
    snr = np.abs(med_all) / np.maximum(sigma_cols / row_factor, 1e-6)
    det = np.isfinite(med_all) & ((np.abs(med_all) > float(detect_sigma) * sigma_global) | (snr > float(detect_snr)))
    # Ignore edges where wavelength solution / coverage is commonly unstable.
    if det.size >= 8:
        det[:2] = False
        det[-2:] = False
    regions = _merge_boolean_regions(det, min_width=int(min_width_px), bridge=1)
    return regions, {
        "median_all": med_all,
        "median_low": med_low,
        "median_high": med_high,
        "sigma_cols": sigma_cols,
        "snr": snr,
        "lam_mid": np.asarray(lam_mid, dtype=float),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _plot_diagnostics(
    out_path: Path,
    *,
    lam_mid: np.ndarray,
    med_low_before: np.ndarray,
    med_high_before: np.ndarray,
    med_low_after: np.ndarray,
    med_high_after: np.ndarray,
    windows: list[WindowRecord],
) -> None:
    try:
        import matplotlib.pyplot as plt
        from scorpio_pipe.plot_style import mpl_style
    except Exception:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(lam_mid, dtype=float)
    if not np.isfinite(x).all():
        x = np.arange(x.size, dtype=float)
    with mpl_style():
        fig = plt.figure(figsize=(11, 7))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)
        ax1.plot(x, med_low_before, lw=1.0, label="lower before")
        ax1.plot(x, med_high_before, lw=1.0, label="upper before")
        ax1.plot(x, med_low_after, lw=1.0, alpha=0.9, label="lower after")
        ax1.plot(x, med_high_after, lw=1.0, alpha=0.9, label="upper after")
        ax1.set_ylabel("Residual")
        ax1.set_title("Sky stripes before/after refine")
        ax1.legend(loc="upper right", fontsize=8)

        peak = np.zeros_like(x, dtype=float)
        for rec in windows:
            peak[rec.x0 : rec.x1 + 1] = max(peak[rec.x0 : rec.x1 + 1].max(initial=0.0), rec.peak_snr)
            ax1.axvspan(x[rec.x0], x[rec.x1], alpha=0.10)
        ax2.plot(x, peak, lw=1.0)
        ax2.set_ylabel("Peak S/N")
        ax2.set_xlabel("Wavelength [A]")
        ax2.set_title("Detected problematic windows")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def _read_lambda_map(run_root: Path) -> np.ndarray:
    path = stage_dir(run_root, "wavesol") / "lambda_map.fits"
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(hdul[0].data, dtype=float)


def run_sky_residual_refine(cfg: dict[str, Any], *, out_dir: Path) -> dict[str, Path]:
    if out_dir is None:
        raise ValueError("sky_refine: out_dir is required")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path(str(cfg.get("work_dir") or cfg.get("run_root") or ".")).expanduser().resolve()
    sky_root = stage_dir(work_dir, "sky")
    lam_map = _read_lambda_map(work_dir)

    rcfg = cfg.get("sky_refine") if isinstance(cfg.get("sky_refine"), dict) else {}
    enabled = str(rcfg.get("enabled", "true")).strip().lower() not in {"0", "false", "no", "off"}
    if not enabled:
        done = write_done_json(
            stage="sky_refine",
            stage_dir=out_dir,
            status="skipped",
            params={"enabled": False},
            outputs={},
            metrics={},
            flags=[make_flag("SKY_REFINE_DISABLED", "INFO", "Sky residual refine disabled by configuration")],
        )
        atomic_write_json(out_dir / "qc_sky_refine.json", done, indent=2, ensure_ascii=False)
        return {"sky_refine_done": out_dir / "done.json", "qc_sky_refine_json": out_dir / "qc_sky_refine.json"}

    frames = cfg.get("frames", {}) if isinstance(cfg.get("frames"), dict) else {}
    obj_frames = [Path(str(x)).expanduser() for x in frames.get("obj", [])]
    if not obj_frames:
        raise ValueError("No object frames configured (frames.obj is empty)")

    detect_sigma = float(rcfg.get("detect_sigma", 2.5) or 2.5)
    detect_snr = float(rcfg.get("detect_snr", 4.0) or 4.0)
    min_width_px = int(rcfg.get("min_width_px", 3) or 3)
    top_n_summary = int(rcfg.get("top_n_summary", 8) or 8)
    alpha_lim = float(rcfg.get("max_abs_alpha", 0.35) or 0.35)
    beta_lim = float(rcfg.get("max_abs_beta", 1.50) or 1.50)
    gamma_lim = float(rcfg.get("max_abs_gamma", 25.0) or 25.0)
    strength = float(rcfg.get("strength", 1.0) or 1.0)
    lsf_strength = float(rcfg.get("lsf_strength", min(strength, 0.7)) or min(strength, 0.7))
    max_corr_frac = float(rcfg.get("max_correction_frac", 1.0) or 1.0)

    protect_cfg = rcfg.get("object_protection") if isinstance(rcfg.get("object_protection"), dict) else {}
    geom_cfg = (cfg.get("sky_sub") or {}).get("geometry") if isinstance((cfg.get("sky_sub") or {}).get("geometry"), dict) else {}

    flags: list[dict[str, Any]] = []
    per_exp: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for raw_p in obj_frames:
        stem = raw_p.stem
        skysub_p = sky_root / f"{stem}_skysub_raw.fits"
        skymodel_p = sky_root / f"{stem}_skymodel_raw.fits"
        if not skysub_p.exists() or not skymodel_p.exists():
            flags.append(make_flag("SKY_REFINE_INPUT_MISSING", "ERROR", f"Missing sky products for {stem}", stem=stem))
            continue

        skysub, var, mask, hdr = _read_mef(skysub_p)
        model, _, _, _ = _read_mef(skymodel_p)
        if var is None:
            var = np.ones_like(skysub, dtype=np.float32)
        if mask is None:
            mask = np.zeros_like(skysub, dtype=np.uint16)
        if lam_map.shape != skysub.shape:
            raise ValueError(f"lambda_map shape {lam_map.shape} does not match {stem} frame {skysub.shape}")

        geom = compute_sky_geometry(
            skysub,
            var,
            mask,
            roi=roi_from_cfg(cfg),
            roi_policy=str(geom_cfg.get("roi_policy") or "prefer_user"),
            fatal_bits=0,
            edge_margin_px=int(geom_cfg.get("edge_margin_px", 16) or 16),
            profile_x_percentile=float(geom_cfg.get("profile_x_percentile", 50.0) or 50.0),
            thresh_sigma=float(geom_cfg.get("thresh_sigma", 3.0) or 3.0),
            dilation_px=int(geom_cfg.get("dilation_px", 3) or 3),
            min_obj_width_px=int(geom_cfg.get("min_obj_width_px", 6) or 6),
            min_sky_width_px=int(geom_cfg.get("min_sky_width_px", 12) or 12),
            contamination_sigma=float(geom_cfg.get("contamination_sigma", 3.0) or 3.0),
            contamination_frac_warn=float(geom_cfg.get("contamination_frac_warn", 0.15) or 0.15),
        )
        base_obj = np.asarray(geom.mask_obj_y, dtype=bool)
        obj_mask, protect_info = expand_object_protection_mask(
            skysub,
            base_obj,
            edge_margin_px=int(protect_cfg.get("edge_margin_px", geom_cfg.get("edge_margin_px", 16) or 16) or 16),
            continuum_sigma=float(protect_cfg.get("continuum_sigma", 2.2) or 2.2),
            line_sigma=float(protect_cfg.get("line_sigma", 3.8) or 3.8),
            asym_low_px=int(protect_cfg.get("asym_low_px", 2) or 2),
            asym_high_px=int(protect_cfg.get("asym_high_px", 4) or 4),
            max_extra_frac=float(protect_cfg.get("max_extra_frac", 0.35) or 0.35),
        )
        sky_mask = np.asarray(geom.mask_sky_y, dtype=bool) & (~obj_mask)
        sky_rows = np.flatnonzero(sky_mask)
        if sky_rows.size < 4:
            flags.append(make_flag("SKY_REFINE_NO_SKY", "WARN", f"No usable sky rows after object protection for {stem}", stem=stem))
            refined = skysub.copy()
            corr = np.zeros_like(skysub, dtype=np.float32)
            recs: list[WindowRecord] = []
            lower_rows = sky_rows
            upper_rows = sky_rows
        else:
            lower_rows = sky_rows[sky_rows < int(np.median(sky_rows))]
            upper_rows = sky_rows[sky_rows >= int(np.median(sky_rows))]
            if lower_rows.size == 0:
                lower_rows = sky_rows[: max(1, sky_rows.size // 2)]
            if upper_rows.size == 0:
                upper_rows = sky_rows[-max(1, sky_rows.size // 2) :]

            residual = np.asarray(skysub[sky_rows, :], dtype=float)
            lam_mid = np.asarray(lam_map[lam_map.shape[0] // 2, :], dtype=float)
            regions, diag = _find_windows(
                residual,
                lower_rows=np.searchsorted(sky_rows, lower_rows),
                upper_rows=np.searchsorted(sky_rows, upper_rows),
                lam_mid=lam_mid,
                detect_sigma=detect_sigma,
                detect_snr=detect_snr,
                min_width_px=min_width_px,
            )

            corr = np.zeros_like(skysub, dtype=np.float32)
            recs = []
            for x0, x1 in regions:
                sl = slice(x0, x1 + 1)
                res_win = skysub[sky_rows, sl]
                mod_win = model[sky_rows, sl]
                var_win = var[sky_rows, sl] if var is not None else None
                alpha, beta, gamma = _solve_window_coeffs(
                    res_win,
                    mod_win,
                    var_win,
                    max_abs_alpha=alpha_lim,
                    max_abs_beta=beta_lim,
                    max_abs_gamma=gamma_lim,
                )
                med_all = float(np.nanmedian(diag["median_all"][sl]))
                med_low = float(np.nanmedian(diag["median_low"][sl])) if np.isfinite(diag["median_low"][sl]).any() else med_all
                med_high = float(np.nanmedian(diag["median_high"][sl])) if np.isfinite(diag["median_high"][sl]).any() else med_all
                peak_abs = float(np.nanmax(np.abs(diag["median_all"][sl])))
                peak_snr = float(np.nanmax(diag["snr"][sl]))
                denom = max(abs(med_all), 1e-6)
                asym = float(abs(med_low - med_high) / denom)
                classification, action = _classify_window(
                    width_px=int(x1 - x0 + 1),
                    peak_snr=peak_snr,
                    asymmetry=asym,
                    robust_mean=med_all,
                    lower_mean=med_low,
                    upper_mean=med_high,
                    alpha=alpha,
                    beta=beta,
                )
                taper = _taper(int(x1 - x0 + 1))[None, :]
                gain_metric = None
                if action == "apply" and alpha is not None and beta is not None and gamma is not None:
                    model_sl = model[:, sl].astype(float)
                    grad_full = np.gradient(model_sl, axis=1) if model_sl.ndim >= 2 and model_sl.shape[1] >= 2 else np.zeros_like(model_sl, dtype=float)
                    local_strength = lsf_strength if classification == "lsf_mismatch" else strength
                    corr_win = (alpha * model[:, sl] + beta * grad_full + gamma).astype(np.float32)
                    # Never correct more than a fixed fraction of the measured residual amplitude in sky rows.
                    ref_amp = np.nanpercentile(np.abs(res_win), 95) if np.isfinite(res_win).any() else 0.0
                    if np.isfinite(ref_amp) and ref_amp > 0:
                        corr_amp = np.nanpercentile(np.abs(corr_win[sky_rows, :]), 95)
                        if np.isfinite(corr_amp) and corr_amp > (max_corr_frac * ref_amp):
                            corr_win *= float((max_corr_frac * ref_amp) / max(corr_amp, 1e-6))
                    corr[:, sl] += (local_strength * corr_win * taper).astype(np.float32)
                    after_tmp = res_win - corr[sky_rows, sl]
                    gain_metric = float(_robust_sigma(res_win) - _robust_sigma(after_tmp))
                lam0 = float(lam_mid[x0]) if np.isfinite(lam_mid[x0]) else None
                lam1 = float(lam_mid[x1]) if np.isfinite(lam_mid[x1]) else None
                lamc = float(np.nanmedian(lam_mid[sl])) if np.isfinite(lam_mid[sl]).any() else None
                recs.append(
                    WindowRecord(
                        stem=stem,
                        x0=int(x0),
                        x1=int(x1),
                        lam0_A=lam0,
                        lam1_A=lam1,
                        lam_c_A=lamc,
                        width_px=int(x1 - x0 + 1),
                        peak_abs=peak_abs,
                        peak_snr=peak_snr,
                        robust_mean=med_all,
                        lower_mean=med_low,
                        upper_mean=med_high,
                        asymmetry=asym,
                        sign_flip=bool(np.sign(med_low) != np.sign(med_high) and abs(med_low) > 0 and abs(med_high) > 0),
                        classification=classification,
                        action=action,
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                        gain_metric=gain_metric,
                    )
                )
            refined = (skysub - corr).astype(np.float32)
            med_low_before = np.nanmedian(skysub[lower_rows, :], axis=0)
            med_high_before = np.nanmedian(skysub[upper_rows, :], axis=0)
            med_low_after = np.nanmedian(refined[lower_rows, :], axis=0)
            med_high_after = np.nanmedian(refined[upper_rows, :], axis=0)
            _plot_diagnostics(
                out_dir / f"{stem}_sky_diagnostics.png",
                lam_mid=lam_mid,
                med_low_before=med_low_before,
                med_high_before=med_high_before,
                med_low_after=med_low_after,
                med_high_after=med_high_after,
                windows=recs,
            )

        qc_before = float(_robust_sigma(skysub[sky_rows, :])) if sky_rows.size else None
        qc_after = float(_robust_sigma(refined[sky_rows, :])) if sky_rows.size else None
        obj_rows = np.flatnonzero(obj_mask)
        object_delta = float(np.nanmedian((refined - skysub)[obj_rows, :])) if obj_rows.size else None
        if object_delta is not None and object_delta < float(rcfg.get("object_delta_warn", -0.75) or -0.75):
            flags.append(make_flag("SKY_REFINE_OBJECT_RISK", "WARN", "Refine may be suppressing object flux", stem=stem, object_delta=object_delta))
        if qc_before is not None and qc_after is not None and np.isfinite(qc_before) and np.isfinite(qc_after) and qc_after >= qc_before:
            flags.append(make_flag("SKY_REFINE_NO_GAIN", "WARN", "Refine did not improve robust sky scatter", stem=stem, sigma_before=qc_before, sigma_after=qc_after))

        hdr0 = fits.Header(hdr)
        hdr0["SKYRFINE"] = (True, "Sky residual refine applied")
        hdr0 = add_provenance(hdr0, cfg, stage="sky_refine")
        corr_p = out_dir / f"{stem}_skycorr_raw.fits"
        ref_p = out_dir / f"{stem}_skysub_refined_raw.fits"
        write_sci_var_mask(corr_p, corr.astype(np.float32), var=var, mask=mask, header=hdr0)
        write_sci_var_mask(ref_p, refined.astype(np.float32), var=var, mask=mask, header=hdr0)

        rows = [r.as_dict() for r in recs]
        _write_csv(out_dir / f"{stem}_sky_windows.csv", rows)
        atomic_write_json(out_dir / f"{stem}_sky_windows.json", rows, indent=2, ensure_ascii=False)
        all_rows.extend(rows)

        per_exp.append({
            "stem": stem,
            "inputs": {"skysub_raw": str(skysub_p), "skymodel_raw": str(skymodel_p)},
            "outputs": {
                "skycorr_raw": str(corr_p),
                "skysub_refined_raw": str(ref_p),
                "sky_windows_csv": str(out_dir / f"{stem}_sky_windows.csv"),
                "sky_windows_json": str(out_dir / f"{stem}_sky_windows.json"),
                "sky_diagnostics_png": str(out_dir / f"{stem}_sky_diagnostics.png"),
            },
            "metrics": {
                "sky_sigma_before": qc_before,
                "sky_sigma_after": qc_after,
                "sky_sigma_gain": (float(qc_before - qc_after) if qc_before is not None and qc_after is not None and np.isfinite(qc_before) and np.isfinite(qc_after) else None),
                "n_windows": int(len(rows)),
                "n_corrected_windows": int(sum(1 for r in recs if r.action == "apply")),
                "object_delta_median": object_delta,
            },
            "object_protection": protect_info,
            "windows": rows,
        })

    # aggregate observer pack
    all_rows_sorted = sorted(all_rows, key=lambda r: float(r.get("peak_snr") or 0.0), reverse=True)
    _write_csv(out_dir / "sky_residual_windows.csv", all_rows_sorted)
    atomic_write_json(out_dir / "sky_residual_windows.json", all_rows_sorted, indent=2, ensure_ascii=False)
    lines = ["# Sky refine observer pack", ""]
    lines.append(f"Frames processed: {len(per_exp)}")
    if all_rows_sorted:
        lines.append("")
        lines.append("## Top problematic windows")
        lines.append("")
        for row in all_rows_sorted[: max(1, top_n_summary)]:
            lines.append(
                f"- `{row['stem']}` λ≈{row.get('lam_c_A'):.1f} Å, S/N={row.get('peak_snr'):.2f}, "
                f"class={row.get('classification')}, action={row.get('action')}"
            )
    else:
        lines.append("")
        lines.append("No problematic skyline windows were detected.")
    (out_dir / "sky_observer_pack.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _med_metric(key: str) -> float | None:
        vals = []
        for row in per_exp:
            v = ((row.get("metrics") or {}).get(key))
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if np.isfinite(fv):
                vals.append(fv)
        if not vals:
            return None
        return float(np.nanmedian(np.asarray(vals, dtype=float)))

    status = "ok"
    sev = max_severity(flags)
    if sev == "ERROR":
        status = "fail"
    elif sev == "WARN":
        status = "warn"

    qc = {
        "stage": "sky_refine",
        "version": PIPELINE_VERSION,
        "status": status,
        "per_exposure": per_exp,
        "flags": flags,
        "max_severity": sev,
        "n_windows_total": int(len(all_rows_sorted)),
    }
    atomic_write_json(out_dir / "qc_sky_refine.json", qc, indent=2, ensure_ascii=False)
    write_done_json(
        stage="sky_refine",
        stage_dir=out_dir,
        status=status,
        inputs={"sky_stage": str(sky_root), "lambda_map": str(stage_dir(work_dir, 'wavesol') / 'lambda_map.fits')},
        params={
            "detect_sigma": detect_sigma,
            "detect_snr": detect_snr,
            "strength": strength,
            "lsf_strength": lsf_strength,
            "object_protection": protect_cfg,
        },
        outputs={
            "skycorr_raw": [str(out_dir / f"{Path(str(x)).stem}_skycorr_raw.fits") for x in obj_frames if (out_dir / f"{Path(str(x)).stem}_skycorr_raw.fits").exists()],
            "skysub_refined_raw": [str(out_dir / f"{Path(str(x)).stem}_skysub_refined_raw.fits") for x in obj_frames if (out_dir / f"{Path(str(x)).stem}_skysub_refined_raw.fits").exists()],
            "sky_residual_windows_csv": str(out_dir / "sky_residual_windows.csv"),
            "sky_residual_windows_json": str(out_dir / "sky_residual_windows.json"),
            "sky_observer_pack_md": str(out_dir / "sky_observer_pack.md"),
            "qc_sky_refine_json": str(out_dir / "qc_sky_refine.json"),
        },
        metrics={
            "sky_sigma_before_median": _med_metric("sky_sigma_before"),
            "sky_sigma_after_median": _med_metric("sky_sigma_after"),
            "sky_sigma_gain_median": _med_metric("sky_sigma_gain"),
            "n_windows_median": _med_metric("n_windows"),
            "n_corrected_windows_median": _med_metric("n_corrected_windows"),
            "object_delta_median": _med_metric("object_delta_median"),
        },
        flags=flags,
        qc=qc,
        extra={"per_exposure": per_exp},
    )
    return {
        "sky_refine_done": out_dir / "done.json",
        "qc_sky_refine_json": out_dir / "qc_sky_refine.json",
        "sky_residual_windows_csv": out_dir / "sky_residual_windows.csv",
        "sky_residual_windows_json": out_dir / "sky_residual_windows.json",
        "sky_observer_pack_md": out_dir / "sky_observer_pack.md",
    }
