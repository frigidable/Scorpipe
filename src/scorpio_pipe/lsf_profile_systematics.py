"""Diagnostic model-systematics checks for original-ARC two-line profiles.

Fits separated original lamps, never a merged SuperNeon. Continuum degree
and window sensitivity are *not* calibrated uncertainties and cannot license
a science-object LSF or a wavelength correction.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import least_squares

_GAUSSIAN_FWHM = 2.354820045


def fit_neon_pair_hypothesis(
    wavelength_air_A: np.ndarray,
    arc_profile: np.ndarray,
    laboratory_air_A: tuple[float, float],
    *,
    half_window_px: int = 16,
    continuum_degree: int = 1,
    min_component_peak: float = 20.0,
    max_normalized_rmse: float = 0.18,
) -> dict[str, Any]:
    """Common-width Ne Gaussians + background, with explicit failure reasons.

    Laboratory separation is fixed from reviewed LineID; component strengths
    are nonnegative and independently fitted, and common pixel shift is free.
    """
    wave = np.asarray(wavelength_air_A, dtype=float)
    obs = np.asarray(arc_profile, dtype=float)
    if (
        wave.ndim != 1 or obs.shape != wave.shape or wave.size < 50
        or not np.all(np.isfinite(wave)) or not np.all(np.isfinite(obs))
        or not np.all(np.diff(wave) > 0)
    ):
        raise ValueError("ARC needs finite 1D observed profile and increasing wavelength map")
    try:
        lab_a, lab_b = [float(v) for v in laboratory_air_A]
    except (TypeError, ValueError) as exc:
        raise ValueError("Exactly two laboratory air wavelengths required") from exc
    if not (
        np.isfinite(lab_a) and np.isfinite(lab_b)
        and wave[0] < lab_a < lab_b < wave[-1]
    ):
        raise ValueError("ARC laboratory air wavelengths must be ordered within detector")
    if (
        not isinstance(half_window_px, int) or half_window_px < 10
        or continuum_degree not in (1, 2)
        or not np.isfinite(min_component_peak) or min_component_peak <= 0
        or not np.isfinite(max_normalized_rmse) or max_normalized_rmse <= 0
    ):
        raise ValueError("Invalid profile model parameters")
    xfull = np.arange(wave.size, dtype=float)
    ca, cb = np.interp([lab_a, lab_b], wave, xfull)
    if cb - ca < 6.0:
        return {"status": "REJECT", "reason": "components_too_close_for_independent_pair_fit"}
    lo = int(np.floor(ca-half_window_px))
    hi = int(np.ceil(cb+half_window_px))+1
    if lo < 0 or hi > wave.size:
        return {"status": "REJECT", "reason": "detector_edge_truncation"}
    x = xfull[lo:hi]
    obs_local = obs[lo:hi]
    center = 0.5*(ca+cb)
    bg = float(np.median(np.r_[obs_local[:4], obs_local[-4:]]))
    # Second-difference scale is an optimizer roughness proxy, NOT a CCD VAR.
    d2 = np.diff(obs_local, n=2)
    rough_noise = max(
        2.0,
        float(1.4826*np.median(np.abs(d2-np.median(d2)))/np.sqrt(6.0)),
    )
    amp_a = max(0.0, float(obs[int(round(ca))]-bg))
    amp_b = max(0.0, float(obs[int(round(cb))]-bg))
    if min(amp_a, amp_b) < min_component_peak:
        return {"status": "REJECT", "reason": "weak_component"}
    extra = continuum_degree == 2

    def model(q: np.ndarray) -> np.ndarray:
        a, b, shift, sigma, intercept, slope = q[:6]
        t = x-center
        return (
            a*np.exp(-0.5*((x-ca-shift)/sigma)**2)
            + b*np.exp(-0.5*((x-cb-shift)/sigma)**2)
            + intercept + slope*t
            + (q[6]*t*t if extra else 0.0)
        )

    p0 = [amp_a, amp_b, 0.0, 2.8, bg, 0.0] + ([0.0] if extra else [])
    lower = [0., 0., -3.5, 1., -np.inf, -np.inf] + ([-np.inf] if extra else [])
    upper = [np.inf, np.inf, 3.5, 5.5, np.inf, np.inf] + ([np.inf] if extra else [])
    try:
        fit = least_squares(
            lambda q: model(q)-obs_local, p0, bounds=(lower, upper),
            loss="soft_l1", f_scale=rough_noise, max_nfev=170,
        )
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        return {"status": "REJECT", "reason": "optimizer_exception", "detail": str(exc)}
    if not fit.success or not np.isfinite(fit.x).all():
        return {"status": "REJECT", "reason": "optimizer_failure"}
    a, b, shift, sigma = map(float, fit.x[:4])
    if sigma <= 1.05 or sigma >= 5.45 or abs(shift) >= 3.45:
        return {"status": "REJECT", "reason": "parameter_bound"}
    if half_window_px < 3.3*sigma:
        return {"status": "REJECT", "reason": "insufficient_wing_support"}
    residual = model(fit.x)-obs_local
    rms = float(np.sqrt(np.mean(residual**2)))
    nrms = rms/max(a, b, 1.0)
    if min(a, b) < min_component_peak or nrms > max_normalized_rmse:
        return {"status": "REJECT", "reason": "weak_or_bad_profile"}
    dispersion = float(np.median(np.diff(wave[int(ca):int(cb)+1])))
    z = (x-ca-shift)/sigma
    odd = np.exp(-0.5*z*z)*(z**3-3*z)
    odd_projection = float(
        np.dot(residual, odd)/(max(a,b)*max(float(np.dot(odd,odd)),1e-9))
    )
    return {
        "status": "DIAGNOSTIC_FIT",
        "fwhm_A": float(_GAUSSIAN_FWHM*sigma*dispersion),
        "fwhm_px": float(_GAUSSIAN_FWHM*sigma),
        "shift_px": shift,
        "normalized_rms": nrms,
        "residual_odd_projection": odd_projection,
        "continuum_degree": continuum_degree,
        "half_window_px": half_window_px,
        "amplitude_ratio": float(a/b),
        "rough_noise_proxy_adu": rough_noise,
        "science_lsf_validated": False,
        "auto_apply_allowed": False,
    }


def evaluate_neon_pair_systematics(
    wavelength_air_A: np.ndarray,
    arc_profile: np.ndarray,
    laboratory_air_A: tuple[float, float],
    *,
    half_windows_px: Sequence[int] = (11, 13, 16, 20, 24),
    continuum_degrees: Sequence[int] = (1, 2),
) -> dict[str, Any]:
    """Run a multi-hypothesis profile sensitivity audit; no science gate."""
    windows = tuple(half_windows_px)
    orders = tuple(continuum_degrees)
    if len(windows) < 3 or len(set(windows)) != len(windows) or any(
        not isinstance(v,int) or v < 10 for v in windows
    ):
        raise ValueError("At least three distinct windows >=10 px required")
    if orders != (1, 2):
        raise ValueError("Both linear and quadratic continuum models required")
    trials = []
    for hw in windows:
        for deg in orders:
            fit = fit_neon_pair_hypothesis(
                wavelength_air_A, arc_profile, laboratory_air_A,
                half_window_px=hw, continuum_degree=deg,
            )
            trials.append({"half_window_px":hw,"continuum_degree":deg,**fit})
    accepted = [q for q in trials if q["status"]=="DIAGNOSTIC_FIT"]
    if len(accepted) < 4:
        return {
            "status":"REJECT","reason":"too_few_valid_hypotheses",
            "trials":trials,"science_lsf_validated":False,
            "auto_apply_allowed":False,
        }
    widths = np.asarray([q["fwhm_A"] for q in accepted],float)
    contrasts = []
    for hw in windows:
        a = [q for q in accepted if
             q["half_window_px"]==hw and q["continuum_degree"]==1]
        b = [q for q in accepted if
             q["half_window_px"]==hw and q["continuum_degree"]==2]
        if a and b:
            contrasts.append({
                "half_window_px":hw,
                "quadratic_minus_linear_FWHM_A":float(b[0]["fwhm_A"]-a[0]["fwhm_A"])
            })
    return {
        "status":"DIAGNOSTIC_ONLY",
        "n_trials":len(trials),
        "n_accepted":len(accepted),
        "fwhm_hypothesis_range_A":[float(np.min(widths)),float(np.max(widths))],
        "fwhm_hypothesis_spread_A":float(np.ptp(widths)),
        "continuum_model_differences":contrasts,
        "trials":trials,
        "uncertainty_status":"MODEL_SENSITIVITY_ONLY_NOT_STATISTICAL_CI",
        "science_lsf_validated":False,
        "auto_apply_allowed":False,
    }
