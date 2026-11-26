"""Robust local Gaussian-equivalent ARC line profile measurement.

Measured ARC line FWHM is not automatically a verified science-object LSF.
Avoid the biased second moment of a truncated and background-subtracted line.
"""
from __future__ import annotations
from typing import Any

import numpy as np
from scipy.optimize import least_squares


def fit_gaussian_arc_profile(
    profile: np.ndarray,
    *,
    center_guess_px: float,
    half_window_px: int,
    min_snr: float = 4.0,
    max_center_displacement_px: float = 2.8,
    max_fractional_residual: float = 0.16,
) -> tuple[float | None, dict[str, Any]]:
    """Robust Gaussian plus linear continuum, with explicit science gates.

    If the profile wings exceed the selected window, reject rather than
    return a biased moment. Blended lines must be excluded before calling.
    """
    p = np.asarray(profile, dtype=np.float64)
    x0 = float(center_guess_px)
    hw = int(half_window_px)
    if p.ndim != 1 or hw < 5 or not np.isfinite(x0):
        raise ValueError("ARC fit needs 1D profile, half-window>=5 and finite seed")
    ix = int(round(x0))
    lo, hi = ix-hw, ix+hw+1
    if lo < 0 or hi > p.size:
        return None, {"reason": "insufficient_detector_edges"}
    xx = np.arange(lo, hi, dtype=float)
    ff = p[lo:hi]
    if not np.isfinite(ff).all():
        return None, {"reason": "nonfinite"}
    edge = min(4, max(2, ff.size//6))
    outside = np.r_[ff[:edge], ff[-edge:]]
    bg = float(np.median(outside))
    noise = float(1.4826*np.median(np.abs(outside-bg)))
    if not np.isfinite(noise) or noise <= 0:
        noise = float(np.std(outside))
    if not np.isfinite(noise) or noise <= 0:
        noise = 1.0
    amp = float(np.max(ff)-bg)
    if not np.isfinite(amp) or amp <= 0 or amp/max(noise,1.0) < min_snr:
        return None, {"reason": "low_snr", "amplitude": amp, "noise": noise}
    first_center = float(xx[np.argmax(ff)])
    if abs(first_center-x0) > max_center_displacement_px:
        return None, {"reason": "peak_far_from_manual_seed"}
    def residual(q: np.ndarray) -> np.ndarray:
        a, mu, sig, b, slope = q
        return a*np.exp(-0.5*((xx-mu)/sig)**2)+b+slope*(xx-x0)-ff
    p0 = np.asarray([amp, first_center, min(2.7,float(hw)/3.0), bg, 0.0])
    lower = [0.0, x0-max_center_displacement_px, 0.55, -np.inf, -np.inf]
    upper = [np.inf, x0+max_center_displacement_px, 8.0, np.inf, np.inf]
    try:
        fit = least_squares(
            residual, p0, bounds=(lower,upper), loss="soft_l1",
            f_scale=max(3.0,noise), max_nfev=150,
        )
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        return None, {"reason": "optimization_error", "detail": str(exc)}
    a, mu, sig, b, slope = map(float,fit.x)
    if not fit.success or not np.isfinite(fit.x).all() or sig <= 0.6 or sig >= 7.95:
        return None, {"reason": "fit_not_converged"}
    if hw < 3.0*sig:
        return None, {"reason": "window_truncates_profile_wings",
                      "half_window_px": hw,"sigma_pix": sig}
    rmse = float(np.sqrt(np.mean(residual(fit.x)**2)))
    if rmse/a > max_fractional_residual:
        return None, {"reason": "non_gaussian_or_blended_profile",
                      "normalized_rmse": rmse/a}
    return sig, {
        "reason": "accepted",
        "fit_kind": "gaussian_plus_linear_background_robust",
        "center_pix": mu, "sigma_pix": sig,
        "fwhm_pix": 2.354820045*sig,
        "amp": a,"noise": noise,
        "norm_rmse": rmse/a,"window_half_px": hw,
        "fit_background": b,"fit_slope": slope,
    }


def isolated_arc_centers_in_pixels(
    local_line_centers_pix: np.ndarray,
    *,
    half_window_px: int,
    separation_extra_px: float = 2.0,
) -> np.ndarray:
    """Reject known lines too close to other anchors for isolated profile fit.

    Unidentified neighbours may still contaminate the fit and must be
    rejected separately by the fit residual gate.
    """
    positions = np.asarray(local_line_centers_pix, float)
    if positions.ndim != 1 or not np.isfinite(positions).all():
        raise ValueError("ARC line positions must be a finite vector")
    if half_window_px < 5 or not np.isfinite(separation_extra_px) or separation_extra_px < 0:
        raise ValueError("Invalid ARC line separation configuration")
    if positions.size == 0:
        return np.empty(0,dtype=bool)
    sep = np.abs(positions[:,None]-positions[None,:])
    np.fill_diagonal(sep,np.inf)
    return np.min(sep,axis=1) >= 2.0*half_window_px+separation_extra_px



def bracketed_arc_wavelength_support(
    wavelength_map_air_A: np.ndarray,
    empirically_fitted_line_wavelengths_A: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | list[float] | None]]:
    """Conservative wavelength-span marker for *measured* ARC LSF.

    This mask asserts only that a pixel is between the extreme fitted
    laboratory lines. It does NOT assert adequate density of lines, a
    validated line profile, or a verified LSF for the science object.
    """
    wave = np.asarray(wavelength_map_air_A, dtype=float)
    lines = np.sort(np.unique(np.asarray(
        empirically_fitted_line_wavelengths_A, dtype=float
    )))
    if wave.ndim != 2 or lines.ndim != 1 or lines.size < 3:
        raise ValueError(
            "ARC LSF support needs a 2D wavelength map and at least three "
            "independently modeled line centers"
        )
    if not np.all(np.isfinite(lines)) or np.any(lines <= 0):
        raise ValueError("ARC LSF line centers must be finite and positive")
    lo, hi = float(lines[0]), float(lines[-1])
    support = np.isfinite(wave) & (wave >= lo) & (wave <= hi)
    valid = np.isfinite(wave)
    frac = float(np.count_nonzero(support)/max(np.count_nonzero(valid),1))
    return support, {
        "measured_line_span_A": [lo,hi],
        "empirical_span_fraction_of_finite_wave_map": frac,
        "extrapolated_fraction_of_finite_wave_map": 1.0 - frac,
        "maximum_gap_between_modeled_lines_A": float(np.max(np.diff(lines))),
        "interpretation": (
            "Line-span bracketing only; no verified LSF shape, gap "
            "interpolation quality, point-source illumination or sky flexure"
        ),
    }
