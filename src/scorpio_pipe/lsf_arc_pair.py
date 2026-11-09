"""Diagnostic Gaussian-equivalent fits to a known two-line original ARC blend.

This is NOT automatically a science-object LSF. The air-wavelength centers
are external LineID references, not free parameters to re-identify lines.
Individual component amplitudes are fitted independently, with shared
width and a single displacement relative to a frozen 2D wavelength map.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares


def fit_arc_pair_diagnostic(
    wavelength_air_A: np.ndarray,
    arc_signal: np.ndarray,
    pair_air_A: tuple[float, float],
    *,
    half_window_px: int = 13,
    max_shift_px: float = 3.5,
    min_peak_amplitude: float = 20.0,
    max_normalized_rmse: float = 0.15,
) -> dict[str, Any]:
    """Fit two Gaussian components + linear continuum, or return REJECT.

    Only one original lamp is accepted at a time. A fitted pair must not be
    mistaken for a laboratory wavelength solution, a transferred sky LSF
    or a science-valid stellar line-spread function.
    """
    wave = np.asarray(wavelength_air_A, dtype=float)
    intensity = np.asarray(arc_signal, dtype=float)
    if (wave.ndim != 1 or intensity.ndim != 1 or wave.shape != intensity.shape
            or wave.size < 40 or not np.isfinite(wave).all()
            or not np.all(np.diff(wave) > 0)
            or not np.isfinite(intensity).all()):
        raise ValueError(
            "Joint ARC pair needs finite 1D data and increasing air wavelengths"
        )
    try:
        lab_a, lab_b = float(pair_air_A[0]), float(pair_air_A[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("ARC pair needs two laboratory-air wavelengths") from exc
    if not np.isfinite([lab_a, lab_b]).all() or not (
        wave[0] < lab_a < lab_b < wave[-1]
    ):
        raise ValueError("ARC pair must be finite, ordered and on the detector")
    if (
        half_window_px < 10
        or not np.isfinite(max_shift_px) or max_shift_px <= 0
        or not np.isfinite(min_peak_amplitude) or min_peak_amplitude <= 0
        or not np.isfinite(max_normalized_rmse) or max_normalized_rmse <= 0
    ):
        raise ValueError("Invalid ARC pair fitting configuration")
    xfull = np.arange(wave.size, dtype=float)
    ca, cb = [float(np.interp(lam, wave, xfull)) for lam in (lab_a, lab_b)]
    if cb - ca < 6.0:
        return {"status": "REJECT", "reason": "pair_centers_unresolved_for_stable_fit"}
    lo = int(np.floor(ca - half_window_px))
    hi = int(np.ceil(cb + half_window_px)) + 1
    if lo < 0 or hi > wave.size:
        return {"status": "REJECT", "reason": "insufficient_detector_edge_support"}
    xx, yy = xfull[lo:hi], intensity[lo:hi]
    edge = np.r_[yy[:4], yy[-4:]]
    bg = float(np.median(edge))
    noise = float(1.4826 * np.median(np.abs(edge - bg)))
    if not np.isfinite(noise) or noise <= 0:
        noise = float(np.std(edge))
    noise = max(noise, 2.0)
    aa = max(0.0, float(intensity[int(round(ca))] - bg))
    bb = max(0.0, float(intensity[int(round(cb))] - bg))
    if min(aa, bb) < min_peak_amplitude:
        return {"status": "REJECT", "reason": "insufficient_both_component_peaks"}
    center = (ca + cb) / 2.0

    def profile(q: np.ndarray) -> np.ndarray:
        a, b, shift, sigma, offset, slope = q
        return (
            a * np.exp(-0.5 * ((xx - ca - shift) / sigma) ** 2)
            + b * np.exp(-0.5 * ((xx - cb - shift) / sigma) ** 2)
            + offset + slope * (xx - center)
        )

    try:
        opt = least_squares(
            lambda q: profile(q) - yy,
            [aa, bb, 0.0, 2.8, bg, 0.0],
            bounds=(
                [0.0, 0.0, -max_shift_px, 1.0, -np.inf, -np.inf],
                [np.inf, np.inf, max_shift_px, 5.5, np.inf, np.inf],
            ),
            loss="soft_l1", f_scale=noise, max_nfev=180,
        )
    except (RuntimeError, FloatingPointError, ValueError) as exc:
        return {"status": "REJECT", "reason": "optimization_error", "detail": str(exc)}
    if not opt.success or not np.isfinite(opt.x).all():
        return {"status": "REJECT", "reason": "optimizer_no_convergence"}
    a, b, shift, sigma, _, _ = map(float, opt.x)
    if sigma <= 1.03 or sigma >= 5.45 or abs(shift) >= 0.99 * max_shift_px:
        return {"status": "REJECT", "reason": "fit_hit_physical_or_position_boundary"}
    if float(half_window_px) < 3.5 * sigma:
        return {"status": "REJECT", "reason": "insufficient_fit_wing_support"}
    rms = float(np.sqrt(np.mean((profile(opt.x) - yy) ** 2)))
    rel = rms / max(a, b, 1.0)
    if min(a, b) < min_peak_amplitude or rel > max_normalized_rmse:
        return {
            "status": "REJECT", "reason": "weak_or_mismatched_component_profile",
            "normalized_rmse": rel,
        }
    dispersion = float(np.median(np.diff(wave[int(ca):int(cb) + 1])))
    return {
        "status": "DIAGNOSTIC_ONLY",
        "line_centers_air_A": [lab_a, lab_b],
        "center_x_predicted_px": [ca, cb],
        "common_shift_px": shift,
        "fwhm_pix": float(2.354820045 * sigma),
        "fwhm_A": float(2.354820045 * sigma * dispersion),
        "dispersion_A_per_px": dispersion,
        "component_amplitudes_free": [a, b],
        "amplitude_ratio_measured": a / b if b > 0 else None,
        "normalized_rmse": rel,
        "window_half_px": int(half_window_px),
        "source": "single_original_neon_exposure",
        "wave_ref": "air",
        "science_lsf_validated": False,
        "safe_to_apply_to_sky_or_science": False,
        "warnings": [
            "Gaussian-equivalent ARC pair fit: wings/asymmetry not certified",
            "ARC LSF is not automatically stellar or night-sky LSF",
            "Requires independent lamp repeatability and LineID review",
        ],
    }
