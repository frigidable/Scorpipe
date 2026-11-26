"""Synthetic-only Monte Carlo on SAME-DATA optimal profile estimation.

Compares a known fixed spatial-profile Gaussian WLS estimator against
production Extract1D._optimal_extract, which re-estimates its own profile
from noisy input SCI each time. The covariance of a data-derived profile
is not contained in the conditional 1/denominator variance.

No real data, no SCI/VAR/MASK mutation, no observational qualification.
"""
from __future__ import annotations

from math import erf, sqrt
from typing import Any

import numpy as np

from scorpio_pipe.stages.extract1d import _optimal_extract
from scorpio_pipe.maskbits import NO_COVERAGE


def synthetic_optimal_profile_coverage(
    *,
    true_flux: float = 0.0,
    n_realizations: int = 400,
    seed: int = 20261104,
    smooth_lambda_sigma: float = 0.0,
) -> dict[str, Any]:
    """Compare independent known-profile WLS with adaptive Horne-like fit.

    The simulated trace, true profile, aperture, read variance and mask
    are known and constant. The *production* optimal extraction estimates
    its positive-clipped profile anew from every noisy science frame;
    this probes the estimator's endogeneity, not Stack2D or SkySub.

    Exactly one fixed central wavelength bin is used for marginal
    coverage so smoothing-induced adjacent-bin dependence does not
    masquerade as extra independent Monte Carlo trials.
    """
    if (isinstance(true_flux, (bool, np.bool_))
            or not isinstance(true_flux, (int, float, np.integer, np.floating))
            or not np.isfinite(true_flux) or not 0 <= true_flux <= 1000):
        raise ValueError("true_flux must be a finite nonnegative synthetic count")
    if (isinstance(n_realizations, (bool, np.bool_))
            or not isinstance(n_realizations, (int, np.integer))
            or not 200 <= int(n_realizations) <= 1500):
        raise ValueError("n_realizations must be an integer in [200,1500]")
    if (isinstance(seed, (bool, np.bool_)) or
            not isinstance(seed, (int, np.integer)) or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    if (isinstance(smooth_lambda_sigma, (bool, np.bool_))
            or not isinstance(smooth_lambda_sigma,
                              (int, float, np.integer, np.floating))
            or not np.isfinite(smooth_lambda_sigma)
            or not 0 <= smooth_lambda_sigma <= 10):
        raise ValueError("smooth_lambda_sigma must be in [0,10]")

    ny, nlam, yc, ap_hw = 9, 25, 4, 3
    wave_idx = nlam//2
    rows = np.arange(yc-ap_hw, yc+ap_hw+1)
    # An exactly known, wavelength-invariant normalized object profile.
    true_p = np.exp(-.5*((rows-yc)/1.1)**2)
    true_p /= np.sum(true_p)
    signal = np.zeros((ny, nlam))
    signal[rows, :] = float(true_flux)*true_p[:, None]
    variance = np.ones_like(signal)
    mask = np.zeros_like(signal, dtype=np.uint16)
    trace = np.full(nlam, float(yc))
    denom = float(np.dot(true_p, true_p))
    expected_fixed_var = 1. / denom
    analytic_weights = true_p/denom

    rng = np.random.default_rng(int(seed))
    n = int(n_realizations)
    fixed_errors = np.empty(n, dtype=float)
    adaptive_errors = np.empty(n, dtype=float)
    adaptive_variances = np.empty(n, dtype=float)
    meta_fallback = np.empty(n, dtype=float)
    adjacent_adaptive = np.empty((n, 2), dtype=float)
    fixed_adjacent = np.empty((n, 2), dtype=float)

    for i in range(n):
        draw = signal + rng.standard_normal((ny, nlam))
        # Independent reference estimator: known profile is NOT learned
        # from the realization. Its variance is exactly 1/(P.T P).
        ref = analytic_weights @ draw[rows, :]
        fixed_errors[i] = float(ref[wave_idx] - true_flux)
        fixed_adjacent[i] = ref[wave_idx:wave_idx+2] - true_flux

        flux, predicted_var, out_mask, _sky, npix, meta = _optimal_extract(
            draw, variance, mask, trace, ap_hw=ap_hw,
            fatal_bits=0, min_good_frac=1.0,
            sky_resid=None, apply_local_sky=False,
            smooth_lambda_sigma=float(smooth_lambda_sigma),
            clip_negative_profile=True, backend="numpy",
        )
        if (not np.isfinite(flux[wave_idx:wave_idx+2]).all()
                or not np.isfinite(predicted_var[wave_idx:wave_idx+2]).all()
                or np.any(predicted_var[wave_idx:wave_idx+2] <= 0)
                or np.any(out_mask[wave_idx:wave_idx+2] & NO_COVERAGE)
                or np.any(npix[wave_idx:wave_idx+2] != len(rows))):
            raise RuntimeError("Production optimal extraction unexpectedly lost clean synthetic coverage")
        adaptive_errors[i] = float(flux[wave_idx] - true_flux)
        adjacent_adaptive[i] = flux[wave_idx:wave_idx+2] - true_flux
        adaptive_variances[i] = float(predicted_var[wave_idx])
        meta_fallback[i] = float(meta["profile_fallback_frac"])

    empirical_fixed_var = float(np.var(fixed_errors, ddof=1))
    empirical_adaptive_var = float(np.var(adaptive_errors, ddof=1))
    nominal68 = erf(1./sqrt(2.))
    fixed_cov68 = float(np.mean(np.abs(fixed_errors)
                                <= sqrt(expected_fixed_var)))
    fixed_cov95 = float(np.mean(np.abs(fixed_errors)
                                <= 1.959963984540054*sqrt(expected_fixed_var)))
    adaptive_z = adaptive_errors/np.sqrt(adaptive_variances)
    adaptive_cov68 = float(np.mean(np.abs(adaptive_z) <= 1.))
    adaptive_cov95 = float(np.mean(np.abs(adaptive_z) <= 1.959963984540054))
    fixed_bias = float(np.mean(fixed_errors))
    adaptive_bias = float(np.mean(adaptive_errors))
    mean_reported_var = float(np.mean(adaptive_variances))
    var_tol = max(.08, 5.*sqrt(2./(n-1)))
    mean_tol = 5.*sqrt(expected_fixed_var/n)
    cov68_tol = max(.03, 5.*sqrt(nominal68*(1.-nominal68)/n))
    cov95_tol = max(.025, 5.*sqrt(.95*.05/n))

    reference_ok = bool(
        abs(empirical_fixed_var/expected_fixed_var-1) <= var_tol
        and abs(fixed_bias) <= mean_tol
        and abs(fixed_cov68-nominal68) <= cov68_tol
        and abs(fixed_cov95-.95) <= cov95_tol
    )
    detected_bias = bool(abs(adaptive_bias) > mean_tol)
    detected_undercoverage = bool(adaptive_cov95 < .95-cov95_tol)
    if not reference_ok:
        status = "REFERENCE_MC_MISMATCH"
    elif detected_bias or detected_undercoverage:
        status = "ADAPTIVE_PROFILE_DISCREPANCY"
    else:
        status = "NO_DETECTED_ADAPTIVE_DISCREPANCY"

    return {
        "status": status,
        "scope": "SYNTHETIC_SAME_DATA_OPTIMAL_PROFILE_MC",
        "n_realizations": n,
        "true_flux": float(true_flux),
        "smooth_lambda_sigma": float(smooth_lambda_sigma),
        "true_profile": true_p.tolist(),
        "known_fixed_profile_variance": expected_fixed_var,
        "fixed_empirical_variance": empirical_fixed_var,
        "fixed_mean_error": fixed_bias,
        "fixed_coverage_1sigma": fixed_cov68,
        "fixed_coverage_95percent": fixed_cov95,
        "fixed_reference_consistent": reference_ok,
        "adaptive_empirical_variance": empirical_adaptive_var,
        "adaptive_mean_reported_conditional_variance": mean_reported_var,
        "adaptive_mean_error": adaptive_bias,
        "adaptive_mean_bias_in_fixed_sigma": adaptive_bias/sqrt(expected_fixed_var),
        "adaptive_coverage_1sigma": adaptive_cov68,
        "adaptive_coverage_95percent": adaptive_cov95,
        "adaptive_bias_detected": detected_bias,
        "adaptive_undercoverage_detected": detected_undercoverage,
        "adaptive_profile_fallback_fraction": float(np.mean(meta_fallback)),
        "adaptive_adjacent_wavelength_covariance": float(np.cov(
            adjacent_adaptive, rowvar=False, ddof=1
        )[0, 1]),
        "fixed_adjacent_wavelength_covariance": float(np.cov(
            fixed_adjacent, rowvar=False, ddof=1
        )[0, 1]),
        "reference_tolerances": {
            "relative_variance": var_tol,
            "mean_absolute": mean_tol,
            "coverage_1sigma": cov68_tol,
            "coverage_95percent": cov95_tol,
        },
        "profile_reestimated_per_draw": True,
        "science_uncertainty_qualified": False,
        "production_variance_release_allowed": False,
    }
