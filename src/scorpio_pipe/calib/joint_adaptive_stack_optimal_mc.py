"""Synthetic paired Monte Carlo for adaptive Stack2D then optimal Extract1D.

Compares four estimators on IDENTICAL noisy exposures:
    invvar stack    + known fixed extraction profile
    Huber stack     + known fixed extraction profile
    invvar stack    + production data-derived optimal profile
    Huber stack     + production data-derived optimal profile

The actual production Stack2D streaming combiner and optimal extractor
are run on every Monte Carlo draw. A known-profile WLS reference and an
exact grouped-MasterBias variance positive control are independent.

This is a small synthetic diagnostic, not a production reduction path.
Neither observational calibration nor release of scientific VAR follows.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from math import erf, sqrt
from typing import Any

import numpy as np
from astropy.io import fits

from scorpio_pipe.maskbits import NO_COVERAGE
from scorpio_pipe.stages.extract1d import _optimal_extract
from scorpio_pipe.stages.stack2d import _stack_chunk_streaming


def synthetic_joint_stack_optimal_coverage(
    *,
    group_ids: Sequence[str] = ("A", "B", "C"),
    true_flux: float = 0.,
    n_realizations: int = 350,
    seed: int = 20261104,
    huber_c: float = 1.,
    robust_iter: int = 2,
    smooth_lambda_sigma: float = 0.,
) -> dict[str, Any]:
    """Assess adaptive combination and profile self-estimation in one trial.

    Three SCI exposures have known identical object truth and independent
    N(0,1) noise. Each additionally has one group-specific MasterBias
    N(0,1) noise field, shared at the exact same CCD pixels within a
    named calibration group. Each input therefore has diagonal VAR=2.

    Since inputs have equal variance and no rejected pixels, the
    inverse-variance Stack2D + true-profile reference has EXACT grouped
    sampling variance [1/3 + sum_g (n_g/3)**2]/sum(P_true**2).
    By contrast the *published conditional independent-input* formula
    reports [2/3]/sum(P_true**2). The two agree only when all
    MasterBias groups are different.

    Huber residual weights and Extract1D spatial profiles are re-learned
    independently for each noise draw. Observables are evaluated at one
    fixed central wavelength; neighboring spectral pixels are not
    treated as additional independent Monte Carlo samples.
    """
    labels = tuple(group_ids)
    if (len(labels) != 3 or
            any(not isinstance(g, str) or not g or g.strip() != g
                for g in labels)):
        raise ValueError("Exactly three explicit nonblank bias group IDs required")
    if (isinstance(true_flux, (bool, np.bool_))
            or not isinstance(true_flux, (int, float, np.integer, np.floating))
            or not np.isfinite(true_flux) or not 0 <= true_flux <= 1000.):
        raise ValueError("true_flux must be a bounded nonnegative count")
    if (isinstance(n_realizations, (bool, np.bool_))
            or not isinstance(n_realizations, (int, np.integer))
            or not 250 <= n_realizations <= 1500):
        raise ValueError("n_realizations must be an integer in [250,1500]")
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    if (isinstance(huber_c, (bool, np.bool_))
            or not isinstance(huber_c, (int, float, np.integer, np.floating))
            or not np.isfinite(huber_c) or not .1 <= huber_c <= 20):
        raise ValueError("huber_c must be finite in [0.1,20]")
    if (isinstance(robust_iter, (bool, np.bool_))
            or not isinstance(robust_iter, (int, np.integer))
            or not 0 <= robust_iter <= 5):
        raise ValueError("robust_iter must be integer in [0,5]")
    if (isinstance(smooth_lambda_sigma, (bool, np.bool_))
            or not isinstance(smooth_lambda_sigma,
                              (int, float, np.integer, np.floating))
            or not np.isfinite(smooth_lambda_sigma)
            or not 0 <= smooth_lambda_sigma <= 10):
        raise ValueError("smooth_lambda_sigma must be finite in [0,10]")

    ny, nlam, n_exposures = 9, 21, 3
    center, half_width = 4, 3
    rows = np.arange(center-half_width, center+half_width+1)
    channel = nlam//2
    profile = np.exp(-.5*((rows-center)/1.1)**2)
    profile /= np.sum(profile)
    profile_sq = float(np.dot(profile, profile))
    source = np.zeros((ny, nlam), dtype=np.float64)
    source[rows] = float(true_flux)*profile[:, None]
    raw_var = np.full((ny, nlam), 2., dtype=np.float32)
    mask_in = np.zeros((ny, nlam), dtype=np.uint16)
    trace = np.full(nlam, float(center))

    counts = Counter(labels)
    variance_exact_grouped = (1./3. + sum(
        (amount/3.)**2 for amount in counts.values()
    )) / profile_sq
    variance_naive = (2./3.) / profile_sq
    n = int(n_realizations)
    rng = np.random.default_rng(int(seed))
    bias_noise = {
        g: rng.standard_normal((n, ny, nlam))
        for g in sorted(set(labels))
    }
    independent_noise = rng.standard_normal((n, n_exposures, ny, nlam))

    hduls = [
        fits.HDUList([
            fits.PrimaryHDU(data=np.zeros((ny, nlam), dtype=np.float32)),
            fits.ImageHDU(data=raw_var.copy(), name="VAR"),
            fits.ImageHDU(data=mask_in.copy(), name="MASK"),
        ])
        for _ in range(n_exposures)
    ]
    labels_mode = ("invvar", "huber")
    labels_profile = ("known", "adaptive")
    fluxes = np.zeros((n, 2, 2), dtype=np.float64)
    formal_vars = np.zeros_like(fluxes)
    rejected = np.zeros((n, 2), dtype=np.float64)

    try:
        for k in range(n):
            for e, hdul in enumerate(hduls):
                hdul[0].data[:] = (
                    source + independent_noise[k, e] + bias_noise[labels[e]][k]
                ).astype(np.float32)
            for mode_id, method in enumerate(("invvar", "invvar_huber")):
                stacked, conditional_var, stack_mask, coverage, n_suppress, total = (
                    _stack_chunk_streaming(
                        hduls=hduls, y0=0, y1=ny,
                        method=method,
                        robust_iter=int(robust_iter) if mode_id else 0,
                        huber_c=float(huber_c), clip_sigma=4.,
                        y_shifts=[0., 0., 0.], y_align_enabled=False,
                        sci_scales=[1., 1., 1.],
                        var_scales=[1., 1., 1.],
                        fatal_bits=np.uint16(NO_COVERAGE), var_floor=None,
                    )
                )
                if (stacked.shape != (ny, nlam) or
                        not np.isfinite(stacked).all() or
                        not np.isfinite(conditional_var).all() or
                        np.any(conditional_var <= 0) or
                        not np.all(coverage == n_exposures) or
                        total != ny*nlam):
                    raise RuntimeError("Synthetic Stack2D lost valid input coverage")
                rejected[k, mode_id] = n_suppress / total

                # Independent known-profile WLS estimator, conditional
                # on the *reported* post-stack per-pixel diagonal VAR.
                pixel_v = np.asarray(
                    conditional_var[rows, channel], dtype=np.float64,
                )
                pixel_s = np.asarray(stacked[rows, channel], dtype=np.float64)
                denominator = float(np.sum(profile**2/pixel_v))
                if not denominator > 0 or not np.isfinite(denominator):
                    raise RuntimeError("Invalid fixed-profile WLS denominator")
                fluxes[k, mode_id, 0] = float(
                    np.sum(profile*pixel_s/pixel_v)/denominator
                )
                formal_vars[k, mode_id, 0] = 1./denominator

                # For a verified stacked product, Stack2D REJECTED is
                # contributor provenance, not fatal input for Extract1D.
                # NO_COVERAGE remains fatal. No actual output FITS are
                # created in this synthetic experiment.
                extracted, extracted_var, extracted_mask, _sky, npix, _meta = (
                    _optimal_extract(
                        stacked, conditional_var, stack_mask, trace,
                        ap_hw=half_width, fatal_bits=int(NO_COVERAGE),
                        min_good_frac=1., sky_resid=None,
                        apply_local_sky=False,
                        smooth_lambda_sigma=float(smooth_lambda_sigma),
                        clip_negative_profile=True, backend="numpy",
                    )
                )
                if (not np.isfinite(extracted[channel])
                        or not np.isfinite(extracted_var[channel])
                        or extracted_var[channel] <= 0
                        or extracted_mask[channel] & NO_COVERAGE
                        or npix[channel] != len(rows)):
                    raise RuntimeError("Synthetic production Extract1D lost central aperture")
                fluxes[k, mode_id, 1] = float(extracted[channel])
                formal_vars[k, mode_id, 1] = float(extracted_var[channel])
    finally:
        for hdul in hduls:
            hdul.close()

    errors = fluxes - float(true_flux)
    two_sigma = 1.959963984540054
    nominal68 = erf(1./sqrt(2.))
    var_tol = max(.10, 5.*sqrt(2./(n-1)))
    nominal95_tol = max(.025, 5.*sqrt(.95*.05/n))
    mean_tol = 5.*sqrt(variance_exact_grouped/n)
    results: dict[str, Any] = {}
    for m, mode in enumerate(labels_mode):
        for j, profile_mode in enumerate(labels_profile):
            err = errors[:, m, j]
            vv = formal_vars[:, m, j]
            key = f"{mode}_{profile_mode}"
            results[key] = {
                "mean_error": float(err.mean()),
                "empirical_variance": float(np.var(err, ddof=1)),
                "mean_reported_conditional_variance": float(vv.mean()),
                "variance_ratio": float(np.var(err, ddof=1) / vv.mean()),
                "coverage_68percent": float(np.mean(np.abs(err) <= np.sqrt(vv))),
                "coverage_95percent": float(np.mean(np.abs(err) <= two_sigma*np.sqrt(vv))),
                "mean_standardized_error": float(np.mean(err/np.sqrt(vv))),
            }

    # This known-profile + nonrobust control must agree with a grouped
    # Gaussian reference, not automatically with the published conditional
    # VAR which lacks the cross-exposure calibration covariance.
    base_err = errors[:, 0, 0]
    true_coverage95 = float(
        np.mean(np.abs(base_err) <= two_sigma*sqrt(variance_exact_grouped))
    )
    reference_ok = bool(
        abs(np.var(base_err, ddof=1)/variance_exact_grouped-1.) <= var_tol
        and abs(np.mean(base_err)) <= mean_tol
        and abs(true_coverage95-.95) <= nominal95_tol
        and abs(formal_vars[:, 0, 0].mean()/variance_naive-1.) < 1e-6
    )

    def _paired(effect: np.ndarray) -> dict[str, float | bool]:
        difference = np.asarray(effect, dtype=np.float64)
        stderr = float(np.std(difference, ddof=1)/sqrt(n))
        mean = float(np.mean(difference))
        return {
            "mean": mean,
            "standard_error": stderr,
            "detected_at_5sigma": bool(abs(mean) > 5.*stderr),
        }

    # Paired contrasts on EXACTLY the same noise draws.
    # D_profile: effect of estimating the profile in a given stack regime.
    # D_stack: effect of robust weights for a given extraction regime.
    # D_interaction: joint effect minus both separate contrasts.
    profile_effect_invvar = _paired(errors[:, 0, 1]-errors[:, 0, 0])
    profile_effect_huber = _paired(errors[:, 1, 1]-errors[:, 1, 0])
    stack_effect_known = _paired(errors[:, 1, 0]-errors[:, 0, 0])
    stack_effect_adaptive = _paired(errors[:, 1, 1]-errors[:, 0, 1])
    interaction = _paired(
        (errors[:, 1, 1]-errors[:, 1, 0])
        - (errors[:, 0, 1]-errors[:, 0, 0])
    )

    return {
        "status": "PAIRED_MC_REPORTED" if reference_ok else "REFERENCE_MC_MISMATCH",
        "scope": "SYNTHETIC_ACTUAL_STACK2D_AND_OPTIMAL_EXTRACT_JOINT_MC",
        "n_realizations": n,
        "group_ids": list(labels),
        "true_flux": float(true_flux),
        "huber_c": float(huber_c),
        "robust_iter": int(robust_iter),
        "smooth_lambda_sigma": float(smooth_lambda_sigma),
        "known_true_profile": profile.tolist(),
        "exact_invvar_known_profile_grouped_variance": float(
            variance_exact_grouped
        ),
        "naive_invvar_known_profile_variance": float(variance_naive),
        "known_grouped_reference_coverage95": true_coverage95,
        "known_grouped_reference_consistent": reference_ok,
        "estimators": results,
        "paired_effects": {
            "adaptive_profile_after_invvar": profile_effect_invvar,
            "adaptive_profile_after_huber": profile_effect_huber,
            "adaptive_huber_with_known_profile": stack_effect_known,
            "adaptive_huber_with_data_profile": stack_effect_adaptive,
            "huber_profile_interaction": interaction,
        },
        "mean_suppression_fraction": {
            "invvar": float(rejected[:, 0].mean()),
            "huber": float(rejected[:, 1].mean()),
        },
        "reference_tolerances": {
            "variance_relative": float(var_tol),
            "mean_absolute": float(mean_tol),
            "reference_coverage95": float(nominal95_tol),
        },
        "nominal_coverage_68percent": float(nominal68),
        "adaptive_profile_relearned_per_draw": True,
        "huber_weights_relearned_per_draw": True,
        "science_uncertainty_qualified": False,
        "production_variance_release_allowed": False,
    }
