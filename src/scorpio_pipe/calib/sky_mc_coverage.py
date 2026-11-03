"""Synthetic-only coverage test for a fixed Kelson sky fit with shared MasterBias.

An *independent generative oracle* draws all pre-sky CCD pixel noise and
re-fits each exposure's 2D sky with dense QR least squares. This is
deliberately independent of the sparse FrozenSkyWLS inverse/adjoint used
to PREDICT covariance. No real data or production FITS/VAR are touched.

Under the specified Gaussian, per-CCD-pixel independent noise and shared
MasterBias fields, the analytic conditional covariance should match Monte
Carlo sample covariance and marginal normal-interval coverage.
"""
from __future__ import annotations

from collections.abc import Sequence
from math import erf, sqrt

import numpy as np
from scipy.linalg import solve_triangular

from scorpio_pipe.calib.sky_bias_covariance import (
    covariance_of_frozen_sky_grouped,
    frozen_kelson_bspline_operator,
)
from scorpio_pipe.stages.sky_sub import _make_bspline_knot_vector


def synthetic_grouped_sky_coverage(
    *,
    group_ids: Sequence[str] = ("A", "A"),
    y_order: int = 0,
    n_realizations: int = 8000,
    seed: int = 20261104,
) -> dict[str, object]:
    """Verify finite-sample variance, cross-covariance and interval coverage.

    Model assumptions are intentionally explicit:
    - two independent science exposures, each with its own final fixed
      Kelson spline fit and diagonal positive pre-sky variance;
    - each realization draws a NEW MasterBias error map for each group;
      every exposure referencing a group reuses its same CCD error map;
    - MasterBias errors are independent between detector pixels and groups;
      independent science pixel errors are independent between exposures;
    - the 2D lambda map, sky ROI, pixel weights and knots never change;
    - no clipping decisions, flat-estimator noise, Poisson sky variance
      re-estimation, data-dependent geometry or pixel resampling.

    The independent Monte Carlo fitter uses dense QR on each weighted
    design, not the sparse normal-equation factorization used to predict
    the covariance. This avoids an inverse-versus-inverse tautology.
    """
    labels = tuple(group_ids)
    if (len(labels) != 2 or
            any(not isinstance(x, str) or not x.strip() or x != x.strip()
                for x in labels)):
        raise ValueError("Exactly two nonblank explicit MasterBias group IDs required")
    if isinstance(y_order, (bool, np.bool_)) or y_order not in (0, 1):
        raise ValueError("Synthetic Kelson y_order must be 0 or 1")
    if (isinstance(n_realizations, (bool, np.bool_)) or
            not isinstance(n_realizations, (int, np.integer)) or
            not 1000 <= n_realizations <= 20000):
        raise ValueError("Monte Carlo realizations must be an integer in [1000,20000]")
    if (isinstance(seed, (bool, np.bool_)) or
            not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("Finite nonnegative Monte Carlo seed required")

    ny, nx = 12, 20
    n_pixels = ny * nx
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    candidate = np.zeros((ny, nx), dtype=bool)
    candidate[0:4, :] = True
    candidate[8:12, :] = True

    # Two non-identical science apertures with overlapping spectral bins.
    left_aperture = np.zeros((ny, nx), dtype=np.float64)
    right_aperture = np.zeros((ny, nx), dtype=np.float64)
    left_aperture[4:8, :] = 1. / (4. * nx * 2.)
    right_aperture[4:8, 3:13] = 1. / (4. * 10. * 2.)

    # Distinct but known detector-pixel variances; no estimation uncertainty
    # in these variance maps is included in this controlled oracle.
    bias_variance = (1.30 + .30 * xx / (nx-1)
                     + .08 * yy / (ny-1)).ravel()
    source_variance = []
    sensitivities = []
    predicted_operators = []
    dense_fitters = []
    functionals_a = []
    functionals_b = []
    true_skies = []

    for e in range(2):
        wave = 5100. + (1.32 + .012*e)*xx + (.035 + .01*e)*yy
        knots = _make_bspline_knot_vector(
            float(np.min(wave)), float(np.max(wave)),
            knot_step_A=4.0, degree=2,
        )
        v_ind = (0.85 + .16*e + .20*xx/(nx-1)
                 + .10*yy/(ny-1)).ravel()
        sensitivity = -(0.82 + .08*e + .09*xx/(nx-1)
                        + .03*yy/(ny-1)).ravel()
        v_fit = (v_ind + sensitivity**2 * bias_variance).reshape(ny, nx)
        op = frozen_kelson_bspline_operator(
            wave, 2.*(np.arange(ny) - .5*(ny-1))/(ny-1),
            knots=knots, degree=2, y_order=y_order,
            final_fit_mask=candidate, variance=v_fit,
        )
        x_full = op.design.toarray()
        chosen = op.fit_indices
        fit_inv_sigma = 1. / np.sqrt(v_fit.ravel()[chosen])
        fit_matrix = x_full[chosen, :] * fit_inv_sigma[:, None]
        q, r = np.linalg.qr(fit_matrix, mode="reduced")
        if (q.shape[1] != x_full.shape[1] or
                not np.isfinite(r).all() or
                np.linalg.cond(r) > 1e8):
            raise ValueError("Synthetic QR sky basis ill-conditioned")

        # A non-zero sky is fitted and removed on each realization.
        ncoef = x_full.shape[1]
        truth_coeff = 15. + 2.*np.sin(np.arange(ncoef)*.3)
        if y_order:
            truth_coeff[ncoef//2:] *= .08
        sky_true = x_full @ truth_coeff
        true_skies.append(sky_true)
        dense_fitters.append((x_full, chosen, fit_inv_sigma, q, r))
        predicted_operators.append(op)
        source_variance.append(v_ind)
        sensitivities.append(sensitivity)
        functionals_a.append((1. if e == 0 else .88) *
                             left_aperture.ravel().copy())
        functionals_b.append((1. if e == 0 else 1.10) *
                             right_aperture.ravel().copy())

    bias_by_group = {gid: bias_variance.copy() for gid in set(labels)}
    analytic = np.empty((2, 2), dtype=np.float64)
    independent = np.empty((2, 2), dtype=np.float64)
    shared = np.empty((2, 2), dtype=np.float64)
    funs = [functionals_a, functionals_b]
    for i in range(2):
        for j in range(2):
            ci, cs, ct = covariance_of_frozen_sky_grouped(
                predicted_operators, funs[i], funs[j],
                source_variance, sensitivities, labels, bias_by_group,
            )
            independent[i, j] = ci
            shared[i, j] = cs
            analytic[i, j] = ct
    if (not np.isfinite(analytic).all()
            or np.min(np.linalg.eigvalsh(analytic)) <= 0
            or np.any(np.diag(analytic) <= 0)):
        raise ValueError("Predicted synthetic covariance is not positive definite")

    # Crucially this NEVER calls FrozenSkyWLS.apply/adjoint. We simulate raw
    # pixel-level CCD errors then fit each sample by an independent dense QR
    # weighted least-squares solution, including the original nonzero sky.
    rng = np.random.default_rng(seed)
    masterbias_noise = {
        gid: rng.standard_normal((n_realizations, n_pixels))
            * np.sqrt(bias_variance)[None, :]
        for gid in sorted(set(labels))
    }
    measurements = np.zeros((n_realizations, 2), dtype=np.float64)
    for e in range(2):
        v_ind = source_variance[e]
        noise = rng.standard_normal((n_realizations, n_pixels)) * np.sqrt(v_ind)
        signal = (true_skies[e][None, :] + noise
                  + sensitivities[e][None, :] * masterbias_noise[labels[e]])
        x_full, chosen, weight, q, r = dense_fitters[e]
        weighted_right = (signal[:, chosen] * weight[None, :]) @ q
        fitted_coefficients = solve_triangular(
            r, weighted_right.T, lower=False, check_finite=True
        ).T
        sky_subtracted = signal - fitted_coefficients @ x_full.T
        measurements[:, 0] += sky_subtracted @ functionals_a[e]
        measurements[:, 1] += sky_subtracted @ functionals_b[e]

    empirical = np.cov(measurements, rowvar=False, ddof=1)
    z = measurements / np.sqrt(np.diag(analytic))[None, :]
    z_independent_only = measurements / np.sqrt(np.diag(independent))[None, :]
    coverage_1sigma = np.mean(np.abs(z) <= 1., axis=0)
    coverage_95 = np.mean(np.abs(z) <= 1.95996398454, axis=0)
    coverage_95_independent_only = np.mean(
        np.abs(z_independent_only) <= 1.95996398454, axis=0
    )
    expected_1sigma = erf(1. / sqrt(2.))
    expected_95 = .95
    variance_relative_error = np.abs(
        np.diag(empirical) / np.diag(analytic) - 1.
    )
    cross_covariance_error = (
        abs(empirical[0, 1] - analytic[0, 1])
        / sqrt(analytic[0, 0] * analytic[1, 1])
    )
    mean_z = np.mean(z, axis=0)
    # Five-sigma finite-Monte-Carlo bounds, with a numerical floor.
    variance_limit = max(.05, 5.*sqrt(2. / (n_realizations-1)))
    cov_limit = max(.05, 5./sqrt(n_realizations-1))
    cov1_limit = max(.025, 5.*sqrt(expected_1sigma*(1.-expected_1sigma)/n_realizations))
    cov95_limit = max(.02, 5.*sqrt(expected_95*(1.-expected_95)/n_realizations))
    consistent = (
        np.max(variance_relative_error) <= variance_limit
        and cross_covariance_error <= cov_limit
        and np.max(np.abs(coverage_1sigma-expected_1sigma)) <= cov1_limit
        and np.max(np.abs(coverage_95-expected_95)) <= cov95_limit
        and np.max(np.abs(mean_z)) <= 5. / sqrt(n_realizations)
    )
    return {
        "status": "MC_CONSISTENT" if consistent else "MC_MISMATCH",
        "scope": "SYNTHETIC_GAUSSIAN_FIXED_KELSON_WLS",
        "group_ids": list(labels),
        "y_order": int(y_order),
        "n_realizations": int(n_realizations),
        "analytic_covariance": analytic.tolist(),
        "analytic_independent": independent.tolist(),
        "analytic_shared_masterbias": shared.tolist(),
        "empirical_covariance": empirical.tolist(),
        "variance_relative_error": variance_relative_error.tolist(),
        "cross_covariance_scaled_error": float(cross_covariance_error),
        "coverage_1sigma": coverage_1sigma.tolist(),
        "coverage_95percent": coverage_95.tolist(),
        "coverage_95percent_independent_only": coverage_95_independent_only.tolist(),
        "standardized_mean": mean_z.tolist(),
        "variance_relative_tolerance": variance_limit,
        "cross_covariance_scaled_tolerance": cov_limit,
        "coverage_1sigma_tolerance": cov1_limit,
        "coverage_95percent_tolerance": cov95_limit,
        "science_uncertainty_qualified": False,
    }
