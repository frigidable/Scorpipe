"""Synthetic sampling-coverage audit with ACTUAL Kelson sigma clipping re-run.

This is deliberately different from the fixed-mask/dense-QR oracle in
sky_mc_coverage: every noisy realization is reprocessed with production
_fit_kelson_bspline, including its clip decisions and finite-iteration LSQR.

It is a diagnostic of a specified synthetic experiment. It does not
provide a science variance product and never reads real observations.
"""
from __future__ import annotations

from collections.abc import Sequence
from math import erf, sqrt
from typing import Any

import numpy as np

from scorpio_pipe.calib.sky_bias_covariance import (
    covariance_of_frozen_sky_grouped,
    frozen_kelson_bspline_operator,
)
from scorpio_pipe.stages.sky_sub import (
    _bspline_design_matrix,
    _fit_kelson_bspline,
    _make_bspline_knot_vector,
)


def synthetic_reclipped_sky_coverage(
    *,
    group_ids: Sequence[str] = ("A", "A"),
    y_order: int = 0,
    n_realizations: int = 400,
    seed: int = 20261104,
    clip_sigma: float = 7.0,
    contamination_sigma: float = 0.0,
) -> dict[str, Any]:
    """Compare frozen analytic covariance with independent full re-fit draws.

    Both exposures have fixed synthetic wavelength solutions, fixed
    independent CCD and shared MasterBias variances, fixed initial sky ROI
    and detector-to-bias responses. New Gaussian errors are drawn for each
    realization, and full production Kelson fit + iterative clipping is
    called independently for each exposure and each noisy CCD.

    `contamination_sigma` is a fixed artificial sky-ROI outlier in only
    one exposure; clipping is permitted to accept/reject it across trials.
    Sky truth belongs exactly to the generated B-spline basis. Observed
    statistics are centered against the KNOWN injected object truth,
    not their empirical mean, so persistent clipping bias affects coverage.
    """
    labels = tuple(group_ids)
    if (len(labels) != 2
            or any(not isinstance(g, str) or not g or g.strip() != g for g in labels)):
        raise ValueError("Exactly two explicit nonblank shared-bias groups are required")
    if isinstance(y_order, (bool, np.bool_)) or y_order not in (0, 1):
        raise ValueError("Synthetic y_order must be 0 or 1")
    if (isinstance(n_realizations, (bool, np.bool_))
            or not isinstance(n_realizations, (int, np.integer))
            or not 200 <= int(n_realizations) <= 3000):
        raise ValueError("n_realizations must be an integer in [200, 3000]")
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or int(seed) < 0):
        raise ValueError("Seed must be a nonnegative integer")
    if (not np.isfinite(clip_sigma) or not 1. <= clip_sigma <= 10.
            or not np.isfinite(contamination_sigma) or contamination_sigma < 0
            or contamination_sigma > 20.):
        raise ValueError("Invalid synthetic clipping or contamination amplitude")

    ny, nx = 12, 24
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    y_norm = 2. * (np.arange(ny) - .5*(ny-1)) / (ny-1)
    n_pixels = ny * nx
    sky_window = np.zeros((ny, nx), dtype=bool)
    sky_window[0:4, :] = True
    sky_window[8:12, :] = True
    fit_idx = np.flatnonzero(sky_window.ravel())
    y_fit = y_norm[fit_idx // nx]

    # 2D row-and-wavelength dependent CCD noise and common-bias variance.
    bias_var = (0.8 + .20 * xx/(nx-1) + .09 * yy/(ny-1)).ravel()
    obs_u = np.zeros((2, ny, nx), dtype=np.float64)
    obs_u[0, 4:8, 4:18] = 1. / (4. * 14 * 2.)
    obs_u[1, 4:8, 9:22] = 1. / (4. * 13 * 2.)
    u = [obs_u[q].ravel() for q in range(2)]

    instruments = []
    for e in range(2):
        wave = 5100. + (1.35 + .03*e) * xx + (.035 + .009*e)*yy
        knot = _make_bspline_knot_vector(
            float(np.min(wave.ravel()[fit_idx])),
            float(np.max(wave.ravel()[fit_idx])),
            knot_step_A=5.0, degree=2,
        )
        B = _bspline_design_matrix(wave.ravel(), knot, 2).tocsr()
        nb = B.shape[1]
        c0 = 16. + 2. * np.sin(np.arange(nb) * .57)
        true_sky = np.asarray(B @ c0).ravel()
        if y_order == 1:
            c1 = .7 * np.cos(np.arange(nb) * .37)
            true_sky += y_norm.repeat(nx) * np.asarray(B @ c1).ravel()
        v_ind = (0.7 + .13*e + .12*xx/(nx-1)
                 + .09*yy/(ny-1)).ravel()
        sensitivity = -(0.75 + .09*e + .08*xx/(nx-1)
                        + .02*yy/(ny-1)).ravel()
        total_var = v_ind + sensitivity**2 * bias_var
        true_object = np.zeros((ny, nx), dtype=np.float64)
        true_object[4:8, :] = (
            4. + (5. + e) * np.exp(-.5 * ((xx[4:8] - 11.)/3.)**2)
        )

        # A fixed nuisance pixel is present in the sky-window sample for
        # every realization. It is not a fresh stochastic MasterBias error.
        contamination = np.zeros(n_pixels, dtype=np.float64)
        if e == 0:
            contaminated_id = 1 * nx + 11
            contamination[contaminated_id] = (
                float(contamination_sigma) * sqrt(total_var[contaminated_id])
            )
        noiseless = true_sky + true_object.ravel() + contamination

        def_fit = _fit_kelson_bspline(
            wave.ravel()[fit_idx], y_fit,
            noiseless[fit_idx], total_var[fit_idx],
            knot_step_A=5., degree=2, y_order=int(y_order),
            clip_sigma=float(clip_sigma), max_iter=4,
            min_pixels=64, rng=np.random.default_rng(0),
        )
        baseline_keep = np.asarray(def_fit["_cache"]["keep"], dtype=bool)
        if baseline_keep.shape != fit_idx.shape:
            raise RuntimeError("Baseline clip selection geometry invalid")
        if not np.array_equal(np.asarray(def_fit["basis"]["t"]), knot):
            raise RuntimeError("Baseline synthetic spline knot geometry mismatch")
        final_mask = np.zeros(n_pixels, dtype=bool)
        final_mask[fit_idx[baseline_keep]] = True
        frozen = frozen_kelson_bspline_operator(
            wave, y_norm, knots=knot, degree=2, y_order=int(y_order),
            final_fit_mask=final_mask.reshape(ny, nx),
            variance=total_var.reshape(ny, nx),
        )
        instruments.append({
            "wave_fit": wave.ravel()[fit_idx],
            "knot": knot,
            "basis": B,
            "true_sky": true_sky,
            "true_object": true_object.ravel(),
            "noiseless": noiseless,
            "var_fit": total_var[fit_idx],
            "v_ind": v_ind,
            "sens": sensitivity,
            "baseline_keep": baseline_keep,
            "frozen": frozen,
        })

    groups = {g: bias_var.copy() for g in set(labels)}
    funs = [u, [np.roll(q, 1) for q in u]]
    # Two distinct synthetic object estimators. The second one has a
    # different wavelength weighting, not a duplicate measurement.
    prediction = np.zeros((2, 2), dtype=np.float64)
    pred_ind = np.zeros((2, 2), dtype=np.float64)
    pred_shared = np.zeros((2, 2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            ai, ab, total = covariance_of_frozen_sky_grouped(
                [v["frozen"] for v in instruments],
                funs[i], funs[j], [v["v_ind"] for v in instruments],
                [v["sens"] for v in instruments], labels, groups,
            )
            prediction[i, j] = total
            pred_ind[i, j] = ai
            pred_shared[i, j] = ab
    if (not np.isfinite(prediction).all()
            or np.min(np.linalg.eigvalsh(prediction)) <= 0):
        raise RuntimeError("Invalid synthetic conditional covariance")

    rng = np.random.default_rng(int(seed))
    n = int(n_realizations)
    # Fresh independent MasterBias realization PER Monte Carlo draw/group;
    # same per-pixel realization shared across both exposures of a group.
    bias_draws = {
        g: rng.standard_normal((n, n_pixels)) * np.sqrt(bias_var)[None, :]
        for g in sorted(set(labels))
    }
    independent_draws = [
        rng.standard_normal((n, n_pixels)) * np.sqrt(inst["v_ind"])[None, :]
        for inst in instruments
    ]

    errors = np.zeros((n, 2), dtype=np.float64)
    changes = np.zeros((n, 2), dtype=bool)
    clipped_fractions = np.zeros((n, 2), dtype=np.float64)
    nonconverged = np.zeros((n, 2), dtype=bool)
    for k in range(n):
        for e, inst in enumerate(instruments):
            noisy = (
                inst["noiseless"] + independent_draws[e][k]
                + inst["sens"] * bias_draws[labels[e]][k]
            )
            fit = _fit_kelson_bspline(
                inst["wave_fit"], y_fit, noisy[fit_idx],
                inst["var_fit"], knot_step_A=5., degree=2,
                y_order=int(y_order), clip_sigma=float(clip_sigma),
                max_iter=4, min_pixels=64,
                rng=np.random.default_rng(0),
            )
            kept = np.asarray(fit["_cache"]["keep"], dtype=bool)
            if kept.shape != inst["baseline_keep"].shape:
                raise RuntimeError("Monte Carlo final-fit sky selection shape changed")
            if not np.array_equal(np.asarray(fit["basis"]["t"]), inst["knot"]):
                raise RuntimeError("Unexpected reselected wavelength knots")
            changes[k, e] = np.any(kept != inst["baseline_keep"])
            clipped_fractions[k, e] = float(1. - np.mean(kept))
            nonconverged[k, e] = int(fit["solver_diagnostics"]["stop_code"]) not in (1, 2)
            c0 = np.asarray(fit["coeff"]["c0"], dtype=np.float64)
            model = np.asarray(inst["basis"] @ c0, dtype=np.float64).ravel()
            if y_order == 1:
                c1 = np.asarray(fit["coeff"]["c1"], dtype=np.float64)
                model += y_norm.repeat(nx) * np.asarray(inst["basis"] @ c1).ravel()
            error_frame = noisy - model - inst["true_object"]
            for j in range(2):
                errors[k, j] += float(np.dot(funs[j][e], error_frame))

    empirical = np.cov(errors, rowvar=False, ddof=1)
    z = errors / np.sqrt(np.diag(prediction))[None, :]
    nominal_1 = erf(1./sqrt(2.))
    coverage1 = np.mean(np.abs(z) <= 1., axis=0)
    coverage95 = np.mean(np.abs(z) <= 1.959963984540054, axis=0)
    var_error = np.abs(np.diag(empirical) / np.diag(prediction) - 1.)
    cross_error = (abs(empirical[0, 1] - prediction[0, 1])
                   / sqrt(prediction[0, 0]*prediction[1, 1]))
    mean_z = np.mean(z, axis=0)

    # Finite-sample, deliberately conservative Monte Carlo gates;
    # passing these gates is not a claim about all synthetic scenes.
    var_limit = max(.10, 5.*sqrt(2./(n-1)))
    cross_limit = max(.10, 5./sqrt(n-1))
    mean_limit = 5./sqrt(n)
    cov1_limit = max(.03, 5.*sqrt(nominal_1*(1-nominal_1)/n))
    cov95_limit = max(.025, 5.*sqrt(.95*.05/n))
    coverage_ok = (
        np.max(var_error) <= var_limit
        and cross_error <= cross_limit
        and np.max(np.abs(mean_z)) <= mean_limit
        and np.max(np.abs(coverage1 - nominal_1)) <= cov1_limit
        and np.max(np.abs(coverage95 - .95)) <= cov95_limit
    )
    any_changes = bool(np.any(changes))
    all_stops_nominal = not bool(np.any(nonconverged))
    if not all_stops_nominal:
        status = "NUMERIC_UNSTABLE"
    elif any_changes:
        status = "SELECTION_VARIABLE"
    elif coverage_ok:
        status = "MC_CONSISTENT"
    else:
        status = "MC_MISMATCH"

    return {
        "status": status,
        "scope": "SYNTHETIC_PRODUCTION_KELSON_RECLIPPED",
        "group_ids": list(labels),
        "y_order": int(y_order),
        "n_realizations": n,
        "clip_sigma": float(clip_sigma),
        "contamination_sigma": float(contamination_sigma),
        "analytic_covariance": prediction.tolist(),
        "analytic_independent": pred_ind.tolist(),
        "analytic_shared_masterbias": pred_shared.tolist(),
        "empirical_covariance": empirical.tolist(),
        "variance_relative_error": var_error.tolist(),
        "cross_covariance_scaled_error": float(cross_error),
        "coverage_1sigma": coverage1.tolist(),
        "coverage_95percent": coverage95.tolist(),
        "standardized_mean": mean_z.tolist(),
        "variance_relative_tolerance": var_limit,
        "cross_covariance_scaled_tolerance": cross_limit,
        "mean_standardized_tolerance": mean_limit,
        "coverage_1sigma_tolerance": cov1_limit,
        "coverage_95percent_tolerance": cov95_limit,
        "conditional_mc_consistent": bool(coverage_ok),
        "n_selection_changed_per_exposure": np.sum(changes, axis=0).astype(int).tolist(),
        "selection_change_fraction_any_exposure": float(np.mean(np.any(changes, axis=1))),
        "mean_clipped_fraction_per_exposure": np.mean(clipped_fractions, axis=0).tolist(),
        "n_nonconverged_per_exposure": np.sum(nonconverged, axis=0).astype(int).tolist(),
        "science_uncertainty_qualified": False,
    }
