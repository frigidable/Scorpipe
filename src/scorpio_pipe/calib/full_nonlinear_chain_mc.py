"""Synthetic-only fully chained nonlinear Kelson -> optimal extraction MC.

Three native CCD images are re-fitted with production Kelson clipping,
linearized through real COUNT_V1 row plans, combined with production
streaming Stack2D (invvar and Huber), and optionally extracted with
production same-data optimal profiles on every draw.

A *different* frozen-adjoint oracle calculates exact conditional
known-profile + invvar reference variance, including shared MasterBias.
This conditional reference is only a gate when sky selection stays fixed.
No real observations, FITS writes, or changes to published science VAR.
"""
from __future__ import annotations

from collections.abc import Sequence
from math import erf, sqrt
from typing import Any

import numpy as np
from astropy.io import fits

from scorpio_pipe.calib.sky_bias_covariance import frozen_kelson_bspline_operator
from scorpio_pipe.calib.sky_chain_grouped_covariance import (
    covariance_of_frozen_sky_chain_grouped_pair,
)
from scorpio_pipe.covariance_chain import FrozenCovarianceExposure
from scorpio_pipe.maskbits import NO_COVERAGE
from scorpio_pipe.stages.extract1d import _optimal_extract
from scorpio_pipe.stages.linearize import (
    _apply_linearize_row_plan,
    _apply_linearize_row_plan_var,
    _build_linearize_row_plan,
)
from scorpio_pipe.stages.sky_sub import (
    _bspline_design_matrix,
    _fit_kelson_bspline,
    _make_bspline_knot_vector,
)
from scorpio_pipe.stages.stack2d import _stack_chunk_streaming


def synthetic_full_nonlinear_chain_coverage(
    *,
    group_ids: Sequence[str] = ("A", "B", "C"),
    y_order: int = 0,
    true_flux: float = 0.,
    n_realizations: int = 250,
    seed: int = 20261104,
    clip_sigma: float = 7.,
    contamination_sigma: float = 0.,
    huber_c: float = 1.,
    robust_iter: int = 2,
    smooth_lambda_sigma: float = 0.,
) -> dict[str, Any]:
    """Joint same-draw test from native synthetic CCD to 1-D scalar flux.

    Object truth is known and stays outside the initial sky-training ROI.
    Each draw uses fresh detector Gaussian noise plus per-group correlated
    MasterBias fields. The fixed-mask WLS reference is independently
    backprojected through actual COUNT_V1 plans and equal-weight
    nonrobust Stack2D. Profiles and Huber weights are data-adaptive in
    the other three paired estimators.
    """
    labels = tuple(group_ids)
    if (len(labels) != 3 or any(
            not isinstance(g, str) or not g or g.strip() != g for g in labels)):
        raise ValueError("Three explicit nonblank MasterBias group IDs required")
    if isinstance(y_order, (bool, np.bool_)) or y_order not in (0, 1):
        raise ValueError("y_order must be 0 or 1")
    if (isinstance(true_flux, (bool, np.bool_))
            or not isinstance(true_flux, (int, float, np.integer, np.floating))
            or not np.isfinite(true_flux) or not 0. <= true_flux <= 1000.):
        raise ValueError("true_flux must be a finite nonnegative count")
    if (isinstance(n_realizations, (bool, np.bool_))
            or not isinstance(n_realizations, (int, np.integer))
            or not 200 <= int(n_realizations) <= 800):
        raise ValueError("n_realizations must be an integer in [200,800]")
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    if (isinstance(clip_sigma, (bool, np.bool_))
            or not isinstance(clip_sigma, (int, float, np.integer, np.floating))
            or not np.isfinite(clip_sigma) or not 1. <= clip_sigma <= 10.):
        raise ValueError("clip_sigma must be finite in [1,10]")
    if (isinstance(contamination_sigma, (bool, np.bool_))
            or not isinstance(contamination_sigma, (int, float, np.integer, np.floating))
            or not np.isfinite(contamination_sigma)
            or not 0. <= contamination_sigma <= 20.):
        raise ValueError("contamination_sigma must be finite in [0,20]")
    if (isinstance(huber_c, (bool, np.bool_))
            or not isinstance(huber_c, (int, float, np.integer, np.floating))
            or not np.isfinite(huber_c) or not .1 <= huber_c <= 20.):
        raise ValueError("huber_c must be finite in [0.1,20]")
    if (isinstance(robust_iter, (bool, np.bool_))
            or not isinstance(robust_iter, (int, np.integer))
            or not 0 <= robust_iter <= 5):
        raise ValueError("robust_iter must be integer in [0,5]")
    if (isinstance(smooth_lambda_sigma, (bool, np.bool_))
            or not isinstance(smooth_lambda_sigma,
                              (int, float, np.integer, np.floating))
            or not np.isfinite(smooth_lambda_sigma)
            or not 0. <= smooth_lambda_sigma <= 10.):
        raise ValueError("smooth_lambda_sigma must be finite in [0,10]")

    n_exposures, ny, nx = 3, 10, 20
    yy, xx = np.indices((ny, nx), dtype=float)
    yn = np.linspace(-1., 1., ny)
    npx = ny * nx
    wave = 5100. + 1.27*xx + .025*yy
    edges = np.linspace(5099.5, 5126., 27)
    nlam = edges.size-1
    target = 12
    training = np.zeros((ny, nx), dtype=bool)
    training[[0, 1, 2, 7, 8, 9]] = True
    index = np.flatnonzero(training.ravel())
    yfit = yn[index // nx]
    knots = _make_bspline_knot_vector(
        float(wave.ravel()[index].min()),
        float(wave.ravel()[index].max()), knot_step_A=4., degree=2,
    )
    B = _bspline_design_matrix(wave.ravel(), knots, 2).tocsr()
    coeff = 15. + 2.*np.sin(.47*np.arange(B.shape[1]))
    true_sky = np.asarray(B @ coeff).ravel()
    if y_order == 1:
        c1 = .45*np.cos(.29*np.arange(B.shape[1]))
        true_sky += yn.repeat(nx)*np.asarray(B @ c1).ravel()

    true_profile = np.zeros(ny, dtype=float)
    true_profile[3:7] = np.exp(-.5*((np.arange(3, 7)-4.5)/.95)**2)
    true_profile /= np.sum(true_profile)
    raw_object = float(true_flux)*true_profile[:, None]*np.ones((1, nx))
    fixed_outlier = np.zeros(npx)
    base_var = np.full((ny, nx), 2., dtype=float)  # V_ind=1 + V_bias=1
    fixed_outlier[1*nx+10] = float(contamination_sigma)*sqrt(2.)
    base_signal = true_sky + raw_object.ravel() + fixed_outlier

    fit_kwargs = dict(
        knot_step_A=4., degree=2, y_order=int(y_order),
        clip_sigma=float(clip_sigma), max_iter=4,
        min_pixels=64,
    )

    def fit_sky(samples: np.ndarray) -> dict[str, Any]:
        return _fit_kelson_bspline(
            wave.ravel()[index], yfit, samples, base_var.ravel()[index],
            **fit_kwargs, rng=np.random.default_rng(0),
        )

    fit0 = fit_sky(base_signal[index])
    baseline_keep = np.asarray(fit0["_cache"]["keep"], dtype=bool)
    if (baseline_keep.shape != index.shape
            or not np.array_equal(np.asarray(fit0["basis"]["t"]), knots)):
        raise RuntimeError("Kelson baseline sky training geometry changed")
    fit_mask = np.zeros(npx, dtype=bool)
    fit_mask[index[baseline_keep]] = True
    sky_operator = frozen_kelson_bspline_operator(
        wave, yn, knots=knots, degree=2, y_order=int(y_order),
        final_fit_mask=fit_mask.reshape(ny, nx), variance=base_var,
    )

    plans = [_build_linearize_row_plan(wave[y], edges) for y in range(ny)]
    if any(p is None for p in plans):
        raise RuntimeError("COUNT_V1 plans missing for synthetic native rows")

    def transform_native(frame: np.ndarray) -> np.ndarray:
        return np.asarray([
            _apply_linearize_row_plan(frame.reshape(ny, nx)[y], plans[y])[0]
            for y in range(ny)
        ], dtype=np.float64)

    object_linear = transform_native(raw_object)
    linear_variance = np.asarray([
        _apply_linearize_row_plan_var(base_var[y], plans[y])
        for y in range(ny)
    ])
    linear_coverage = np.asarray([
        _apply_linearize_row_plan(
            np.ones(nx, dtype=float), plans[y],
        )[1] for y in range(ny)
    ])
    good = (linear_coverage >= .999999) & (linear_variance > 0)
    if not np.all(good[:, target]):
        raise RuntimeError("Synthetic science wavelength has incomplete row coverage")

    # The fixed known-profile reference is derived ONLY from injected
    # signal and the known uncorrelated propagation, never from noisy SCI.
    p = object_linear[:, target].copy()
    if np.sum(p) <= 0:
        p = true_profile.copy()
    p /= np.sum(p)
    inv_pixel_var = np.divide(1., linear_variance[:, target]/3.,
                              out=np.zeros(ny), where=good[:, target])
    normalizer = float(np.dot(p*p, inv_pixel_var))
    if normalizer <= 0:
        raise RuntimeError("Fixed extraction profile lacks valid information")
    fixed_weights = p*inv_pixel_var/normalizer
    known_true_flux = float(np.dot(fixed_weights, object_linear[:, target]))

    extraction = np.zeros((ny, nlam), dtype=float)
    extraction[:, target] = fixed_weights
    one_lambda = np.zeros(nlam, dtype=float)
    one_lambda[target] = 1.
    frozen_exposures = [
        FrozenCovarianceExposure(
            row_plans=plans,
            input_variance=base_var,
            valid_mask=np.ones_like(base_var, dtype=bool),
            stack_weights=np.full((ny, nlam), 1./n_exposures),
            y_shift=0., output_scale=1.,
        ) for _ in range(n_exposures)
    ]
    frozen_ind, frozen_bias, frozen_total = (
        covariance_of_frozen_sky_chain_grouped_pair(
            exposures=frozen_exposures,
            sky_operators=[sky_operator]*n_exposures,
            left_extraction_weights=extraction,
            right_extraction_weights=extraction,
            fluxcal_transfer=np.ones(nlam),
            left_weights=one_lambda, right_weights=one_lambda,
            independent_variances=[np.ones((ny, nx))]*n_exposures,
            bias_sensitivities=[-np.ones((ny, nx))]*n_exposures,
            group_ids=labels,
            masterbias_variances={g: np.ones(npx) for g in set(labels)},
        )
    )
    if not np.isfinite(frozen_total) or frozen_total <= 0:
        raise RuntimeError("Frozen full-chain sky covariance invalid")

    mask_out = np.zeros((ny, nlam), dtype=np.uint16)
    mask_out[~good] = np.uint16(NO_COVERAGE)
    var_out = np.asarray(linear_variance, dtype=np.float32)
    # Invalid wavelength tails never supply a statistical observation.
    var_out[~good] = 0.
    hduls = [fits.HDUList([
        fits.PrimaryHDU(data=np.zeros((ny, nlam), dtype=np.float32)),
        fits.ImageHDU(data=var_out.copy(), name="VAR"),
        fits.ImageHDU(data=mask_out.copy(), name="MASK"),
    ]) for _ in range(n_exposures)]

    n = int(n_realizations)
    rng = np.random.default_rng(int(seed))
    bias_draw = {
        g: rng.standard_normal((n, npx))
        for g in sorted(set(labels))
    }
    separate = rng.standard_normal((n, n_exposures, npx))
    data = np.zeros((n, 2, 2), dtype=float)
    conditional = np.zeros_like(data)
    clip_changes = np.zeros((n, n_exposures), dtype=bool)
    lsqr_fail = np.zeros((n, n_exposures), dtype=bool)
    suppressed = np.zeros((n, 2), dtype=float)
    trace = np.full(nlam, 4.5, dtype=float)
    try:
        for k in range(n):
            for e, h in enumerate(hduls):
                noisy = base_signal + separate[k, e] - bias_draw[labels[e]][k]
                fit = fit_sky(noisy[index])
                kept = np.asarray(fit["_cache"]["keep"], dtype=bool)
                if (kept.shape != baseline_keep.shape or
                        not np.array_equal(np.asarray(fit["basis"]["t"]), knots)):
                    raise RuntimeError("Refitted Kelson basis/selection geometry changed")
                clip_changes[k, e] = bool(np.any(kept != baseline_keep))
                lsqr_fail[k, e] = (
                    int(fit["solver_diagnostics"]["stop_code"]) not in (1, 2)
                )
                fitted = np.asarray(B @ np.asarray(fit["coeff"]["c0"])).ravel()
                if y_order == 1:
                    fitted += yn.repeat(nx)*np.asarray(
                        B @ np.asarray(fit["coeff"]["c1"])
                    ).ravel()
                h[0].data[:] = transform_native(noisy-fitted).astype(np.float32)

            for mode_id, method in enumerate(("invvar", "invvar_huber")):
                sci, vari, mask, covered, nsupp, ntotal = _stack_chunk_streaming(
                    hduls=hduls, y0=0, y1=ny,
                    method=method,
                    robust_iter=int(robust_iter) if mode_id else 0,
                    huber_c=float(huber_c), clip_sigma=4.,
                    y_shifts=[0.]*n_exposures, y_align_enabled=False,
                    sci_scales=[1.]*n_exposures,
                    var_scales=[1.]*n_exposures,
                    fatal_bits=np.uint16(NO_COVERAGE), var_floor=None,
                )
                if (not np.all(covered[:, target] == n_exposures)
                        or not np.isfinite(sci[:, target]).all()
                        or not np.isfinite(vari[:, target]).all()
                        or np.any(vari[:, target] <= 0)):
                    raise RuntimeError("Native-to-Stack2D central sky coverage invalid")
                suppressed[k, mode_id] = nsupp/max(ntotal, 1)
                # Known profile WLS on the output SCI. Use the REAL
                # conditional stack variance for the adaptive Huber case.
                denom = float(np.sum(p*p/vari[:, target]))
                data[k, mode_id, 0] = float(
                    np.sum(p*sci[:, target]/vari[:, target])/denom
                )
                conditional[k, mode_id, 0] = 1./denom

                flux, vflux, mm, _sky, n_used, _meta = _optimal_extract(
                    sci, vari, mask, trace, ap_hw=3,
                    fatal_bits=int(NO_COVERAGE), min_good_frac=1.,
                    sky_resid=None, apply_local_sky=False,
                    smooth_lambda_sigma=float(smooth_lambda_sigma),
                    clip_negative_profile=True, backend="numpy",
                )
                if (not np.isfinite(flux[target])
                        or not np.isfinite(vflux[target])
                        or vflux[target] <= 0
                        or (mm[target] & NO_COVERAGE)
                        or n_used[target] != 7):
                    raise RuntimeError("Native-to-Extract1D central aperture invalid")
                data[k, mode_id, 1] = float(flux[target])
                conditional[k, mode_id, 1] = float(vflux[target])
    finally:
        for h in hduls:
            h.close()

    # All estimators share the same known transformed source truth, but
    # adaptive-profile fluxes can be biased even when source=0.
    errors = data-known_true_flux
    results: dict[str, dict[str, float]] = {}
    for mode_id, stack_label in enumerate(("invvar", "huber")):
        for profile_id, profile_label in enumerate(("known", "adaptive")):
            err = errors[:, mode_id, profile_id]
            formal = conditional[:, mode_id, profile_id]
            results[f"{stack_label}_{profile_label}"] = {
                "mean_error": float(np.mean(err)),
                "empirical_variance": float(np.var(err, ddof=1)),
                "mean_reported_conditional_variance": float(np.mean(formal)),
                "variance_ratio": float(np.var(err, ddof=1)/np.mean(formal)),
                "coverage_95percent": float(np.mean(
                    np.abs(err) <= 1.959963984540054*np.sqrt(formal)
                )),
            }

    known = errors[:, 0, 0]
    tol_var = max(.10, 5.*sqrt(2./(n-1)))
    tol_mean = 5.*sqrt(frozen_total/n)
    tol_cov95 = max(.025, 5.*sqrt(.95*.05/n))
    ref_coverage95 = float(np.mean(
        np.abs(known) <= 1.959963984540054*sqrt(frozen_total)
    ))
    ref_consistent = bool(
        abs(np.var(known, ddof=1)/frozen_total-1.) <= tol_var
        and abs(np.mean(known)) <= tol_mean
        and abs(ref_coverage95-.95) <= tol_cov95
    )
    baseline_stop = int(fit0["solver_diagnostics"]["stop_code"])
    any_numerical_problem = bool(
        baseline_stop not in (1, 2) or np.any(lsqr_fail)
    )
    any_selection_change = bool(np.any(clip_changes))
    # A frozen reference cannot validate a reselected-data estimator.
    if any_numerical_problem:
        status = "NUMERIC_UNSTABLE"
    elif any_selection_change:
        status = "SKY_SELECTION_VARIABLE"
    elif not ref_consistent:
        status = "FROZEN_REFERENCE_MC_MISMATCH"
    else:
        status = "JOINT_ADAPTIVE_MC_REPORTED"

    def contrast(diff: np.ndarray) -> dict[str, float | bool]:
        mean = float(np.mean(diff))
        se = float(np.std(diff, ddof=1)/sqrt(n))
        return {"mean": mean, "standard_error": se,
                "detected_at_5sigma": bool(abs(mean)>5.*se)}

    return {
        "status": status,
        "scope": "SYNTHETIC_NATIVE_KELSON_TO_ADAPTIVE_STACK_EXTRACT",
        "n_realizations": n,
        "group_ids": list(labels),
        "y_order": int(y_order),
        "true_flux": float(true_flux),
        "known_transformed_true_flux": known_true_flux,
        "clip_sigma": float(clip_sigma),
        "contamination_sigma": float(contamination_sigma),
        "huber_c": float(huber_c),
        "robust_iter": int(robust_iter),
        "smooth_lambda_sigma": float(smooth_lambda_sigma),
        "frozen_reference_variance": float(frozen_total),
        "frozen_reference_independent": float(frozen_ind),
        "frozen_reference_shared_masterbias": float(frozen_bias),
        "frozen_reference_coverage95": ref_coverage95,
        "frozen_reference_mc_consistent": ref_consistent,
        "baseline_lsqr_stop": baseline_stop,
        "per_exposure_nonconverged": np.sum(lsqr_fail, axis=0).astype(int).tolist(),
        "per_exposure_sky_reselection": np.sum(clip_changes, axis=0).astype(int).tolist(),
        "any_reselection_fraction": float(np.mean(np.any(clip_changes, axis=1))),
        "mean_suppression_fraction": {
            "invvar": float(np.mean(suppressed[:, 0])),
            "huber": float(np.mean(suppressed[:, 1])),
        },
        "estimators": results,
        "paired_effects": {
            "profile_invvar": contrast(errors[:, 0, 1]-errors[:, 0, 0]),
            "profile_huber": contrast(errors[:, 1, 1]-errors[:, 1, 0]),
            "huber_known": contrast(errors[:, 1, 0]-errors[:, 0, 0]),
            "huber_adaptive": contrast(errors[:, 1, 1]-errors[:, 0, 1]),
            "interaction": contrast(
                (errors[:, 1, 1]-errors[:, 1, 0])
                -(errors[:, 0, 1]-errors[:, 0, 0])
            ),
        },
        "tolerances": {
            "reference_relative_variance": tol_var,
            "reference_mean_error": tol_mean,
            "reference_coverage95": tol_cov95,
        },
        "scope_limits": "FROZEN_LAMBDA_GEOMETRY_TRACE_MASK_INPUT_VAR",
        "science_uncertainty_qualified": False,
        "production_variance_release_allowed": False,
    }
