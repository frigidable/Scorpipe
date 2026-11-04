"""Synthetic Monte Carlo: re-clipped production Kelson through fixed 2D-to-1D chain.

Uses real _fit_kelson_bspline and COUNT_V1 row-plan execution on EVERY
noise draw. Subpixel y registration, Stack2D coefficients, extraction and
fluxcal transfer are explicitly FROZEN; they do not adapt to noise.
Scientific FITS uncertainty/VAR status is never modified or qualified.
"""
from __future__ import annotations

from collections.abc import Sequence
from math import erf, sqrt
from typing import Any

import numpy as np

from scorpio_pipe.calib.sky_bias_covariance import frozen_kelson_bspline_operator
from scorpio_pipe.calib.sky_chain_grouped_covariance import (
    covariance_of_frozen_sky_chain_grouped_pair,
)
from scorpio_pipe.covariance_chain import FrozenCovarianceExposure
from scorpio_pipe.stages.linearize import (
    _apply_linearize_row_plan,
    _build_linearize_row_plan,
)
from scorpio_pipe.stages.sky_sub import (
    _bspline_design_matrix,
    _fit_kelson_bspline,
    _make_bspline_knot_vector,
)


def synthetic_reclipped_sky_to_1d_coverage(
    *,
    group_ids: Sequence[str] = ("A", "A"),
    y_order: int = 0,
    n_realizations: int = 250,
    seed: int = 20261104,
    clip_sigma: float = 7.,
    contamination_sigma: float = 0.,
) -> dict[str, Any]:
    """Run actual clipping+LSQR per draw, then a frozen downstream 1D map.

    Sky truth is constructed from the same deterministic spline family.
    Object signal is exactly known and excluded from initial sky windows.
    The covariance oracle is the *baseline* frozen SkySub + chain
    operator; selection changes and finite-iteration effects are measured
    rather than silently absorbed into its estimated variance.
    """
    labels = tuple(group_ids)
    if (len(labels) != 2 or
            any(not isinstance(x, str) or not x or x.strip() != x
                for x in labels)):
        raise ValueError("Two explicit nonempty MasterBias group identifiers required")
    if (isinstance(y_order, (bool, np.bool_)) or
            not isinstance(y_order, (int, np.integer))
            or int(y_order) not in (0, 1)):
        raise ValueError("y_order must be 0 or 1")
    if (isinstance(n_realizations, (bool, np.bool_)) or
            not isinstance(n_realizations, (int, np.integer))
            or not 200 <= int(n_realizations) <= 1000):
        raise ValueError("MC draws must be an integer in [200,1000]")
    if (isinstance(seed, (bool, np.bool_)) or
            not isinstance(seed, (int, np.integer)) or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    if (not np.isfinite(clip_sigma) or not 1. <= clip_sigma <= 10.
            or not np.isfinite(contamination_sigma)
            or not 0 <= contamination_sigma <= 20.):
        raise ValueError("Invalid clipping or contamination controls")

    ny, nx = 10, 20
    yy, xx = np.indices((ny, nx), dtype=float)
    npx = ny*nx
    yn = np.linspace(-1., 1., ny)
    candidate = np.zeros((ny, nx), dtype=bool)
    candidate[[0, 1, 2, 7, 8, 9]] = True
    idx = np.flatnonzero(candidate.ravel())
    yfit = yn[idx//nx]
    edges = np.linspace(5099.5, 5126., 27)
    nlam = len(edges)-1
    transfer = np.linspace(.6, 1.3, nlam)
    u = np.zeros(nlam)
    v = np.zeros(nlam)
    u[5:16] = np.linspace(.5, 1.5, 11)/11.
    v[11:22] = np.linspace(1.4, .6, 11)/11.
    Eleft = np.zeros((ny, nlam))
    Eright = np.zeros((ny, nlam))
    Eleft[3:7] = np.linspace(.7, 1.2, nlam)
    Eright[3:7] = np.linspace(1.3, .5, nlam)[None, :] * np.array(
        [1., .6, .75, 1.2],
    )[:, None]
    bvar = (.65 + .13*xx/(nx-1) + .07*yy/(ny-1)).ravel()
    shift = (0., .25)

    states: list[dict[str, Any]] = []
    for e in range(2):
        wavelength = 5100. + (1.3 + .018*e)*xx + (.025+.004*e)*yy
        knot = _make_bspline_knot_vector(
            float(wavelength.ravel()[idx].min()),
            float(wavelength.ravel()[idx].max()),
            knot_step_A=4., degree=2,
        )
        basis = _bspline_design_matrix(wavelength.ravel(), knot, 2).tocsr()
        coeff = 16. + 2.*np.sin(.51*np.arange(basis.shape[1]))
        sky_true = np.asarray(basis @ coeff).ravel()
        if y_order == 1:
            coeff_y = .4*np.cos(.35*np.arange(basis.shape[1]))
            sky_true = sky_true + yn.repeat(nx)*np.asarray(
                basis @ coeff_y
            ).ravel()
        obj = np.zeros((ny, nx), dtype=float)
        obj[3:7] = 3.5 + (4. + .3*e)*np.exp(
            -.5*((xx[3:7]-10.)/3.)**2
        )
        v_ind = (.8+.1*e+.1*xx/(nx-1)+.06*yy/(ny-1)).ravel()
        sensitivity = -(.8+.045*e+.04*xx/(nx-1)+.02*yy/(ny-1)).ravel()
        total_var = v_ind + sensitivity**2*bvar
        contamination = np.zeros(npx)
        if e == 0:
            position = 1*nx + 10
            contamination[position] = (
                float(contamination_sigma)*sqrt(total_var[position])
            )
        base_signal = sky_true + obj.ravel() + contamination

        def refit(sky_samples: np.ndarray):
            return _fit_kelson_bspline(
                wavelength.ravel()[idx], yfit, sky_samples,
                total_var[idx], knot_step_A=4., degree=2,
                y_order=int(y_order), clip_sigma=float(clip_sigma),
                max_iter=4, min_pixels=64, rng=np.random.default_rng(0),
            )

        fit0 = refit(base_signal[idx])
        keep0 = np.asarray(fit0["_cache"]["keep"], dtype=bool)
        if (keep0.shape != idx.shape
                or not np.array_equal(np.asarray(fit0["basis"]["t"]), knot)):
            raise RuntimeError("Baseline Kelson selection or basis mismatch")
        final_mask = np.zeros(npx, dtype=bool)
        final_mask[idx[keep0]] = True
        sky_op = frozen_kelson_bspline_operator(
            wavelength, yn, knots=knot, degree=2, y_order=int(y_order),
            final_fit_mask=final_mask.reshape(ny, nx),
            variance=total_var.reshape(ny, nx),
        )
        plans = [_build_linearize_row_plan(wavelength[j], edges)
                 for j in range(ny)]
        if any(plan is None for plan in plans):
            raise RuntimeError("Synthetic COUNT_V1 row plan is unsupported")

        stack = np.full((ny, nlam), .55 if e == 0 else .45)
        if e == 0:
            stack[0] = 1.
        else:
            stack[0] = 0.  # y-shift .25 has no top-row geometric support
        ex = FrozenCovarianceExposure(
            row_plans=plans,
            input_variance=total_var.reshape(ny, nx),
            valid_mask=np.ones((ny, nx), dtype=bool),
            stack_weights=stack,
            y_shift=shift[e],
            output_scale=1./(10.+e),
        )
        states.append({
            "wave": wavelength, "basis": basis, "sky_true": sky_true,
            "obj": obj.ravel(), "base": base_signal, "var_ind": v_ind,
            "sensitivity": sensitivity, "total_var": total_var,
            "baseline_keep": keep0, "sky_operator": sky_op,
            "exposure": ex, "refit": refit,
        })

    def predicted(left_E: np.ndarray, right_E: np.ndarray,
                  left_w: np.ndarray, right_w: np.ndarray) -> tuple[float, float, float]:
        return covariance_of_frozen_sky_chain_grouped_pair(
            [q["exposure"] for q in states],
            [q["sky_operator"] for q in states],
            left_E, right_E, transfer, left_w, right_w,
            [q["var_ind"].reshape(ny, nx) for q in states],
            [q["sensitivity"].reshape(ny, nx) for q in states],
            labels, {g: bvar for g in set(labels)},
        )

    independent, shared, _ = predicted(Eleft, Eright, u, v)
    covariance = np.array([
        [predicted(Eleft, Eleft, u, u)[2], independent+shared],
        [independent+shared, predicted(Eright, Eright, v, v)[2]],
    ])
    if (not np.isfinite(covariance).all() or
            np.linalg.eigvalsh(covariance)[0] <= 0):
        raise RuntimeError("Nonpositive conditional SkySub-to-1D covariance")

    def measure_native(native_frames: list[np.ndarray]) -> np.ndarray:
        # COUNT_V1 uses production per-row overlap integration, while
        # the subsequent deterministic coefficients are held fixed.
        combined = np.zeros((ny, nlam))
        for e, state in enumerate(states):
            rows = [
                _apply_linearize_row_plan(
                    native_frames[e].reshape(ny, nx)[j],
                    state["exposure"].row_plans[j],
                )[0] for j in range(ny)
            ]
            rebin = np.asarray(rows)/(10.+e)
            registered = np.zeros_like(rebin)
            for y in range(ny):
                source_y = y-shift[e]
                below = int(np.floor(source_y))
                fraction = source_y-below
                if (below >= 0 and below < ny and
                        (fraction <= 1e-12 or below+1 < ny)):
                    registered[y] = (1.-fraction)*rebin[below]
                    if fraction > 1e-12:
                        registered[y] += fraction*rebin[below+1]
            combined += state["exposure"].stack_weights*registered
        output = combined*transfer[None, :]
        return np.array([
            float(u @ np.sum(output*Eleft, axis=0)),
            float(v @ np.sum(output*Eright, axis=0)),
        ])

    truth = measure_native([x["obj"] for x in states])
    rng = np.random.default_rng(int(seed))
    n = int(n_realizations)
    common_bias = {
        g: rng.standard_normal((n, npx))*np.sqrt(bvar)[None, :]
        for g in sorted(set(labels))
    }
    indep_noise = [
        rng.standard_normal((n, npx))*np.sqrt(q["var_ind"])[None, :]
        for q in states
    ]
    residuals = np.zeros((n, 2))
    changed = np.zeros((n, 2), dtype=bool)
    bad_stop = np.zeros((n, 2), dtype=bool)
    for k in range(n):
        outputs = []
        for e, state in enumerate(states):
            noisy = (state["base"] + indep_noise[e][k]
                     + state["sensitivity"]*common_bias[labels[e]][k])
            fit = state["refit"](noisy[idx])
            kept = np.asarray(fit["_cache"]["keep"], dtype=bool)
            if kept.shape != state["baseline_keep"].shape:
                raise RuntimeError("Synthetic refit sky sample size changed")
            if not np.array_equal(
                np.asarray(fit["basis"]["t"]), knot if False else
                # The exact saved knot vector is checked per exposure.
                state["sky_operator"].knots
            ):
                raise RuntimeError("Synthetic refit sky wavelength knots changed")
            changed[k, e] = np.any(kept != state["baseline_keep"])
            bad_stop[k, e] = (
                int(fit["solver_diagnostics"]["stop_code"]) not in (1, 2)
            )
            c0 = np.asarray(fit["coeff"]["c0"], dtype=float)
            model = np.asarray(state["basis"] @ c0).ravel()
            if y_order == 1:
                c1 = np.asarray(fit["coeff"]["c1"], dtype=float)
                model += yn.repeat(nx)*np.asarray(state["basis"] @ c1).ravel()
            outputs.append(noisy-model)
        residuals[k] = measure_native(outputs)-truth

    observed_cov = np.cov(residuals, rowvar=False, ddof=1)
    z = residuals/np.sqrt(np.diag(covariance))[None, :]
    coverage95 = np.mean(np.abs(z) <= 1.959963984540054, axis=0)
    nominal68 = erf(1./sqrt(2.))
    coverage68 = np.mean(np.abs(z) <= 1., axis=0)
    mean_z = np.mean(z, axis=0)
    relative_diag = np.abs(np.diag(observed_cov)/np.diag(covariance)-1.)
    relative_cross = abs(observed_cov[0, 1]-covariance[0, 1])/sqrt(
        covariance[0, 0]*covariance[1, 1]
    )
    tol_var = max(.10, 5.*sqrt(2./(n-1)))
    tol_cross = max(.10, 5./sqrt(n-1))
    tol_mean = 5./sqrt(n)
    tol_68 = max(.03, 5.*sqrt(nominal68*(1-nominal68)/n))
    tol_95 = max(.025, 5.*sqrt(.95*.05/n))
    consistent = bool(
        np.max(relative_diag) <= tol_var
        and relative_cross <= tol_cross
        and np.max(np.abs(mean_z)) <= tol_mean
        and np.max(np.abs(coverage68-nominal68)) <= tol_68
        and np.max(np.abs(coverage95-.95)) <= tol_95
    )
    n_bad = np.sum(bad_stop, axis=0).astype(int).tolist()
    n_changed = np.sum(changed, axis=0).astype(int).tolist()
    if any(n_bad):
        status = "NUMERIC_UNSTABLE"
    elif any(n_changed):
        status = "SELECTION_VARIABLE"
    elif consistent:
        status = "MC_CONSISTENT"
    else:
        status = "MC_MISMATCH"

    return {
        "status": status,
        "scope": "SYNTHETIC_KELSON_RECLIPPED_FIXED_2D_TO_1D",
        "n_realizations": n, "group_ids": list(labels),
        "y_order": int(y_order), "clip_sigma": float(clip_sigma),
        "contamination_sigma": float(contamination_sigma),
        "analytic_covariance": covariance.tolist(),
        "analytic_cross_independent": float(independent),
        "analytic_cross_shared_bias": float(shared),
        "empirical_covariance": observed_cov.tolist(),
        "relative_variance_errors": relative_diag.tolist(),
        "scaled_cross_covariance_error": float(relative_cross),
        "coverage_1sigma": coverage68.tolist(),
        "coverage_95percent": coverage95.tolist(),
        "standardized_mean": mean_z.tolist(),
        "conditional_mc_consistent": consistent,
        "selection_change_fraction_any_exposure": float(np.mean(np.any(changed, axis=1))),
        "selection_changed_per_exposure": n_changed,
        "nonconverged_per_exposure": n_bad,
        "tolerances": {
            "relative_variance": tol_var,
            "scaled_cross_covariance": tol_cross,
            "standardized_mean": tol_mean,
            "coverage_68": tol_68,
            "coverage_95": tol_95,
        },
        "science_uncertainty_qualified": False,
        "production_variance_release_allowed": False,
    }
