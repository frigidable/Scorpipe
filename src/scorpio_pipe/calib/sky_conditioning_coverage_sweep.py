"""Synthetic conditioning-versus-coverage oracle for frozen sky estimators.

Two independent ways to solve the SAME known-truth synthetic WLS task:
LU on normal equations to predict covariance, and dense SVD to
generate re-fitted Monte Carlo realizations. No real SCORPIO data or
production FITS are read or changed.
"""
from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from typing import Any

import numpy as np

from scorpio_pipe.calib.sky_bias_covariance import (
    FrozenSkyWLS,
    covariance_of_frozen_sky_grouped,
)
from scorpio_pipe.calib.sky_wls_conditioning import audit_frozen_wls_numerics


def synthetic_wls_conditioning_coverage_sweep(
    *,
    perturbations: Sequence[float] = (1e-1, 1e-3, 1e-5, 1e-7),
    group_ids: Sequence[str] = ("A", "A"),
    n_realizations: int = 5000,
    seed: int = 20261104,
    max_condition: float = 1e5,
    max_numerical_error: float = 1e-5,
) -> dict[str, Any]:
    """Quantify conditioning, covariance error, and MC coverage separately.

    A 16-pixel algebraic proxy has progressively near-collinear fitting
    columns; it is not a physically simulated SCORPIO slit or claim about
    its actual condition spectrum. Two exposures use separate noise but
    share a per-draw MasterBias map iff their group IDs match.

    The SVD condition gate is deliberately a conservative *engineering*
    control, not a threshold inferred from empirical confidence coverage.
    """
    try:
        eps = tuple(perturbations)
    except TypeError as exc:
        raise ValueError("Perturbations must be a finite numeric sequence") from exc
    if (not 2 <= len(eps) <= 8
            or any(isinstance(v, (bool, np.bool_))
                   or not isinstance(v, (int, float, np.integer, np.floating))
                   or not np.isfinite(v) or v <= 0 for v in eps)
            or any(a <= b for a, b in zip(eps[:-1], eps[1:]))):
        raise ValueError("Provide 2-8 strictly descending positive finite perturbations")
    labels = tuple(group_ids)
    if (len(labels) != 2 or
            any(not isinstance(v, str) or not v or v.strip() != v
                for v in labels)):
        raise ValueError("Exactly two explicit nonblank MasterBias groups required")
    if (isinstance(n_realizations, (bool, np.bool_)) or
            not isinstance(n_realizations, (int, np.integer)) or
            not 2000 <= n_realizations <= 20000):
        raise ValueError("n_realizations must be an integer in [2000,20000]")
    if (isinstance(seed, (bool, np.bool_)) or
            not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("Seed must be a nonnegative integer")
    if (not np.isfinite(max_condition) or max_condition <= 1 or
            not np.isfinite(max_numerical_error) or max_numerical_error < 0):
        raise ValueError("Invalid SVD numerical qualification limits")

    n = 16
    rows = np.arange(n, dtype=np.float64)
    xx = np.linspace(-1., 1., n)
    sky_idx = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 14, 15])
    sampler = np.eye(n)[sky_idx]
    bvar = .6 + .2 * rows / (n-1.)
    functionals = [
        [np.eye(n)[5] * .5, np.eye(n)[5] * .6],
        [np.eye(n)[10] * .5, np.eye(n)[10] * .6],
    ]
    rng = np.random.default_rng(seed)
    trials: list[dict[str, Any]] = []
    for severity in eps:
        instruments = []
        for e in range(2):
            design = np.column_stack([
                np.ones(n), xx,
                xx + float(severity)*(1. + .1*e)*np.sin(rows*1.31),
            ])
            source_var = 1. + .08*rows/(n-1.) + .1*e
            sensitivity = -.7 - .08*e - .03*xx
            weights = 1./(source_var[sky_idx]
                          + sensitivity[sky_idx]**2*bvar[sky_idx])
            rootw = np.sqrt(weights)
            whitened = design[sky_idx]*rootw[:, None]
            U, singular, Vt = np.linalg.svd(whitened, full_matrices=False)
            if not np.isfinite(singular).all() or singular[-1] <= 0:
                raise ValueError("Cannot resolve synthetic reference design")
            condition = float(singular[0]/singular[-1])
            if not np.isfinite(condition):
                raise ValueError("Nonfinite synthetic design condition")
            # Independent dense SVD residual projector in native pixels.
            reference_R = (
                np.eye(n) - design @ (Vt.T/singular) @ U.T
                @ (rootw[:, None]*sampler)
            )
            object_truth = np.zeros(n)
            object_truth[5] = 3. + e
            object_truth[10] = 2.5 + .3*e
            sky_truth = design @ np.array([14., 2., -1.])
            try:
                op = FrozenSkyWLS(design, sky_idx, weights)
                audit = audit_frozen_wls_numerics(
                    op, n_probes=2, seed=42,
                    max_condition=max_condition,
                    max_relative_error=max_numerical_error,
                )
                gate = audit["status"]
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                op = None
                gate = "FACTORIZATION_FAILED"
                audit = {"error": f"{type(exc).__name__}: {exc}"}
            instruments.append({
                "op": op, "gate": gate, "audit": audit,
                "condition": condition, "design": design,
                "svd": (U, singular, Vt), "Rref": reference_R,
                "sqrt_weights": rootw, "source_var": source_var,
                "sensitivity": sensitivity, "sky_truth": sky_truth,
                "object_truth": object_truth,
            })

        reference_cov = np.zeros((2, 2), dtype=np.float64)
        for i in range(2):
            for j in range(2):
                left = [d["Rref"].T @ functionals[i][e]
                        for e, d in enumerate(instruments)]
                right = [d["Rref"].T @ functionals[j][e]
                         for e, d in enumerate(instruments)]
                ind = sum(np.dot(d["source_var"], left[e]*right[e])
                          for e, d in enumerate(instruments))
                shared = 0.
                for gid in set(labels):
                    lgroup = sum(
                        (d["sensitivity"]*left[e]
                         for e, d in enumerate(instruments)
                         if labels[e] == gid), np.zeros(n),
                    )
                    rgroup = sum(
                        (d["sensitivity"]*right[e]
                         for e, d in enumerate(instruments)
                         if labels[e] == gid), np.zeros(n),
                    )
                    shared += float(np.dot(bvar, lgroup*rgroup))
                reference_cov[i, j] = ind + shared
        if (not np.isfinite(reference_cov).all() or
                np.min(np.linalg.eigvalsh(reference_cov)) <= 0):
            raise ValueError("SVD reference covariance must be positive definite")

        predicted_cov = None
        if all(d["op"] is not None for d in instruments):
            predicted_cov = np.zeros((2, 2), dtype=np.float64)
            for i in range(2):
                for j in range(2):
                    _, _, total = covariance_of_frozen_sky_grouped(
                        [d["op"] for d in instruments],
                        functionals[i], functionals[j],
                        [d["source_var"] for d in instruments],
                        [d["sensitivity"] for d in instruments],
                        labels, {g: bvar for g in set(labels)},
                    )
                    predicted_cov[i, j] = total

        # Each Monte Carlo draw has a fresh independent detector field and
        # a single shared bias field per group, reused within that draw.
        bias_draws = {
            g: rng.standard_normal((n_realizations, n))*np.sqrt(bvar)
            for g in sorted(set(labels))
        }
        errors = np.zeros((n_realizations, 2), dtype=np.float64)
        for e, d in enumerate(instruments):
            independent = (rng.standard_normal((n_realizations, n))
                           *np.sqrt(d["source_var"]))
            signal = (
                d["sky_truth"][None, :] + d["object_truth"][None, :]
                + independent
                + d["sensitivity"][None, :]*bias_draws[labels[e]]
            )
            U, S, Vt = d["svd"]
            coeff = (
                (signal[:, sky_idx]*d["sqrt_weights"][None, :]) @ U / S
            ) @ Vt
            residual = signal - coeff @ d["design"].T
            residual -= d["object_truth"][None, :]
            for j in range(2):
                errors[:, j] += residual @ functionals[j][e]
        empirical = np.cov(errors, rowvar=False, ddof=1)
        z_reference = errors/np.sqrt(np.diag(reference_cov))[None, :]
        ref_cov95 = np.mean(np.abs(z_reference) <= 1.959963984540054, axis=0)
        ref_var_error = np.abs(np.diag(empirical)/np.diag(reference_cov)-1.)
        ref_coverage_limit = max(.02, 5.*sqrt(.95*.05/n_realizations))
        ref_var_limit = max(.05, 5.*sqrt(2./(n_realizations-1)))
        reference_ok = bool(
            np.max(ref_var_error) <= ref_var_limit
            and np.max(np.abs(ref_cov95-.95)) <= ref_coverage_limit
        )
        predicted_coverage = None
        variance_ratios = None
        if predicted_cov is not None and np.all(np.diag(predicted_cov) > 0):
            z_pred = errors/np.sqrt(np.diag(predicted_cov))[None, :]
            predicted_coverage = np.mean(
                np.abs(z_pred) <= 1.959963984540054, axis=0,
            ).tolist()
            variance_ratios = (
                np.diag(predicted_cov)/np.diag(reference_cov)
            ).tolist()
        status = (
            "REFERENCE_MC_MISMATCH" if not reference_ok else
            ("NUMERIC_SAFE" if all(d["gate"] == "STABLE"
                                   for d in instruments)
             else "NUMERIC_BLOCKED")
        )
        trials.append({
            "perturbation": float(severity),
            "status": status,
            "weighted_design_conditions": [d["condition"]
                                           for d in instruments],
            "svd_gate_statuses": [d["gate"] for d in instruments],
            "svd_numerical_gate_details": [d["audit"] for d in instruments],
            "reference_covariance": reference_cov.tolist(),
            "lu_predicted_covariance": (None if predicted_cov is None
                                        else predicted_cov.tolist()),
            "empirical_covariance": empirical.tolist(),
            "lu_to_svd_variance_ratios": variance_ratios,
            "reference_coverage95": ref_cov95.tolist(),
            "lu_predicted_coverage95": predicted_coverage,
            "reference_mc_consistent": reference_ok,
            "reference_mc_variance_error": ref_var_error.tolist(),
        })
    return {
        "scope": "SYNTHETIC_NEAR_COLLINEAR_WLS_SVD_MC",
        "status": (
            "REFERENCE_MC_MISMATCH" if any(
                x["status"] == "REFERENCE_MC_MISMATCH" for x in trials
            ) else "REPORTED"
        ),
        "n_realizations": int(n_realizations),
        "group_ids": list(labels),
        "max_condition": float(max_condition),
        "max_numerical_error": float(max_numerical_error),
        "trials": trials,
        "science_uncertainty_qualified": False,
    }
