"""Bounded, read-only SVD reference for frozen SkySub covariance numerics.

Tests the normal-equation LU used by FrozenSkyWLS against independent
singular-value decomposition of the weighted training matrix. This is a
numerical diagnostic, NOT an astrophysical uncertainty qualification.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from scorpio_pipe.calib.sky_bias_covariance import FrozenSkyWLS


def audit_frozen_wls_numerics(
    operator: FrozenSkyWLS,
    *,
    n_probes: int = 3,
    seed: int = 20261104,
    max_condition: float = 1e5,
    max_relative_error: float = 1e-6,
    max_fit_entries: int = 300_000,
    max_native_pixels: int = 100_000,
) -> dict[str, Any]:
    """Audit a fixed-fit covariance operator against an independent SVD.

    Only the weighted FITTING matrix is densified, after a strict resource
    bound. The full native CCD design remains sparse. A fit with very high
    condition number, numerical rank deficiency, or inaccurate LU forward /
    adjoint action does not pass, even when sparse LU returned a result.

    max_condition is a user-specified conservative diagnostic threshold
    on cond(sqrt(W) Xfit), NOT an empirically calibrated universal limit.
    """
    if not isinstance(operator, FrozenSkyWLS):
        raise ValueError("Expected actual FrozenSkyWLS operator")
    if operator.ridge != 0.:
        raise ValueError("SVD oracle supports only unregularized frozen WLS")
    if (isinstance(n_probes, (bool, np.bool_))
            or not isinstance(n_probes, (int, np.integer))
            or not 1 <= int(n_probes) <= 8):
        raise ValueError("n_probes must be an integer in [1, 8]")
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    if (not np.isfinite(max_condition) or max_condition <= 1
            or not np.isfinite(max_relative_error) or max_relative_error < 0):
        raise ValueError("Finite positive conditioning and precision limits required")
    if (isinstance(max_fit_entries, (bool, np.bool_))
            or not isinstance(max_fit_entries, (int, np.integer))
            or max_fit_entries < 1
            or isinstance(max_native_pixels, (bool, np.bool_))
            or not isinstance(max_native_pixels, (int, np.integer))
            or max_native_pixels < 1):
        raise ValueError("Positive integer resource ceilings required")

    m, p = operator.fit_design.shape
    n = operator.n_pixels
    if m*p > max_fit_entries or n > max_native_pixels:
        raise ValueError("SVD reference resource ceiling exceeded")

    # Crucially, the oracle uses a DIFFERENT numerical factorization from
    # production's normal-equation LU (condition approximately squared).
    root_weights = np.sqrt(operator.fit_weights)
    whitened = operator.fit_design.toarray() * root_weights[:, None]
    uu, ss, vt = np.linalg.svd(whitened, full_matrices=False)
    if ss.shape != (p,) or not np.isfinite(ss).all() or ss[0] <= 0:
        raise ValueError("SVD of weighted sky design invalid")

    rank_cut = max(m, p) * np.finfo(np.float64).eps * ss[0]
    rank = int(np.count_nonzero(ss > rank_cut))
    cond = float(ss[0] / ss[-1]) if ss[-1] > 0 else float("inf")
    if rank < p:
        return {
            "status": "RANK_DEFICIENT",
            "scope": "SYNTHETIC_OR_FROZEN_FIXED_WLS_NUMERICS",
            "n_pixels": n,
            "n_fit_samples": m,
            "n_coefficients": p,
            "effective_rank": rank,
            "condition_weighted_design": cond,
            "normal_equation_condition_estimate": float(cond * cond),
            "n_probes": 0,
            "max_forward_relative_error": None,
            "max_model_relative_error": None,
            "max_adjoint_relative_error": None,
            "max_duality_relative_error": None,
            "science_uncertainty_qualified": False,
        }

    rng = np.random.default_rng(int(seed))
    max_forward = 0.
    max_model = 0.
    max_adjoint = 0.
    max_duality = 0.
    # SVD pseudo-inverse action is a high-accuracy reference for the
    # *fully converged* weighted least-squares solution on a full-rank fit.
    for _ in range(int(n_probes)):
        v = rng.standard_normal(n)
        q = rng.standard_normal(n)
        weighted_v = root_weights * v[operator.fit_indices]
        c = vt.T @ ((uu.T @ weighted_v) / ss)
        predicted_model = np.asarray(operator.design @ c).ravel()
        expected_forward = v - predicted_model

        estimated_forward = operator.apply(v)
        estimated_model = v - estimated_forward
        forward_err = float(
            np.linalg.norm(estimated_forward - expected_forward)
            / max(np.linalg.norm(expected_forward), 1e-12)
        )
        model_err = float(
            np.linalg.norm(estimated_model - predicted_model)
            / max(np.linalg.norm(predicted_model), 1e-12)
        )

        # Independent transpose reference for the same SVD projector.
        d = vt @ np.asarray(operator.design.T @ q).ravel()
        sky_contribution = root_weights * (uu @ (d / ss))
        expected_adjoint = q.copy()
        np.add.at(expected_adjoint, operator.fit_indices, -sky_contribution)
        estimated_adjoint = operator.adjoint(q)
        adjoint_err = float(
            np.linalg.norm(estimated_adjoint - expected_adjoint)
            / max(np.linalg.norm(expected_adjoint), 1e-12)
        )
        lhs = float(np.dot(q, estimated_forward))
        rhs = float(np.dot(estimated_adjoint, v))
        duality_err = float(abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.))
        max_forward = max(max_forward, forward_err)
        max_model = max(max_model, model_err)
        max_adjoint = max(max_adjoint, adjoint_err)
        max_duality = max(max_duality, duality_err)

    checks = (max_forward, max_model, max_adjoint, max_duality)
    if not all(np.isfinite(x) for x in checks):
        status = "NUMERIC_MISMATCH"
    elif cond > max_condition:
        status = "ILL_CONDITIONED"
    elif any(x > max_relative_error for x in checks):
        status = "NUMERIC_MISMATCH"
    else:
        status = "STABLE"

    return {
        "status": status,
        "scope": "SYNTHETIC_OR_FROZEN_FIXED_WLS_NUMERICS",
        "n_pixels": n,
        "n_fit_samples": m,
        "n_coefficients": p,
        "effective_rank": rank,
        "condition_weighted_design": cond,
        "normal_equation_condition_estimate": float(cond * cond),
        "n_probes": int(n_probes),
        "max_forward_relative_error": max_forward,
        "max_model_relative_error": max_model,
        "max_adjoint_relative_error": max_adjoint,
        "max_duality_relative_error": max_duality,
        "max_allowed_condition": float(max_condition),
        "max_allowed_relative_error": float(max_relative_error),
        "science_uncertainty_qualified": False,
    }
