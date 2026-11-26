"""Opt-in LSQR-vs-converged-WLS directional derivative fidelity diagnostic.

Checks a single REAL frozen fitted sky against SciPy's production LSQR
algorithm under *frozen* sky pixel selection, weights, knots and geometry.

This is not an unconditional Jacobian for sigma clipping or sky geometry.
Only SCI output sensitivity is tested; no production VAR is rewritten.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from scipy.sparse.linalg import lsqr

from scorpio_pipe.calib.sky_fit_audit import audit_frozen_kelson_products
from scorpio_pipe.calib.sky_fit_capture import load_frozen_kelson_fit


def assess_kelson_lsqr_directional_derivatives(
    frozen_fit_path: str | Path,
    skymodel_fits: str | Path,
    skysub_fits: str | Path,
    *,
    seed: int = 20261104,
    n_probes: int = 3,
    epsilon_sigma: float = 0.1,
    relative_tolerance: float = 0.02,
    replay_tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Compare the finite-iteration LSQR directional response to frozen WLS.

    Central finite differences recompute the *actual SciPy LSQR* with its
    archived tolerances and iteration limit, keeping the final sky-pixel
    selection and variances fixed. Probe directions are random independent
    Gaussian science pixel noise with a scale determined by the median
    selected sky variance (in the same detector units as SCI).

    A 'MATCH' is only a local numerical derivative diagnostic. Derivative
    discontinuities from sigma-clipping, mask/ROI estimation, or covariance
    of shared MasterFlat are completely excluded.
    """
    if (not isinstance(n_probes, (int, np.integer)) or
            not 1 <= n_probes <= 20 or
            not np.isfinite(float(epsilon_sigma)) or epsilon_sigma <= 0 or
            not np.isfinite(float(relative_tolerance)) or relative_tolerance < 0 or
            not np.isfinite(float(replay_tolerance)) or replay_tolerance < 0):
        raise ValueError("Invalid finite-difference probe controls")

    # First require that the stored fit reproduces the actual FITS model and
    # the actual terminal LSQR residual, and that the archive identity matches.
    baseline = audit_frozen_kelson_products(
        Path(frozen_fit_path), Path(skymodel_fits), Path(skysub_fits)
    )
    op, (c0, c1) = load_frozen_kelson_fit(Path(frozen_fit_path))
    with np.load(Path(frozen_fit_path), allow_pickle=False) as d:
        atol = float(d["lsqr_atol"])
        btol = float(d["lsqr_btol"])
        iter_limit = int(d["lsqr_iter_limit"])
        saved_istop = int(d["lsqr_stop"])
        saved_condition = float(d["lsqr_condition"])

    # Require a numerically plausible original fit; never regard an
    # iteration cap or condition-stop as certified linear-WLS behavior.
    fit = op.fit_design
    sqrtw = np.sqrt(op.fit_weights)
    Aw = fit.multiply(sqrtw[:, None]).tocsr()
    idx = op.fit_indices
    c_saved = np.r_[c0, c1] if c1.size else c0.copy()

    with fits.open(skymodel_fits, memmap=True) as fm, fits.open(skysub_fits, memmap=True) as fs:
        model = np.asarray(fm["SCI"].data, dtype=np.float64)
        residual = np.asarray(fs["SCI"].data, dtype=np.float64)
        flat_model = model.ravel()
        flat_resid = residual.ravel()
        if (flat_model.size != op.n_pixels or flat_resid.size != op.n_pixels):
            raise ValueError("Sky output pixel count differs from frozen operator")
        observed_fit = flat_model[idx] + flat_resid[idx]
    if not np.isfinite(observed_fit).all():
        raise ValueError("Nonfinite selected data in real SkySub output")

    replay = lsqr(
        Aw, observed_fit * sqrtw, atol=atol, btol=btol,
        iter_lim=iter_limit,
    )
    replay_model_difference = float(
        np.linalg.norm(op.design @ (np.asarray(replay[0]) - c_saved))
        / max(np.linalg.norm(op.design @ c_saved), 1.)
    )
    observed_rms = float(np.sqrt(np.mean(op.fit_weights ** -1)))
    epsilon = float(epsilon_sigma) * observed_rms
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Invalid finite difference scale")

    rng = np.random.default_rng(seed)
    comparisons = []
    for j in range(n_probes):
        # One frozen native-detector perturbation; only the selected sky
        # pixels influence the LSQR response, but EVERY output pixel is
        # checked after sky-model subtraction.
        direction = rng.normal(size=op.n_pixels)
        d_fit = direction[idx] * sqrtw
        plus = lsqr(
            Aw, observed_fit * sqrtw + epsilon * d_fit,
            atol=atol, btol=btol, iter_lim=iter_limit,
        )
        minus = lsqr(
            Aw, observed_fit * sqrtw - epsilon * d_fit,
            atol=atol, btol=btol, iter_lim=iter_limit,
        )
        # Native pixel derivative: d/dt [D + t v - X c(D+t v)]
        coeff_response = (np.asarray(plus[0]) - np.asarray(minus[0])) / (2. * epsilon)
        finite_diff = direction - np.asarray(op.design @ coeff_response).ravel()
        ideal = op.apply(direction)
        difference = finite_diff - ideal
        relative = float(
            np.linalg.norm(difference) / max(np.linalg.norm(ideal), 1e-12)
        )
        # Direct detector noise can hide a large model derivative error
        # when averaged over the entire residual frame.
        ideal_model_response = direction - ideal
        finite_model_response = direction - finite_diff
        model_relative = float(
            np.linalg.norm(finite_model_response - ideal_model_response)
            / max(np.linalg.norm(ideal_model_response), 1e-12)
        )
        comparisons.append({
            "probe": j,
            "relative_l2_difference": relative,
            "model_response_relative_l2_difference": model_relative,
            "plus_lsqr_stop": int(plus[1]),
            "minus_lsqr_stop": int(minus[1]),
            "plus_iterations": int(plus[2]),
            "minus_iterations": int(minus[2]),
        })

    stops = [saved_istop, int(replay[1])]
    for rec in comparisons:
        stops.extend([rec["plus_lsqr_stop"], rec["minus_lsqr_stop"]])
    numerical_convergence = all(x in (1, 2) for x in stops)
    max_difference = max(x["relative_l2_difference"] for x in comparisons)
    max_model_difference = max(
        x["model_response_relative_l2_difference"] for x in comparisons
    )
    matched = (numerical_convergence and
               replay_model_difference <= replay_tolerance and
               max_difference <= relative_tolerance and
               max_model_difference <= relative_tolerance)
    return {
        "status": "MATCH" if matched else "UNSTABLE",
        "scope": "CONDITIONAL_FINAL_SKY_MASK_ONLY",
        "n_probes": n_probes,
        "epsilon_native_signal": epsilon,
        "max_relative_l2_difference": max_difference,
        "max_model_response_relative_l2_difference": max_model_difference,
        "replay_model_relative_difference": replay_model_difference,
        "original_lsqr_stop": saved_istop,
        "original_lsqr_condition_estimate": saved_condition,
        "replay_lsqr_stop": int(replay[1]),
        "all_lsqr_stops_nominal": numerical_convergence,
        "fidelity_audit": baseline["status"],
        "probes": comparisons,
        "science_uncertainty_qualified": False,
    }
