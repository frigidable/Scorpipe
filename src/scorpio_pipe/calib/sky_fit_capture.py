"""Versioned opt-in capture of ACTUAL Kelson raw fit inputs and selection.

This does not claim an exact derivative of sigma-clipping decisions or finite-
iteration LSQR, and never modifies science SCI/VAR/MASK. The resulting
artifact is a frozen conditional fit *provenance*, not a covariance product.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


_VERSION = 1
_REQUIRED = frozenset({
    "version", "status", "lambda_map", "y_norm_rows", "fit_indices",
    "fit_variance", "knots", "degree", "y_order", "coeff0", "coeff1",
    "lsqr_stop", "lsqr_iterations", "lsqr_residual", "lsqr_condition",
    "lsqr_atol", "lsqr_btol", "lsqr_iter_limit",
})


def capture_kelson_fit(
    path: Path,
    *,
    fit: dict[str, Any],
    lambda_map: np.ndarray,
    y_norm_rows: np.ndarray,
    candidate_flat_indices: np.ndarray,
    candidate_lambda: np.ndarray,
    candidate_y: np.ndarray,
    candidate_data: np.ndarray,
    candidate_variance: np.ndarray,
) -> dict[str, Any]:
    """Write a complete, explicit *frozen WLS* snapshot of one real sky fit.

    The final clipping selection in fit['_cache']['keep'] indexes the
    production fit's finite-positive subset, not the original sky-window
    candidates. Remap it to original CCD pixels before publishing.

    The snapshot includes evaluated wavelength geometry; it does not rely
    on a mutable external wavelength file for reconstruction.
    """
    lam = np.asarray(lambda_map, dtype=np.float64)
    yn = np.asarray(y_norm_rows, dtype=np.float64)
    candidate = np.asarray(candidate_flat_indices)
    cl = np.asarray(candidate_lambda, dtype=np.float64)
    cy = np.asarray(candidate_y, dtype=np.float64)
    cd = np.asarray(candidate_data, dtype=np.float64)
    cv = np.asarray(candidate_variance, dtype=np.float64)
    if lam.ndim != 2 or not all(x > 0 for x in lam.shape):
        raise ValueError("Frozen sky snapshot requires 2-D nonempty wavelength geometry")
    if yn.shape != (lam.shape[0],) or not np.isfinite(yn).all():
        raise ValueError("Frozen sky y geometry mismatch")
    if (candidate.ndim != 1 or not np.issubdtype(candidate.dtype, np.integer)
            or not all(v.shape == candidate.shape for v in (cl, cy, cd, cv))):
        raise ValueError("Sky fit candidate arrays must have matching 1-D geometry")
    if (not candidate.size or np.any(candidate < 0)
            or np.any(candidate >= lam.size)
            or np.unique(candidate).size != candidate.size):
        raise ValueError("Sky fit candidate pixel IDs invalid or duplicated")
    finite = (
        np.isfinite(cl) & np.isfinite(cy) & np.isfinite(cd)
        & np.isfinite(cv) & (cv > 0)
    )
    cache = fit.get("_cache") or {}
    kept = np.asarray(cache.get("keep"))
    if kept.dtype != np.dtype(bool) or kept.shape != (int(finite.sum()),):
        raise ValueError("Actual Kelson final clipping selection unavailable/inconsistent")
    if not kept.any():
        raise ValueError("Cannot capture an empty final sky fit")
    selected = candidate[finite][kept].astype(np.int64)
    fit_variance = cv[finite][kept].astype(np.float64)
    fit_lambda = cl[finite][kept]
    fit_y = cy[finite][kept]
    if (not np.allclose(lam.ravel()[selected], fit_lambda, rtol=0, atol=1e-8)
            or not np.allclose(yn[selected // lam.shape[1]], fit_y, rtol=0, atol=1e-12)):
        raise ValueError("Frozen fit pixels disagree with actual wavelength/y geometry")
    if "var" not in cache or "lam" not in cache:
        raise ValueError("Kelson cache missing selected science variance/wavelength")
    if (not np.allclose(np.asarray(cache["var"])[kept], fit_variance, rtol=5e-7, atol=1e-7)
            or not np.allclose(np.asarray(cache["lam"])[kept], fit_lambda, rtol=5e-7, atol=1e-3)):
        raise ValueError("Kelson solver cache disagrees with selected fit data")

    basis = fit.get("basis") or {}
    coeff = fit.get("coeff") or {}
    knot = np.asarray(basis.get("t"), dtype=np.float64)
    c0 = np.asarray(coeff.get("c0"), dtype=np.float64)
    order = int(fit.get("y_order", -1))
    degree = int(basis.get("degree", -1))
    if (knot.ndim != 1 or knot.size < degree + 2 or degree < 1
            or order not in (0, 1) or not np.isfinite(knot).all()
            or np.any(np.diff(knot) < 0)):
        raise ValueError("Invalid actual Kelson B-spline geometry")
    n_basis = knot.size - degree - 1
    if c0.shape != (n_basis,) or not np.isfinite(c0).all():
        raise ValueError("Actual Kelson coefficient geometry invalid")
    c1_raw = coeff.get("c1")
    if order == 1:
        c1 = np.asarray(c1_raw, dtype=np.float64)
        if c1.shape != (n_basis,) or not np.isfinite(c1).all():
            raise ValueError("Missing/invalid spatial Kelson coefficients")
    else:
        if c1_raw is not None:
            raise ValueError("Unexpected spatial coefficients in y_order=0 fit")
        c1 = np.empty(0, dtype=np.float64)

    solver = fit.get("solver_diagnostics") or {}
    if str(solver.get("method") or "").strip().lower() != "lsqr":
        raise ValueError("Actual Kelson LSQR diagnostics missing")
    stop = int(solver["stop_code"])
    iterations = int(solver["iterations"])
    resnorm = float(solver["residual_norm"])
    cond = float(solver["condition_estimate"])
    atol = float(solver["atol"])
    btol = float(solver["btol"])
    itlim = int(solver["iter_lim"])
    if (stop not in range(8) or iterations < 0 or iterations > itlim
            or itlim <= 0 or not np.isfinite(resnorm) or resnorm < 0
            or not np.isfinite(cond) or cond < 0
            or not np.isfinite(atol) or atol <= 0
            or not np.isfinite(btol) or btol <= 0):
        raise ValueError("Invalid actual LSQR numerical diagnostics")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": np.array(_VERSION, dtype=np.int16),
        "status": np.array("FROZEN_CONDITIONAL"),
        "lambda_map": lam,
        "y_norm_rows": yn,
        "fit_indices": selected,
        "fit_variance": fit_variance,
        "knots": knot,
        "degree": np.array(degree, dtype=np.int16),
        "y_order": np.array(order, dtype=np.int16),
        "coeff0": c0,
        "coeff1": c1,
        "lsqr_stop": np.array(stop, dtype=np.int16),
        "lsqr_iterations": np.array(iterations, dtype=np.int32),
        "lsqr_residual": np.array(resnorm, dtype=np.float64),
        "lsqr_condition": np.array(cond, dtype=np.float64),
        "lsqr_atol": np.array(atol, dtype=np.float64),
        "lsqr_btol": np.array(btol, dtype=np.float64),
        "lsqr_iter_limit": np.array(itlim, dtype=np.int32),
    }
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=str(path.parent), prefix="." + path.stem + ".",
            suffix=".npz", delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    return {
        "path": str(path),
        "version": _VERSION,
        "status": "FROZEN_CONDITIONAL",
        "n_candidates": int(candidate.size),
        "n_finite_candidates": int(finite.sum()),
        "n_final_fit_pixels": int(selected.size),
        "n_coefficients": int(n_basis * (order + 1)),
        "solver": "LSQR_FINITE_ITERATION",
        "lsqr_stop_code": stop,
        "lsqr_iterations": iterations,
        "lsqr_condition": cond,
        "selection": "ACTUAL_POST_CLIP",
        "covariance_qualified": False,
    }


def load_frozen_kelson_fit(path: Path):
    """Load and validate the frozen fit for a *conditional* WLS oracle.

    Returns (FrozenSkyWLS, stored fitted coefficients). The caller must
    separately verify the sky stage's real LSQR solver output; a frozen WLS
    adjoint does NOT include clipping/geometry/weight selection derivatives.
    """
    from scorpio_pipe.calib.sky_bias_covariance import frozen_kelson_bspline_operator

    with np.load(Path(path), allow_pickle=False) as data:
        if set(data.files) != _REQUIRED:
            raise ValueError("Frozen sky fit schema incomplete or unexpected")
        if int(data["version"]) != _VERSION:
            raise ValueError("Unsupported frozen sky fit version")
        if str(data["status"].item()) != "FROZEN_CONDITIONAL":
            raise ValueError("Frozen sky fit is not conditional WLS")
        lam = np.asarray(data["lambda_map"], dtype=np.float64)
        y = np.asarray(data["y_norm_rows"], dtype=np.float64)
        ids = np.asarray(data["fit_indices"])
        vs = np.asarray(data["fit_variance"], dtype=np.float64)
        knot = np.asarray(data["knots"], dtype=np.float64)
        degree = int(data["degree"])
        y_order = int(data["y_order"])
        c0 = np.asarray(data["coeff0"], dtype=np.float64)
        c1 = np.asarray(data["coeff1"], dtype=np.float64)
        stop = int(data["lsqr_stop"])
        iterations = int(data["lsqr_iterations"])
        residual = float(data["lsqr_residual"])
        condition = float(data["lsqr_condition"])
        atol = float(data["lsqr_atol"])
        btol = float(data["lsqr_btol"])
        max_iter = int(data["lsqr_iter_limit"])
    if (stop not in range(8) or iterations < 0 or iterations > max_iter
            or max_iter <= 0 or not np.isfinite(residual) or residual < 0
            or not np.isfinite(condition) or condition < 0
            or not np.isfinite(atol) or atol <= 0
            or not np.isfinite(btol) or btol <= 0):
        raise ValueError("Invalid stored LSQR diagnostics")
    if lam.ndim != 2 or ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Invalid frozen CCD geometry/index type")
    if (not ids.size or len(np.unique(ids)) != ids.size or
            np.any(ids < 0) or np.any(ids >= lam.size)
            or vs.shape != ids.shape or np.any(~np.isfinite(vs) | (vs <= 0))):
        raise ValueError("Invalid stored sky pixel selection/weights")
    if y.shape != (lam.shape[0],) or not np.isfinite(y).all():
        raise ValueError("Stored sky y-coordinates inconsistent")
    if degree < 1 or y_order not in (0, 1):
        raise ValueError("Invalid frozen spline order")
    nb = knot.size - degree - 1
    if c0.shape != (nb,) or c1.shape != ((nb,) if y_order else (0,)):
        raise ValueError("Frozen sky fit coefficients have incompatible geometry")
    full_var = np.ones(lam.shape, dtype=np.float64)
    full_var.ravel()[ids] = vs
    final_mask = np.zeros(lam.size, dtype=bool)
    final_mask[ids] = True
    op = frozen_kelson_bspline_operator(
        lam, y, knots=knot, degree=degree, y_order=y_order,
        final_fit_mask=final_mask.reshape(lam.shape),
        variance=full_var,
    )
    return op, (c0, c1)
