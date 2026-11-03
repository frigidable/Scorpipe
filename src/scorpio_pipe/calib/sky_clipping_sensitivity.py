"""Synthetic-only audit of data-dependent Kelson RAW sigma clipping.

Re-fit the ACTUAL production B-spline+LSQR+clipping algorithm at each
perturbation. Compare the result to the conditional frozen-WLS operator.
Never modifies the stage, sky FITS products, VAR, or calibration metadata.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from scorpio_pipe.calib.sky_bias_covariance import frozen_kelson_bspline_operator
from scorpio_pipe.stages.sky_sub import _fit_kelson_bspline


def audit_synthetic_kelson_clipping(
    science: np.ndarray,
    variance: np.ndarray,
    wavelength: np.ndarray,
    sky_candidate_mask: np.ndarray,
    *,
    y_normalized_rows: np.ndarray | None = None,
    knot_step_A: float = 8.0,
    degree: int = 3,
    y_order: int = 0,
    clip_sigma: float = 4.5,
    max_iter: int = 4,
    min_pixels: int = 64,
    seed: int = 20261104,
    n_probes: int = 3,
    perturbation_sigma: float = 0.1,
    residual_tolerance: float = 0.05,
    model_tolerance: float = 0.05,
    probe_vectors: np.ndarray | None = None,
    max_pixels: int = 100_000,
) -> dict[str, Any]:
    """Diagnose when a frozen sky residual Jacobian is invalidated by re-clipping.

    Four possible statuses: FROZEN_STABLE, SELECTION_CHANGED,
    NUMERIC_UNSTABLE and REFIT_FAILED. FROZEN_STABLE means ONLY that all
    tested finite perturbations retained the baseline sky sample AND
    matched the converged frozen WLS derivative, conditionally on the
    supplied 2-D wavelength/y geometry and fixed variance.

    This is not an empirical confidence interval or a calibrated VAR.
    """
    image = np.asarray(science, dtype=np.float64)
    var = np.asarray(variance, dtype=np.float64)
    wave = np.asarray(wavelength, dtype=np.float64)
    selected = np.asarray(sky_candidate_mask)
    if (image.ndim != 2 or min(image.shape) == 0
            or not 0 < image.size <= int(max_pixels)
            or var.shape != image.shape or wave.shape != image.shape
            or selected.shape != image.shape
            or selected.dtype != np.dtype(bool)):
        raise ValueError("Valid matching small 2-D science/variance/wavelength/mask required")
    if (not np.isfinite(image).all() or not np.isfinite(var).all()
            or np.any(var <= 0) or not np.isfinite(wave).all()):
        raise ValueError("Finite SCI and wavelength and positive finite variance required")
    ny, nx = image.shape
    if y_normalized_rows is None:
        y_rows = 2. * (np.arange(ny) - .5 * (ny - 1)) / max(ny - 1, 1)
    else:
        y_rows = np.asarray(y_normalized_rows, dtype=np.float64)
    if y_rows.shape != (ny,) or not np.isfinite(y_rows).all():
        raise ValueError("Invalid spatial row coordinates")
    if (int(y_order) not in (0, 1) or int(degree) < 1 or
            int(max_iter) < 1 or int(min_pixels) < 1 or
            not np.isfinite(knot_step_A) or knot_step_A <= 0 or
            not np.isfinite(clip_sigma) or clip_sigma <= 0):
        raise ValueError("Invalid Kelson fit controls")
    if (not isinstance(n_probes, (int, np.integer)) or not 1 <= n_probes <= 20
            or not np.isfinite(perturbation_sigma) or perturbation_sigma <= 0
            or not np.isfinite(residual_tolerance) or residual_tolerance < 0
            or not np.isfinite(model_tolerance) or model_tolerance < 0):
        raise ValueError("Invalid perturbation and comparison controls")

    ids = np.flatnonzero(selected.ravel())
    if ids.size < min_pixels:
        raise ValueError("Too few selected synthetic sky candidates")
    wave_sample = wave.ravel()[ids]
    y_sample = y_rows[ids // nx]
    var_sample = var.ravel()[ids]
    science_sample = image.ravel()[ids]

    def run_fit(data: np.ndarray) -> dict[str, Any]:
        return _fit_kelson_bspline(
            wave_sample, y_sample, data, var_sample,
            knot_step_A=float(knot_step_A), degree=int(degree),
            y_order=int(y_order), clip_sigma=float(clip_sigma),
            max_iter=int(max_iter), min_pixels=int(min_pixels),
            rng=np.random.default_rng(0),
        )

    baseline = run_fit(science_sample)
    keep = np.asarray(baseline["_cache"]["keep"], dtype=bool)
    if keep.shape != ids.shape or not np.any(keep):
        raise ValueError("Invalid actual baseline post-clip selection")
    final_mask = np.zeros(image.size, dtype=bool)
    final_mask[ids[keep]] = True
    op = frozen_kelson_bspline_operator(
        wave, y_rows,
        knots=np.asarray(baseline["basis"]["t"], dtype=np.float64),
        degree=int(baseline["basis"]["degree"]),
        y_order=int(baseline["y_order"]),
        final_fit_mask=final_mask.reshape(image.shape),
        variance=var,
    )
    if op.fit_indices.size != int(keep.sum()):
        raise ValueError("Frozen WLS selection and actual fit disagree")

    if probe_vectors is None:
        directions = np.random.default_rng(seed).normal(
            size=(int(n_probes), ny, nx),
        )
    else:
        directions = np.asarray(probe_vectors, dtype=np.float64)
        if (directions.shape != (int(n_probes), ny, nx)
                or not np.isfinite(directions).all()
                or np.any(np.linalg.norm(directions.reshape(n_probes, -1), axis=1) == 0)):
            raise ValueError("Explicit perturbations must be nonzero finite full-grid vectors")
    epsilon = float(perturbation_sigma * np.sqrt(np.median(var_sample)))
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Invalid native-signal perturbation amplitude")

    def fitted_model(fit: dict[str, Any]) -> np.ndarray:
        c0 = np.asarray(fit["coeff"]["c0"], dtype=np.float64)
        c1raw = fit["coeff"]["c1"]
        coefficients = c0 if c1raw is None else np.concatenate([
            c0, np.asarray(c1raw, dtype=np.float64)
        ])
        if coefficients.size != op.design.shape[1]:
            raise ValueError("B-spline basis changed between perturbations")
        if not np.array_equal(np.asarray(fit["basis"]["t"]), baseline["basis"]["t"]):
            raise ValueError("B-spline knots changed between perturbations")
        return np.asarray(op.design @ coefficients, dtype=np.float64).ravel()

    records: list[dict[str, Any]] = []
    for probe_index, direction in enumerate(directions):
        v = direction.ravel()
        current: dict[str, Any] = {"probe": int(probe_index), "status": "REFIT_FAILED"}
        try:
            plus = run_fit(science_sample + epsilon * v[ids])
            minus = run_fit(science_sample - epsilon * v[ids])
            plus_keep = np.asarray(plus["_cache"]["keep"], dtype=bool)
            minus_keep = np.asarray(minus["_cache"]["keep"], dtype=bool)
            if plus_keep.shape != keep.shape or minus_keep.shape != keep.shape:
                raise ValueError("Refitted clipping geometry changed")
            nplus = int(np.count_nonzero(plus_keep != keep))
            nminus = int(np.count_nonzero(minus_keep != keep))
            finite_model = (fitted_model(plus) - fitted_model(minus)) / (2 * epsilon)
            ideal_residual = op.apply(v)
            ideal_model = v - ideal_residual
            finite_residual = v - finite_model
            model_error = float(
                np.linalg.norm(finite_model - ideal_model) /
                max(np.linalg.norm(ideal_model), 1e-12)
            )
            residual_error = float(
                np.linalg.norm(finite_residual - ideal_residual) /
                max(np.linalg.norm(ideal_residual), 1e-12)
            )
            lsqr_stops = [
                int(f["solver_diagnostics"]["stop_code"])
                for f in (baseline, plus, minus)
            ]
            numeric_ok = (all(code in (1, 2) for code in lsqr_stops)
                          and model_error <= model_tolerance
                          and residual_error <= residual_tolerance)
            current.update({
                "plus_selection_changes": nplus,
                "minus_selection_changes": nminus,
                "model_derivative_relative_error": model_error,
                "residual_derivative_relative_error": residual_error,
                "lsqr_stop_codes": lsqr_stops,
                "status": (
                    "SELECTION_CHANGED" if nplus or nminus else
                    ("FROZEN_STABLE" if numeric_ok else "NUMERIC_UNSTABLE")
                ),
            })
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            current["error"] = f"{type(exc).__name__}: {exc}"
        records.append(current)

    statuses = {x["status"] for x in records}
    if "REFIT_FAILED" in statuses:
        overall = "REFIT_FAILED"
    elif "SELECTION_CHANGED" in statuses:
        overall = "SELECTION_CHANGED"
    elif "NUMERIC_UNSTABLE" in statuses:
        overall = "NUMERIC_UNSTABLE"
    else:
        overall = "FROZEN_STABLE"

    return {
        "status": overall,
        "scope": "SYNTHETIC_FIXED_GEOMETRY_REFIT_CLIPPING",
        "n_candidates": int(ids.size),
        "n_baseline_kept": int(keep.sum()),
        "n_baseline_rejected": int(ids.size - keep.sum()),
        "epsilon_native_signal": epsilon,
        "n_probes": int(n_probes),
        "baseline_lsqr_stop": int(baseline["solver_diagnostics"]["stop_code"]),
        "probes": records,
        "science_uncertainty_qualified": False,
    }
