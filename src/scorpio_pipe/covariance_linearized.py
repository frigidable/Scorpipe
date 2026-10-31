"""On-demand covariance of linearized spectral measurements (P1.4b).

This module uses the exact COUNT_V1 overlap coefficients already stored in a
linearize row plan.  It computes a *conditional, input-diagonal* covariance
without assembling the huge output-bin covariance matrix.

Excluded: input pixel correlations, uncertainty of wavelength solutions,
sky-model fitting, cosmic inpainting, optimal-extraction profile estimation,
adaptive Huber/clip weights, and y registration.  This module does not alter
the production SCI/VAR FITS contract.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _project_linearized_functional(
    row_plan: Any,
    input_variance: np.ndarray,
    valid_mask: np.ndarray,
    output_weights: np.ndarray,
    output_scale: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a linear functional to input CCD pixels with strict validation.

    For output measurement F = sum_k u_k S_k (sum_i R_ki x_i), return
    g_i = sum_k u_k S_k R_ki so Cov(F,G) = sum_i g_i h_i Var(x_i).
    """
    var = np.asarray(input_variance, dtype=np.float64)
    valid = np.asarray(valid_mask)
    if var.ndim != 1 or valid.shape != var.shape or valid.dtype != np.dtype(bool):
        raise ValueError("Input variance and Boolean valid_mask must be matching 1D arrays")

    finite_idx = np.asarray(row_plan.finite_idx)
    src_idx = np.asarray(row_plan.src_idx)
    starts = np.asarray(row_plan.starts)
    counts = np.asarray(row_plan.counts)
    overlap_weights = np.asarray(row_plan.weights, dtype=np.float64)

    for name, indices in (
        ("finite_idx", finite_idx), ("src_idx", src_idx),
        ("starts", starts), ("counts", counts),
    ):
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"Invalid COUNT_V1 row plan: {name} is not 1D integer")
    if (np.any(finite_idx < 0) or np.any(finite_idx >= var.size)
            or np.unique(finite_idx).size != finite_idx.size):
        raise ValueError("Invalid COUNT_V1 row plan: finite_idx is out of range or duplicated")
    if counts.size == 0:
        raise ValueError("Invalid COUNT_V1 row plan: empty output grid")
    if (np.any(counts < 0) or np.any(starts < 0)
            or not np.array_equal(
                starts, np.r_[0, np.cumsum(counts[:-1], dtype=np.int64)]
            )):
        raise ValueError("Invalid COUNT_V1 row plan: noncontiguous output-bin offsets")
    if (src_idx.size != int(np.sum(counts, dtype=np.int64))
            or overlap_weights.shape != src_idx.shape
            or np.any(src_idx < 0) or np.any(src_idx >= finite_idx.size)):
        raise ValueError("Invalid COUNT_V1 row plan: invalid overlap indices")
    if (not np.isfinite(overlap_weights).all()
            or np.any(overlap_weights < 0.0)
            or np.any(overlap_weights > 1.0 + 1e-10)):
        raise ValueError("Invalid COUNT_V1 row plan: invalid input-overlap fractions")

    u = np.asarray(output_weights, dtype=np.float64)
    if u.shape != counts.shape or not np.isfinite(u).all():
        raise ValueError("Output functional weights must be finite and match output bins")
    scale = np.asarray(output_scale, dtype=np.float64)
    if (scale.ndim > 1 or (scale.ndim == 1 and scale.shape != counts.shape)
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("Output scale must be finite and positive, scalar or per-bin")

    # Invalid pixels may have nonfinite VAR, but are excluded rather than
    # silently repaired.  Every contributing valid source needs honest VAR.
    v = var[finite_idx]
    good = valid[finite_idx]
    if np.any(good & (~np.isfinite(v) | (v < 0))):
        raise ValueError("Valid CCD inputs must have finite, nonnegative VAR")

    bin_ids = np.repeat(np.arange(counts.size, dtype=np.intp), counts)
    per_overlap = overlap_weights * (u * scale)[bin_ids]
    # np.bincount accumulates repeated contributions from the same input CCD
    # sample across adjacent output bins; this is precisely the covariance.
    g = np.bincount(
        src_idx, weights=per_overlap, minlength=finite_idx.size
    ).astype(np.float64, copy=False)
    g[~good] = 0.0
    return g, np.where(good, v, 0.0)


def covariance_of_linearized_functionals(
    row_plan: Any,
    input_variance: np.ndarray,
    valid_mask: np.ndarray,
    left_weights: np.ndarray,
    right_weights: np.ndarray | None = None,
    *,
    output_scale: float | np.ndarray = 1.0,
) -> float:
    """Exact conditional covariance for two linearized 1D row measurements.

    `right_weights=None` returns the variance of `left_weights`. The caller
    must use the *same* validity mask as the production SCI/VAR rectification
    and explicitly supply density/exposure scaling when applicable.

    This contract is valid when the pre-resampling noise is diagonal and the
    overlap plan/masks/scaling are treated as fixed. It is not the complete
    instrumental or post-stack covariance.
    """
    g, v = _project_linearized_functional(
        row_plan, input_variance, valid_mask, left_weights, output_scale
    )
    if right_weights is None:
        h = g
    else:
        h, _ = _project_linearized_functional(
            row_plan, input_variance, valid_mask, right_weights, output_scale
        )
    return float(np.dot(v, g * h))
