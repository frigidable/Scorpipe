"""Conditional matrix-free covariance across frozen Scorpipe linear operators.

P1.4d1 is an opt-in diagnostic, not a production FITS covariance product.
It assumes independent pre-rectification CCD pixels and *fixed* masks,
resampling, y-registration, stacking, extraction, and flux-calibration weights.
Actual Huber/clip, optimal-profile and sky-fitting uncertainties are excluded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from scorpio_pipe.covariance_linearized import covariance_of_linearized_functionals


@dataclass(frozen=True)
class FrozenCovarianceExposure:
    """One independent exposure's explicitly frozen, per-pixel operators.

    ``row_plans`` are COUNT_V1 row plans, one for each input CCD row.
    ``input_variance`` and ``valid_mask`` refer to those original CCD pixels.
    ``stack_weights`` are FINAL, normalized, frozen coefficients on the
    registered output (y, wavelength) grid, not pre-normalization IVARs.
    ``y_shift`` uses Stack2D's convention: output[y + shift] <- input[y].
    ``output_scale`` converts COUNT_V1 integrated outputs into the units
    accepted by the stack (including any exposure-time scaling). It can
    be a positive scalar, per-wavelength vector, or per-input-row matrix.
    The caller is responsible for obtaining coefficients from the exact
    science run and ensuring upstream masks are identical to production.
    """

    row_plans: Sequence[Any]
    input_variance: np.ndarray
    valid_mask: np.ndarray
    stack_weights: np.ndarray
    y_shift: float = 0.0
    output_scale: float | np.ndarray = 1.0


def _validate_exposure(
    exposure: FrozenCovarianceExposure, *, ny_out: int, nlam: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    var = np.asarray(exposure.input_variance, dtype=np.float64)
    valid = np.asarray(exposure.valid_mask)
    weight = np.asarray(exposure.stack_weights, dtype=np.float64)
    if var.ndim != 2 or var.shape[0] == 0 or var.shape[1] == 0:
        raise ValueError("input_variance must be a nonempty 2-D CCD array")
    if valid.shape != var.shape or valid.dtype != np.dtype(bool):
        raise ValueError("valid_mask must be Boolean with input CCD shape")
    if np.any(valid & (~np.isfinite(var) | (var < 0))):
        raise ValueError("Valid CCD variance must be finite and nonnegative")
    if len(exposure.row_plans) != var.shape[0]:
        raise ValueError("One COUNT_V1 row plan is required per input CCD row")
    if weight.shape != (ny_out, nlam) or not np.isfinite(weight).all() or np.any(weight < 0):
        raise ValueError("Frozen stack weights must be finite, nonnegative and (ny_out,nlam)")
    shift = float(exposure.y_shift)
    if not np.isfinite(shift) or abs(shift) > 1e6:
        raise ValueError("Spatial shift must be finite and within integer index range")
    scale = np.asarray(exposure.output_scale, dtype=np.float64)
    if (scale.ndim > 2 or
        (scale.ndim == 1 and scale.shape != (nlam,)) or
        (scale.ndim == 2 and scale.shape != (var.shape[0], nlam)) or
        not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("output_scale must be positive scalar, wavelength vector or CCD-row matrix")
    return var, valid, weight, shift, scale


def _y_adjoint_links(ny_in: int, ny_out: int, shift: float, weight: np.ndarray) -> list[list[tuple[int, float]]]:
    """Sparse transpose of Stack2D's (fixed) linear y interpolator.

    Also refuses nonzero stack weights on geometrically unobserved rows.
    Memory is O(ny_in + ny_out), not O(ny_in * ny_out).
    """
    links: list[list[tuple[int, float]]] = [[] for _ in range(ny_in)]
    for y in range(ny_out):
        x = float(y) - shift
        i0 = int(np.floor(x))
        f = x - i0
        use_second = f > 1e-12
        supported = 0 <= i0 < ny_in and (not use_second or i0 + 1 < ny_in)
        if not supported:
            if np.any(weight[y] != 0.0):
                raise ValueError("Frozen stack weights use spatially uncovered pixels")
            continue
        if use_second:
            links[i0].append((y, 1.0 - f))
            links[i0 + 1].append((y, f))
        else:
            links[i0].append((y, 1.0))
    return links


def covariance_of_frozen_chain_functionals(
    exposures: Sequence[FrozenCovarianceExposure],
    extraction_weights: np.ndarray,
    fluxcal_transfer: np.ndarray,
    left_weights: np.ndarray,
    right_weights: np.ndarray | None = None,
) -> float:
    """Exact conditional covariance of two linear 1D measurements.

    Chain: input CCD -> COUNT_V1 (R,S) -> subpixel registration Y ->
    frozen normalized stack A -> frozen extraction E -> fixed fluxcal D -> u.
    
    ``extraction_weights`` are 0/1 boxcar or frozen *effective* optimal
    coefficients, independently specified for TRACE or FIXED. A caller may
    also compute their cross-covariance by applying this function to distinct
    final functionals only when the extraction map is shared; for different
    maps use ``covariance_of_frozen_chain_functionals_pair``.
    
    The resulting number is conditional detector variance/covariance, NOT
    the full uncertainty of the data-dependent reduction or calibration.
    No standard FITS COV HDU is used (that is coverage, not covariance).
    """
    return covariance_of_frozen_chain_functionals_pair(
        exposures, extraction_weights, extraction_weights,
        fluxcal_transfer, left_weights, right_weights,
    )


def covariance_of_frozen_chain_functionals_pair(
    exposures: Sequence[FrozenCovarianceExposure],
    left_extraction_weights: np.ndarray,
    right_extraction_weights: np.ndarray,
    fluxcal_transfer: np.ndarray,
    left_weights: np.ndarray,
    right_weights: np.ndarray | None = None,
) -> float:
    """Conditional covariance allowing distinct TRACE and FIXED extraction maps."""
    if not exposures:
        raise ValueError("At least one independent exposure is required")
    left_E = np.asarray(left_extraction_weights, dtype=np.float64)
    right_E = np.asarray(right_extraction_weights, dtype=np.float64)
    if left_E.ndim != 2 or left_E.shape != right_E.shape:
        raise ValueError("Extraction coefficients must have matching 2-D shapes")
    ny_out, nlam = left_E.shape
    if not ny_out or not nlam or not np.isfinite(left_E).all() or not np.isfinite(right_E).all():
        raise ValueError("Extraction coefficients must be nonempty and finite")
    transfer = np.asarray(fluxcal_transfer, dtype=np.float64)
    u = np.asarray(left_weights, dtype=np.float64)
    v = np.asarray(right_weights if right_weights is not None else left_weights, dtype=np.float64)
    if (transfer.shape != (nlam,) or u.shape != (nlam,) or v.shape != (nlam,)
            or not np.isfinite(transfer).all() or np.any(transfer <= 0)
            or not np.isfinite(u).all() or not np.isfinite(v).all()):
        raise ValueError("Functional and calibration vectors must be finite wavelength arrays; calibration > 0")
    validated = [_validate_exposure(e, ny_out=ny_out, nlam=nlam) for e in exposures]
    # The stack is a weighted mean. Missing spectral bins must not be
    # mistaken for noise-free measurements with a formal variance of zero.
    # Track support separately for the left and right extraction maps.
    left_supported = np.zeros(nlam, dtype=bool)
    right_supported = np.zeros(nlam, dtype=bool)
    # Validate without allocating another full (ny_out,nlam) image.
    for y in range(ny_out):
        wsum = np.zeros(nlam, dtype=np.float64)
        for _var, _valid, w, _s, _scale in validated:
            wsum += w[y]
        present = wsum > 0
        left_supported |= present & (left_E[y] != 0.0)
        right_supported |= present & (right_E[y] != 0.0)
        if np.any(~np.isclose(wsum[present], 1.0, rtol=1e-7, atol=1e-9)):
            raise ValueError("Frozen stack coefficients must sum to one on supported pixels")
    if np.any((u != 0.0) & ~left_supported) or np.any((v != 0.0) & ~right_supported):
        raise ValueError("Requested functional uses wavelength bins without extraction support")

    left_post = transfer * u
    right_post = transfer * v
    total = 0.0
    for exposure, (var, valid, stack, shift, scale) in zip(exposures, validated):
        links = _y_adjoint_links(var.shape[0], ny_out, shift, stack)
        for j, rows in enumerate(links):
            if not rows:
                continue
            p = np.zeros(nlam, dtype=np.float64)
            q = np.zeros(nlam, dtype=np.float64)
            for y, coef in rows:
                p += coef * stack[y] * left_E[y] * left_post
                q += coef * stack[y] * right_E[y] * right_post
            if not np.any(p) and not np.any(q):
                continue
            plan = exposure.row_plans[j]
            if plan is None:
                raise ValueError("Missing COUNT_V1 row plan under nonzero measurement weights")
            if np.asarray(plan.counts).shape != (nlam,):
                raise ValueError("COUNT_V1 row plan/output wavelength grid mismatch")
            row_scale = scale[j] if scale.ndim == 2 else scale
            total += covariance_of_linearized_functionals(
                plan, var[j], valid[j], p, q, output_scale=row_scale
            )
    if not np.isfinite(total):
        raise ValueError("Computed covariance is not finite")
    return float(total)
