"""Diagnostic line-window flux with conditional noise covariance (P1.4c).

This is deliberately separate from the production median-continuum line
estimator. It constructs an exactly linear estimator, *conditional on fixed
sideband inverse-variance weights and a fixed wavelength grid*, and exposes
its full linear functional so an externally qualified covariance operator
can propagate the noise. It makes no claims about post-stack covariance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from scorpio_pipe.science.lines import LineSpec

CovarianceOperator = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class LineFluxCovarianceDiagnostic:
    """Diagnostic result; does not replace the science line catalogue."""

    flux: float
    continuum_at_center: float
    error_diag: float
    error_conditional: float | None
    covariance_status: str
    continuum_order: int
    wavelength_coverage: float
    line_pixel_count: int
    sideband_pixel_count: int
    # Includes both line integration and continuum estimation.
    functional_weights: np.ndarray


def _bin_edges_from_centers(wave: np.ndarray) -> np.ndarray:
    """Center-based pixel-edge convention matched to COUNT_V1."""
    if wave.ndim != 1 or wave.size < 3 or not np.isfinite(wave).all():
        raise ValueError("Wavelength grid must have at least 3 finite centers")
    if np.any(np.diff(wave) <= 0.0):
        raise ValueError("Wavelength grid must be strictly increasing")
    edges = np.empty(wave.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (wave[:-1] + wave[1:])
    edges[0] = wave[0] - (edges[1] - wave[0])
    edges[-1] = wave[-1] + (wave[-1] - edges[-2])
    return edges


def measure_line_flux_linear_continuum(
    wave: np.ndarray,
    flux_density: np.ndarray,
    variance_density: np.ndarray,
    *,
    spec: LineSpec,
    redshift: float = 0.0,
    valid_mask: np.ndarray | None = None,
    continuum_order: int = 1,
    covariance_operator: CovarianceOperator | None = None,
) -> LineFluxCovarianceDiagnostic:
    """Return line-window integrated flux with full continuum-fit functional.

    Input flux is density per wavelength, with variance in squared density
    units. Integration weights are exact pixel-bin overlaps. Sideband samples
    define a fixed-weight diagonal-WLS continuum estimator of degree 0 or 1.
    The resulting *total* linear functional propagates both noise in the
    line window and uncertainty of the fitted continuum.

    If provided, covariance_operator(u, v) returns u.T @ C @ v for the same
    output spectrum in the SAME flux-density units. It can be backed by
    P1.4b for a single raw-rectified row, but not directly for a stacked 1D
    spectrum whose upstream transformation has not been reconstructed.

    Conditional covariance excludes sky fitting, flux calibration, cosmic
    inpainting, trace/profile estimation and adaptive stacking uncertainty.
    """
    wave = np.asarray(wave, dtype=np.float64)
    flux = np.asarray(flux_density, dtype=np.float64)
    var = np.asarray(variance_density, dtype=np.float64)
    if wave.ndim != 1 or flux.shape != wave.shape or var.shape != wave.shape:
        raise ValueError("Wavelength, flux density and variance must be matching 1D vectors")
    edges = _bin_edges_from_centers(wave)
    if valid_mask is None:
        valid = np.ones(wave.size, dtype=bool)
    else:
        valid = np.asarray(valid_mask)
        if valid.shape != wave.shape or valid.dtype != np.dtype(bool):
            raise ValueError("valid_mask must be a Boolean vector matching wavelength")
    if continuum_order not in (0, 1):
        raise ValueError("continuum_order must be zero or one")
    if not np.isfinite(redshift) or redshift <= -1.0:
        raise ValueError("redshift must be finite and greater than -1")
    if not (np.isfinite(spec.window_half_width) and spec.window_half_width > 0):
        raise ValueError("Line half-width must be finite and positive")

    center = float(spec.rest_wavelength) * (1.0 + float(redshift))
    half = float(spec.window_half_width)
    lo, hi = center - half, center + half
    if lo < edges[0] or hi > edges[-1]:
        raise ValueError("Line window extends beyond the wavelength grid")

    a = np.maximum(0.0, np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo))
    in_line = a > 0.0
    if np.count_nonzero(in_line) < 3:
        raise ValueError("Insufficient line-window sampling")
    if not np.isclose(a.sum(), hi - lo, rtol=1e-12, atol=1e-10):
        raise ValueError("Line-window wavelength coverage is incomplete")
    if np.any(in_line & (~valid | ~np.isfinite(flux) | ~np.isfinite(var) | (var < 0))):
        raise ValueError("Invalid or masked pixel inside line window")

    (left_a, left_b), (right_a, right_b) = spec.continuum_left, spec.continuum_right
    if not (left_a < left_b < -half < half < right_a < right_b):
        raise ValueError("Continuum sidebands must be ordered and separated from the line")
    left = (wave >= center + left_a) & (wave <= center + left_b)
    right = (wave >= center + right_a) & (wave <= center + right_b)
    side_good = valid & np.isfinite(flux) & np.isfinite(var) & (var > 0)
    sb_left = left & side_good
    sb_right = right & side_good
    if np.count_nonzero(sb_left) < 2 or np.count_nonzero(sb_right) < 2:
        raise ValueError("Two valid continuum samples required in each sideband")
    sb = sb_left | sb_right
    x = wave - center
    sb_x = x[sb]
    X = (np.ones((sb_x.size, 1), dtype=float) if continuum_order == 0
         else np.column_stack([np.ones(sb_x.size), sb_x]))

    # Diagonal WLS determines the linear continuum operator. These weights
    # are frozen; a supplied covariance describes its *actual* uncertainty,
    # but does not retroactively make the WLS fit a GLS estimator.
    sqrt_w = 1.0 / np.sqrt(var[sb])
    WX = sqrt_w[:, None] * X
    if np.linalg.matrix_rank(WX) < X.shape[1]:
        raise ValueError("Continuum design matrix has deficient rank")
    H = np.linalg.pinv(WX) * sqrt_w[None, :]
    beta = H @ flux[sb]
    k = np.array([a.sum(), np.dot(a, x)])[:continuum_order + 1]
    functional = a.copy()
    functional[sb] -= k @ H

    active = functional != 0.0
    if np.any(active & (~valid | ~np.isfinite(flux) | ~np.isfinite(var) | (var < 0))):
        raise ValueError("Invalid input under the line/continuum functional")
    # Unused masked bins may contain NaN. Never contract them (0*NaN is NaN).
    line_flux = float(np.dot(functional[active], flux[active]))
    diag_variance = float(np.dot(functional[active] ** 2, var[active]))
    if not np.isfinite(diag_variance) or diag_variance < 0:
        raise ValueError("Cannot compute finite nonnegative diagonal diagnostic variance")

    cov_error: float | None = None
    if covariance_operator is not None:
        cov_var = float(covariance_operator(functional.copy(), functional.copy()))
        if not np.isfinite(cov_var) or cov_var < -1e-10 * max(1.0, diag_variance):
            raise ValueError("Covariance operator returned invalid variance")
        cov_error = float(np.sqrt(max(0.0, cov_var)))
    return LineFluxCovarianceDiagnostic(
        flux=line_flux,
        continuum_at_center=float(beta[0]),
        error_diag=float(np.sqrt(diag_variance)),
        error_conditional=cov_error,
        covariance_status=("CONDITIONAL_USER_OPERATOR" if cov_error is not None else "DIAG_ONLY"),
        continuum_order=int(continuum_order),
        wavelength_coverage=float(a.sum() / (hi - lo)),
        line_pixel_count=int(np.count_nonzero(in_line)),
        sideband_pixel_count=int(np.count_nonzero(sb)),
        functional_weights=functional,
    )
