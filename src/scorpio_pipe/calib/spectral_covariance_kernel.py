"""Small frozen SkySub-to-1D spectral covariance kernel (synthetic QA).

A diagonal 1-D VAR cannot represent spectral-channel correlations caused by
COUNT_V1 overlaps, sky-model fitting, and shared MasterBias. This bounded
diagnostic evaluates the existing conditional matrix-free adjoint for pairs
of selected 1-D bins, without materializing native CCD covariance matrices.

The caller must supply the exact known frozen noise/geometry provenance.
The numerical matrix is NOT an observationally calibrated SCI/VAR product.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from scorpio_pipe.calib.sky_bias_covariance import FrozenSkyWLS
from scorpio_pipe.calib.sky_chain_grouped_covariance import (
    covariance_of_frozen_sky_chain_grouped_pair,
)
from scorpio_pipe.covariance_chain import FrozenCovarianceExposure


def frozen_sky_1d_spectral_covariance_kernel(
    *,
    exposures: Sequence[FrozenCovarianceExposure],
    sky_operators: Sequence[FrozenSkyWLS],
    extraction_weights: np.ndarray,
    fluxcal_transfer: np.ndarray,
    independent_variances: Sequence[np.ndarray],
    bias_sensitivities: Sequence[np.ndarray],
    group_ids: Sequence[str],
    masterbias_variances: Mapping[str, np.ndarray],
    wavelength_indices: Sequence[int],
    integrated_line_weights: Sequence[float],
) -> dict[str, Any]:
    """Return conditional covariance and integrated-line variance.

    For line weights a_k, the integrated spectral observable is
    L = sum_k a_k F_k and Var(L) = a.T @ C @ a.
    The naive diagonal-only shortcut sum_k a_k**2 C_kk is reported
    separately. Off-diagonal values can have EITHER SIGN; thus the
    diagonal-only shortcut can either under- or over-estimate Var(L).

    The smallest supported kernel has two channels; up to twelve
    ordered distinct channels are allowed to bound quadratic work.
    Each element uses the *same* frozen model, extraction map,
    wavelength solution, calibration transfer, and grouping as its peers.
    """
    E = np.asarray(extraction_weights, dtype=np.float64)
    if E.ndim != 2 or min(E.shape) == 0 or not np.isfinite(E).all():
        raise ValueError("Finite nonempty 2D extraction map is required")
    nlam = E.shape[1]
    try:
        raw_idx = tuple(wavelength_indices)
        raw_weight = tuple(integrated_line_weights)
    except TypeError as exc:
        raise ValueError("Spectral indices and integration weights must be sequences") from exc

    if (not 2 <= len(raw_idx) <= 12
            or len(raw_weight) != len(raw_idx)
            or any(isinstance(x, (bool, np.bool_))
                   or not isinstance(x, (int, np.integer)) for x in raw_idx)):
        raise ValueError("Need 2-12 distinct integer bin indices and matching line weights")
    bins = np.asarray(raw_idx, dtype=np.int64)
    if (np.any(bins < 0) or np.any(bins >= nlam)
            or np.any(np.diff(bins) <= 0)):
        raise ValueError("Indices must be strictly increasing in the extraction wavelength grid")
    if any(isinstance(x, (bool, np.bool_))
           or not isinstance(x, (int, float, np.integer, np.floating))
           for x in raw_weight):
        raise ValueError("Integration weights must be numeric, not bool")
    weights = np.asarray(raw_weight, dtype=np.float64)
    if (not np.isfinite(weights).all()
            or not np.any(weights != 0)):
        raise ValueError("Nonzero finite integration weights are required")
    m = bins.size
    channels = np.zeros((m, nlam), dtype=np.float64)
    channels[np.arange(m), bins] = 1.

    cov_ind = np.zeros((m, m), dtype=np.float64)
    cov_bias = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(i, m):
            ind, shared, total = covariance_of_frozen_sky_chain_grouped_pair(
                exposures=exposures, sky_operators=sky_operators,
                left_extraction_weights=E,
                right_extraction_weights=E,
                fluxcal_transfer=fluxcal_transfer,
                left_weights=channels[i],
                right_weights=channels[j],
                independent_variances=independent_variances,
                bias_sensitivities=bias_sensitivities,
                group_ids=group_ids,
                masterbias_variances=masterbias_variances,
            )
            if not np.isfinite([ind, shared, total]).all():
                raise ValueError("Nonfinite frozen sky covariance term")
            cov_ind[i, j] = cov_ind[j, i] = ind
            cov_bias[i, j] = cov_bias[j, i] = shared
    total = cov_ind + cov_bias
    diag = np.diag(total)
    maximum = max(float(np.max(diag)), 1e-20)
    for name, matrix in (
        ("Independent", cov_ind),
        ("Shared MasterBias", cov_bias),
        ("Combined", total),
    ):
        if (not np.isfinite(matrix).all()
                or np.min(np.diag(matrix)) < -1e-12*maximum
                or np.min(np.linalg.eigvalsh(matrix)) < -1e-9*maximum):
            raise ValueError(f"{name} covariance is numerically non-PSD")
    if np.any(diag <= 0):
        raise ValueError("All requested spectral-bin variances must be positive")

    variance_ind = float(weights @ cov_ind @ weights)
    variance_bias = float(weights @ cov_bias @ weights)
    variance_total = float(weights @ total @ weights)
    diagonal_only = float(np.dot(weights**2, diag))
    if variance_total <= 0 or diagonal_only <= 0:
        raise ValueError("Integrated-line variance must be strictly positive")
    factor = variance_total / diagonal_only

    return {
        "status": "CONDITIONAL_SPECTRAL_KERNEL_ONLY",
        "scope": "FROZEN_SKYSUB_COUNT_V1_STACK_EXTRACT_FLUXCAL_GROUPED",
        "wavelength_bin_indices": bins.astype(int).tolist(),
        "integrated_line_weights": weights.tolist(),
        "independent_covariance": cov_ind.tolist(),
        "shared_masterbias_covariance": cov_bias.tolist(),
        "total_covariance": total.tolist(),
        "integrated_line_variance_independent": variance_ind,
        "integrated_line_variance_masterbias": variance_bias,
        "integrated_line_variance_full": variance_total,
        "integrated_line_variance_diagonal_only": diagonal_only,
        "line_variance_ratio_full_to_diagonal": float(factor),
        "line_sigma_ratio_full_to_diagonal": float(np.sqrt(factor)),
        "offdiagonal_covariance_sum": float(
            variance_total-diagonal_only
        ),
        "has_nonzero_interchannel_covariance": bool(
            np.max(np.abs(total-np.diag(diag))) > 1e-12*maximum
        ),
        "selection_masks_and_downstream_weights_frozen": True,
        "science_uncertainty_qualified": False,
        "production_variance_release_allowed": False,
    }
