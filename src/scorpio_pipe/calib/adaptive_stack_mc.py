"""Synthetic-only Stack2D adaptive Huber weighting and shared-bias coverage.

Calls the actual production streaming stacker on fresh per-exposure
noise realizations, contrasting conditional fixed-weight output VAR with
frequentist sampling covariance. No real data and no science FITS I/O.

A shared MasterBias random field is applied ONCE per independent draw,
then reused for every exposure belonging to that explicit group.
The per-frame VAR includes that calibration variance. Stack2D's
sum(w**2 * VAR)/sum(w)**2 does not by itself restore cross-exposure
MasterBias covariance.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from math import sqrt
from typing import Any

import numpy as np
from astropy.io import fits

from scorpio_pipe.stages.stack2d import (
    _stack_chunk_streaming,
)


def synthetic_adaptive_stack2d_coverage(
    *,
    group_ids: Sequence[str] = ("A", "A", "A"),
    method: str = "invvar_huber",
    n_realizations: int = 600,
    seed: int = 20261104,
    huber_c: float = 1.0,
    robust_iter: int = 2,
    true_flux: float = 0.,
) -> dict[str, Any]:
    """Test data-derived Stack2D weights and cross-exposure covariance.

    Three 1x2 synthetic rectified images, each with independent N(0,1)
    noise and signed response +1 to its group's N(0,1) MasterBias field,
    have input variance 2 per pixel. Every draw uses new independent
    noise AND new MasterBias fields, shared only within named groups.

    Real `_stack_chunk_streaming` executes inverse-variance or adaptive
    Huber iteration independently on every realization. Fixed spatial
    shifts, scaling, masks, per-pixel variance and input exposure count
    let us isolate adaptive stacking and grouped calibration covariance.

    For plain inverse variance (all equal weights), the exact model
    variance is 1/3 + sum_group (n_group/3)**2. The published
    fixed-weight independent-input variance is always 2/3; the two
    models agree ONLY for three distinct bias groups. For Huber weights,
    the same fixed-weight reference is NOT treated as an exact sampling
    law because weights are random and estimated from noisy SCI.
    """
    labels = tuple(group_ids)
    if (len(labels) != 3 or
            any(not isinstance(v, str) or not v or v.strip() != v
                for v in labels)):
        raise ValueError("Exactly three explicit nonblank MasterBias groups are required")
    if method not in ("invvar_huber", "invvar"):
        raise ValueError("Synthetic method must be invvar_huber or invvar")
    if (isinstance(n_realizations, (bool, np.bool_))
            or not isinstance(n_realizations, (int, np.integer))
            or not 200 <= int(n_realizations) <= 2000):
        raise ValueError("n_realizations must be an integer in [200,2000]")
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    if (isinstance(robust_iter, (bool, np.bool_))
            or not isinstance(robust_iter, (int, np.integer))
            or not 0 <= int(robust_iter) <= 5):
        raise ValueError("robust_iter must be an integer in [0,5]")
    if (isinstance(huber_c, (bool, np.bool_))
            or not isinstance(huber_c, (int, float, np.integer, np.floating))
            or not np.isfinite(huber_c) or not 0.1 <= huber_c <= 20.):
        raise ValueError("huber_c must be finite and in [0.1,20]")
    if (isinstance(true_flux, (bool, np.bool_))
            or not isinstance(true_flux, (int, float, np.integer, np.floating))
            or not np.isfinite(true_flux) or abs(true_flux) > 1000.):
        raise ValueError("Synthetic true_flux must be finite and bounded")

    n = int(n_realizations)
    rng = np.random.default_rng(int(seed))
    group_map = {
        g: rng.standard_normal((n, 2)) for g in sorted(set(labels))
    }
    independent = rng.standard_normal((n, 3, 2))
    data = np.empty((n, 3, 2), dtype=np.float64)
    for e, gid in enumerate(labels):
        data[:, e] = (float(true_flux) + independent[:, e]
                       + group_map[gid])

    hduls: list[fits.HDUList] = []
    for _e in range(3):
        hduls.append(fits.HDUList([
            fits.PrimaryHDU(data=np.zeros((1, 2), dtype=np.float32)),
            fits.ImageHDU(data=np.full((1, 2), 2., dtype=np.float32),
                          name="VAR"),
            fits.ImageHDU(data=np.zeros((1, 2), dtype=np.uint16),
                          name="MASK"),
        ]))

    fluxes = np.empty((n, 2), dtype=np.float64)
    reported_variances = np.empty((n, 2), dtype=np.float64)
    suppression_fractions = np.empty(n, dtype=np.float64)
    try:
        for k in range(n):
            for e, hdul in enumerate(hduls):
                hdul[0].data[0] = data[k, e].astype(np.float32)
            combined, variance, flags, coverage, suppressed, total = (
                _stack_chunk_streaming(
                    hduls=hduls, y0=0, y1=1,
                    method=method, robust_iter=int(robust_iter),
                    huber_c=float(huber_c), clip_sigma=4.,
                    y_shifts=[0., 0., 0.],
                    y_align_enabled=False,
                    sci_scales=[1., 1., 1.],
                    var_scales=[1., 1., 1.],
                    fatal_bits=np.uint16(0), var_floor=None,
                )
            )
            if (combined.shape != (1, 2) or variance.shape != (1, 2)
                    or not np.isfinite(combined).all()
                    or not np.isfinite(variance).all()
                    or np.any(variance <= 0) or not np.all(coverage == 3)
                    or total != 2):
                raise RuntimeError("Production Stack2D lost synthetic coverage")
            fluxes[k] = combined[0]
            reported_variances[k] = variance[0]
            suppression_fractions[k] = suppressed/total
    finally:
        for hdul in hduls:
            hdul.close()

    residuals = fluxes-float(true_flux)
    empirical = np.cov(residuals, rowvar=False, ddof=1)
    predicted_mean_diag = np.mean(reported_variances, axis=0)
    ratios = np.diag(empirical)/predicted_mean_diag
    standardized = residuals/np.sqrt(reported_variances)
    coverage95 = np.mean(np.abs(standardized) <= 1.959963984540054, axis=0)
    coverage68 = np.mean(np.abs(standardized) <= 1., axis=0)
    mean_z = np.mean(standardized, axis=0)
    counts = Counter(labels)

    # Exact theoretical reference for the UNWEIGHTED independent exposures
    # with known shared calibration grouping. This is a positive control
    # only for the plain inverse-variance method, never for adaptive Huber.
    true_fixed_invvar = (1./3. + sum(
        (amount/3.)**2 for amount in counts.values()
    ))
    conditional_independent_invvar = 2./3.
    tol_var = max(.10, 5.*sqrt(2./(n-1)))
    tol_cov = max(.03, 5.*sqrt(.95*.05/n))
    tol_mean = 5./sqrt(n)
    consistent = bool(
        np.max(np.abs(ratios-1.)) <= tol_var
        and np.max(np.abs(coverage95-.95)) <= tol_cov
        and np.max(np.abs(mean_z)) <= tol_mean
    )
    undercoverage = bool(
        np.max(ratios) > 1. + tol_var
        and np.min(coverage95) < .95-tol_cov
    )
    if consistent:
        status = "MC_CONSISTENT"
    elif undercoverage:
        status = "CONDITIONAL_VAR_UNDERCOVERAGE"
    else:
        status = "MC_DISCREPANCY_OR_INCONCLUSIVE"

    return {
        "scope": "SYNTHETIC_PRODUCTION_ADAPTIVE_STACK2D_SHARED_BIAS_MC",
        "status": status,
        "group_ids": list(labels),
        "method": method,
        "robust_iter": int(robust_iter),
        "huber_c": float(huber_c),
        "true_flux": float(true_flux),
        "n_realizations": n,
        "per_exposure_independent_variance": 1.,
        "per_group_masterbias_variance": 1.,
        "published_input_variance": 2.,
        "empirical_covariance": empirical.tolist(),
        "mean_reported_conditional_variance": predicted_mean_diag.tolist(),
        "empirical_to_conditional_variance_ratios": ratios.tolist(),
        "mean_standardized_error": mean_z.tolist(),
        "coverage_1sigma": coverage68.tolist(),
        "coverage_95percent": coverage95.tolist(),
        "mean_fraction_of_pixels_with_robust_suppression": float(
            np.mean(suppression_fractions)
        ),
        "known_fixed_invvar_grouped_variance": float(true_fixed_invvar),
        "known_fixed_invvar_independent_only_variance": float(
            conditional_independent_invvar
        ),
        "statistically_consistent_with_published_conditional_var": consistent,
        "undercoverage_detected": undercoverage,
        "tolerances": {
            "variance_ratio": tol_var,
            "coverage_95percent": tol_cov,
            "mean_standardized_error": tol_mean,
        },
        "science_uncertainty_qualified": False,
        "production_variance_release_allowed": False,
    }
