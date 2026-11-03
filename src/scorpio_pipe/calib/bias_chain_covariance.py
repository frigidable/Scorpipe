"""Opt-in shared-MasterBias covariance through the FROZEN linear chain.

Not used by production Stack2D. Derivatives are projected through the
same wavelength/registration/extraction/fluxcal operators as P1.4d1.

The input noise variance MUST EXCLUDE shared MasterBias uncertainty.
The sensitivity maps must be supplied from the actual calibration
operators; post-FlatField combined VAR cannot recover them.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np

from scorpio_pipe.covariance_chain import (
    FrozenCovarianceExposure,
    _validate_exposure,
    _y_adjoint_links,
    covariance_of_frozen_chain_functionals_pair,
)
from scorpio_pipe.covariance_linearized import _project_linearized_functional


def covariance_of_frozen_chain_with_shared_bias(
    exposures: Sequence[FrozenCovarianceExposure],
    group_ids: Sequence[str],
    masterbias_variances: Mapping[str, np.ndarray],
    bias_sensitivities: Sequence[np.ndarray],
    left_extraction_weights: np.ndarray,
    right_extraction_weights: np.ndarray,
    fluxcal_transfer: np.ndarray,
    left_weights: np.ndarray,
    right_weights: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Independent, shared-MasterBias, and total conditional covariance.

    Treat each calibration group as a common *pixelwise* random field with
    diagonal native-detector covariance and an independent realization from
    every other group. All science operators and weights are held fixed.
    Bias sensitivity for each exposure is the explicitly supplied derivative
    of its pre-rectified science image w.r.t. the group's bias estimator
    in matched units (typically negative with a flat/gain response).

    The shared component contracts each group's SUM of exposure derivatives
    against its one calibration covariance map, never sums its diagonal
    contribution independently per exposure.
    """
    if not exposures:
        raise ValueError("At least one frozen exposure is required")
    if len(group_ids) != len(exposures) or len(bias_sensitivities) != len(exposures):
        raise ValueError("One bias group and sensitivity map per exposure are required")
    ids = [str(g).strip() for g in group_ids]
    if not all(ids):
        raise ValueError("MasterBias group identifiers must be nonempty")
    if set(masterbias_variances) != set(ids):
        raise ValueError("MasterBias variance map keys must match all group identifiers")
    le = np.asarray(left_extraction_weights, dtype=np.float64)
    re = np.asarray(right_extraction_weights, dtype=np.float64)
    if le.ndim != 2 or le.shape != re.shape:
        raise ValueError("Both extraction maps must share nonempty 2-D shape")
    ny_out, nlam = le.shape
    if ny_out == 0 or nlam == 0:
        raise ValueError("Extraction maps cannot be empty")
    transfer = np.asarray(fluxcal_transfer, dtype=np.float64)
    u = np.asarray(left_weights, dtype=np.float64)
    v = np.asarray(right_weights if right_weights is not None else left_weights, dtype=np.float64)
    if (transfer.shape != (nlam,) or u.shape != (nlam,) or v.shape != (nlam,)):
        raise ValueError("Flux calibration and both functionals must match the wavelength grid")
    # The existing P1.4d1 contracts validate normalization, gaps, positive
    # transfer, and all frozen geometry before the shared-calibration oracle.
    independent = covariance_of_frozen_chain_functionals_pair(
        exposures, le, re, transfer, u, v,
    )
    group_derivatives: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for exposure, gid, sensitivity in zip(exposures, ids, bias_sensitivities):
        var, valid, weight, shift, scale = _validate_exposure(
            exposure, ny_out=ny_out, nlam=nlam
        )
        derivative = np.asarray(sensitivity, dtype=np.float64)
        if derivative.shape != var.shape or not np.isfinite(derivative).all():
            raise ValueError("MasterBias sensitivity must be finite and match CCD geometry")
        if gid not in group_derivatives:
            group_derivatives[gid] = (
                np.zeros(var.shape, dtype=np.float64),
                np.zeros(var.shape, dtype=np.float64),
            )
        gl, gr = group_derivatives[gid]
        if gl.shape != var.shape:
            raise ValueError("Shared MasterBias group requires common CCD pixel geometry")
        links = _y_adjoint_links(var.shape[0], ny_out, shift, weight)
        for j, rows in enumerate(links):
            if not rows:
                continue
            p = np.zeros(nlam, dtype=np.float64)
            q = np.zeros(nlam, dtype=np.float64)
            for y, coef in rows:
                p += coef * weight[y] * le[y] * transfer * u
                q += coef * weight[y] * re[y] * transfer * v
            if not np.any(p) and not np.any(q):
                continue
            plan = exposure.row_plans[j]
            if plan is None or np.asarray(plan.counts).shape != (nlam,):
                raise ValueError("Missing or incompatible COUNT_V1 row plan")
            row_scale = scale[j] if scale.ndim == 2 else scale
            a, _ = _project_linearized_functional(
                plan, var[j], valid[j], p, row_scale
            )
            b, _ = _project_linearized_functional(
                plan, var[j], valid[j], q, row_scale
            )
            # P1.4b returns coefficients on finite_idx. Expand them back to
            # native CCD coordinates BEFORE combining repeated exposures.
            idx = np.asarray(plan.finite_idx, dtype=np.intp)
            gl[j, idx] += derivative[j, idx] * a
            gr[j, idx] += derivative[j, idx] * b
    common = 0.0
    for gid, (gl, gr) in group_derivatives.items():
        bias_var = np.asarray(masterbias_variances[gid], dtype=np.float64)
        if bias_var.shape != gl.shape:
            raise ValueError(f"MasterBias variance for {gid!r} has incorrect CCD geometry")
        if not np.isfinite(bias_var).all() or np.any(bias_var < 0):
            raise ValueError(f"MasterBias variance for {gid!r} must be finite nonnegative")
        common += float(np.sum(bias_var * gl * gr, dtype=np.float64))
    result = float(independent + common)
    if not np.isfinite(common) or not np.isfinite(result):
        raise ValueError("Nonfinite shared MasterBias covariance")
    return float(independent), float(common), result
