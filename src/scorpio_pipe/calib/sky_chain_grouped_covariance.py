"""Conditional grouped covariance across SkySub -> Linearize -> Stack2D -> Extract1D.

Diagnostic only. Every operator, selection, weight and group is frozen.
This does not modify FITS VAR, SCI, MASK or qualify scientific uncertainties.

For exposure e, let P_e be the adjoint of the fixed post-sky chain and
R_e be the frozen SkySub residual operator. Then the measurement gradient
on the pre-sky CCD is g_e = R_e.T @ P_e. Independent detector variance is
sum_e g_e.T diag(V_e) h_e. For a MasterBias group g, shared covariance is
(sum_e s_e*g_e).T diag(B_g) (sum_e s_e*h_e), summing only exposures
with that explicit group identity.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from scorpio_pipe.calib.sky_bias_covariance import FrozenSkyWLS
from scorpio_pipe.covariance_chain import (
    FrozenCovarianceExposure,
    _validate_exposure,
    _y_adjoint_links,
)
from scorpio_pipe.covariance_linearized import _project_linearized_functional


def _chain_pre_sky_functional(
    exposure: FrozenCovarianceExposure,
    *,
    ny_out: int,
    nlam: int,
    extraction: np.ndarray,
    fluxcal_transfer: np.ndarray,
    science_weights: np.ndarray,
    validated: tuple,
) -> np.ndarray:
    var, valid, stack, shift, scale = validated
    links = _y_adjoint_links(var.shape[0], ny_out, shift, stack)
    p = np.zeros_like(var, dtype=np.float64)
    post = fluxcal_transfer * science_weights
    for j, rows in enumerate(links):
        if not rows:
            continue
        out_weights = np.zeros(nlam, dtype=np.float64)
        for y, coefficient in rows:
            out_weights += coefficient * stack[y] * extraction[y] * post
        if not np.any(out_weights):
            continue
        plan = exposure.row_plans[j]
        if plan is None:
            raise ValueError("Missing COUNT_V1 row plan under nonzero measurement weight")
        if np.asarray(plan.counts).shape != (nlam,):
            raise ValueError("COUNT_V1 row plan wavelength dimension mismatch")
        row_scale = scale[j] if scale.ndim == 2 else scale
        back, _ = _project_linearized_functional(
            plan, var[j], valid[j], out_weights, row_scale,
        )
        if back.shape != np.asarray(plan.finite_idx).shape:
            raise ValueError("COUNT_V1 projection geometry is inconsistent")
        p[j, np.asarray(plan.finite_idx)] = back
    return p.ravel()


def covariance_of_frozen_sky_chain_grouped_pair(
    exposures: Sequence[FrozenCovarianceExposure],
    sky_operators: Sequence[FrozenSkyWLS],
    left_extraction_weights: np.ndarray,
    right_extraction_weights: np.ndarray,
    fluxcal_transfer: np.ndarray,
    left_weights: np.ndarray,
    right_weights: np.ndarray,
    independent_variances: Sequence[np.ndarray],
    bias_sensitivities: Sequence[np.ndarray],
    group_ids: Sequence[str],
    masterbias_variances: Mapping[str, np.ndarray],
) -> tuple[float, float, float]:
    """Exact conditional covariance for two final 1D linear functionals.

    `independent_variances` is the FULL pre-sky CCD independent noise,
    even at pixels ignored by later Linearize (these pixels may still
    influence a sky model). Do NOT pass a post-sky diagonal variance or
    silently replace invalid upstream variances with zero.

    `bias_sensitivities` must describe pre-sky response to the identified
    *same* MasterBias error field for every exposure in its group. This
    method never infers calibration-group provenance.
    """
    n = len(exposures)
    if n == 0 or any(len(item) != n for item in (
        sky_operators, independent_variances, bias_sensitivities, group_ids
    )):
        raise ValueError("All frozen chain/sky/bias inputs must have one entry per exposure")
    if (not isinstance(masterbias_variances, Mapping)
            or not all(isinstance(x, str) and x and x.strip() == x for x in group_ids)
            or set(group_ids) != set(masterbias_variances)):
        raise ValueError("Explicit, complete MasterBias group identity mapping required")

    E_left = np.asarray(left_extraction_weights, dtype=np.float64)
    E_right = np.asarray(right_extraction_weights, dtype=np.float64)
    if E_left.ndim != 2 or E_right.shape != E_left.shape or min(E_left.shape) == 0:
        raise ValueError("Both extraction maps must be matching nonempty 2-D grids")
    if not np.isfinite(E_left).all() or not np.isfinite(E_right).all():
        raise ValueError("Extraction coefficients must be finite")
    ny_out, nlam = E_left.shape
    d = np.asarray(fluxcal_transfer, dtype=np.float64)
    u = np.asarray(left_weights, dtype=np.float64)
    v = np.asarray(right_weights, dtype=np.float64)
    if (d.shape != (nlam,) or u.shape != (nlam,) or v.shape != (nlam,)
            or not np.isfinite(d).all() or np.any(d <= 0)
            or not np.isfinite(u).all() or not np.isfinite(v).all()):
        raise ValueError("Final functional and positive flux calibration must match wavelength grid")

    validated = [
        _validate_exposure(e, ny_out=ny_out, nlam=nlam)
        for e in exposures
    ]
    left_support = np.zeros(nlam, dtype=bool)
    right_support = np.zeros(nlam, dtype=bool)
    for y in range(ny_out):
        total = np.zeros(nlam)
        for _var, _valid, stack, _shift, _scale in validated:
            total += stack[y]
        supported = total > 0
        left_support |= supported & (E_left[y] != 0)
        right_support |= supported & (E_right[y] != 0)
        if not np.allclose(total[supported], 1., rtol=1e-7, atol=1e-9):
            raise ValueError("Stack coefficients must sum to one on supported pixels")
    if np.any((u != 0) & ~left_support) or np.any((v != 0) & ~right_support):
        raise ValueError("Final measurement uses wavelength bins without extraction support")

    independent = 0.
    groups: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i, (exposure, sky, prevar, sensitivity, gid, valid) in enumerate(zip(
        exposures, sky_operators, independent_variances, bias_sensitivities,
        group_ids, validated,
    )):
        shape = valid[0].shape
        prevar = np.asarray(prevar, dtype=np.float64)
        sensitivity = np.asarray(sensitivity, dtype=np.float64)
        if (not isinstance(sky, FrozenSkyWLS) or sky.n_pixels != int(np.prod(shape))
                or prevar.shape != shape or sensitivity.shape != shape
                or not np.isfinite(prevar).all() or np.any(prevar < 0)
                or not np.isfinite(sensitivity).all()):
            raise ValueError("Pre-sky noise, sensitivity and frozen SkySub CCD geometry invalid")
        p = _chain_pre_sky_functional(
            exposure, ny_out=ny_out, nlam=nlam,
            extraction=E_left, fluxcal_transfer=d,
            science_weights=u, validated=valid,
        )
        q = _chain_pre_sky_functional(
            exposure, ny_out=ny_out, nlam=nlam,
            extraction=E_right, fluxcal_transfer=d,
            science_weights=v, validated=valid,
        )
        a = sky.adjoint(p)
        b = sky.adjoint(q)
        independent += float(np.dot(prevar.ravel(), a*b))
        if gid not in groups:
            groups[gid] = (np.zeros_like(a), np.zeros_like(b))
        ga, gb = groups[gid]
        if ga.shape != a.shape:
            raise ValueError("Same MasterBias group has incompatible CCD geometry")
        ga += sensitivity.ravel() * a
        gb += sensitivity.ravel() * b

    shared = 0.
    for gid, (ga, gb) in groups.items():
        bvar = np.asarray(masterbias_variances[gid], dtype=np.float64)
        if (bvar.shape != ga.shape or not np.isfinite(bvar).all()
                or np.any(bvar < 0)):
            raise ValueError("MasterBias variance geometry or values invalid")
        shared += float(np.dot(bvar, ga*gb))
    total = float(independent+shared)
    if not np.isfinite(total) or not np.isfinite(independent) or not np.isfinite(shared):
        raise ValueError("Nonfinite grouped frozen-chain covariance")
    return float(independent), float(shared), total
