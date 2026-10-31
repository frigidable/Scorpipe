"""P1.4d1: independent dense/noise oracles for frozen cross-stage covariance."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scorpio_pipe.covariance_chain import (
    FrozenCovarianceExposure,
    covariance_of_frozen_chain_functionals,
    covariance_of_frozen_chain_functionals_pair,
)


# The dense oracle below uses these plain 2-D coefficient arrays. The code
# under test instead receives a sparse COUNT_V1 plan for each raw CCD row.
R0 = np.array([[.75, .25, 0., 0.], [0., .5, .4, 0.], [0., 0., .6, .5]])
R1 = np.array([[.5, .5, 0., 0.], [0., .25, .75, 0.], [0., 0., .25, .75]])


def _plan(R):
    indices = [np.flatnonzero(R[k]) for k in range(R.shape[0])]
    counts = np.array([len(x) for x in indices], dtype=np.int32)
    return SimpleNamespace(
        finite_idx=np.arange(R.shape[1], dtype=np.int32),
        src_idx=np.concatenate(indices).astype(np.int32),
        starts=np.r_[0, np.cumsum(counts[:-1])].astype(np.int32),
        counts=counts,
        weights=np.concatenate([R[k, indices[k]] for k in range(len(indices))]),
    )


def _fixtures():
    ny, nlam, nx = 4, 3, 4
    vars = [np.arange(1., ny * nx + 1.).reshape(ny, nx) / 2.5,
            np.arange(2., ny * nx + 2.).reshape(ny, nx) / 1.7]
    mask = [np.ones((ny, nx), dtype=bool), np.ones((ny, nx), dtype=bool)]
    mask[0][2, 0] = False
    vars[0][2, 0] = np.nan  # Invalid CCD variance must not be used.
    mask[1][0, 3] = False
    shifts = [.25, -.5]
    a0 = np.full((ny, nlam), .3)
    a1 = np.full((ny, nlam), .7)
    a0[0] = 0.
    a1[0] = 1.
    a0[-1] = 1.
    a1[-1] = 0.
    a0[2, 1], a1[2, 1] = .15, .85
    scales = [np.array([[.3, .6, 1.2], [.4, .7, 1.3],
                        [.5, .8, 1.4], [.6, .9, 1.5]]),
              np.array([.45, .8, 1.3])]
    exposures = [
        FrozenCovarianceExposure(
            row_plans=[_plan(R)] * ny, input_variance=vv, valid_mask=mm,
            stack_weights=ww, y_shift=shift, output_scale=scale
        )
        for R, vv, mm, ww, shift, scale in zip(
            (R0, R1), vars, mask, (a0, a1), shifts, scales
        )
    ]
    Etrace = np.array([[.1, .2, .25], [.6, .45, .3],
                       [1.1, .75, .85], [.2, .4, .5]])
    Efixed = np.array([[0., .1, .1], [.7, .8, .65],
                       [.7, .6, .8], [.5, .1, .15]])
    d = np.array([.3, 1.1, 2.4])
    u = np.array([-.5, 1.25, .7])
    v = np.array([.8, -.3, 1.7])
    return exposures, (R0, R1), Etrace, Efixed, d, u, v


def _dense_output_coefficient_matrices(exposures, Rlist, E, d):
    # Independently compose *forward* dense maps without row-plan helpers.
    out = []
    ny, nlam = E.shape
    for e, R in zip(exposures, Rlist):
        raw_var = np.asarray(e.input_variance)
        ny_in, nx = raw_var.shape
        full = np.zeros((nlam, ny_in * nx), dtype=float)
        for y in range(ny):
            x = y - e.y_shift
            j0 = int(np.floor(x))
            f = x - j0
            if not 0 <= j0 < ny_in or (f > 1e-12 and j0 + 1 >= ny_in):
                continue
            contributors = ((j0, 1. - f), (j0 + 1, f)) if f > 1e-12 else ((j0, 1.),)
            for j, ycoef in contributors:
                for k in range(nlam):
                    scale = e.output_scale
                    sc = (scale if np.ndim(scale) == 0 else
                          scale[k] if np.ndim(scale) == 1 else scale[j, k])
                    for i in range(nx):
                        if not e.valid_mask[j, i]:
                            continue
                        full[k, j * nx + i] += (
                            d[k] * E[y, k] * e.stack_weights[y, k]
                            * ycoef * sc * R[k, i]
                        )
        out.append(full)
    return out


def _dense_cov(exposures, Rlist, left_E, right_E, d, u, v):
    Mleft = _dense_output_coefficient_matrices(exposures, Rlist, left_E, d)
    Mright = _dense_output_coefficient_matrices(exposures, Rlist, right_E, d)
    cov = 0.
    for e, L, Q in zip(exposures, Mleft, Mright):
        variance = np.where(e.valid_mask, e.input_variance, 0.).ravel()
        cov += np.dot(variance * (u @ L), (v @ Q))
    return float(cov)


def test_dense_independent_oracle_trace_fixed_cross_covariance():
    e, Rs, et, ef, d, u, v = _fixtures()
    result = covariance_of_frozen_chain_functionals_pair(e, et, ef, d, u, v)
    expect = _dense_cov(e, Rs, et, ef, d, u, v)
    assert result == pytest.approx(expect, abs=1e-12, rel=1e-12)
    assert covariance_of_frozen_chain_functionals(e, et, d, u, v) == pytest.approx(
        _dense_cov(e, Rs, et, et, d, u, v), rel=1e-12
    )


def test_variance_nonnegative_and_monte_carlo_with_cross_mode_noise():
    e, Rs, et, ef, d, u, v = _fixtures()
    expect_trace = _dense_cov(e, Rs, et, et, d, u, u)
    expect_fixed = _dense_cov(e, Rs, ef, ef, d, v, v)
    expect_cross = _dense_cov(e, Rs, et, ef, d, u, v)
    assert covariance_of_frozen_chain_functionals(e, et, d, u) == pytest.approx(expect_trace, rel=1e-12)
    assert covariance_of_frozen_chain_functionals(e, ef, d, v) == pytest.approx(expect_fixed, rel=1e-12)
    assert expect_trace > 0 and expect_fixed > 0
    rng = np.random.default_rng(20261031)
    n = 50000
    Ft = np.zeros(n)
    Ff = np.zeros(n)
    for ex, L, Q in zip(e,
                        _dense_output_coefficient_matrices(e, Rs, et, d),
                        _dense_output_coefficient_matrices(e, Rs, ef, d)):
        var = np.where(ex.valid_mask, ex.input_variance, 0.).ravel()
        z = rng.normal(size=(n, var.size)) * np.sqrt(var)[None, :]
        Ft += z @ (u @ L)
        Ff += z @ (v @ Q)
    assert np.var(Ft, ddof=1) == pytest.approx(expect_trace, rel=.04)
    assert np.var(Ff, ddof=1) == pytest.approx(expect_fixed, rel=.04)
    assert np.cov(Ft, Ff, ddof=1)[0, 1] == pytest.approx(expect_cross, rel=.05)


@pytest.mark.parametrize("change,matched", [
    ("bad_normalization", "sum to one"),
    ("spatial_gap", "spatially uncovered"),
    ("negative_stack", "nonnegative"),
    ("bad_scale", "output_scale"),
    ("bad_shift", "Spatial shift"),
    ("bad_valid", "valid_mask"),
    ("bad_rowplan", "row plan"),
    ("bad_variance", "variance"),
])
def test_bad_contracts_fail_closed(change, matched):
    e, _Rs, et, _ef, d, u, _v = _fixtures()
    e = list(e)
    old = e[0]
    kw = dict(row_plans=old.row_plans, input_variance=old.input_variance,
              valid_mask=old.valid_mask, stack_weights=old.stack_weights,
              y_shift=old.y_shift, output_scale=old.output_scale)
    if change == "bad_normalization":
        kw["stack_weights"] = old.stack_weights.copy()
        kw["stack_weights"][1, 0] = .4
    elif change == "spatial_gap":
        kw["stack_weights"] = old.stack_weights.copy()
        kw["stack_weights"][0, 0] = .1
        e1 = e[1]
        w1 = e1.stack_weights.copy()
        w1[0, 0] = .9
        e[1] = FrozenCovarianceExposure(e1.row_plans, e1.input_variance,
                e1.valid_mask, w1, e1.y_shift, e1.output_scale)
    elif change == "negative_stack":
        kw["stack_weights"] = -old.stack_weights
    elif change == "bad_scale":
        kw["output_scale"] = np.array([.1, .2])
    elif change == "bad_shift":
        kw["y_shift"] = np.nan
    elif change == "bad_valid":
        kw["valid_mask"] = old.valid_mask.astype(np.uint8)
    elif change == "bad_rowplan":
        kw["row_plans"] = list(old.row_plans)[:-1]
    elif change == "bad_variance":
        kw["input_variance"] = old.input_variance.copy()
        kw["input_variance"][1, 1] = np.nan
    e[0] = FrozenCovarianceExposure(**kw)
    with pytest.raises(ValueError, match=matched):
        covariance_of_frozen_chain_functionals(e, et, d, u)


def test_exact_integer_shift_and_zero_support_geometry():
    e, _Rs, et, _ef, d, u, _v = _fixtures()
    E = e[0]
    # Both input and output have four rows; pure +1 shift has valid y=1..3.
    w = np.zeros_like(E.stack_weights)
    w[1:] = 1.
    solo = FrozenCovarianceExposure(E.row_plans, E.input_variance, E.valid_mask,
                                    w, y_shift=1., output_scale=E.output_scale)
    got = covariance_of_frozen_chain_functionals([solo], et, d, u)
    assert np.isfinite(got) and got > 0


def test_wide_spectrum_matrix_free_identity():
    nlam = 90000
    plan = SimpleNamespace(finite_idx=np.arange(nlam, dtype=np.int32),
        src_idx=np.arange(nlam, dtype=np.int32),
        starts=np.arange(nlam, dtype=np.int32),
        counts=np.ones(nlam, dtype=np.int32), weights=np.ones(nlam))
    e = FrozenCovarianceExposure([plan], np.ones((1, nlam)),
            np.ones((1, nlam), dtype=bool), np.ones((1, nlam)))
    E = np.ones((1, nlam))
    d = np.linspace(.75, 1.25, nlam)
    u = np.ones(nlam) / nlam
    got = covariance_of_frozen_chain_functionals([e], E, d, u)
    assert got == pytest.approx(np.dot(d * u, d * u), rel=1e-11)
