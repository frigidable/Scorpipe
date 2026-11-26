"""Independent dense and Monte Carlo checks for the frozen SkySub covariance."""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.calib.sky_bias_covariance import (
    FrozenSkyWLS, covariance_of_frozen_sky_grouped,
    frozen_kelson_bspline_operator,
)


def _example():
    n = 9
    x = np.column_stack([
        np.ones(n),
        np.linspace(-1., 1., n),
        np.linspace(-1., 1., n)**2,
    ])
    idx = np.array([0, 1, 2, 4, 6, 7, 8], dtype=int)
    w = np.array([1.2, .7, 1.5, 1., .8, 2., 1.3])
    return x, idx, w


def _dense_residual_matrix(x, indices, weights):
    # Independent forward oracle: explicitly form H and R in small dimensions.
    fit = x[indices]
    normal = fit.T @ (fit * weights[:, None])
    sampler = np.eye(x.shape[0])[indices]
    hat = x @ np.linalg.solve(normal, fit.T * weights[None, :]) @ sampler
    return np.eye(x.shape[0]) - hat


def test_frozen_wls_forward_and_adjoint_match_dense_reference():
    x, idx, w = _example()
    model = FrozenSkyWLS(x, idx, w)
    expected = _dense_residual_matrix(x, idx, w)
    r = np.random.default_rng(141).normal(size=x.shape[0])
    u = np.random.default_rng(142).normal(size=x.shape[0])
    np.testing.assert_allclose(model.apply(r), expected @ r, atol=1e-11)
    np.testing.assert_allclose(model.adjoint(u), expected.T @ u, atol=1e-11)
    assert np.dot(u, model.apply(r)) == pytest.approx(
        np.dot(model.adjoint(u), r), abs=1e-11,
    )


def test_frozen_sky_removes_basis_modes_but_not_other_uncertainty():
    x, idx, w = _example()
    op = FrozenSkyWLS(x, idx, w)
    for column in range(x.shape[1]):
        np.testing.assert_allclose(op.apply(x[:, column]), 0., atol=1e-10)
    # An object-pixel measurement inherits independent sky-fitting noise.
    u = np.zeros(x.shape[0])
    u[3] = 1.
    projected = op.adjoint(u)
    assert np.count_nonzero(np.abs(projected) > 1e-10) >= 2
    assert projected[3] == pytest.approx(1.)


@pytest.mark.parametrize("group_ids", [["A", "A"], ["A", "B"]])
def test_grouped_sky_bias_covariance_matches_dense_independent_oracle(group_ids):
    x, idx, w = _example()
    operators = [
        FrozenSkyWLS(x, idx, w),
        FrozenSkyWLS(x, idx[:-1], w[:-1] * 1.17),
    ]
    n = x.shape[0]
    u = [np.linspace(.2, 1., n), np.linspace(-.5, .7, n)]
    v = [np.linspace(-1., .3, n), np.linspace(.2, -.4, n)]
    var = [np.linspace(1., 3., n), np.linspace(2., 4., n)]
    sensitivity = [-np.linspace(1., 1.5, n), -np.linspace(.5, 2., n)]
    bvars = {g: np.linspace(.3, 1.4, n) for g in group_ids}
    ind, shared, total = covariance_of_frozen_sky_grouped(
        operators, u, v, var, sensitivity, group_ids, bvars,
    )
    rmat = [_dense_residual_matrix(x, op.fit_indices, op.fit_weights)
            for op in operators]
    projected_left = [R.T @ q for R, q in zip(rmat, u)]
    projected_right = [R.T @ q for R, q in zip(rmat, v)]
    expected_ind = sum(
        np.dot(v_i, q_l * q_r)
        for v_i, q_l, q_r in zip(var, projected_left, projected_right)
    )
    expected_bias = 0.
    for gid in set(group_ids):
        left = sum((s * g for s, g, lab in zip(
            sensitivity, projected_left, group_ids) if lab == gid),
            np.zeros(n),
        )
        right = sum((s * g for s, g, lab in zip(
            sensitivity, projected_right, group_ids) if lab == gid),
            np.zeros(n),
        )
        expected_bias += np.dot(bvars[gid], left * right)
    np.testing.assert_allclose(
        [ind, shared, total],
        [expected_ind, expected_bias, expected_ind + expected_bias],
        rtol=1e-11, atol=1e-11,
    )


def test_grouped_sky_bias_covariance_monte_carlo():
    x, idx, w = _example()
    ops = [FrozenSkyWLS(x, idx, w), FrozenSkyWLS(x, idx, w * 1.15)]
    n = x.shape[0]
    vi = [np.linspace(.2, 1., n), np.linspace(.4, .8, n)]
    s = [-np.ones(n), -np.linspace(.8, 1.3, n)]
    bias = np.linspace(.7, 1.2, n)
    u = [np.eye(n)[3], np.eye(n)[5]]
    independent, shared, predicted = covariance_of_frozen_sky_grouped(
        ops, u, u, vi, s, ["A", "A"], {"A": bias},
    )
    rng = np.random.default_rng(2026149)
    k = 24000
    common = rng.normal(size=(k, n)) * np.sqrt(bias)[None, :]
    empirical = np.zeros(k)
    for op, var, sensitivity, functional in zip(ops, vi, s, u):
        draw = rng.normal(size=(k, n)) * np.sqrt(var)[None, :]
        # Independent dense *forward* residual operator as Monte Carlo oracle.
        R = _dense_residual_matrix(x, op.fit_indices, op.fit_weights)
        empirical += (draw + common * sensitivity) @ (R.T @ functional)
    actual = np.var(empirical, ddof=1)
    assert predicted > independent
    assert shared > 0.
    np.testing.assert_allclose(actual, predicted, rtol=.045)


def test_frozen_kelson_sparse_basis_matches_actual_sky_model_basis():
    from scorpio_pipe.stages.sky_sub import _make_bspline_knot_vector
    rows, cols = 8, 12
    lam = 5000. + np.linspace(0., 10., cols)[None, :] + .04 * np.arange(rows)[:, None]
    y = np.linspace(-1., 1., rows)
    mask = np.ones((rows, cols), dtype=bool)
    mask[3:5, :] = False
    var = np.ones_like(lam) * 4.
    knots = _make_bspline_knot_vector(
        float(np.min(lam)), float(np.max(lam)),
        knot_step_A=3., degree=2,
    )
    op = frozen_kelson_bspline_operator(
        lam, y, knots=knots, degree=2, y_order=1,
        final_fit_mask=mask, variance=var,
    )
    assert op.n_pixels == rows * cols
    assert op.design.shape[1] >= 4
    rng = np.random.default_rng(321)
    coeff = rng.normal(size=op.design.shape[1])
    # A frozen exact linear sky model is removed even on object rows.
    np.testing.assert_allclose(
        op.apply(np.asarray(op.design @ coeff).ravel()),
        np.zeros(rows * cols), atol=5e-8,
    )


@pytest.mark.parametrize("damage", [
    "duplicate_fit", "out_of_bounds", "negative_weights",
    "nan_weights", "singular", "nan_design",
])
def test_invalid_frozen_sky_fit_rejected(damage):
    x, idx, w = _example()
    x, idx, w = x.copy(), idx.copy(), w.copy()
    if damage == "duplicate_fit":
        idx[1] = idx[0]
    elif damage == "out_of_bounds":
        idx[-1] = 100
    elif damage == "negative_weights":
        w[0] = -1.
    elif damage == "nan_weights":
        w[0] = np.nan
    elif damage == "singular":
        x[:, 1] = x[:, 0]
    elif damage == "nan_design":
        x[0, 0] = np.nan
    with pytest.raises(ValueError):
        FrozenSkyWLS(x, idx, w)


@pytest.mark.parametrize("damage", [
    "missing_group", "wrong_variance", "wrong_sensitivity",
    "wrong_functional", "negative_bias_var", "unknown_group",
])
def test_invalid_sky_covariance_lineage_fails_closed(damage):
    x, idx, w = _example()
    op = FrozenSkyWLS(x, idx, w)
    n = x.shape[0]
    args = {
        "sky_operators": [op],
        "left_functionals": [np.ones(n)],
        "right_functionals": [np.ones(n)],
        "independent_variances": [np.ones(n)],
        "bias_sensitivities": [np.ones(n)],
        "group_ids": ["A"],
        "masterbias_variances": {"A": np.ones(n)},
    }
    if damage == "missing_group":
        args["group_ids"] = [""]
    elif damage == "wrong_variance":
        args["independent_variances"] = [np.ones(n - 1)]
    elif damage == "wrong_sensitivity":
        args["bias_sensitivities"] = [np.ones(n - 1)]
    elif damage == "wrong_functional":
        args["left_functionals"] = [np.ones(n - 1)]
    elif damage == "negative_bias_var":
        args["masterbias_variances"] = {"A": -np.ones(n)}
    else:
        args["masterbias_variances"] = {"B": np.ones(n)}
    with pytest.raises(ValueError):
        covariance_of_frozen_sky_grouped(**args)
