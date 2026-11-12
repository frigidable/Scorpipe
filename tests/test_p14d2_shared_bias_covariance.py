"""P1.4d2: independently validate grouped MasterBias through frozen operators.

These are analytic/dense and Monte Carlo numerical ORACLES only.
They neither certify production variance nor modify science FITS.
"""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.calib.bias_chain_covariance import (
    covariance_of_frozen_chain_with_shared_bias,
)
from tests.test_p14d1_frozen_chain_covariance import (
    _fixtures, _dense_output_coefficient_matrices, _dense_cov,
)


def _inputs():
    e, rs, et, ef, d, u, v = _fixtures()
    sensitivity = [
        -np.linspace(0.4, 1.6, 16).reshape(4, 4),
        -np.linspace(0.8, 1.9, 16).reshape(4, 4),
    ]
    variance = np.arange(1., 17.).reshape(4, 4) / 5.
    return e, rs, et, ef, d, u, v, sensitivity, variance


def _dense_bias_covariance(exposures, response, group_ids, group_var,
                           r_matrices, left_E, right_E, transfer, u, v):
    left = _dense_output_coefficient_matrices(
        exposures, r_matrices, left_E, transfer,
    )
    right = _dense_output_coefficient_matrices(
        exposures, r_matrices, right_E, transfer,
    )
    total = 0.
    for gid in set(group_ids):
        dl = sum(
            (u @ left[i]).reshape(response[i].shape) * response[i]
            for i, name in enumerate(group_ids) if gid == name
        )
        dr = sum(
            (v @ right[i]).reshape(response[i].shape) * response[i]
            for i, name in enumerate(group_ids) if gid == name
        )
        total += np.sum(group_var[gid] * dl * dr)
    return float(total)


@pytest.mark.parametrize("groups", [
    ["groupA", "groupA"],
    ["groupA", "groupB"],
])
def test_shared_bias_frozen_chain_matches_independent_dense_oracle(groups):
    e, rs, et, ef, d, u, v, response, bvar = _inputs()
    maps = {g: bvar * (2. if g == "groupB" else 1.) for g in set(groups)}
    ind, shared, total = covariance_of_frozen_chain_with_shared_bias(
        e, groups, maps, response, et, ef, d, u, v,
    )
    expected_ind = _dense_cov(e, rs, et, ef, d, u, v)
    expected_common = _dense_bias_covariance(
        e, response, groups, maps, rs, et, ef, d, u, v,
    )
    np.testing.assert_allclose(ind, expected_ind, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(shared, expected_common, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(total, ind + shared, rtol=1e-12)


def test_shared_bias_same_group_is_not_naively_divided_by_exposure_count():
    from types import SimpleNamespace
    from scorpio_pipe.covariance_chain import FrozenCovarianceExposure
    n = 7
    plan = SimpleNamespace(
        finite_idx=np.array([0]), src_idx=np.array([0]),
        starts=np.array([0]), counts=np.array([1]),
        weights=np.array([1.]),
    )
    exposures = [
        FrozenCovarianceExposure(
            row_plans=[plan],
            input_variance=np.array([[9.]]),
            valid_mask=np.array([[True]]),
            stack_weights=np.array([[1. / n]]),
        ) for _ in range(n)
    ]
    vi, vb, total = covariance_of_frozen_chain_with_shared_bias(
        exposures, ["A"] * n, {"A": np.array([[4.]])},
        [np.array([[-1.]])] * n,
        np.ones((1, 1)), np.ones((1, 1)),
        np.ones(1), np.ones(1),
    )
    np.testing.assert_allclose([vi, vb, total], [9. / n, 4., 9. / n + 4.])


def test_monte_carlo_dense_joint_noise_matches_shared_and_independent():
    e, rs, et, ef, d, u, v, response, bvar = _inputs()
    left = _dense_output_coefficient_matrices(e, rs, et, d)
    right = _dense_output_coefficient_matrices(e, rs, ef, d)
    g_left = [u @ mat for mat in left]
    g_right = [v @ mat for mat in right]
    rng = np.random.default_rng(20261103)
    draws = 25000
    shared_bias_noise = rng.normal(size=(draws, 16)) * np.sqrt(bvar.ravel())
    value_left = np.zeros(draws)
    value_right = np.zeros(draws)
    for ex, grad_l, grad_r, sens in zip(e, g_left, g_right, response):
        independent_var = np.where(ex.valid_mask, ex.input_variance, 0.)
        independent = rng.normal(size=(draws, 16)) * np.sqrt(independent_var.ravel())
        raw_noise = independent + shared_bias_noise * sens.ravel()
        value_left += raw_noise @ grad_l
        value_right += raw_noise @ grad_r
    actual = np.cov(value_left, value_right, ddof=1)[0, 1]
    independent, shared, predicted = covariance_of_frozen_chain_with_shared_bias(
        e, ["A", "A"], {"A": bvar}, response, et, ef, d, u, v,
    )
    np.testing.assert_allclose(actual, predicted, rtol=.055, atol=.03)
    assert shared != 0.
    assert np.isfinite(independent)


@pytest.mark.parametrize("broken", [
    "missing_group", "missing_map", "shape_mismatch",
    "negative_bias_variance", "nan_bias_variance",
    "bad_sensitivity", "bad_response_shape",
])
def test_shared_bias_operator_fails_closed(broken):
    e, rs, et, ef, d, u, v, response, bvar = _inputs()
    groups = ["A", "A"]
    maps = {"A": bvar.copy()}
    response = [x.copy() for x in response]
    if broken == "missing_group":
        groups[0] = ""
    elif broken == "missing_map":
        maps = {}
    elif broken == "shape_mismatch":
        maps = {"A": np.ones((2, 4))}
    elif broken == "negative_bias_variance":
        maps["A"][0, 0] = -1
    elif broken == "nan_bias_variance":
        maps["A"][0, 0] = np.nan
    elif broken == "bad_sensitivity":
        response[0][0, 0] = np.nan
    elif broken == "bad_response_shape":
        response[1] = np.ones((3, 4))
    with pytest.raises(ValueError):
        covariance_of_frozen_chain_with_shared_bias(
            e, groups, maps, response, et, ef, d, u, v,
        )


def test_zero_bias_variance_reduces_to_p14d1():
    e, rs, et, ef, d, u, v, response, bvar = _inputs()
    vi, vb, total = covariance_of_frozen_chain_with_shared_bias(
        e, ["A", "A"], {"A": np.zeros_like(bvar)},
        response, et, ef, d, u, v,
    )
    np.testing.assert_allclose(vb, 0.)
    np.testing.assert_allclose(total, vi)


def test_distinct_groups_do_not_cross_correlate_identical_inputs():
    from types import SimpleNamespace
    from scorpio_pipe.covariance_chain import FrozenCovarianceExposure
    plan = SimpleNamespace(
        finite_idx=np.array([0]), src_idx=np.array([0]), starts=np.array([0]),
        counts=np.array([1]), weights=np.array([1.])
    )
    e = [
        FrozenCovarianceExposure([plan], np.array([[1.]]), np.array([[True]]),
                                 np.array([[.5]])),
        FrozenCovarianceExposure([plan], np.array([[1.]]), np.array([[True]]),
                                 np.array([[.5]])),
    ]
    params = [np.ones((1, 1)), np.ones((1, 1)), np.ones(1), np.ones(1)]
    shared = covariance_of_frozen_chain_with_shared_bias(
        e, ["A", "A"], {"A": np.array([[4.]])},
        [np.ones((1, 1))] * 2, *params,
    )
    distinct = covariance_of_frozen_chain_with_shared_bias(
        e, ["A", "B"], {"A": np.array([[4.]]), "B": np.array([[4.]])},
        [np.ones((1, 1))] * 2, *params,
    )
    np.testing.assert_allclose(shared, [0.5, 4., 4.5])
    np.testing.assert_allclose(distinct, [0.5, 2., 2.5])
