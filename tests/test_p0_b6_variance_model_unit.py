import numpy as np

from scorpio_pipe.variance_model import (
    ensure_var_contract,
    propagate_divide,
    propagate_multiply,
    propagate_weighted_mean,
)


def test_variance_propagate_multiply_scale():
    var = np.array([1.0, 4.0], dtype=float)
    out = propagate_multiply(var, 2.0)
    assert np.allclose(out, np.array([4.0, 16.0], dtype=float))


def test_variance_propagate_divide_constant_divisor():
    sci = np.array([10.0, 10.0], dtype=float)
    var = np.array([4.0, 9.0], dtype=float)
    d = np.full_like(sci, 2.0)

    out = propagate_divide(sci=sci, var_sci=var, divisor=d)
    assert np.allclose(out, np.array([1.0, 2.25], dtype=float))


def test_variance_propagate_divide_with_divisor_variance():
    sci = np.array([10.0], dtype=float)
    var_sci = np.array([4.0], dtype=float)
    divisor = np.array([2.0], dtype=float)
    var_divisor = np.array([0.25], dtype=float)

    out = propagate_divide(sci=sci, var_sci=var_sci, divisor=divisor, var_divisor=var_divisor)

    # Var(S/D) = Var(S)/D^2 + S^2*Var(D)/D^4
    expected = (4.0 / 4.0) + (10.0**2 * 0.25 / 16.0)
    assert np.allclose(out, np.array([expected], dtype=float))


def test_variance_propagate_weighted_mean_stack():
    # Two samples, explicit weights, same pixel grid.
    vars_ = [np.array([1.0, 4.0], dtype=float), np.array([9.0, 1.0], dtype=float)]
    weights = [np.array([1.0, 2.0], dtype=float), np.array([3.0, 4.0], dtype=float)]

    out = propagate_weighted_mean(vars_, weights)

    # Var(y) = sum_i(w_i^2 * Var_i) / (sum_i w_i)^2, pixel-wise.
    wsum = weights[0] + weights[1]
    num = (weights[0] ** 2) * vars_[0] + (weights[1] ** 2) * vars_[1]
    expected = num / (wsum**2)
    assert np.allclose(out, expected)


def test_variance_ensure_var_contract_sanitizes_nonfinite_and_negative():
    sci = np.ones(3, dtype=float)
    var = np.array([np.nan, -1.0, 5.0], dtype=float)

    out = ensure_var_contract(sci=sci, var=var, fill_value=0.0)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)
    assert np.allclose(out, np.array([0.0, 0.0, 5.0], dtype=float))
