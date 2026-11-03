"""Native MasterBias pixel-response decomposition, prior to sky modeling."""
import numpy as np
import pytest

from scorpio_pipe.calib.bias_components import decompose_flat_masterbias_variance


def test_flat_bias_decomposition_with_gain_and_flat_response():
    # ADU bias estimator variance is shared, independent photon variance is 9 ADU^2.
    flat = np.array([[2., 4.], [1., 2.]])
    bvar = np.full((2, 2), 4.)
    sens = -2. / flat
    expected_bias = sens**2 * bvar
    expected_rest = 9. * (2. / flat)**2
    vb, vr, bs = decompose_flat_masterbias_variance(
        expected_bias + expected_rest, bvar, flat, 2.,
    )
    np.testing.assert_allclose(vb, expected_bias)
    np.testing.assert_allclose(vr, expected_rest)
    np.testing.assert_allclose(bs, sens)


def test_incomplete_flat_pixels_have_zero_calibration_sensitivity():
    flat = np.array([[1., 0.], [2., np.nan]])
    bias = np.ones((2, 2)) * 3
    total = np.array([[8., 0.], [4., 0.]])
    b, r, s = decompose_flat_masterbias_variance(total, bias, flat, 1.)
    np.testing.assert_allclose(s, [[-1., 0.], [-0.5, 0.]])
    np.testing.assert_allclose(b, [[3., 0.], [.75, 0.]])
    np.testing.assert_allclose(r, [[5., 0.], [3.25, 0.]])


@pytest.mark.parametrize("case", [
    "bad_gain", "negative_bias", "missing_bias", "nan_bias",
    "negative_total", "bad_shape", "unclosed_total",
])
def test_invalid_flat_bias_component_contract_fails_closed(case):
    total = np.ones((2, 3)) * 10.
    bias = np.ones((2, 3))
    flat = np.ones((2, 3))
    gain = 2.
    if case == "bad_gain":
        gain = np.nan
    elif case == "negative_bias":
        bias[0, 0] = -1.
    elif case == "missing_bias":
        bias[0, 0] = np.inf
    elif case == "nan_bias":
        bias[0, 0] = np.nan
    elif case == "negative_total":
        total[0, 0] = -1.
    elif case == "bad_shape":
        bias = np.ones((2, 4))
    elif case == "unclosed_total":
        total[:] = 1.
        bias[:] = 10.
    with pytest.raises(ValueError):
        decompose_flat_masterbias_variance(total, bias, flat, gain)


def test_one_input_group_bias_not_fictitiously_divided():
    n = 7
    var_b, var_rest, sensitivity = decompose_flat_masterbias_variance(
        np.ones((1, 1)) * 13., np.ones((1, 1)) * 4.,
        np.ones((1, 1)), 1.,
    )
    assert var_b.item() == pytest.approx(4.)
    assert var_rest.item() == pytest.approx(9.)
    # The shared input perturbation is present once in a weighted sum.
    alpha = np.ones(n) / n
    assert (alpha.sum() * sensitivity.item())**2 * 4. == pytest.approx(4.)
