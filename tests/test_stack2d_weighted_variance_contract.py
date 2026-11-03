"""Stack2D: correct conditional variance under fixed, arbitrary weights.

The old 1/sum(weights) expression is valid only for pure inverse-variance
weights. Robust Huber weights and floors change the actual averaging weights.
These tests certify *conditional fixed-weight* propagation, not the sampling
variance of weights estimated from noisy data.
"""

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.maskbits import NO_COVERAGE
from scorpio_pipe.stages.stack2d import _WeightedMeanAccumulator, _stack_chunk_streaming, _huber_weight_block


def test_accumulator_huber_weights_use_w_squared_times_detector_var():
    acc = _WeightedMeanAccumulator((1, 1))
    acc.update(np.asarray([[10.0]]), np.asarray([[1.0]]), variances=np.asarray([[1.0]]))
    acc.update(np.asarray([[20.0]]), np.asarray([[0.2]]), variances=np.asarray([[1.0]]))
    mean, var = acc.finalize()
    np.testing.assert_allclose(mean, [[(10 + .2 * 20) / 1.2]], rtol=1e-6)
    np.testing.assert_allclose(var, [[(1.0 + .2**2) / 1.2**2]], rtol=1e-6)
    assert not np.isclose(var[0, 0], 1 / 1.2)


def test_accumulator_inverse_variance_reduces_to_old_formula():
    acc = _WeightedMeanAccumulator((1, 1))
    for x, v in [(5.0, 1.0), (7.0, 4.0), (10.0, 9.0)]:
        acc.update(
            np.asarray([[x]]),
            np.asarray([[1 / v]]),
            variances=np.asarray([[v]]),
        )
    _, var = acc.finalize()
    np.testing.assert_allclose(var, [[1 / (1 + 1 / 4 + 1 / 9)]], rtol=1e-6)


def test_accumulator_nonoptimal_floor_uses_true_input_var():
    acc = _WeightedMeanAccumulator((1, 1))
    acc.update(np.asarray([[0.0]]), np.asarray([[.25]]), variances=np.asarray([[1.0]]))
    acc.update(np.asarray([[10.0]]), np.asarray([[1 / 9]]), variances=np.asarray([[9.0]]))
    _, var = acc.finalize()
    expected = (.25**2 * 1.0 + (1 / 9)**2 * 9.0) / (.25 + 1 / 9)**2
    np.testing.assert_allclose(var, [[expected]], rtol=1e-6)


def test_accumulator_zero_coverage_and_zero_weight_dont_propagate_nan():
    acc = _WeightedMeanAccumulator((1, 2))
    acc.update(
        np.asarray([[2.0, np.nan]]),
        np.asarray([[1.0, 0.0]]),
        variances=np.asarray([[3.0, np.nan]]),
    )
    mean, var = acc.finalize()
    np.testing.assert_allclose(mean[0, 0], 2.0)
    np.testing.assert_allclose(var[0, 0], 3.0)
    assert np.isnan(mean[0, 1]) and np.isnan(var[0, 1])


def _hdul(value, variance):
    return fits.HDUList([
        fits.PrimaryHDU(data=np.asarray([[value]], dtype=np.float32)),
        fits.ImageHDU(data=np.asarray([[variance]], dtype=np.float32), name="VAR"),
        fits.ImageHDU(data=np.zeros((1, 1), dtype=np.uint16), name="MASK"),
    ])


def test_real_streaming_huber_stage_propagates_conditional_variance():
    # Three readings (0, 0, 10) with variance=1 each. Initial inverse-var
    # mean=10/3, Huber c=1 => weights (.3,.3,.15) in final streaming pass.
    hduls = [_hdul(0.0, 1.0), _hdul(0.0, 1.0), _hdul(10.0, 1.0)]
    try:
        mu, var, mask, cov, *_ = _stack_chunk_streaming(
            hduls=hduls, y0=0, y1=1, method="invvar_huber",
            robust_iter=0, huber_c=1.0, clip_sigma=4.0,
            y_shifts=[0.0] * 3, y_align_enabled=False,
            sci_scales=[1.0] * 3, var_scales=[1.0] * 3,
            fatal_bits=np.uint16(0), var_floor=None,
        )
    finally:
        for hdul in hduls:
            hdul.close()
    np.testing.assert_allclose(mu[0, 0], 2.0, rtol=1e-6)
    np.testing.assert_allclose(var[0, 0], .36, rtol=1e-6)
    assert cov[0, 0] == 3
    assert mask[0, 0] != 0  # Rejection provenance; not a fatal output pixel.


def test_real_streaming_floor_does_not_replace_detector_variance():
    hduls = [_hdul(0.0, 1.0), _hdul(10.0, 9.0)]
    try:
        _, var, _, cov, *_ = _stack_chunk_streaming(
            hduls=hduls, y0=0, y1=1, method="invvar",
            robust_iter=0, huber_c=2.0, clip_sigma=4.0,
            y_shifts=[0.0, 0.0], y_align_enabled=False,
            sci_scales=[1.0, 1.0], var_scales=[1.0, 1.0],
            fatal_bits=np.uint16(0), var_floor=4.0,
        )
    finally:
        for hdul in hduls:
            hdul.close()
    expected = (.25**2 * 1.0 + (1 / 9)**2 * 9.0) / (.25 + 1 / 9)**2
    np.testing.assert_allclose(var[0, 0], expected, rtol=1e-6)
    assert cov[0, 0] == 2


def test_huber_weights_never_multiply_zero_by_nan():
    w0 = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32)
    residual = np.asarray([[np.nan, 0.0, np.inf]], dtype=float)
    weights, hub = _huber_weight_block(w0, residual, 2.0)
    np.testing.assert_array_equal(weights, np.asarray([[0.0, 1.0, 0.0]]))
    assert np.isfinite(weights).all()
    assert np.isnan(hub[0, 0])  # NaN in diagnostic cannot enter the weight


def test_real_streaming_huber_handles_missing_frame_without_poisoning_neighbor():
    # In a single Y chunk, the first exposure lacks one wavelength bin
    # entirely; another exposure measures it correctly. The old w0*hub
    # produced 0*NaN=NaN and poisoned sum_w despite another valid exposure.
    invalid = fits.HDUList([
        fits.PrimaryHDU(data=np.asarray([[np.nan, 0.0]], dtype=np.float32)),
        fits.ImageHDU(
            data=np.asarray([[np.nan, 1.0]], dtype=np.float32),
            name="VAR",
        ),
        fits.ImageHDU(
            data=np.asarray([[int(NO_COVERAGE), 0]], dtype=np.uint16),
            name="MASK",
        ),
    ])
    valid = fits.HDUList([
        fits.PrimaryHDU(data=np.asarray([[4.0, 0.0]], dtype=np.float32)),
        fits.ImageHDU(data=np.ones((1, 2), dtype=np.float32), name="VAR"),
        fits.ImageHDU(data=np.zeros((1, 2), dtype=np.uint16), name="MASK"),
    ])
    try:
        for n_iter in [0, 3]:
            sci, var, mask, cov, *_ = _stack_chunk_streaming(
                hduls=[invalid, valid], y0=0, y1=1,
                method="invvar_huber", robust_iter=n_iter, huber_c=2.0,
                clip_sigma=4.0, y_shifts=[0.0, 0.0],
                y_align_enabled=False, sci_scales=[1.0, 1.0],
                var_scales=[1.0, 1.0],
                fatal_bits=np.uint16(NO_COVERAGE), var_floor=None,
            )
            np.testing.assert_allclose(sci, [[4.0, 0.0]], atol=1e-7)
            np.testing.assert_allclose(var, [[1.0, 0.5]], atol=1e-7)
            np.testing.assert_array_equal(cov, [[1, 2]])
            np.testing.assert_array_equal(mask, [[0, 0]])
    finally:
        invalid.close()
        valid.close()


@pytest.mark.parametrize("bad_weight", [np.nan, np.inf, -1.0])
def test_accumulator_refuses_invalid_weights_before_sum_is_poisoned(bad_weight):
    acc = _WeightedMeanAccumulator((1, 2))
    with pytest.raises(ValueError, match="finite nonnegative weights"):
        acc.update(
            np.asarray([[1.0, 2.0]]),
            np.asarray([[1.0, bad_weight]]),
            variances=np.ones((1, 2)),
        )
    assert np.array_equal(acc.sum_w, np.zeros((1, 2)))
