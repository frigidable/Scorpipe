"""Stack2D contributor rejection must not invalidate accepted 1D pixels.

Stack2D's REJECTED bit is a diagnostic of *any* rejected/downweighted
constituent exposure, not a fatal property of the weighted stack SCI/VAR.
"""

import numpy as np
import pytest

from scorpio_pipe.maskbits import NO_COVERAGE, REJECTED
from scorpio_pipe.stages.extract1d import (
    _boxcar_extract,
    _extraction_mask_policy,
    _optimal_extract,
    _strip_informational_1d_bits,
)


def _synthetic_robust_stack():
    ny, nlam = 7, 21
    sci = np.full((ny, nlam), 10.0, dtype=float)
    var = np.full_like(sci, 4.0)
    # One input exposure was downweighted at this location while good
    # exposures contributed. The resulting stacked SCI remains usable.
    mask = np.zeros_like(sci, dtype=np.uint16)
    mask[3, 10] = np.uint16(REJECTED)
    yc = np.full(nlam, 3.0)
    return sci, var, mask, yc


def test_stack_rejected_contributor_is_informational_not_fatal():
    fatal, info = _extraction_mask_policy(is_stack=True)
    assert not (fatal & int(REJECTED))
    assert info == int(REJECTED)

    sci, var, mask, yc = _synthetic_robust_stack()
    flux, variance, one_dq, _, npix = _boxcar_extract(
        sci, var, mask, yc,
        ap_hw=2, fatal_bits=fatal, min_good_frac=0.6,
        sky_resid=None, apply_local_sky=False,
    )
    np.testing.assert_allclose(flux, 50.0)
    np.testing.assert_allclose(variance, 20.0)
    assert np.all(npix == 5)
    assert int(one_dq[10]) & int(REJECTED)
    final_dq = _strip_informational_1d_bits(one_dq, bits=info)
    assert not (int(final_dq[10]) & int(REJECTED))
    assert int(mask[3, 10]) & int(REJECTED)  # Original 2D provenance retained.


@pytest.mark.parametrize("backend", ["numpy", "auto"])
def test_optimal_extraction_uses_valid_stacked_rejected_pixel(backend):
    fatal, info = _extraction_mask_policy(is_stack=True)
    sci, var, mask, yc = _synthetic_robust_stack()
    flux, variance, one_dq, _, npix, meta = _optimal_extract(
        sci, var, mask, yc,
        ap_hw=2, fatal_bits=fatal, min_good_frac=0.6,
        sky_resid=None, apply_local_sky=False,
        backend=backend,
    )
    assert np.isfinite(flux[10])
    assert variance[10] > 0
    assert npix[10] == 5
    final_dq = _strip_informational_1d_bits(one_dq, bits=info)
    assert (int(final_dq[10]) & int(REJECTED)) == 0
    assert meta["backend_used"] in {"numpy", "numba"}


def test_single_frame_rejected_pixel_remains_fatal():
    fatal, info = _extraction_mask_policy(is_stack=False)
    assert fatal & int(REJECTED)
    assert info == 0
    sci, var, mask, yc = _synthetic_robust_stack()
    flux, variance, one_dq, _, npix = _boxcar_extract(
        sci, var, mask, yc,
        ap_hw=2, fatal_bits=fatal, min_good_frac=0.6,
        sky_resid=None, apply_local_sky=False,
    )
    assert npix[10] == 4
    assert flux[10] == 40.0
    assert variance[10] == 16.0
    final_dq = _strip_informational_1d_bits(one_dq, bits=info)
    assert int(final_dq[10]) & int(REJECTED)


def test_fatal_output_bits_are_never_cleared():
    m = np.asarray([REJECTED | NO_COVERAGE, REJECTED, NO_COVERAGE], dtype=np.uint16)
    out = _strip_informational_1d_bits(m, bits=int(REJECTED))
    np.testing.assert_array_equal(
        out, np.asarray([NO_COVERAGE, 0, NO_COVERAGE], dtype=np.uint16)
    )
    np.testing.assert_array_equal(
        m, np.asarray([REJECTED | NO_COVERAGE, REJECTED, NO_COVERAGE], dtype=np.uint16)
    )
