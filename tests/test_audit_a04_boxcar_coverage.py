"""A04: one missing aperture row is not a missing output wavelength bin."""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.contracts.maskbits import NO_COVERAGE, EDGE, BADPIX
from scorpio_pipe.stages.extract1d import _boxcar_extract, _optimal_extract


def _planes():
    sci = np.full((7, 8), 4., dtype=float)
    var = np.ones_like(sci)
    mask = np.zeros_like(sci, dtype=np.uint16)
    centers = np.full(sci.shape[1], 3.)
    return sci, var, mask, centers


def _box(sci, var, mask, centers, *, minimum=.6):
    return _boxcar_extract(
        sci, var, mask, centers, ap_hw=2,
        fatal_bits=int(NO_COVERAGE | BADPIX),
        min_good_frac=minimum, sky_resid=None, apply_local_sky=False,
    )


def test_one_no_coverage_input_row_still_yields_valid_boxcar_spectrum():
    sci, var, mask, centers = _planes()
    mask[1, 3] = NO_COVERAGE
    flux, out_var, out_mask, _, npix = _box(sci, var, mask, centers)
    assert (int(out_mask[3]) & int(NO_COVERAGE)) == 0
    assert (int(out_mask[3]) & int(EDGE)) == 0
    assert int(npix[3]) == 4
    assert flux[3] == pytest.approx(16.)
    assert out_var[3] == pytest.approx(4.)
    assert flux[2] == pytest.approx(20.)
    assert int(npix[2]) == 5


def test_output_no_coverage_is_set_when_fraction_is_insufficient():
    sci, var, mask, centers = _planes()
    mask[1:5, 3] |= NO_COVERAGE
    flux, out_var, out_mask, _, npix = _box(sci, var, mask, centers)
    assert (int(out_mask[3]) & int(NO_COVERAGE)) != 0
    assert int(npix[3]) == 1
    assert flux[3] == pytest.approx(4.)
    assert out_var[3] == pytest.approx(1.)


def test_all_samples_missing_reliably_mark_no_coverage():
    sci, var, mask, centers = _planes()
    mask[1:6, 3] |= NO_COVERAGE
    flux, out_var, out_mask, _, npix = _box(sci, var, mask, centers)
    assert np.isnan(flux[3]) and np.isnan(out_var[3])
    assert int(npix[3]) == 0
    assert (int(out_mask[3]) & int(NO_COVERAGE)) != 0


def test_other_aperture_bits_still_propagate_as_provenance():
    sci, var, mask, centers = _planes()
    mask[1, 3] = np.uint16(int(NO_COVERAGE) | int(EDGE) | int(BADPIX))
    flux, _, out_mask, _, npix = _box(sci, var, mask, centers)
    assert (int(out_mask[3]) & int(NO_COVERAGE)) == 0
    assert (int(out_mask[3]) & int(EDGE)) != 0
    assert (int(out_mask[3]) & int(BADPIX)) != 0
    assert int(npix[3]) == 4
    assert flux[3] == pytest.approx(16.)


def test_boxcar_and_optimal_agree_on_partial_coverage_status():
    sci, var, mask, centers = _planes()
    mask[1, 3] |= NO_COVERAGE
    flux, var1, m1, _, n1 = _box(sci, var, mask, centers)
    _, _, m2, _, n2, _ = _optimal_extract(
        sci, var, mask, centers, ap_hw=2,
        fatal_bits=int(NO_COVERAGE | BADPIX),
        min_good_frac=.6, sky_resid=None, apply_local_sky=False,
        backend="numpy",
    )
    assert (int(m1[3]) & int(NO_COVERAGE)) == 0
    assert (int(m2[3]) & int(NO_COVERAGE)) == 0
    assert int(n1[3]) == int(n2[3]) == 4
