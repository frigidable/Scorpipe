"""Regression tests for 2D->1D quality masks in optimal extraction."""

from __future__ import annotations

import numpy as np
import pytest

import scorpio_pipe.stages.extract1d as extraction
from scorpio_pipe.maskbits import (
    BADPIX, COSMIC, EDGE, NO_COVERAGE, SKYMODEL_FAIL, USER,
)


def _case():
    ny, nlam = 7, 16
    sci = np.full((ny, nlam), 20.0, dtype=np.float64)
    var = np.full((ny, nlam), 9.0, dtype=np.float64)
    mask = np.zeros((ny, nlam), dtype=np.uint16)
    # Aperture is y=2..4. A sky-failure bit may coexist with usable pixels.
    mask[2, 2] |= SKYMODEL_FAIL
    mask[3, 2] |= EDGE
    mask[4, 4] |= COSMIC
    # Only one missing aperture pixel: output bin still has usable coverage.
    mask[2, 5] |= NO_COVERAGE
    # No bit outside the extraction aperture may leak into the 1D mask.
    mask[0, 6] |= BADPIX
    mask[6, 9] |= USER
    # No valid aperture pixels at this wavelength.
    mask[2:5, 11] |= NO_COVERAGE
    yc = np.full(nlam, 3.0, dtype=float)
    return sci, var, mask, yc


def _extract(backend: str):
    sci, var, mask, yc = _case()
    return extraction._optimal_extract(
        sci, var, mask, yc,
        ap_hw=1,
        fatal_bits=int(NO_COVERAGE | BADPIX | COSMIC | USER),
        min_good_frac=0.6,
        sky_resid=None,
        apply_local_sky=False,
        smooth_lambda_sigma=0,
        backend=backend,
    )


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_optimal_extraction_propagates_aperture_quality_bits(backend: str) -> None:
    if backend == "numba" and not extraction._NUMBA_OK:
        pytest.skip("Numba backend is optional")
    flux, var, mask, _, npix, meta = _extract(backend)

    assert mask[2] & SKYMODEL_FAIL
    assert mask[2] & EDGE
    assert mask[4] & COSMIC
    assert not (mask[5] & NO_COVERAGE), "Partial coverage must not mark the entire bin missing"
    assert npix[5] == 2
    assert np.isfinite(flux[5]) and np.isfinite(var[5]) and var[5] > 0
    assert mask[11] & NO_COVERAGE
    assert npix[11] == 0
    assert not (mask[6] & BADPIX), "Outside-aperture bits leaked into spectrum"
    assert not (mask[9] & USER), "Outside-aperture bits leaked into spectrum"
    assert meta["backend_used"] == backend


def test_optimal_numpy_and_numba_mask_contracts_are_identical() -> None:
    if not extraction._NUMBA_OK:
        pytest.skip("Numba backend is optional")
    np_result = _extract("numpy")
    nb_result = _extract("numba")
    np.testing.assert_array_equal(np_result[2], nb_result[2])
    np.testing.assert_array_equal(np_result[4], nb_result[4])
    np.testing.assert_allclose(np_result[0], nb_result[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np_result[1], nb_result[1], rtol=1e-12, atol=1e-12)


def test_optimal_without_input_mask_does_not_create_spurious_flags() -> None:
    sci, var, _mask, yc = _case()
    flux, out_var, out_mask, _, npix, _meta = extraction._optimal_extract(
        sci, var, None, yc,
        ap_hw=1,
        fatal_bits=int(NO_COVERAGE | BADPIX | COSMIC | USER),
        min_good_frac=0.6,
        sky_resid=None,
        apply_local_sky=False,
        smooth_lambda_sigma=0,
        backend="numpy",
    )
    assert np.count_nonzero(out_mask) == 0
    assert np.all(npix == 3)
    assert np.all(np.isfinite(flux))
    assert np.all(np.isfinite(out_var))
