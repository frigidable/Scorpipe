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


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("smoothing", [0.0, 3.0])
def test_masked_cosmic_preserves_full_aperture_flux_and_neighbouring_channels(
    backend: str, smoothing: float,
) -> None:
    """Known 3-row object: an excluded CR must not turn 30 into 20 or 13."""
    if backend == "numba" and not extraction._NUMBA_OK:
        pytest.skip("Numba backend is optional")
    ny, nlam = 9, 65
    sci = np.zeros((ny, nlam), dtype=float)
    sci[3:6] = 10.0
    var = np.ones_like(sci)
    mask = np.zeros((ny, nlam), dtype=np.uint16)
    center = nlam // 2
    sci[4, center] += 1000.0
    mask[4, center] = np.uint16(COSMIC)
    before = sci.copy()

    flux, variance, flags, _sky, npix, meta = extraction._optimal_extract(
        sci, var, mask, np.full(nlam, 4.0),
        ap_hw=1,
        fatal_bits=int(COSMIC | BADPIX | USER | NO_COVERAGE),
        min_good_frac=0.6,
        sky_resid=None,
        apply_local_sky=False,
        smooth_lambda_sigma=smoothing,
        backend=backend,
    )

    np.testing.assert_allclose(flux, 30.0, atol=1e-9, rtol=0)
    np.testing.assert_allclose(variance[np.arange(nlam) != center], 3.0, atol=1e-9)
    assert variance[center] == pytest.approx(4.5, abs=1e-9)
    assert npix[center] == 2
    assert flags[center] & COSMIC
    assert not (flags[center] & NO_COVERAGE)
    assert meta["profile_normalization"] == "full_geometric_aperture"
    assert meta["profile_unconstrained_frac"] == 0.0
    np.testing.assert_array_equal(sci, before)


def test_optimal_flags_profile_that_cannot_be_recovered_from_any_wavelength() -> None:
    ny, nlam = 9, 40
    sci = np.zeros((ny, nlam), dtype=float)
    sci[3:6] = 10.0
    var = np.ones_like(sci)
    mask = np.zeros((ny, nlam), dtype=np.uint16)
    mask[4] = np.uint16(COSMIC)
    _flux, _var, flags, _sky, _npix, meta = extraction._optimal_extract(
        sci, var, mask, np.full(nlam, 4.0),
        ap_hw=1, fatal_bits=int(COSMIC),
        min_good_frac=0.6, sky_resid=None, apply_local_sky=False,
        smooth_lambda_sigma=3.0, backend="numpy",
    )
    assert np.all((flags & NO_COVERAGE) != 0)
    assert meta["profile_unconstrained_frac"] == 1.0
