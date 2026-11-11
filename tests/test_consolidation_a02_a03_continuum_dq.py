"""A02/A03: reliable continuum and positive signed line significance.

The DQ1D v1 fatal MASK is already applied by LineMeasure; excluded pixels
must not leave a one-sided extrapolated continuum labelled a detection.
"""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.maskbits import COSMIC
from scorpio_pipe.science.lines import LineSpec, measure_line


LINE = LineSpec("synthetic_line", 5000.0, 8.0, (-30.0, -15.0), (15.0, 30.0))


def _spectrum():
    wave = np.arange(4950.0, 5051.0)
    flux = np.ones_like(wave)
    var = np.ones_like(wave)
    mask = np.zeros_like(wave, dtype=np.uint16)
    return wave, flux, var, mask


def _measure(wave, flux, var, mask):
    return measure_line(
        wave, flux, var, spec=LINE,
        calsys_rel=None, calsys_tell=None, calsys_abs=None,
        mask=mask,
    )


@pytest.mark.parametrize("side", ["left", "right"])
def test_one_wholly_masked_continuum_sideband_fails_closed(side):
    wave, flux, var, mask = _spectrum()
    flux += 20.0 * np.exp(-0.5 * ((wave - 5000.0) / 1.5) ** 2)
    if side == "left":
        mask[(wave >= 4970.0) & (wave <= 4985.0)] = COSMIC
    else:
        mask[(wave >= 5015.0) & (wave <= 5030.0)] = COSMIC
    result = _measure(wave, flux, var, mask)
    assert result["detected"] is False
    assert result["reason"] == "insufficient_unmasked_continuum"
    assert result["upper_limit_3sigma"] is None
    assert result["n_continuum_left" if side == "left" else "n_continuum_right"] == 0


def test_only_one_valid_sideband_sample_is_not_sufficient():
    wave, flux, var, mask = _spectrum()
    flux += 20.0 * np.exp(-0.5 * ((wave - 5000.0) / 1.5) ** 2)
    mask[(wave >= 4970.0) & (wave <= 4985.0)] = COSMIC
    mask[wave == 4975.0] = 0
    result = _measure(wave, flux, var, mask)
    assert result["detected"] is False
    assert result["reason"] == "insufficient_unmasked_continuum"
    assert result["n_continuum_left"] == 1
    assert result["n_continuum_right"] >= 2


def test_valid_sidebands_and_strong_line_still_detected():
    wave, flux, var, mask = _spectrum()
    flux += 20.0 * np.exp(-0.5 * ((wave - 5000.0) / 1.5) ** 2)
    result = _measure(wave, flux, var, mask)
    assert result["detected"] is True
    assert result["snr_stat"] >= 3
    assert result["n_continuum_left"] >= 2
    assert result["n_continuum_right"] >= 2
    assert abs(result["centroid"] - 5000.0) < 0.2


def test_negative_absorption_does_not_create_positive_detection():
    wave, flux, var, mask = _spectrum()
    flux -= 15.0 * np.exp(-0.5 * ((wave - 5000.0) / 1.5) ** 2)
    result = _measure(wave, flux, var, mask)
    assert result["flux"] < 0
    assert result["snr_stat"] < 0
    assert result["detected"] is False


def test_fatal_cosmic_outlier_is_excluded_without_expanding_bin_width():
    wave, flux, var, mask = _spectrum()
    flux[(wave >= 4998.0) & (wave <= 5002.0)] += 10.0
    flux[wave == 5000.0] = 10000.0
    mask[wave == 5000.0] = COSMIC
    result = _measure(wave, flux, var, mask)
    assert result["n_masked_in_window"] == 1
    assert result["detected"] is True
    assert result["flux"] == pytest.approx(40.0, abs=1e-9)


@pytest.mark.parametrize("values", [
    [5000.0, 5000.0, 5001.0],
    [5000.0, 4999.0, 5001.0],
    [5000.0, float("nan"), 5001.0],
])
def test_nonmonotonic_or_nonfinite_wavelengths_fail_closed(values):
    wave = np.asarray(values, dtype=float)
    with pytest.raises(ValueError, match="strictly increasing"):
        _measure(wave, np.ones_like(wave), np.ones_like(wave),
                 np.zeros_like(wave, dtype=np.uint16))
