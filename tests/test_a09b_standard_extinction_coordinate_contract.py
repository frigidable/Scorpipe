"""A09b: standard-star atmospheric law must use native air coordinates."""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.calib.common import convert_wavelength_convention
from scorpio_pipe.calib.sensfunc import _interpolate_standard_extinction_air


def test_standard_vacuum_nm_samples_recover_native_air_extinction():
    air = np.arange(5000.0, 5011.0)
    vacuum_nm = convert_wavelength_convention(
        air, from_ref="air", to_ref="vacuum",
    ) / 10.0
    curve = np.array([4900.0, 5100.0])
    k = np.array([0.31, 0.11])
    got = _interpolate_standard_extinction_air(
        vacuum_nm, wave_unit="nm", wave_ref="vacuum",
        extinction_wave_A=curve, k_lambda=k,
    )
    expected = np.interp(air, curve, k)
    np.testing.assert_allclose(got, expected, rtol=2e-9, atol=1e-11)


def test_air_angstrom_samples_interpolate_without_conversion():
    wave = 5000.0 + np.arange(5, dtype=float)
    got = _interpolate_standard_extinction_air(
        wave, wave_unit="Angstrom", wave_ref="air",
        extinction_wave_A=np.array([4900.0, 5100.0]),
        k_lambda=np.array([0.3, 0.1]),
    )
    np.testing.assert_allclose(got, 0.2 - (wave - 5000) / 1000.0)


@pytest.mark.parametrize("curve_wave,curve_k", [
    ([5000.0, 5000.0, 5100.0], [0.2, 0.2, 0.1]),
    ([5100.0, 5000.0], [0.1, 0.2]),
    ([5000.0, 5100.0], [0.2, -0.1]),
    ([5000.0, 5100.0], [0.2, float("nan")]),
])
def test_invalid_tabulated_extinction_curve_fails_closed(curve_wave, curve_k):
    with pytest.raises(ValueError, match="Invalid atmospheric extinction reference"):
        _interpolate_standard_extinction_air(
            np.arange(5020.0, 5025.0),
            wave_unit="Angstrom", wave_ref="air",
            extinction_wave_A=np.asarray(curve_wave),
            k_lambda=np.asarray(curve_k),
        )


def test_missing_extinction_wavelength_coverage_fails_instead_of_median_padding():
    with pytest.raises(ValueError, match="does not cover"):
        _interpolate_standard_extinction_air(
            np.arange(5020.0, 5025.0), wave_unit="Angstrom",
            wave_ref="air",
            extinction_wave_A=np.array([5021.0, 5023.0]),
            k_lambda=np.array([0.2, 0.19]),
        )


@pytest.mark.parametrize("medium,unit", [
    ("vac", "nm"),
    ("vacuum", "meter"),
    ("", "Angstrom"),
])
def test_undeclared_or_invalid_medium_units_block_standard_transfer(medium, unit):
    with pytest.raises(ValueError, match="explicit"):
        _interpolate_standard_extinction_air(
            np.arange(5020.0, 5025.0), wave_unit=unit,
            wave_ref=medium,
            extinction_wave_A=np.array([5000.0, 5100.0]),
            k_lambda=np.array([0.2, 0.1]),
        )
