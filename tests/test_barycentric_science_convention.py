"""Astropy returns a positive optical barycentric radial-velocity correction.

For wavelengths (optical convention), applying it multiplies the observed
wavelength by (1+v_bary/c). It does NOT divide wavelength by that factor.
"""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.barycentric import apply_barycentric_to_wavelength_grid


@pytest.mark.parametrize("velocity_kms", [-30.0, -15.0, 0.0, 15.0, 30.0])
def test_barycentric_wavelength_shift_sign_matches_astropy_optical_convention(
    velocity_kms: float,
) -> None:
    c_kms = 299792.458
    factor = 1.0 + velocity_kms / c_kms
    lambda0 = 6562.8
    disp = 0.87
    corrected, step = apply_barycentric_to_wavelength_grid(
        lambda0, disp, doppler_factor=factor
    )
    assert corrected == pytest.approx(lambda0 * factor, rel=1e-13)
    assert step == pytest.approx(disp * factor, rel=1e-13)
    # Observed rest frame line at wavelength lambda0 must gain the same
    # optical radial-velocity correction, including the negative sign.
    got_v = c_kms * (corrected / lambda0 - 1.0)
    assert got_v == pytest.approx(velocity_kms, abs=1e-9)


@pytest.mark.parametrize(
    "factor", [0.0, -1.0, np.nan, np.inf, -np.inf]
)
def test_barycentric_wavelength_transform_rejects_invalid_factor(
    factor: float,
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        apply_barycentric_to_wavelength_grid(
            5000.0, 1.0, doppler_factor=factor
        )


def test_barycentric_wavelength_transform_rejects_invalid_wavelength_grid() -> None:
    with pytest.raises(ValueError, match="wavelength grid"):
        apply_barycentric_to_wavelength_grid(
            np.nan, 1.0, doppler_factor=1.0001
        )
    with pytest.raises(ValueError, match="wavelength grid"):
        apply_barycentric_to_wavelength_grid(
            5000.0, 0.0, doppler_factor=1.0001
        )
