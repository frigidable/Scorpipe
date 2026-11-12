"""Physical regression for Astropy's optical barycentric wavelength correction.

Reference: https://docs.astropy.org/en/stable/coordinates/velocities.html
The observed optical wavelength (and dispersion) is multiplied by
1 + v_corr / c. No science spectrum is calibrated in this unit test.
"""
from __future__ import annotations

import pytest

from scorpio_pipe.barycentric import apply_barycentric_to_wavelength_grid


C_KMS = 299792.458


@pytest.mark.parametrize("corr_kms", [-30.0, -3.0, 0.0, 3.0, 30.0])
def test_barycentric_grid_matches_optical_velocity_identity(corr_kms: float) -> None:
    rest_a = 5006.84
    v_observed = 1500.0
    lam0_topo = rest_a * (1.0 + v_observed / C_KMS)
    dl_topo = 0.85
    factor = 1.0 + corr_kms / C_KMS

    lam0_bary, dl_bary = apply_barycentric_to_wavelength_grid(
        lam0_topo, dl_topo, doppler_factor=factor
    )
    expected_velocity = (
        v_observed + corr_kms + v_observed * corr_kms / C_KMS
    )
    actual_velocity = C_KMS * (lam0_bary / rest_a - 1.0)

    assert actual_velocity == pytest.approx(expected_velocity, abs=1e-9)
    assert dl_bary == pytest.approx(dl_topo * factor, rel=1e-14)
    for pixel in (0, 1, 20, 500):
        assert lam0_bary + pixel * dl_bary == pytest.approx(
            (lam0_topo + pixel * dl_topo) * factor, abs=1e-10
        )


@pytest.mark.parametrize(
    "factor",
    [0.0, -1.0, float("nan"), float("inf"), -float("inf")],
)
def test_invalid_barycentric_factor_fails_closed(factor: float) -> None:
    with pytest.raises(ValueError, match="Doppler factor"):
        apply_barycentric_to_wavelength_grid(5000.0, 1.0, doppler_factor=factor)


@pytest.mark.parametrize(
    "lam0,dlambda",
    [
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (5000.0, float("nan")),
        (5000.0, float("inf")),
        (5000.0, 0.0),
    ],
)
def test_invalid_wavelength_grid_fails_closed(lam0: float, dlambda: float) -> None:
    with pytest.raises(ValueError, match="wavelength grid"):
        apply_barycentric_to_wavelength_grid(
            lam0, dlambda, doppler_factor=1.0001
        )


def test_positive_correction_increases_optical_wavelength() -> None:
    lam0, _ = apply_barycentric_to_wavelength_grid(
        5000.0, 1.0, doppler_factor=1.0 + 30.0 / C_KMS
    )
    assert lam0 > 5000.0
