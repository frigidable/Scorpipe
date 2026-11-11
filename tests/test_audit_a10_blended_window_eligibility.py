"""A10: a fixed window integral is not a deblended line-component flux."""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.calib.common import air_to_vacuum
from scorpio_pipe.science.lines import (
    DEFAULT_LINES, LineSpec, _window_overlap_partners, measure_lines,
)
from scorpio_pipe.science.reddening import balmer_decrement_summary


def _data():
    wave = np.arange(4800.0, 6810.0, 0.25)
    flux = np.ones_like(wave)
    var = np.full_like(wave, 0.0025)
    for center, amplitude in (
        (4861.33, 20.0), (6562.80, 30.0),
        (6548.05, 9.0), (6583.45, 27.0),
        (5006.84, 18.0), (6716.44, 7.0), (6730.82, 8.0),
    ):
        flux += amplitude * np.exp(-0.5 * ((wave - center) / 1.5) ** 2)
    return wave, flux, var


def _rows(select=None, **kwargs):
    wave, flux, var = _data()
    return measure_lines(
        wave, flux, var,
        calsys_rel=None, calsys_tell=None, calsys_abs=None,
        lines=DEFAULT_LINES if select is None else select,
        **kwargs,
    )


def test_halpha_nii_integration_windows_are_blended_not_component_flux():
    measured = {row["name"]: row for row in _rows()}
    ha = measured["Halpha"]
    assert ha["detected"] is True
    assert ha["blend_flag"] is True
    assert set(ha["blend_partners"]) >= {"[NII]6548", "[NII]6583"}
    assert ha["flux_interpretation"] == "BLENDED_DIAGNOSTIC_ONLY"
    assert ha["component_flux_qualified"] is False
    assert ha["flux_kind"] == "window_integral_not_deblended"
    for name in ("[NII]6548", "[NII]6583"):
        assert measured[name]["blend_flag"] is True
        assert measured[name]["component_flux_qualified"] is False


def test_sii_doublet_windows_are_not_independent_component_fluxes():
    measured = {row["name"]: row for row in _rows()}
    for name, partner in (
        ("[SII]6716", "[SII]6731"),
        ("[SII]6731", "[SII]6716"),
    ):
        row = measured[name]
        assert partner in row["blend_partners"]
        assert row["blend_flag"] is True
        assert row["blend_warning"] is True
        assert row["component_flux_qualified"] is False
        assert row["flux_kind"] == "window_integral_not_deblended"


def test_unblended_window_is_still_only_diagnostic():
    measured = {row["name"]: row for row in _rows()}
    oi = measured["[OIII]5007"]
    assert oi["detected"] is True
    assert oi["blend_partners"] == []
    assert oi["blend_flag"] is False
    assert oi["blend_window_overlap_free"] is True
    # Lack of catalogued overlap is NOT evidence of qualified line flux.
    assert oi["component_flux_qualified"] is False
    assert oi["flux_interpretation"] == "ISOLATED_WINDOW_DIAGNOSTIC"


def test_requesting_halpha_alone_still_checks_unrequested_nii_catalogue():
    selected = (next(s for s in DEFAULT_LINES if s.name == "Halpha"),)
    row = _rows(select=selected)[0]
    assert row["name"] == "Halpha"
    assert set(row["blend_partners"]) >= {"[NII]6548", "[NII]6583"}
    assert row["component_flux_qualified"] is False


@pytest.mark.parametrize("medium", ["air", "vacuum"])
def test_blend_classification_respects_declared_wavelength_medium(medium):
    halpha = next(s for s in DEFAULT_LINES if s.name == "Halpha")
    partners = _window_overlap_partners(
        halpha, DEFAULT_LINES, redshift=0.015, waveref=medium,
    )
    assert set(partners) >= {"[NII]6548", "[NII]6583"}
    assert "[SII]6716" not in partners


def test_one_custom_requested_line_is_compared_to_default_catalogue():
    custom = LineSpec("Halpha_custom", 6562.8, 18.0, (-60., -25.), (25., 60.))
    row = _rows(select=(custom,))[0]
    assert "Halpha" in row["blend_partners"]
    assert "[NII]6583" in row["blend_partners"]


def test_vacuum_observed_spectrum_retains_blend_classification():
    wave, flux, var = _data()
    vac_wave = air_to_vacuum(wave)
    rows = measure_lines(
        vac_wave, flux, var,
        calsys_rel=None, calsys_tell=None, calsys_abs=None,
        lines=tuple(s for s in DEFAULT_LINES if s.name == "Halpha"),
        waveref="vacuum",
    )
    assert len(rows) == 1
    assert "[NII]6583" in rows[0]["blend_partners"]


def test_balmer_decrement_rejects_numerical_window_fluxes():
    measured = {row["name"]: row for row in _rows()}
    assert measured["Halpha"]["detected"] is True
    assert measured["Hbeta"]["detected"] is True
    assert balmer_decrement_summary(measured) is None


def test_independently_deblended_external_line_fluxes_are_not_blocked():
    external = {
        "Halpha": {
            "flux": 8.58, "detected": True, "blend_warning": False,
            "blend_flag": False, "component_flux_qualified": True,
        },
        "Hbeta": {
            "flux": 3.0, "detected": True, "blend_warning": False,
            "blend_flag": False, "component_flux_qualified": True,
        },
    }
    result = balmer_decrement_summary(external)
    assert result is not None
    assert result["line_ratio_Ha_Hb"] == pytest.approx(2.86)


def test_explicit_unqualified_component_vetoes_physical_ratio():
    external = {
        "Halpha": {"flux": 8.58, "detected": True, "component_flux_qualified": False},
        "Hbeta": {"flux": 3.0, "detected": True, "component_flux_qualified": True},
    }
    assert balmer_decrement_summary(external) is None


@pytest.mark.parametrize("z", [float("nan"), -1.0, float("inf")])
def test_invalid_redshift_refused_in_blend_catalogue(z):
    halpha = next(s for s in DEFAULT_LINES if s.name == "Halpha")
    with pytest.raises(ValueError, match="redshift"):
        _window_overlap_partners(halpha, DEFAULT_LINES, redshift=z, waveref="air")
