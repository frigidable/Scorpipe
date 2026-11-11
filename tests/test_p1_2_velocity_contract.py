"""P1.2 compute-only velocity and FITS metadata contracts.

No actual science data, wave grids or pipeline stage products are modified.
The direct Astropy calculation provides an independent oracle for both kinds.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, solar_system_ephemeris
from astropy.io import fits
from astropy.time import Time
from astropy.utils import iers

from scorpio_pipe.barycentric import (
    _parse_obstime,
    _parse_radec,
    apply_barycentric_to_wavelength_grid,
    compute_barycentric_velocity_mps,
)
from scorpio_pipe.schema import ProjectBlock
from scorpio_pipe.stages.linearize import _write_mef


def _header(**updates):
    d = {
        "RA": 10.0,
        "DEC": 20.0,
        "MJD-OBS": 59000.0,
        "EXPTIME": 600.0,
        "SPECSYS": "TOPOCENT",
    }
    d.update(updates)
    return d


def test_ra_numeric_unit_not_inferred_from_fk5_or_magnitude():
    with pytest.raises(ValueError, match="Numeric RA is ambiguous"):
        _parse_radec(_header(RADESYS="FK5"))
    with pytest.raises(ValueError, match="Numeric RA is ambiguous"):
        _parse_radec(_header())
    assert _parse_radec(_header(RAUNIT="deg")) == (10.0, 20.0)
    assert _parse_radec(_header(RAUNIT="hourangle")) == (150.0, 20.0)


def test_colon_ra_is_explicit_hourangle_and_nonfinite_rejected():
    ra, dec = _parse_radec({"RA": "12:30:00", "DEC": "+20:00:00"})
    assert ra == pytest.approx(187.5)
    assert dec == pytest.approx(20)
    with pytest.raises(ValueError):
        _parse_radec({"RA": 361, "DEC": 20, "RAUNIT": "deg"})


def test_observation_midpoint_and_conflicting_averages():
    t = _parse_obstime(_header())
    assert t.mjd == pytest.approx(59000 + 300.0 / 86400)
    t2 = _parse_obstime(_header(**{"MJD-AVG": 59000 + 300 / 86400}))
    assert t2.mjd == pytest.approx(t.mjd)
    with pytest.raises(ValueError, match="Average time disagrees"):
        _parse_obstime(_header(**{"MJD-AVG": 59000.5}))
    midpoint_header = _header(EXPTIME=None, **{"MJD-OBS": 59000.5})
    assert _parse_obstime(midpoint_header, time_reference="mid").mjd == pytest.approx(
        59000.5
    )


def test_scorpio_slash_dates_require_instrument_and_utc_clock():
    d = {"DATE-OBS": "2024/06/12", "UT": "15:00:00", "INSTRUME": "SCORPIO-1", "EXPTIME": 600}
    actual = _parse_obstime(d)
    assert actual.isot == "2024-12-06T15:05:00.000"
    d["INSTRUME"] = "OTHER"
    with pytest.raises(ValueError, match="Ambiguous slash date"):
        _parse_obstime(d)
    with pytest.raises(ValueError, match="Need MJD-AVG"):
        _parse_obstime({"DATE": "2024-12-06T15:00:00", "EXPTIME": 600})


@pytest.mark.parametrize("kind,expected_frame", [
    ("barycentric", "BARYCENT"),
    ("heliocentric", "HELIOCEN"),
])
def test_compute_matches_direct_astropy(kind, expected_frame):
    header = _header(RAUNIT="deg")
    preserved = header.copy()
    result = compute_barycentric_velocity_mps(header, kind=kind)
    assert result.ok, result.message
    assert result.specsyst_out == expected_frame
    assert result.applied is False
    assert result.velocity_definition == "optical"
    assert result.ephemeris == "builtin"
    assert "IERS_B" in result.iers_source
    assert header == preserved

    location = EarthLocation.from_geodetic(
        41.4416666667 * u.deg, 43.6533333333 * u.deg, 2070 * u.m
    )
    time = Time(59000 + 300.0 / 86400.0, format="mjd", scale="utc")
    table = iers.IERS_B.open(iers.IERS_B_FILE)
    with (
        iers.conf.set_temp("auto_download", False),
        iers.earth_orientation_table.set(table),
        solar_system_ephemeris.set("builtin"),
    ):
        independent = SkyCoord(10 * u.deg, 20 * u.deg).radial_velocity_correction(
            kind=kind, obstime=time, location=location
        ).to_value(u.m / u.s)
    assert result.v_bary_mps == pytest.approx(independent, abs=1e-6)
    assert result.doppler_factor == pytest.approx(1.0 + independent / 299792458.0, abs=1e-14)


@pytest.mark.parametrize("changes", [
    {"SPECSYS": "BARYCENT"},
    {"SSYSOBS": "HELIOCEN"},
    {"TIMESYS": "TT"},
    {"RAUNIT": "bogus"},
    {"RADESYS": "FK4"},
    {"MJD-OBS": None},
])
def test_unsafe_or_incomplete_compute_fails_closed(changes):
    header = _header(RAUNIT="deg")
    header.update(changes)
    result = compute_barycentric_velocity_mps(header)
    assert not result.ok
    assert math.isnan(result.v_bary_mps)
    assert math.isnan(result.doppler_factor)
    assert result.applied is False
    assert result.message


def test_unknown_location_does_not_silently_use_bta():
    result = compute_barycentric_velocity_mps(
        _header(RAUNIT="deg"), location="invalid coordinates"
    )
    assert not result.ok
    assert "location" in result.message.lower()


@pytest.mark.parametrize("kwargs", [
    {"already_applied": True},
    {"specsys_in": "BARYCENT"},
    {"wave_reference": "air"},
    {"specsys_out": "TOPOCENT"},
])
def test_grid_rejects_double_application_or_wrong_medium(kwargs):
    with pytest.raises(ValueError):
        apply_barycentric_to_wavelength_grid(
            5000.0, 0.85, doppler_factor=1.0001, **kwargs
        )


def test_grid_rejects_quantities_and_matches_optical_multiplier():
    factor = 1.0 + 30.0 / 299792.458
    x0, dx = apply_barycentric_to_wavelength_grid(5000.0, 0.85, doppler_factor=factor)
    assert x0 == pytest.approx(5000 * factor)
    assert dx == pytest.approx(0.85 * factor)
    with pytest.raises(ValueError):
        apply_barycentric_to_wavelength_grid(
            5000 * u.AA, 0.85, doppler_factor=factor
        )


def test_project_schema_rejects_automatic_application_until_stage_exists():
    assert ProjectBlock(barycentric_correction=None).barycentric_correction is None
    assert ProjectBlock(barycentric_correction="off").barycentric_correction == "OFF"
    with pytest.raises(ValueError, match="not implemented"):
        ProjectBlock(barycentric_correction="barycentric")
    with pytest.raises(ValueError, match="not implemented"):
        ProjectBlock(barycentric_correction="heliocentric")


def test_linearize_fits_defaults_observer_frame_independent_of_axis(tmp_path):
    path = tmp_path / "frame.fits"
    header = fits.Header()
    header["SPECSYS"] = "BARYCENT"
    _write_mef(path, np.zeros((2, 3), dtype=float), header, None, None, None)
    with fits.open(path, memmap=False) as hdul:
        assert hdul[0].header["SPECSYS"] == "BARYCENT"
        assert hdul[0].header["SSYSOBS"] == "TOPOCENT"
