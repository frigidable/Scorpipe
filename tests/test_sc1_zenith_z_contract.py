"""Instrument-specific SCORPIO-1 zenith distance header contract."""
from __future__ import annotations

import pytest

from scorpio_pipe.metadata.frame_meta import HeaderContractError, parse_frame_meta


def _sc1(**overrides):
    header = {
        "INSTRUME": "SCORPIO-1",
        "DATE-OBS": "2024/06/12",
        "TIME-OBS": "15:19:12.190",
        "EXPTIME": 600.0016,
        "OBJECT": "N2146sat",
        "IMAGETYP": "obj",
        "MODE": "Spectra",
        "DISPERSE": "VPHG1200B",
        "FILTERS": "slit_1.2",
        "SLITWID": None,
        "BINNING": "1x2",
        "NAXIS1": 2080,
        "NAXIS2": 1032,
        "NODE": "A",
        "RATE": 145.0,
        "GAIN": 0.9,
    }
    header.update(overrides)
    return header


@pytest.mark.parametrize("z", [0.0, 45.5, 51.3, 90.0, 100.2, 180.0])
def test_sc1_z_is_zenith_distance_with_explicit_provenance(z):
    m = parse_frame_meta(_sc1(Z=z), strict=True)
    assert m.zenith_distance_deg == pytest.approx(z)
    assert m.meta_provenance["zenith_distance_deg"] == "header:Z"
    assert "zenith_distance_deg" not in m.meta_missing_optional
    # Do not infer atmospheric airmass for calibrations below the horizon.
    assert m.airmass is None


def test_named_zenith_distance_precedes_z():
    m = parse_frame_meta(_sc1(Z=51.3, ZDIST=49.7), strict=True)
    assert m.zenith_distance_deg == pytest.approx(49.7)
    assert m.meta_provenance["zenith_distance_deg"] == "header:ZDIST"


def test_sc1_missing_zenith_remains_optional_missing():
    m = parse_frame_meta(_sc1(), strict=True)
    assert m.zenith_distance_deg is None
    assert "zenith_distance_deg" in m.meta_missing_optional


@pytest.mark.parametrize("bad", [-0.1, 180.1, float("nan"), float("inf")])
def test_sc1_invalid_zenith_fails_closed(bad):
    with pytest.raises(HeaderContractError, match="zenith distance"):
        parse_frame_meta(_sc1(Z=bad), strict=True)


def test_sc2_z_does_not_inherit_sc1_semantics():
    header = {
        "INSTRUME": "SCORPIO-2",
        "DATE-OBS": "2018-02-15",
        "TIME-OBS": "18:04:32.16",
        "EXPTIME": 5.0,
        "OBJECT": "Ne",
        "IMAGETYP": "NEON",
        "MODE": "SPECTRA",
        "DISPERSE": "VPHG1200@540",
        "SLITWID": 1.0,
        "LAMP": "NEON",
        "GAIN": 1.0,
        "RDNOISE": 3.0,
        "NAXIS1": 2048,
        "NAXIS2": 2048,
        "BINNING": "1 1",
        "READMODE": "1",
        "NODE": "0",
        "RATE": 100.0,
        "READRATE": "100",
        "READGAIN": "1.0",
        "Z": 0.123,
    }
    m = parse_frame_meta(header, strict=True)
    assert m.zenith_distance_deg is None
    assert "zenith_distance_deg" in m.meta_missing_optional
