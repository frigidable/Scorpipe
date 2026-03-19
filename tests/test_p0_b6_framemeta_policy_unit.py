import pytest

from scorpio_pipe.metadata.frame_meta import HeaderContractError
from scorpio_pipe.metadata.frame_meta import parse_frame_meta
from scorpio_pipe.metadata.policy import FieldRequirement, MetadataPolicy, default_metadata_policy
from scorpio_pipe.metadata.sources import FallbackSources


# Minimal SCORPIO-2 long-slit header for spectra mode (adapted from
# tests/test_headers_sc2.py; keep only keys required by the parser).
BASE_SC2_SPECTRA_HEADER = {
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
    # Geometry / binning
    "NAXIS1": 2048,
    "NAXIS2": 2048,
    "BINNING": "1 1",
    # Readout keys (node/rate/gain)
    "READMODE": "1",
        "NODE": "0",
    "RATE": 100.0,
    "READRATE": "100",
    "READGAIN": "1.0",
}


def test_framemeta_missing_required_key_strict_raises():
    hdr = dict(BASE_SC2_SPECTRA_HEADER)
    hdr.pop("DISPERSE", None)

    with pytest.raises(HeaderContractError):
        parse_frame_meta(hdr, strict=True)


def test_framemeta_missing_optional_key_records_missing_optional():
    hdr = dict(BASE_SC2_SPECTRA_HEADER)
    hdr.pop("OBJECT", None)

    m = parse_frame_meta(hdr, strict=True)
    assert "object_name" in (m.meta_missing_optional or [])


def test_framemeta_fallback_records_provenance_and_value():
    hdr = dict(BASE_SC2_SPECTRA_HEADER)
    hdr.pop("DISPERSE", None)

    base = default_metadata_policy()
    req = dict(base.requirements)
    req["disperser"] = FieldRequirement.FALLBACK_ALLOWED
    policy = MetadataPolicy(requirements=req)

    fb = FallbackSources(global_values={"disperser": "VPHG1200@540"})

    m = parse_frame_meta(hdr, strict=True, policy=policy, fallback_sources=fb)

    assert m.disperser == "VPHG1200@540"
    assert "disperser" in (m.meta_fallback_used or [])
    assert (m.meta_provenance or {}).get("disperser") == "fallback:config"


def test_framemeta_calibration_grade_metadata_from_headers_and_derivations():
    hdr = dict(BASE_SC2_SPECTRA_HEADER)
    hdr.update({
        "AIRMASS": 1.20,
        "RA": "12:30:00",
        "DEC": "+15:00:00",
        "PARANG": 23.5,
        "OBSERVAT": "SAO RAS 6m",
        "SITENAME": "BTA",
        "SEEING": 1.4,
    })

    m = parse_frame_meta(hdr, strict=True)

    assert m.airmass == pytest.approx(1.20)
    assert m.zenith_distance_deg == pytest.approx(33.5573, rel=1e-4)
    assert m.parallactic_angle_deg == pytest.approx(23.5)
    assert m.ra_deg == pytest.approx(187.5)
    assert m.dec_deg == pytest.approx(15.0)
    assert m.utc_mid_exposure is not None
    assert m.utc_mid_exposure.isoformat().startswith("2018-02-15T18:04:34.660000")
    assert m.observatory == "SAO RAS 6m"
    assert m.site_name == "BTA"
    assert m.seeing_fwhm_arcsec == pytest.approx(1.4)
    assert (m.meta_provenance or {}).get("airmass") == "header:AIRMASS"
    assert (m.meta_provenance or {}).get("zenith_distance_deg") == "derived:airmass"
    assert (m.meta_provenance or {}).get("ra_deg") == "header:RA"
    assert (m.meta_provenance or {}).get("utc_mid_exposure") == "derived:date_time_utc+exptime"


def test_framemeta_calibration_grade_missing_values_are_preserved_as_none():
    hdr = dict(BASE_SC2_SPECTRA_HEADER)

    m = parse_frame_meta(hdr, strict=True)

    assert m.airmass is None
    assert m.zenith_distance_deg is None
    assert m.ra_deg is None
    assert m.dec_deg is None
    assert m.utc_mid_exposure is not None
    assert "airmass" in (m.meta_missing_optional or [])
    assert "ra_deg" in (m.meta_missing_optional or [])
    assert (m.meta_provenance or {}).get("airmass") == "missing"
    assert (m.meta_provenance or {}).get("ra_deg") == "missing"
    assert (m.meta_provenance or {}).get("utc_mid_exposure") == "derived:date_time_utc+exptime"
