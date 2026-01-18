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
