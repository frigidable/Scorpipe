import pytest


def _hdr(**kw):
    # Minimal header mapping for calib_compat tests.
    base = {
        "INSTRUME": "SCORPIO-2",
        "DETECTOR": "CCD",
        "MODE": "Spectra",
        "DISPERSE": "VPHG1200@540",
        "CCDBIN1": "1x1",
        "NAXIS1": 2048,
        "NAXIS2": 2048,
        "SLITWID": "1.0",
        "NODE": "A",
        "ROTANGLE": "0.0",
        "SLITPOS": 0.0,
        "GAIN": 1.0,
        "RATE": 185.0,
        "RDNOISE": 3.5,
    }
    base.update(kw)
    return base


def test_rot_mismatch_is_qc_only_and_does_not_raise():
    from scorpio_pipe.calib_compat import compare_compat_headers

    sci = _hdr(ROTANGLE="10.0")
    cal = _hdr(ROTANGLE="45.0")
    flags = []
    meta = compare_compat_headers(sci, cal, kind="flat", strict=True, stage_flags=flags)

    assert meta["must_diffs"] == {}
    assert "rot" in meta["qc_diffs"]
    assert flags
    assert flags[0].get("code") == "CALIB_ROT_MISMATCH"


def test_binning_mismatch_is_fatal_in_strict_mode():
    from scorpio_pipe.calib_compat import CalibrationMismatchError, compare_compat_headers

    sci = _hdr(CCDBIN1="1x1")
    cal = _hdr(CCDBIN1="2x2")
    with pytest.raises(CalibrationMismatchError):
        compare_compat_headers(sci, cal, kind="flat", strict=True)


def test_disperser_mismatch_is_fatal_in_strict_mode():
    from scorpio_pipe.calib_compat import CalibrationMismatchError, compare_compat_headers

    sci = _hdr(DISPERSE="VPHG1200@540")
    cal = _hdr(DISPERSE="VPHG2300@520")
    with pytest.raises(CalibrationMismatchError):
        compare_compat_headers(sci, cal, kind="flat", strict=True)


def test_slitpos_mismatch_is_qc_only_and_flagged():
    from scorpio_pipe.calib_compat import compare_compat_headers

    sci = _hdr(SLITPOS=0.0)
    cal = _hdr(SLITPOS=1.0)
    flags = []
    meta = compare_compat_headers(sci, cal, kind="flat", strict=True, stage_flags=flags)
    assert "slitpos" in meta["qc_diffs"]
    assert any(f.get("code") == "CALIB_SLITPOS_MISMATCH" for f in flags)
