"""SCORPIO dedicated SKY templates must be bias-calibrated, not native RAW."""
from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.stages.sky_sub import _require_calibrated_scorpio_sky_template


def _write_source(tmp_path, *, instrument="SCORPIO", detector="E2V CCD4240",
                  biassub=None, mef=False):
    p = tmp_path / ("sky_clean.fits" if mef else "s22800604.fts")
    hdr = fits.Header()
    if instrument is not None:
        hdr["INSTRUME"] = instrument
    if detector is not None:
        hdr["DETECTOR"] = detector
    if biassub is not None:
        hdr["BIASSUB"] = biassub
    if mef:
        hdus = fits.HDUList([
            fits.PrimaryHDU(header=hdr),
            fits.ImageHDU(data=np.ones((8, 8), dtype=np.float32), name="SCI"),
        ])
    else:
        hdus = fits.HDUList([
            fits.PrimaryHDU(data=np.ones((8, 8), dtype=np.uint16), header=hdr)
        ])
    hdus.writeto(p)
    return p


@pytest.mark.parametrize("biassub", [None, False, "F", "FALSE", 0])
def test_scorpio_raw_sky_never_accepted_as_scale_template(tmp_path, biassub):
    source = _write_source(tmp_path, biassub=biassub)
    with pytest.raises(RuntimeError, match="Uncalibrated SCORPIO SKY template"):
        _require_calibrated_scorpio_sky_template(source)


@pytest.mark.parametrize("biassub", [True, "T", "TRUE"])
def test_verified_bias_subtracted_scorpio_sky_can_pass_prerequisite(
    tmp_path, biassub,
):
    source = _write_source(tmp_path, biassub=biassub, mef=True)
    _require_calibrated_scorpio_sky_template(source)


def test_detector_id_identifies_scorpio_even_without_instrument(tmp_path):
    source = _write_source(
        tmp_path, instrument=None, detector="E2V CCD4240", biassub=False,
    )
    with pytest.raises(RuntimeError, match="BIASSUB=True is required"):
        _require_calibrated_scorpio_sky_template(source)


def test_nonscorpio_legacy_sky_is_not_reinterpreted(tmp_path):
    source = _write_source(
        tmp_path, instrument="OtherSpectrograph", detector="Other CCD",
        biassub=None,
    )
    _require_calibrated_scorpio_sky_template(source)


def test_biassub_on_sci_only_is_not_trusted_upstream_provenance(tmp_path):
    source = _write_source(tmp_path, biassub=None, mef=True)
    with fits.open(source, mode="update") as hdu:
        hdu["SCI"].header["BIASSUB"] = True
    with pytest.raises(RuntimeError, match="BIASSUB=True is required"):
        _require_calibrated_scorpio_sky_template(source)
