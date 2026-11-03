"""Versioned 1D science-mask, input-provenance, and incomplete-line contracts.

The numerical extraction kernels retain their historical diagnostic mask OR;
the published SPEC1D v1 schema separates provenance from science usability.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.calib.common import read_spec1d
from scorpio_pipe.contracts.validators import validate_spec1d_product
from scorpio_pipe.maskbits import (
    BADPIX, COSMIC, EDGE, INVALID_WAVELENGTH, NO_COVERAGE, REJECTED,
    SKYMODEL_FAIL, USER,
)
from scorpio_pipe.science.lines import LineSpec, measure_line
from scorpio_pipe.stages.extract1d import (
    _boxcar_extract,
    _aperture_input_dq,
    _aperture_fraction_used,
    _science_usable_1d_mask,
    run_extract1d,
)


def test_excluded_cosmic_retains_input_dq_without_invalidating_usable_flux():
    sci = np.full((7, 21), 10., dtype=float)
    var = np.full_like(sci, 4.)
    raw = np.zeros_like(sci, dtype=np.uint16)
    raw[3, 10] = COSMIC
    raw[4, 11] = EDGE
    centers = np.full(21, 3.)
    f, v, legacy, _, npix = _boxcar_extract(
        sci, var, raw, centers,
        ap_hw=2, fatal_bits=int(COSMIC | BADPIX | USER | NO_COVERAGE),
        min_good_frac=.6, sky_resid=None, apply_local_sky=False,
    )
    assert f[10] == pytest.approx(40.)
    assert v[10] == pytest.approx(16.)
    assert npix[10] == 4
    assert legacy[10] & COSMIC
    any_input = _aperture_input_dq(raw, centers, ap_hw=2)
    assert any_input[10] & COSMIC
    assert any_input[11] & EDGE
    usable = _science_usable_1d_mask(legacy)
    assert usable[10] == 0 and usable[11] == 0
    frac = _aperture_fraction_used(npix, ap_hw=2)
    assert frac[10] == pytest.approx(0.8)
    assert frac[11] == pytest.approx(1.)


def test_partial_input_no_coverage_does_not_create_fatal_boxcar_channel():
    sci = np.full((7, 15), 5.)
    var = np.ones_like(sci)
    raw = np.zeros_like(sci, dtype=np.uint16)
    raw[2, 8] = NO_COVERAGE
    _, _, legacy, _, npix = _boxcar_extract(
        sci, var, raw, np.full(15, 3.), ap_hw=1,
        fatal_bits=int(NO_COVERAGE), min_good_frac=0.6,
        sky_resid=None, apply_local_sky=False,
    )
    assert npix[8] == 2
    assert not (legacy[8] & NO_COVERAGE)
    assert _aperture_input_dq(raw, np.full(15, 3.), ap_hw=1)[8] & NO_COVERAGE


def test_no_usable_pixels_and_sky_failure_remain_fatal():
    from scorpio_pipe.maskbits import SATURATED
    mixed = np.asarray([COSMIC, EDGE | BADPIX, NO_COVERAGE, SKYMODEL_FAIL,
                        REJECTED, USER, SATURATED, INVALID_WAVELENGTH,
                        1 << 14], dtype=np.uint16)
    out = _science_usable_1d_mask(mixed)
    np.testing.assert_array_equal(out, [0, 0, NO_COVERAGE, SKYMODEL_FAIL,
                                       0, 0, 0, 0, 1 << 14])
    degraded = _science_usable_1d_mask(np.zeros(3, dtype=np.uint16), upstream_degraded=True)
    assert np.all((degraded & SKYMODEL_FAIL) != 0)


def test_input_dq_is_limited_to_aperture_not_background():
    src = np.zeros((9, 13), dtype=np.uint16)
    src[0, 8] = COSMIC
    src[4, 8] = BADPIX
    raw = _aperture_input_dq(src, np.full(13, 4.), ap_hw=1)
    assert raw[8] == BADPIX
    assert raw[7] == 0


def _write_minimum_2d(path: Path):
    ny, nlam = 32, 128
    spatial = np.exp(-0.5 * ((np.arange(ny) - 16.) / 2.) ** 2)
    sci = (400. * spatial[:, None]) * np.ones((1, nlam))
    var = np.full((ny, nlam), 4., dtype=np.float32)
    mask = np.zeros((ny, nlam), dtype=np.uint16)
    mask[16, 62] = COSMIC
    mask[:, 70] = NO_COVERAGE
    hdr = fits.Header()
    hdr["BUNIT"] = "ADU"
    sh = fits.Header()
    sh["CRVAL1"] = 4950.
    sh["CDELT1"] = 1.
    sh["CRPIX1"] = 1.
    sh["CTYPE1"] = "WAVE"
    sh["CUNIT1"] = "Angstrom"
    fits.HDUList([
        fits.PrimaryHDU(header=hdr),
        fits.ImageHDU(sci.astype("f4"), header=sh, name="SCI"),
        fits.ImageHDU(var, header=sh, name="VAR"),
        fits.ImageHDU(mask, header=sh, name="MASK"),
    ]).writeto(path)
    return mask


def test_published_spec1d_records_separate_provenance_and_validity(tmp_path):
    work = tmp_path / "work"
    raw = tmp_path / "input2d.fits"
    original = _write_minimum_2d(raw)
    cfg = {
        "work_dir": str(work),
        "frames": {"__setup__": {"instrument": "scorpio", "mode": "longslit"}},
        "extract1d": {"input_mode": "single_frame", "method": "boxcar", "save_png": False},
    }
    output = run_extract1d(cfg, in_fits=raw, out_dir=work / "extract")
    specpath = output["spec1d_fits"]
    with fits.open(specpath) as hdul:
        assert hdul[0].header["DQ1DVER"] == 1
        assert hdul[0].header["DQ1DSEM"] == "FATAL"
        for mode in ("TRACE", "FIXED"):
            tab = hdul[f"SPEC_{mode}"].data
            assert f"DQ_ANY_{mode}" in tab.names
            assert f"FRAC_{mode}" in tab.names
            assert (np.asarray(tab[f"FRAC_{mode}"]) >= 0).all()
            assert (np.asarray(tab[f"FRAC_{mode}"]) <= 1).all()
            assert (int(tab[f"MASK_{mode}"][70]) & int(NO_COVERAGE))
        # A masked cosmic inside the aperture is retained as provenance
        # while the output channel remains usable after exclusion.
        for mode in ("TRACE", "FIXED"):
            tab = hdul[f"SPEC_{mode}"].data
            assert int(tab[f"DQ_ANY_{mode}"][62]) & int(COSMIC)
            assert not (int(tab[f"MASK_{mode}"][62]) & int(COSMIC))
        assert "DQANY2" in hdul and "FRAC2" in hdul
    read = read_spec1d(specpath, mode="trace")
    assert read["dq1d_version"] == 1
    assert read["dq_any"][62] & COSMIC
    assert read["frac_used"][62] < 1
    np.testing.assert_array_equal(fits.getdata(raw, "MASK"), original)
    validate_spec1d_product(specpath, stage="extract1d")


def test_versioned_reader_fails_closed_when_provenance_columns_are_missing(tmp_path):
    path = tmp_path / "broken.fits"
    hdr = fits.Header()
    hdr["DQ1DVER"] = 1
    hdr["DQ1DSEM"] = "FATAL"
    fits.HDUList([
        fits.PrimaryHDU(header=hdr),
        fits.BinTableHDU.from_columns([
            fits.Column(name="LAMBDA", format="D", array=np.arange(10.)),
            fits.Column(name="FLUX_TRACE", format="E", array=np.ones(10)),
            fits.Column(name="VAR_TRACE", format="E", array=np.ones(10)),
            fits.Column(name="MASK_TRACE", format="J", array=np.zeros(10, dtype=int)),
        ], name="SPEC_TRACE"),
    ]).writeto(path)
    with pytest.raises(ValueError, match="DQ1D schema requires"):
        read_spec1d(path)


def test_line_measure_partial_aperture_is_not_full_flux_detection():
    wave = np.arange(4950., 5061.)
    profile = 35. * np.exp(-0.5 * ((wave - 5007.) / 2.) ** 2)
    flux = 5. + profile
    var = np.ones_like(wave)
    spec = LineSpec("Synthetic [OIII]", 5006.84, 10., (-35., -15.), (15., 35.))
    kwargs = {
        "calsys_rel": None, "calsys_tell": None,
        "calsys_abs": None, "spec": spec, "waveref": "air",
    }
    full = measure_line(wave, flux, var, coverage_fraction=np.ones_like(wave), **kwargs)
    assert full["detected"] is True and full["coverage_complete"] is True
    partial = np.ones_like(wave)
    partial[wave == 5007] = .8
    result = measure_line(wave, flux, var, coverage_fraction=partial, **kwargs)
    assert result["snr_stat"] > 3
    assert result["detected"] is False
    assert result["reason"] == "partial_aperture_or_spectral_coverage"
    assert result["flux_kind"] == "partial_window_integral_unqualified"
    assert result["upper_limit_3sigma"] is None
    legacy = measure_line(wave, flux, var, **kwargs)
    assert legacy["detected"] is True
    assert legacy["coverage_complete"] is None


def test_line_measure_rejects_invalid_fraction_used():
    wave = np.arange(4950., 5061.)
    spec = LineSpec("test", 5006.84, 10., (-35., -15.), (15., 35.))
    with pytest.raises(ValueError, match="FRAC_USED"):
        measure_line(
            wave, np.ones(wave.size), np.ones(wave.size),
            calsys_rel=None, calsys_tell=None, calsys_abs=None,
            spec=spec, coverage_fraction=np.full(wave.size, np.nan),
        )
