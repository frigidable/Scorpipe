"""Detector saturation provenance: real raw ADC data, never a sky residual."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.detector_saturation import carry_detector_saturation
from scorpio_pipe.maskbits import NO_COVERAGE, SATURATED
from scorpio_pipe.stages.linearize import _build_linearize_grid_plan, _process_linearize_frame
from scorpio_pipe.stages.sky_sub import _prepare_sky_input_planes


def _header(*, unit="ADU", gain=2.0, processed=False):
    hdr = fits.Header()
    hdr["BUNIT"] = unit
    hdr["GAIN"] = gain
    hdr["RDNOISE"] = 3.0
    hdr["EXPTIME"] = 600.0
    if processed:
        hdr["SKYMD"] = "kelson_raw"
    return hdr


@pytest.mark.parametrize("gain", [0.9, 2.0])
def test_raw_adu_threshold_is_independent_of_electron_gain(gain):
    counts = np.array([[100., 490., 495., 501.]], dtype=float)
    hdr = _header(gain=gain)
    mask, out_hdr = carry_detector_saturation(
        counts, None, hdr,
        {"detector": {"saturation_adu": 500., "saturation_margin_adu": 5.}},
    )
    np.testing.assert_array_equal(
        (mask & SATURATED) != 0, [[False, False, True, True]],
    )
    assert out_hdr["SATQ"] == "RAW_EVAL"
    assert out_hdr["SATADU"] == 500.
    assert out_hdr["SATMARG"] == 5.
    assert hdr.get("SATQ") is None


def test_ambiguous_header_saturation_unit_is_not_assumed_adu():
    data = np.array([[65000., 65535.]])
    hdr = _header()
    hdr["SATURATE"] = 65000.   # may mean electron full well
    mask, out_hdr = carry_detector_saturation(data, None, hdr, {})
    assert not np.any(mask)
    assert out_hdr["SATQ"] == "UNKNOWN"
    hdr["SATUNIT"] = "ADU"
    mask, out_hdr = carry_detector_saturation(data, None, hdr, {})
    assert mask[0, 0] & SATURATED
    assert out_hdr["SATQ"] == "RAW_EVAL"


def test_existing_saturation_bits_survive_on_processed_electron_frame():
    hdr = _header(unit="e-", processed=True)
    hdr["SATURATE"] = 100.
    data = np.array([[15000., 0.]])
    original = np.array([[0, SATURATED]], dtype=np.uint16)
    mask, h = carry_detector_saturation(
        data, original, hdr, {"linearize": {"saturation_adu": 500.}},
    )
    assert not (mask[0, 0] & SATURATED)
    assert mask[0, 1] & SATURATED
    assert h["SATQ"] == "PROP"


def test_raw_detection_persists_after_electron_conversion_and_sky_subtraction():
    raw = np.full((3, 12), 100., dtype=float)
    raw[1, 3] = 500.
    sci_e, var_e, mask, hdr = _prepare_sky_input_planes(
        raw, None, None, None, _header(),
        cfg={"linearize": {"saturation_adu": 499.}},
        sky_cfg={},
    )
    assert mask[1, 3] & SATURATED
    assert hdr["SATQ"] == "RAW_EVAL"
    assert hdr["BUNIT"] == "e-"
    assert sci_e[1, 3] == pytest.approx(1000.)
    assert var_e[1, 3] > 0

    residual = sci_e - 990.0
    carried, second = carry_detector_saturation(
        residual, mask, hdr, {"linearize": {"saturation_adu": 499.}},
    )
    assert carried[1, 3] & SATURATED
    assert second["SATQ"] == "RAW_EVAL"


def _linearized(tmp_path, *, with_saturation_bit, level):
    ny, nx = 4, 12
    wave = 5000. + np.arange(nx, dtype=float)
    lam_map = np.broadcast_to(wave, (ny, nx)).copy()
    data = np.full((ny, nx), 10., dtype="f4")
    data[0, 3] = level
    var = np.full((ny, nx), 4., dtype="f4")
    mask = np.zeros((ny, nx), dtype="u2")
    if with_saturation_bit:
        mask[0, 3] = SATURATED
    hdr = _header(unit="e-", processed=True)
    hdr["SATURATE"] = 20.  # ADU: invalid after sky subtraction
    path = tmp_path / "sky_subtracted_mef.fits"
    fits.HDUList([
        fits.PrimaryHDU(header=hdr),
        fits.ImageHDU(data=data, name="SCI"),
        fits.ImageHDU(data=var, name="VAR"),
        fits.ImageHDU(data=mask, name="MASK"),
    ]).writeto(path)
    task = {
        "path": str(path), "lam_map": lam_map,
        "linearize_grid_plan": _build_linearize_grid_plan(
            lam_map, 4999.5 + np.arange(nx + 1, dtype=float)
        ),
        "ny": ny, "nlam": nx, "y0": 0, "y1": ny,
        "mask_dir": str(tmp_path / "missing_mask_dir"),
        "lcfg": {"post_sky_cleanup": "off", "saturation_adu": 20.},
        "cfg": {},
        "gain_override_f": None, "rdnoise_override_f": None,
        "scale_to_bin": False, "normalize_exptime": False,
        "wave0": 4999.5, "dw": 1., "unit": "Angstrom",
    }
    return _process_linearize_frame(task)


def test_linearize_does_not_reclassify_high_sky_residual_as_saturated(tmp_path):
    output = _linearized(tmp_path, with_saturation_bit=False, level=10000.)
    assert not (output["rect_mask"][0, 3] & SATURATED)
    assert output["hdr"]["SATQ"] == "UNKNOWN"
    assert output["rect"][0, 3] == pytest.approx(10000.)


def test_linearize_preserves_upstream_saturation_even_for_low_residual(tmp_path):
    output = _linearized(tmp_path, with_saturation_bit=True, level=1.)
    assert output["rect_mask"][0, 3] & SATURATED
    assert output["rect_mask"][0, 3] & NO_COVERAGE
    assert output["rect"][0, 3] == 0.
    assert output["rect_var"][0, 3] == 0.
    assert output["rect_cov"][0, 3] == 0.
    assert output["hdr"]["SATQ"] == "PROP"


@pytest.mark.parametrize("bad_margin", [-1., float("nan"), 500.])
def test_bad_saturation_margin_refused_on_raw_input(bad_margin):
    with pytest.raises(ValueError, match="saturation_margin_adu"):
        carry_detector_saturation(
            np.ones((3, 4)), None, _header(),
            {"detector": {"saturation_adu": 500., "saturation_margin_adu": bad_margin}},
        )
