"""Regression coverage for the SCI/VAR/MASK handoff into RAW sky subtraction."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.maskbits import BADPIX, COSMIC, NO_COVERAGE
from scorpio_pipe.stages.sky_sub import (
    _prepare_sky_input_planes,
    _read_sky_input_planes,
)


def _header(*, bunit: str = "ADU") -> fits.Header:
    return fits.Header({"BUNIT": bunit, "GAIN": 2.0, "RDNOISE": 3.0})


def test_sky_reader_preserves_mef_planes_and_primary_provenance(tmp_path) -> None:
    path = tmp_path / "flat.fits"
    hdr = _header(bunit="e-")
    hdr["FLATCOR"] = True
    sci = np.full((3, 4), 7.0, dtype=np.float32)
    var = np.full((3, 4), 13.0, dtype=np.float32)
    mask = np.zeros((3, 4), dtype=np.uint16)
    mask[1, 2] = BADPIX
    fits.HDUList([
        fits.PrimaryHDU(header=hdr),
        fits.ImageHDU(data=sci, name="SCI"),
        fits.ImageHDU(data=var, name="VAR"),
        fits.ImageHDU(data=mask, name="MASK"),
    ]).writeto(path)

    got_sci, got_var, got_mask, got_hdr = _read_sky_input_planes(path)

    np.testing.assert_array_equal(got_sci, sci)
    np.testing.assert_array_equal(got_var, var)
    np.testing.assert_array_equal(got_mask, mask)
    assert got_hdr["FLATCOR"]
    assert got_hdr["BUNIT"] == "e-"


def test_sky_reader_accepts_legacy_primary_only_frame(tmp_path) -> None:
    path = tmp_path / "raw.fits"
    sci = np.full((2, 3), 11.0, dtype=np.float32)
    fits.PrimaryHDU(data=sci, header=_header()).writeto(path)

    got_sci, got_var, got_mask, got_hdr = _read_sky_input_planes(path)

    np.testing.assert_array_equal(got_sci, sci)
    assert got_var is None and got_mask is None
    assert got_hdr["BUNIT"] == "ADU"


def test_upstream_variance_is_preserved_without_noise_reestimation(monkeypatch) -> None:
    import scorpio_pipe.noise_model as noise_model

    def forbidden_estimate(*args, **kwargs):
        raise AssertionError("VAR exists: Poisson/RN estimate must not be used")

    monkeypatch.setattr(noise_model, "estimate_variance_auto", forbidden_estimate)
    sci = np.full((3, 4), 10.0)
    upstream_var = np.full((3, 4), 7.0)
    upstream_mask = np.zeros((3, 4), dtype=np.uint16)
    upstream_mask[0, 0] = BADPIX
    cosmic_mask = np.zeros((3, 4), dtype=np.uint16)
    cosmic_mask[0, 0] = COSMIC
    cosmic_mask[1, 1] = COSMIC

    sci_e, var_e, mask, hdr = _prepare_sky_input_planes(
        sci, upstream_var, upstream_mask, cosmic_mask,
        _header(), cfg={}, sky_cfg={},
    )

    np.testing.assert_allclose(sci_e, sci * 2.0, rtol=0, atol=0)
    np.testing.assert_allclose(var_e, upstream_var * 4.0, rtol=0, atol=0)
    assert mask[0, 0] == (BADPIX | COSMIC)
    assert mask[1, 1] == COSMIC
    assert hdr["SKYIVAR"] == "INPUT"
    assert hdr["SKYIMSK"] == "BOTH"
    assert hdr["BUNIT"] == "e-"


def test_var_estimation_only_when_plane_is_missing() -> None:
    sci = np.full((2, 3), 10.0)
    sci_e, var_e, mask, hdr = _prepare_sky_input_planes(
        sci, None, None, None, _header(), cfg={}, sky_cfg={},
    )
    np.testing.assert_allclose(sci_e, 20.0)
    # Variance in ADU^2 is 10/2 + (3/2)^2 = 7.25;
    # conversion to e-^2 multiplies by gain^2.
    np.testing.assert_allclose(var_e, 29.0)
    assert np.count_nonzero(mask) == 0
    assert hdr["SKYIVAR"] == "ESTIM"
    assert hdr["SKYIMSK"] == "NONE"


def test_electron_input_variance_is_not_gain_scaled() -> None:
    sci_e, var_e, _, hdr = _prepare_sky_input_planes(
        np.full((2, 2), 10.0),
        np.full((2, 2), 27.0),
        None, None, _header(bunit="e-"), cfg={}, sky_cfg={},
    )
    np.testing.assert_allclose(sci_e, 10.0)
    np.testing.assert_allclose(var_e, 27.0)
    assert hdr["SKYIVAR"] == "INPUT"


@pytest.mark.parametrize("plane", ["VAR", "MASK", "COSMIC"])
def test_mismatched_input_shapes_fail_closed(plane: str) -> None:
    sci = np.ones((3, 4))
    var = np.ones((3, 4))
    mask = np.zeros((3, 4), dtype=np.uint16)
    cosmic = mask.copy()
    if plane == "VAR":
        var = np.ones((3, 3))
    elif plane == "MASK":
        mask = np.zeros((3, 3), dtype=np.uint16)
    else:
        cosmic = np.zeros((3, 3), dtype=np.uint16)
    with pytest.raises(ValueError, match="shape"):
        _prepare_sky_input_planes(
            sci, var, mask, cosmic, _header(), cfg={}, sky_cfg={},
        )


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_bad_variance_on_unmasked_pixel_is_rejected(value: float) -> None:
    var = np.ones((3, 4))
    var[0, 1] = value
    with pytest.raises(ValueError, match="SCI/VAR/MASK"):
        _prepare_sky_input_planes(
            np.ones((3, 4)), var, None, None,
            _header(), cfg={}, sky_cfg={},
        )


def test_bad_mask_value_rejected_before_uint16_cast() -> None:
    with pytest.raises(ValueError, match="uint16"):
        _prepare_sky_input_planes(
            np.ones((2, 2)), np.ones((2, 2)),
            np.array([[-1, 0], [0, 0]]), None,
            _header(), cfg={}, sky_cfg={},
        )


def test_bad_var_allowed_only_when_fatally_masked() -> None:
    var = np.ones((2, 2))
    var[0, 0] = np.nan
    mask = np.zeros((2, 2), dtype=np.uint16)
    mask[0, 0] = NO_COVERAGE
    _, got_var, got_mask, _ = _prepare_sky_input_planes(
        np.ones((2, 2)), var, mask, None, _header(), cfg={}, sky_cfg={},
    )
    assert np.isnan(got_var[0, 0])
    assert got_mask[0, 0] & NO_COVERAGE
