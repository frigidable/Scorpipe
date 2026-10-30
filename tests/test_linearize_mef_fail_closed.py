"""Linearize must never discard a corrupt upstream SCI/VAR/MASK MEF.

The fixture deliberately puts apparently usable data in PRIMARY: before the
fix this silently defeated upstream VAR and MASK validation.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.maskbits import NO_COVERAGE, SKYMODEL_FAIL
from scorpio_pipe.stages.linearize import _open_science_with_optional_var_mask


def _write(tmp_path, *, var=None, mask=None, primary_data=True):
    shape = (4, 12)
    hdr = fits.Header()
    hdr["BUNIT"] = "e-"
    hdr["GAIN"] = 2.0
    hdr["RDNOISE"] = 3.0
    hdr["EXPTIME"] = 600.0
    sci = np.full(shape, 7.0, dtype=np.float32)
    phdu = fits.PrimaryHDU(
        data=np.full(shape, 999.0, dtype=np.float32) if primary_data else None,
        header=hdr,
    )
    hdus = [phdu, fits.ImageHDU(data=sci, name="SCI")]
    if var is not None:
        hdus.append(fits.ImageHDU(data=var, name="VAR"))
    if mask is not None:
        hdus.append(fits.ImageHDU(data=mask, name="MASK"))
    path = tmp_path / "sky_input.fits"
    fits.HDUList(hdus).writeto(path)
    return path


def test_valid_mef_prefers_sci_and_preserves_upstream_var_mask(tmp_path):
    var = np.full((4, 12), 13.0, dtype=np.float32)
    mask = np.zeros((4, 12), dtype=np.uint16)
    mask[1, 2] = SKYMODEL_FAIL
    path = _write(tmp_path, var=var, mask=mask)
    sci, got_var, got_mask, hdr = _open_science_with_optional_var_mask(path)
    np.testing.assert_allclose(sci, 7.0)
    np.testing.assert_array_equal(got_var, var)
    np.testing.assert_array_equal(got_mask, mask)
    assert hdr["BUNIT"] == "e-"


@pytest.mark.parametrize("defect", ["wrong_var_shape", "nan_var", "negative_var", "wrong_mask_shape"])
def test_corrupt_mef_fails_closed_even_with_readable_primary(tmp_path, defect):
    var = np.ones((4, 12), dtype=np.float32)
    mask = np.zeros((4, 12), dtype=np.uint16)
    if defect == "wrong_var_shape":
        var = np.ones((4, 11), dtype=np.float32)
    if defect == "nan_var":
        var[0, 3] = np.nan
    if defect == "negative_var":
        var[0, 3] = -1.0
    if defect == "wrong_mask_shape":
        mask = np.zeros((3, 12), dtype=np.uint16)
    path = _write(tmp_path, var=var, mask=mask)
    with pytest.raises(ValueError, match="Invalid SCI/VAR/MASK MEF") as exc:
        _open_science_with_optional_var_mask(path)
    assert str(path) in str(exc.value)
    assert exc.value.__cause__ is not None
    assert "MEF contract violation" in str(exc.value.__cause__)


def test_masked_bad_variance_is_valid_in_mef_contract(tmp_path):
    var = np.ones((4, 12), dtype=np.float32)
    var[0, 2] = np.nan
    mask = np.zeros((4, 12), dtype=np.uint16)
    mask[0, 2] = NO_COVERAGE
    path = _write(tmp_path, var=var, mask=mask)
    _sci, got_var, got_mask, _hdr = _open_science_with_optional_var_mask(path)
    assert np.isnan(got_var[0, 2])
    assert got_mask[0, 2] & NO_COVERAGE


def test_legacy_primary_only_raw_image_still_supported(tmp_path):
    path = tmp_path / "raw.fits"
    hdr = fits.Header()
    hdr["BUNIT"] = "ADU"
    data = np.arange(48, dtype=np.float32).reshape(4, 12)
    fits.PrimaryHDU(data=data, header=hdr).writeto(path)
    sci, var, mask, out_header = _open_science_with_optional_var_mask(path)
    np.testing.assert_array_equal(sci, data)
    assert var is None and mask is None
    assert out_header["BUNIT"] == "ADU"


def test_mef_with_empty_primary_still_reads_sci(tmp_path):
    path = _write(tmp_path, var=np.ones((4, 12), dtype=np.float32),
                  mask=np.zeros((4, 12), dtype=np.uint16),
                  primary_data=False)
    sci, var, mask, _ = _open_science_with_optional_var_mask(path)
    assert sci.shape == var.shape == mask.shape == (4, 12)
