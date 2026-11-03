"""Real FlatField FITS component lineage and SkySub schema downgrade.

The current SkySub does NOT propagate a fitted sky model's derivative;
the complete pre-sky component planes must therefore not be advertised
on the downstream sky-subtracted product.
"""
from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.stages import calib, flatfield
from scorpio_pipe.stages.calib import BiasSelection
from scorpio_pipe.stages.sky_sub import _prepare_sky_input_planes, _read_sky_input_planes


def _flat_with_indexed_bias(tmp_path, monkeypatch):
    shape = (6, 8)
    hdr = fits.Header()
    hdr["BUNIT"] = "ADU"
    hdr["GAIN"] = 2.
    hdr["RDNOISE"] = 4.
    hdr["EXPTIME"] = 100.
    hdr["BIASSUB"] = False
    raw = tmp_path / "object.fits"
    fits.HDUList([
        fits.PrimaryHDU(data=np.full(shape, 110., dtype="f4"), header=hdr),
        fits.ImageHDU(data=np.full(shape, 9., dtype="f4"), name="VAR"),
        fits.ImageHDU(data=np.zeros(shape, dtype=np.uint16), name="MASK"),
    ]).writeto(raw)
    bias_path = tmp_path / "master_bias.fits"
    bias_var_path = tmp_path / "master_bias_var.fits"
    flat_path = tmp_path / "flat.fits"
    fits.writeto(bias_path, np.full(shape, 10., dtype="f4"))
    fits.writeto(bias_var_path, np.full(shape, 4., dtype="f4"))
    fits.writeto(flat_path, np.full(shape, 2., dtype="f4"))
    selection = BiasSelection(
        sci_path=bias_path, var_path=bias_var_path, dq_path=None,
        gid="readout_group_1", degraded=False, reason="",
        gain_e_per_adu=2., rdnoise_e=4.,
    )
    monkeypatch.setattr(
        calib, "resolve_master_bias", lambda cfg, hdr, **kwargs: selection
    )
    monkeypatch.setattr(
        flatfield, "ensure_compatible_calib", lambda *a, **kw: None
    )
    dest = tmp_path / "flat_output.fits"
    path, chosen = flatfield.apply_flat(
        raw, flat_path, dest, cfg={}, do_bias_subtract=True
    )
    assert chosen.gid == "readout_group_1"
    return path


def test_real_flatfield_writes_closed_bias_variance_components(tmp_path, monkeypatch):
    out = _flat_with_indexed_bias(tmp_path, monkeypatch)
    with fits.open(out) as hdul:
        assert hdul[0].header["VCMPVER"] == 1
        assert hdul[0].header["VCMPQ"] == "LOCAL"
        assert hdul[0].header["VCMPGID"] == "readout_group_1"
        assert hdul[0].header["SB_GID"] == "readout_group_1"
        assert hdul[0].header["BVARINC"] is True
        assert {"VBIAS", "VREST", "BSENS", "BVARADU"}.issubset(hdul)
        np.testing.assert_allclose(hdul["SCI"].data, 100., rtol=0, atol=1e-5)
        np.testing.assert_allclose(hdul["VAR"].data, 13., rtol=0, atol=1e-5)
        np.testing.assert_allclose(hdul["VBIAS"].data, 4., rtol=0, atol=1e-5)
        np.testing.assert_allclose(hdul["VREST"].data, 9., rtol=0, atol=1e-5)
        np.testing.assert_allclose(hdul["BSENS"].data, -1., rtol=0, atol=1e-5)
        np.testing.assert_allclose(hdul["BVARADU"].data, 4., rtol=0, atol=1e-5)
        assert hdul["VBIAS"].header["BUNIT"] == "e-^2"
        assert hdul["BSENS"].header["BUNIT"] == "e-/ADU"
        assert hdul["BVARADU"].header["BUNIT"] == "ADU^2"


def test_sky_reader_checks_flat_components_then_downgrades_output(tmp_path, monkeypatch):
    path = _flat_with_indexed_bias(tmp_path, monkeypatch)
    sci, var, mask, hdr = _read_sky_input_planes(path)
    assert hdr["VCMPVER"] == 1
    new_sci, new_var, new_mask, new_hdr = _prepare_sky_input_planes(
        sci, var, mask, None, hdr, cfg={}, sky_cfg={}
    )
    np.testing.assert_array_equal(new_var, var)
    np.testing.assert_array_equal(new_sci, sci)
    assert "VCMPVER" not in new_hdr
    assert "VCMPGID" not in new_hdr
    assert new_hdr["VCMPQ"] == "SKY_LOST"
    assert new_hdr["SB_GID"] == "readout_group_1"
    assert new_hdr["BVARINC"] is True


@pytest.mark.parametrize("damage", ["missing_plane", "wrong_shape", "broken_sum", "wrong_group"])
def test_sky_rejects_incorrect_versioned_flat_components(tmp_path, monkeypatch, damage):
    source = _flat_with_indexed_bias(tmp_path, monkeypatch)
    bad = tmp_path / "bad_components.fits"
    with fits.open(source, memmap=False) as hdul:
        if damage == "missing_plane":
            hdul.pop(hdul.index_of("BSENS"))
        elif damage == "wrong_shape":
            hdul["VREST"].data = np.ones((1, 2), dtype=np.float32)
        elif damage == "broken_sum":
            hdul["VREST"].data[0, 0] += 5.
        elif damage == "wrong_group":
            hdul[0].header["VCMPGID"] = "wrong"
        hdul.writeto(bad)
    with pytest.raises(ValueError):
        _read_sky_input_planes(bad)


def test_flatfield_without_source_bias_map_does_not_publish_fake_components(
    tmp_path, monkeypatch
):
    hdr = fits.Header()
    hdr["BUNIT"] = "ADU"
    hdr["GAIN"] = 2.
    hdr["RDNOISE"] = 4.
    hdr["BIASSUB"] = True
    raw = tmp_path / "already_bias_subtracted.fits"
    flat = tmp_path / "flat.fits"
    fits.HDUList([
        fits.PrimaryHDU(data=np.full((4, 6), 100., dtype="f4"), header=hdr),
        fits.ImageHDU(data=np.full((4, 6), 9., dtype="f4"), name="VAR"),
    ]).writeto(raw)
    fits.writeto(flat, np.ones((4, 6), dtype="f4"))
    monkeypatch.setattr(
        flatfield, "ensure_compatible_calib", lambda *a, **kw: None
    )
    out = tmp_path / "already_calibrated_flat.fits"
    flatfield.apply_flat(raw, flat, out, cfg={}, do_bias_subtract=False)
    with fits.open(out) as hdul:
        assert "VCMPVER" not in hdul[0].header
        assert "VBIAS" not in hdul
        assert "VREST" not in hdul
        assert "BSENS" not in hdul
        assert "BVARADU" not in hdul
