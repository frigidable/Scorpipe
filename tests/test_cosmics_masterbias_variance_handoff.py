"""Indexed MasterBias VAR survives primary-only Cosmics CLEAN to Sky/Flatfield."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from scorpio_pipe.cosmics_biasvar import (
    read_clean_biasvar_sidecar,
    validate_bias_var_extra,
    write_clean_biasvar_sidecar,
)
from scorpio_pipe.stages import flatfield
from scorpio_pipe.stages.sky_sub import (
    _prepare_sky_input_planes,
    _read_sky_input_planes,
)


def _clean(tmp_path: Path) -> Path:
    path = tmp_path / "obj_01_clean.fits"
    header = fits.Header()
    header["BIASSUB"] = True
    header["SB_GID"] = "exact_ccd_readout_1"
    header["BUNIT"] = "ADU"
    header["GAIN"] = 0.62
    header["RDNOISE"] = 3.0
    header["EXPTIME"] = 300.0
    header["BVARINC"] = False
    header["BVARREF"] = "obj_01_clean_biasvar.fits"
    fits.writeto(path, np.full((16, 24), 120.0, dtype="f4"), header)
    write_clean_biasvar_sidecar(
        path, var_adu2=np.full((16, 24), 4.0),
        group_id="exact_ccd_readout_1",
    )
    return path


def test_cosmics_bias_variance_sidecar_consumed_once_by_flatfield(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean = _clean(tmp_path)
    sf = tmp_path / "superflat.fits"
    fits.writeto(sf, np.ones((16, 24), dtype="f4"))
    monkeypatch.setattr(flatfield, "ensure_compatible_calib", lambda *a, **kw: None)
    dst = tmp_path / "obj_01_flat.fits"
    output, selection = flatfield.apply_flat(
        clean, sf, dst, cfg={}, do_bias_subtract=True,
    )
    assert output == dst
    assert selection is None  # Must not reselect/reapply any MasterBias.
    with fits.open(dst) as hdul:
        h = hdul[0].header
        assert h["BVARINC"] is True
        assert "BVARREF" not in h  # The referenced sidecar is in another dir.
        assert h["SB_GID"] == "exact_ccd_readout_1"
        assert np.allclose(hdul["SCI"].data, 120.0 * 0.62)
        expected_var_e2 = 120 * 0.62 + 3.0**2 + 4.0 * 0.62**2
        assert np.allclose(hdul["VAR"].data, expected_var_e2, atol=1e-4)


def test_cosmics_bias_variance_sidecar_consumed_once_by_direct_sky(
    tmp_path: Path,
) -> None:
    clean = _clean(tmp_path)
    sci, var, mask, hdr = _read_sky_input_planes(clean)
    assert var is None
    bvar = read_clean_biasvar_sidecar(clean, hdr, sci.shape)
    out_sci, out_var, out_mask, out_hdr = _prepare_sky_input_planes(
        sci, var, mask, None, hdr,
        cfg={}, sky_cfg={}, input_bias_var=bvar,
    )
    assert out_hdr["SKYIVAR"] == "ESTIM+BV"
    assert out_hdr["BVARINC"] is True
    assert "BVARREF" not in out_hdr
    assert np.allclose(out_sci, 120 * 0.62, atol=1e-4)
    assert np.allclose(
        out_var, 120 * 0.62 + 3.0**2 + 4.0 * 0.62**2, atol=1e-4
    )
    assert out_mask.dtype == np.uint16
    with pytest.raises(ValueError, match="missing declared"):
        _prepare_sky_input_planes(sci, var, mask, None, hdr, cfg={}, sky_cfg={})


@pytest.mark.parametrize(
    "damage",
    [
        "missing", "wrong_gid", "wrong_shape", "wrong_unit", "negative",
        "nan", "wrong_ref", "not_biassub", "already_included",
    ],
)
def test_cosmics_bias_variance_handoff_fails_closed(
    tmp_path: Path, damage: str,
) -> None:
    clean = _clean(tmp_path)
    with fits.open(clean, memmap=False) as hdul:
        hdr = hdul[0].header.copy()
        shape = hdul[0].data.shape
    sidecar = clean.with_name(hdr["BVARREF"])
    if damage == "missing":
        sidecar.unlink()
    elif damage == "wrong_ref":
        hdr["BVARREF"] = "../arbitrary_calibration.fits"
    elif damage == "not_biassub":
        hdr["BIASSUB"] = False
    elif damage == "already_included":
        hdr["BVARINC"] = True
    else:
        with fits.open(sidecar, mode="update") as hdul:
            if damage == "wrong_gid":
                hdul[0].header["SB_GID"] = "different"
            elif damage == "wrong_unit":
                hdul[0].header["BUNIT"] = "e-^2"
            elif damage == "wrong_shape":
                hdul[0].data = np.ones((2, 3), dtype="f4")
            elif damage == "negative":
                hdul[0].data[0, 0] = -1.0
            elif damage == "nan":
                hdul[0].data[0, 0] = np.nan
            hdul.flush()
    with pytest.raises((ValueError, OSError)):
        read_clean_biasvar_sidecar(clean, hdr, shape)


def test_biasvar_combiner_rejects_double_counting_and_unphysical_maps() -> None:
    with pytest.raises(ValueError, match="geometry"):
        validate_bias_var_extra(np.ones((3, 4)), np.ones((2, 4)), (3, 4))
    with pytest.raises(ValueError, match="unphysical"):
        validate_bias_var_extra(
            np.ones((3, 4)), np.full((3, 4), -1.0), (3, 4)
        )
