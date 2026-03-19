from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scorpio_pipe.contracts.calibration import CalibrationStatus, stamp_calibration_semantics
from scorpio_pipe.contracts.data_model import stamp_data_model
from scorpio_pipe.contracts.maskbits import header_cards

fits = pytest.importorskip("astropy.io.fits")


def _stamp_mask_schema(hdr):
    for k, v in header_cards().items():
        hdr[k] = v
    return hdr


def _write_fluxcal_mef(path: Path, *, fluxunit: str = "erg/s/cm2/Angstrom") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr0 = fits.Header()
    hdr0["SCORPGN"] = 1.0
    hdr0["SCORPRN"] = 3.0
    hdr0["SCORPNS"] = "header"
    hdr0["CTYPE1"] = "AWAV"
    hdr0["WAVEREF"] = "air"
    hdr0["WAVEUNIT"] = "Angstrom"
    hdr0["SCORP_L0"] = 4000.0
    hdr0["SCORP_DL"] = 1.0
    hdr0["SCORP_NL"] = 4
    stamp_data_model(hdr0)
    stamp_calibration_semantics(
        hdr0,
        status=CalibrationStatus.ABSOLUTE_TRUSTED.value,
        flux_unit=fluxunit,
        reference_context_id="ctx123",
        sensfunc_ref="sensfunc_v1",
        telluric_ref="telluric_v1",
    )
    sci_hdr = fits.Header(); sci_hdr["BUNIT"] = "erg/s/cm2/Angstrom"
    var_hdr = fits.Header(); var_hdr["BUNIT"] = "(erg/s/cm2/Angstrom)^2"
    _stamp_mask_schema(hdr0)
    hdul = fits.HDUList([
        fits.PrimaryHDU(header=hdr0),
        fits.ImageHDU(data=np.ones((4,4), dtype=np.float32), header=sci_hdr, name="SCI"),
        fits.ImageHDU(data=np.ones((4,4), dtype=np.float32)*0.1, header=var_hdr, name="VAR"),
        fits.ImageHDU(data=np.zeros((4,4), dtype=np.uint16), name="MASK"),
    ])
    hdul.writeto(path, overwrite=True)
    return path


def _write_fluxcal_spec1d(
    path: Path,
    *,
    with_telluric: bool = True,
    bunit: str = "erg/s/cm2/Angstrom",
    fluxunit: str = "erg/s/cm2/Angstrom",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr0 = fits.Header()
    hdr0["WAVEREF"] = "air"
    hdr0["BUNIT"] = bunit
    stamp_data_model(hdr0)
    stamp_calibration_semantics(
        hdr0,
        status=CalibrationStatus.RELATIVE_TELLURIC.value,
        flux_unit=fluxunit,
        reference_context_id="ctx123",
        sensfunc_ref="sensfunc_v1",
        telluric_ref=("telluric_v1" if with_telluric else None),
    )
    cols_trace = [
        fits.Column(name="LAMBDA", format="D", array=np.arange(4, dtype=float) + 4000.0),
        fits.Column(name="FLUX_TRACE", format="E", array=np.ones(4, dtype=np.float32)),
        fits.Column(name="VAR_TRACE", format="E", array=np.ones(4, dtype=np.float32) * 0.1),
        fits.Column(name="MASK_TRACE", format="J", array=np.zeros(4, dtype=np.int32)),
    ]
    cols_fixed = [
        fits.Column(name="LAMBDA", format="D", array=np.arange(4, dtype=float) + 4000.0),
        fits.Column(name="FLUX_FIXED", format="E", array=np.ones(4, dtype=np.float32)),
        fits.Column(name="VAR_FIXED", format="E", array=np.ones(4, dtype=np.float32) * 0.1),
        fits.Column(name="MASK_FIXED", format="J", array=np.zeros(4, dtype=np.int32)),
    ]
    hdul = fits.HDUList([
        fits.PrimaryHDU(header=hdr0),
        fits.BinTableHDU.from_columns(cols_trace, name="SPEC_TRACE"),
        fits.BinTableHDU.from_columns(cols_fixed, name="SPEC_FIXED"),
    ])
    hdul.writeto(path, overwrite=True)
    return path


def test_validate_calibrated_mef2d_product_accepts_semantic_headers(tmp_path: Path) -> None:
    from scorpio_pipe.contracts.validators import validate_calibrated_mef2d_product

    p = _write_fluxcal_mef(tmp_path / "fluxcal_2d.fits")
    validate_calibrated_mef2d_product(p, stage="fluxcal_apply")


def test_validate_calibrated_spec1d_product_requires_telluric_ref_when_declared(tmp_path: Path) -> None:
    from scorpio_pipe.contracts.validators import ProductContractError, validate_calibrated_spec1d_product

    p = _write_fluxcal_spec1d(tmp_path / "fluxcal_spec1d.fits", with_telluric=False)
    with pytest.raises(ProductContractError) as e:
        validate_calibrated_spec1d_product(p, stage="fluxcal_apply")
    assert e.value.code == "TELLURIC_MISSING"


def test_validate_calibrated_spec1d_product_rejects_non_flux_units(tmp_path: Path) -> None:
    from scorpio_pipe.contracts.validators import ProductContractError, validate_calibrated_spec1d_product

    p = _write_fluxcal_spec1d(
        tmp_path / "fluxcal_spec1d_bad_units.fits",
        with_telluric=True,
        bunit="electron",
        fluxunit="electron",
    )
    with pytest.raises(ProductContractError) as e:
        validate_calibrated_spec1d_product(p, stage="fluxcal_apply")
    assert e.value.code == "FLUXUNIT_INVALID"
