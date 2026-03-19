from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")


def test_validate_sensfunc_product_accepts_expected_table(tmp_path: Path) -> None:
    from scorpio_pipe.contracts.validators import validate_sensfunc_product

    p = tmp_path / "sensfunc_table.fits"
    hdr = fits.Header()
    hdr["CTYPE1"] = "AWAV"
    hdr["WAVEREF"] = "air"
    hdr["REFCTID"] = "ctx1"
    cols = [
        fits.Column(name="WAVELENGTH", format="D", array=np.array([4000.0, 5000.0, 6000.0, 7000.0])),
        fits.Column(name="RESPONSE_ABS", format="E", array=np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float32)),
        fits.Column(name="RESPONSE_REL", format="E", array=np.array([0.9, 1.0, 1.1, 1.2], dtype=np.float32)),
        fits.Column(name="RESPONSE_UNC", format="E", array=np.array([0.05, 0.05, 0.06, 0.06], dtype=np.float32)),
        fits.Column(name="FIT_MASK", format="J", array=np.array([1, 1, 1, 1], dtype=np.int32)),
    ]
    fits.HDUList([fits.PrimaryHDU(header=hdr), fits.BinTableHDU.from_columns(cols, name="SENSFUNC")]).writeto(p)
    validate_sensfunc_product(p, stage="sensfunc")


def test_validate_telluric_model_product_accepts_expected_table(tmp_path: Path) -> None:
    from scorpio_pipe.contracts.validators import validate_telluric_model_product

    p = tmp_path / "telluric_model.fits"
    hdr = fits.Header()
    hdr["CTYPE1"] = "AWAV"
    hdr["WAVEREF"] = "air"
    hdr["REFCTID"] = "ctx1"
    cols = [
        fits.Column(name="WAVELENGTH", format="D", array=np.array([6800.0, 6900.0, 7600.0, 7700.0])),
        fits.Column(name="TRANSMISSION", format="E", array=np.array([0.98, 0.75, 0.68, 0.96], dtype=np.float32)),
        fits.Column(name="CONFIDENCE", format="E", array=np.array([0.9, 0.8, 0.85, 0.9], dtype=np.float32)),
        fits.Column(name="MODEL_MASK", format="J", array=np.array([0, 1, 1, 0], dtype=np.int32)),
    ]
    fits.HDUList([fits.PrimaryHDU(header=hdr), fits.BinTableHDU.from_columns(cols, name="TELLURIC_MODEL")]).writeto(p)
    validate_telluric_model_product(p, stage="telluric_model")


def test_validate_calibration_systematics_product_rejects_negative_values(tmp_path: Path) -> None:
    from scorpio_pipe.contracts.validators import ProductContractError, validate_calibration_systematics_product

    p = tmp_path / "calibration_systematics.fits"
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=np.arange(4, dtype=np.float32), name="WAVE"),
        fits.ImageHDU(data=np.array([0.1, 0.1, -0.1, 0.2], dtype=np.float32), name="CALSYS_REL"),
        fits.ImageHDU(data=np.array([0.0, 0.1, 0.1, 0.2], dtype=np.float32), name="CALSYS_TELL"),
        fits.ImageHDU(data=np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32), name="CALSYS_ABS"),
    ]).writeto(p)
    with pytest.raises(ProductContractError) as e:
        validate_calibration_systematics_product(p, stage="fluxcal_apply")
    assert e.value.code == "CALSYS_NEGATIVE"
