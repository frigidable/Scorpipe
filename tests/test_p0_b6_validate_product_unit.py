import numpy as np
import pytest

astropy = pytest.importorskip("astropy")
from astropy.io import fits

from scorpio_pipe.contracts.validators import ProductContractError, validate_product
from scorpio_pipe.io.mef import WaveGrid, write_sci_var_mask


def _write_valid_mef(path):
    """Write a minimal valid SCI/VAR/MASK MEF for validator unit tests."""

    hdr = fits.Header()
    # Noise provenance (legacy keys are accepted by the validator)
    hdr["GAIN"] = 1.0
    hdr["RDNOISE"] = 3.0
    hdr["NOISRC"] = "MODEL"
    # Wavelength medium + units
    hdr["WAVEREF"] = "air"
    hdr["BUNIT"] = "ADU"

    sci = np.ones((8, 6), dtype=float)
    var = np.ones_like(sci)
    msk = np.zeros_like(sci, dtype=np.uint16)

    grid = WaveGrid(lambda0=5000.0, dlambda=1.0, nlam=sci.shape[1], unit="Angstrom", wave_ref="air")
    write_sci_var_mask(path, sci, var=var, mask=msk, header=hdr, grid=grid, validate=False)
    return path


def test_validate_product_missing_hdu(tmp_path):
    p = tmp_path / "stack2d.fits"
    fits.PrimaryHDU(data=np.zeros((4, 4), dtype=np.float32)).writeto(p)

    with pytest.raises(ProductContractError) as e:
        validate_product(p, stage="unit")

    assert e.value.code in {"SCI_MISSING", "MEF_CONTRACT"}


def test_validate_product_dtype(tmp_path):
    p = tmp_path / "stack2d.fits"

    hdr0 = fits.Header()
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(header=hdr0),
            fits.ImageHDU(data=np.ones((3, 3), dtype=np.int16), name="SCI"),
            fits.ImageHDU(data=np.ones((3, 3), dtype=np.float32), name="VAR"),
            fits.ImageHDU(data=np.zeros((3, 3), dtype=np.uint16), name="MASK"),
        ]
    )
    hdul.writeto(p)

    with pytest.raises(ProductContractError) as e:
        validate_product(p, stage="unit")

    assert e.value.code == "DTYPE"


def test_validate_product_var_negative(tmp_path):
    p = tmp_path / "stack2d.fits"

    sci = np.ones((3, 3), dtype=np.float32)
    var = np.ones((3, 3), dtype=np.float32)
    var[1, 1] = -1.0
    msk = np.zeros((3, 3), dtype=np.uint16)

    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=sci, name="SCI"),
            fits.ImageHDU(data=var, name="VAR"),
            fits.ImageHDU(data=msk, name="MASK"),
        ]
    ).writeto(p)

    with pytest.raises(ProductContractError) as e:
        validate_product(p, stage="unit")

    assert e.value.code == "VAR_NEGATIVE"


def test_validate_product_var_nan(tmp_path):
    p = tmp_path / "stack2d.fits"

    sci = np.ones((3, 3), dtype=np.float32)
    var = np.ones((3, 3), dtype=np.float32)
    var[0, 0] = np.nan
    msk = np.zeros((3, 3), dtype=np.uint16)

    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=sci, name="SCI"),
            fits.ImageHDU(data=var, name="VAR"),
            fits.ImageHDU(data=msk, name="MASK"),
        ]
    ).writeto(p)

    with pytest.raises(ProductContractError) as e:
        validate_product(p, stage="unit")

    assert e.value.code == "VAR_NONFINITE"


def test_validate_product_shape_mismatch(tmp_path):
    p = tmp_path / "stack2d.fits"

    sci = np.ones((3, 3), dtype=np.float32)
    var = np.ones((3, 4), dtype=np.float32)
    msk = np.zeros((3, 3), dtype=np.uint16)

    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=sci, name="SCI"),
            fits.ImageHDU(data=var, name="VAR"),
            fits.ImageHDU(data=msk, name="MASK"),
        ]
    ).writeto(p)

    with pytest.raises(ProductContractError) as e:
        validate_product(p, stage="unit")

    assert e.value.code == "SHAPE_MISMATCH"


def test_validate_product_missing_required_keys(tmp_path):
    p = tmp_path / "stack2d.fits"
    _write_valid_mef(p)

    # Remove data model version from primary header.
    with fits.open(p, mode="update", memmap=False) as hdul:
        hdr0 = hdul[0].header
        if "SCORPDMV" in hdr0:
            del hdr0["SCORPDMV"]
        hdul.flush()

    with pytest.raises(ProductContractError) as e:
        validate_product(p, stage="unit")

    assert e.value.code == "DATA_MODEL_MISSING"
