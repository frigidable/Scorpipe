import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy.io import fits

from scorpio_pipe.boundary_contract import ProductContractError, validate_lambda_map_product


def _write_lambda_map(path, *, waveref: str | None = "air") -> None:
    data = np.linspace(5000.0, 5010.0, 20, dtype=float).reshape(4, 5)
    hdr = fits.Header()
    hdr["SCORPVER"] = ("6.0.0", "Pipeline version")
    hdr["WAVEUNIT"] = ("Angstrom", "Wavelength unit")
    hdr["CTYPE1"] = ("WAVE", "Axis type")
    if waveref is not None:
        hdr["WAVEREF"] = (waveref, "Air/vacuum")
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


def test_lambda_map_contract_passes_with_minimal_header(tmp_path):
    p = tmp_path / "lambda_map.fits"
    _write_lambda_map(p, waveref="air")
    validate_lambda_map_product(p, stage="wavesolution")


def test_lambda_map_contract_requires_waveref(tmp_path):
    p = tmp_path / "lambda_map.fits"
    _write_lambda_map(p, waveref=None)
    with pytest.raises(ProductContractError) as e:
        validate_lambda_map_product(p, stage="wavesolution")
    assert e.value.code == "LAMBDA_MAP_CONTRACT"
