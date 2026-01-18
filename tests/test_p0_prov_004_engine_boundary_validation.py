import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy.io import fits

from scorpio_pipe.boundary_contract import ProductContractError
from scorpio_pipe.io.mef import WaveGrid, write_sci_var_mask
from scorpio_pipe.pipeline.engine import _validate_task_products


def test_engine_boundary_validation_uses_out_dir_for_stack2d(tmp_path):
    out_dir = tmp_path / "11_stack"
    out_dir.mkdir()

    sci = np.zeros((5, 5), dtype=float)
    var = np.ones_like(sci)
    mask = np.zeros(sci.shape, dtype=np.uint16)
    grid = WaveGrid(lambda0=5000.0, dlambda=1.0, nlam=sci.shape[1], unit="Angstrom", wave_ref="air")

    # Disable writer-side validation: we want to test engine-side validation.
    write_sci_var_mask(
        out_dir / "stack2d.fits",
        sci,
        var=var,
        mask=mask,
        header=fits.Header(),
        grid=grid,
        validate=False,
    )

    with pytest.raises(ProductContractError) as e:
        _validate_task_products("stack2d", out_dir)
    assert e.value.code == "NOISE_PROV_MISSING"


def test_engine_boundary_validation_enforces_lambda_map_after_wavesolution(tmp_path):
    out_dir = tmp_path / "08_wavesol"
    out_dir.mkdir()

    data = np.linspace(5000.0, 5001.0, 25, dtype=float).reshape(5, 5)
    hdr = fits.Header()
    hdr["SCORPVER"] = ("6.0.0", "Pipeline version")
    hdr["WAVEUNIT"] = ("Angstrom", "Wavelength unit")
    hdr["CTYPE1"] = ("WAVE", "Axis type")
    # Intentionally omit WAVEREF
    fits.PrimaryHDU(data=data, header=hdr).writeto(out_dir / "lambda_map.fits", overwrite=True)

    with pytest.raises(ProductContractError) as e:
        _validate_task_products("wavesolution", out_dir)
    assert e.value.code == "LAMBDA_MAP_CONTRACT"


def test_engine_boundary_validation_is_limited_in_migration_stage1(tmp_path):
    # Even if a stage output violates the MEF contract, we do NOT validate it yet
    # (coverage is intentionally limited in P0-B5 stage 1).
    out_dir = tmp_path / "10_linearize"
    out_dir.mkdir()

    sci = np.zeros((2, 2), dtype=float)
    var = np.ones_like(sci)
    mask = np.zeros(sci.shape, dtype=np.uint16)
    grid = WaveGrid(lambda0=5000.0, dlambda=1.0, nlam=sci.shape[1], unit="Angstrom", wave_ref="air")

    write_sci_var_mask(
        out_dir / "linearize.fits",
        sci,
        var=var,
        mask=mask,
        header=fits.Header(),
        grid=grid,
        validate=False,
    )

    # Should not raise
    _validate_task_products("linearize", out_dir)
