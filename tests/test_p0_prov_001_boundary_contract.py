import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy.io import fits

from scorpio_pipe.boundary_contract import ProductContractError, validate_mef_product
from scorpio_pipe.io.mef import WaveGrid, write_sci_var_mask
from scorpio_pipe.maskbits import NO_COVERAGE


def _hdr_with_units() -> fits.Header:
    h = fits.Header()
    h["SCORPUM"] = ("e-", "SCI/VAR units model")
    h["SCORPGN"] = (1.7, "Gain used")
    h["SCORPRN"] = (3.2, "Read-noise used")
    return h


def test_boundary_contract_mef_requires_units(tmp_path):
    sci = np.zeros((4, 5), dtype=float)
    var = np.ones_like(sci)
    mask = np.zeros(sci.shape, dtype=np.uint16)

    # include wavelength axis so WAVEREF is required
    grid = WaveGrid(lambda0=5000.0, dlambda=1.0, nlam=sci.shape[1], unit="Angstrom", wave_ref="air")

    p = tmp_path / "prod.fits"
    write_sci_var_mask(p, sci, var=var, mask=mask, header=_hdr_with_units(), grid=grid)

    # should validate
    validate_mef_product(p, stage="linearize")

    # now without required unit/noise provenance
    p2 = tmp_path / "prod2.fits"
    write_sci_var_mask(p2, sci, var=var, mask=mask, header=fits.Header(), grid=grid)
    with pytest.raises(ProductContractError) as e:
        validate_mef_product(p2, stage="linearize")
    assert e.value.code == "UNITS_MISSING"


def test_boundary_contract_mef_requires_waveref_when_wavelength_exists(tmp_path):
    sci = np.zeros((3, 6), dtype=float)
    var = np.ones_like(sci)
    mask = np.zeros(sci.shape, dtype=np.uint16)

    grid_unknown = WaveGrid(lambda0=6000.0, dlambda=2.0, nlam=sci.shape[1], unit="Angstrom", wave_ref="UNKNOWN")
    p = tmp_path / "prod.fits"
    write_sci_var_mask(p, sci, var=var, mask=mask, header=_hdr_with_units(), grid=grid_unknown)

    with pytest.raises(ProductContractError) as e:
        validate_mef_product(p, stage="linearize")
    assert e.value.code == "WAVEREF_MISSING"


def test_mef_writer_sanitizes_nonfinite_and_marks_no_coverage(tmp_path):
    sci = np.zeros((4, 5), dtype=float)
    var = np.ones_like(sci)
    mask = np.zeros(sci.shape, dtype=np.uint16)

    sci[0, 0] = np.nan
    var[1, 1] = np.inf

    grid = WaveGrid(lambda0=5000.0, dlambda=1.0, nlam=sci.shape[1], unit="Angstrom", wave_ref="air")
    p = tmp_path / "prod.fits"
    write_sci_var_mask(p, sci, var=var, mask=mask, header=_hdr_with_units(), grid=grid)

    with fits.open(p, memmap=False) as hdul:
        hdr0 = hdul[0].header
        assert int(hdr0.get("SCORPNAN", 0)) == 2
        sci2 = hdul["SCI"].data
        var2 = hdul["VAR"].data
        m2 = hdul["MASK"].data

    assert np.isfinite(sci2).all()
    assert np.isfinite(var2).all()

    # pixels should be flagged as NO_COVERAGE
    assert (int(m2[0, 0]) & int(NO_COVERAGE)) != 0
    assert (int(m2[1, 1]) & int(NO_COVERAGE)) != 0
