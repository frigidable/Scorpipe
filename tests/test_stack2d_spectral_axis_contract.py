"""Stack2D must not combine numerically equal yet physically incompatible grids.

FITS WCS WAVE means vacuum wavelength, AWAV means wavelength in air.
SPECSYS is the spectral velocity reference frame. These are scientific
coordinate semantics, independent of identical CRVAL1/CDELT1 numbers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from scorpio_pipe.spectral_axis_contract import read_spectral_axis_contract
from scorpio_pipe.stages.stack2d import run_stack2d
from scorpio_pipe.workspace_paths import stage_dir
from tests.test_perf_40_4_stack2d_streaming import _write_linearized_input


def _write(
    path: Path,
    value: float,
    *,
    waveref: str = "air",
    ctype: str = "AWAV",
    specsys: str = "TOPOCENT",
    instrument: str = "SCORPIO-2",
    dlamap: bool | None = False,
) -> None:
    _write_linearized_input(path, value)
    with fits.open(path, mode="update") as hdul:
        h = hdul[0].header
        if waveref:
            h["WAVEREF"] = waveref
        if ctype:
            h["CTYPE1"] = ctype
        if specsys:
            h["SPECSYS"] = specsys
        if instrument:
            h["INSTRUME"] = instrument
        if dlamap is not None:
            h["DLAMAP"] = dlamap
            h["FLXFLAG"] = "OK" if dlamap else "UNVERIFIED"
            if dlamap:
                h["DLAM_A"] = 0.25
        hdul.flush()


def _run(tmp_path: Path, *, first: dict | None = None, second: dict | None = None):
    work = tmp_path / "night"
    linear = stage_dir(work, "linearize")
    linear.mkdir(parents=True, exist_ok=True)
    a, b = linear / "a_skysub.fits", linear / "b_skysub.fits"
    _write(a, 10.0, **(first or {}))
    _write(b, 14.0, **(second or {}))
    cfg = {
        "work_dir": str(work),
        "stack2d": {
            "method": "invvar",
            "robust_iter": 0,
            "var_floor_enabled": False,
            "eta": {"enabled": False},
            "save_png": False,
        },
    }
    return run_stack2d(cfg, inputs=[a, b], out_dir=stage_dir(work, "stack2d"))


@pytest.mark.parametrize(
    ("first", "second", "message"),
    [
        ({}, {"waveref": "vacuum", "ctype": "WAVE"},
         "incompatible spectral coordinate contracts"),
        ({}, {"specsys": "BARYCENT"},
         "incompatible spectral coordinate contracts"),
        ({}, {"specsys": "HELIOCEN"},
         "incompatible spectral coordinate contracts"),
        ({}, {"dlamap": True},
         "incompatible flexure correction provenance"),
    ],
)
def test_equal_numeric_grid_different_physical_system_is_rejected(
    tmp_path: Path, first: dict, second: dict, message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _run(tmp_path, first=first, second=second)


@pytest.mark.parametrize(
    ("medium", "ctype"),
    [("air", "WAVE"), ("vacuum", "AWAV")],
)
def test_inconsistent_waveref_ctype_fails_even_for_single_frame(
    tmp_path: Path, medium: str, ctype: str,
) -> None:
    with pytest.raises(ValueError, match="contradicts WAVEREF"):
        _run(tmp_path, first={"waveref": medium, "ctype": ctype})


@pytest.mark.parametrize(
    "field", ["waveref", "ctype", "specsys"],
)
def test_scorpio_missing_spectral_metadata_fails_closed(
    tmp_path: Path, field: str,
) -> None:
    with pytest.raises(ValueError, match="SCORPIO Stack2D requires explicit"):
        _run(tmp_path, first={field: ""})


@pytest.mark.parametrize("ctype", ["WAVE-LOG", "AWAV-TAB", "FREQ", "LINEAR"])
def test_stack2d_refuses_nonlinear_or_nonwavelength_wcs(
    tmp_path: Path, ctype: str,
) -> None:
    with pytest.raises(ValueError, match="unsupported spectral CTYPE1"):
        _run(tmp_path, first={"ctype": ctype})


def test_scorpio_rejects_missing_flexure_provenance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires explicit DLAMAP"):
        _run(tmp_path, first={"dlamap": None})


def test_applied_flexure_without_a_calibrated_shift_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "science_skysub.fits"
    _write_linearized_input(path, 10.0)
    with fits.open(path, mode="update") as hdul:
        hdr = hdul[0].header
        hdr["INSTRUME"] = "SCORPIO-2"
        hdr["WAVEREF"] = "air"
        hdr["CTYPE1"] = "AWAV"
        hdr["SPECSYS"] = "TOPOCENT"
        hdr["DLAMAP"] = True
        hdr["FLXFLAG"] = "OK"
        hdul.flush()
    with pytest.raises(ValueError, match="requires finite DLAM_A"):
        read_spectral_axis_contract(fits.getheader(path))


def test_scorpio_rejects_applied_unverified_flexure(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="contradicts unverified flexure"):
        work = tmp_path / "night"
        lin = stage_dir(work, "linearize")
        lin.mkdir(parents=True)
        path = lin / "a_skysub.fits"
        _write(path, 10.0, dlamap=True)
        with fits.open(path, mode="update") as hdul:
            hdul[0].header["FLXFLAG"] = "UNVERIFIED"
            hdul.flush()
        hdr = fits.getheader(path)
        read_spectral_axis_contract(hdr, source=path.name)


def test_consistent_air_topocentric_produces_qualified_stack(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path)
    with fits.open(result["stack2d_fits"]) as hdul:
        assert hdul[0].header["SPCQUAL"] == "QUALIF"
        assert hdul[0].header["WAVEREF"] == "air"
        assert hdul[0].header["CTYPE1"] == "AWAV"
        assert hdul[0].header["SPECSYS"] == "TOPOCENT"
        np.testing.assert_allclose(hdul["SCI"].data, 12.0, atol=1e-5)
    assert result["spectral_axis_contract"]["validation"] == "QUALIFIED"
    assert all(row["spectral_axis_contract"]["validation"] == "QUALIFIED"
               for row in result["inputs"])


def test_consistent_vacuum_barycentric_already_rectified_passes(
    tmp_path: Path,
) -> None:
    values = {"waveref": "vacuum", "ctype": "WAVE", "specsys": "BARYCENT"}
    result = _run(tmp_path, first=values, second=values)
    with fits.open(result["stack2d_fits"]) as hdul:
        assert hdul[0].header["WAVEREF"] == "vacuum"
        assert hdul[0].header["SPECSYS"] == "BARYCENT"
        assert hdul[0].header["SPCQUAL"] == "QUALIF"


def test_legacy_unidentified_spectral_grids_remain_visibly_unqualified(
    tmp_path: Path,
) -> None:
    old = {
        "waveref": "", "ctype": "", "specsys": "",
        "instrument": "", "dlamap": None,
    }
    result = _run(tmp_path, first=old, second=old)
    with fits.open(result["stack2d_fits"]) as hdul:
        assert hdul[0].header["SPCQUAL"] == "UNQUAL"
        assert "WAVEREF" not in hdul[0].header
        assert "SPECSYS" not in hdul[0].header
    assert result["spectral_axis_contract"]["validation"] == "UNQUAL"


def test_partially_labeled_anonymous_fits_never_defaults_implicitly(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="partial spectral WCS"):
        _run(tmp_path, first={
            "instrument": "", "specsys": "",
        })
