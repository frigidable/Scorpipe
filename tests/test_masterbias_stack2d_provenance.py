"""End-to-end shared MasterBias covariance provenance: zero numeric correction.

Keep uncertainty unqualified until independent detector and shared calibration
components are represented separately through each calibration operator.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from astropy.io import fits
from scorpio_pipe.stages.stack2d import run_stack2d
from scorpio_pipe.stages.extract1d import run_extract1d
from scorpio_pipe.workspace_paths import stage_dir
from tests.test_perf_40_4_stack2d_streaming import _write_linearized_input
from tests.test_extract_stacked_rejection_semantics import _write_stack


@pytest.mark.parametrize("ids,status,remove_gid", [
    (["group1", "group1"], "SHARED", False),
    (["group1", "group2"], "DISTINCT", True),
    (["group1", None], "UNKNOWN", True),
])
def test_stack2d_identifies_shared_masterbias_and_does_not_fake_var_correction(
    tmp_path: Path, ids, status, remove_gid,
):
    work = tmp_path / "night"
    lin_dir = stage_dir(work, "linearize")
    lin_dir.mkdir(parents=True)
    files = [
        lin_dir / "a_skysub.fits",
        lin_dir / "b_skysub.fits",
    ]
    for i, path in enumerate(files):
        _write_linearized_input(path, 10. + i * 4., var_value=4.)
        with fits.open(path, mode="update") as hdul:
            if ids[i]:
                hdul[0].header["SB_GID"] = ids[i]
            hdul[0].header["BVARINC"] = True
            hdul.flush()
    cfg = {
        "work_dir": str(work),
        "stack2d": {
            "enabled": True,
            "method": "invvar", "robust_iter": 0,
            "eta": {"enabled": False}, "save_png": False,
        },
    }
    result = run_stack2d(cfg, inputs=files, out_dir=stage_dir(work, "stack2d"))
    assert result["bias_covariance"]["status"] == status
    assert result["bias_covariance"]["variance_qualified"] is False
    with fits.open(result["stack2d_fits"]) as hdul:
        header = hdul[0].header
        assert header["SBGSTAT"] == status
        assert header["SBCOVQ"] == "UNMODEL"
        assert ("SB_GID" not in header) == remove_gid
        np.testing.assert_allclose(hdul["VAR"].data, 2., atol=1e-5)
    if status == "SHARED":
        codes = {f["code"] for f in result["qc"]["flags"]}
        assert "SHARED_MASTERBIAS_COVARIANCE_UNMODELLED" in codes


def test_extraction_preserves_shared_calibration_uncertainty_metadata(tmp_path: Path):
    work = tmp_path / "night"
    stack = tmp_path / "stack2d.fits"
    _write_stack(stack, verified=True)
    with fits.open(stack, mode="update") as hdul:
        header = hdul[0].header
        header["STKVRALG"] = "W2VAR"
        header["STKVARQ"] = "COND_ONLY"
        header["STKXCVQ"] = "UNMODEL"
        header["SBCOVQ"] = "UNMODEL"
        header["SBGSTAT"] = "SHARED"
        header["SBGRPN"] = 1
        hdul.flush()
    cfg = {
        "work_dir": str(work),
        "frames": {"__setup__": {"instrument": "scorpio", "mode": "longslit"}},
        "extract1d": {
            "input_mode": "stack2d",
            "method": "boxcar",
            "save_png": False,
        },
    }
    result = run_extract1d(cfg, in_fits=stack, out_dir=work / "extract")
    with fits.open(result["spec1d_fits"]) as hdul:
        h = hdul[0].header
        assert h["STKVRALG"] == "W2VAR"
        assert h["STKVARQ"] == "COND_ONLY"
        assert h["STKXCVQ"] == "UNMODEL"
        assert h["SBCOVQ"] == "UNMODEL"
        assert h["SBGSTAT"] == "SHARED"
        assert h["SBGRPN"] == 1
