"""A soft Sky pass-through must never become scientific Stack2D signal.

Sky failure is reported by SKYOK=False and/or the SKYMODEL_FAIL bit. These
are mandatory admission checks, not optional operator rejection settings.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from scorpio_pipe.maskbits import SKYMODEL_FAIL
from scorpio_pipe.stages.stack2d import run_stack2d
from scorpio_pipe.workspace_paths import stage_dir
from tests.test_perf_40_4_stack2d_streaming import _write_linearized_input


def _input(path: Path, flux: float, *, skyok=None, fail_bit=False):
    _write_linearized_input(path, flux)
    with fits.open(path, mode="update") as hdul:
        if skyok is not None:
            hdul[0].header["SKYOK"] = skyok
        if fail_bit:
            hdul["MASK"].data[:] = np.uint16(SKYMODEL_FAIL)
        hdul.flush()
    return path


def _cfg(work: Path) -> dict:
    return {
        "work_dir": str(work),
        "stack2d": {
            "enabled": True,
            "method": "invvar",
            "chunk_rows": 4,
            "robust_iter": 0,
            "eta": {"enabled": False},
            "var_floor_enabled": False,
            "save_png": False,
        },
    }


@pytest.mark.parametrize("skyok,fail_bit", [(False, False), (True, True), (False, True)])
def test_stack2d_excludes_failed_sky_even_without_user_reject_bits(
    tmp_path: Path, skyok, fail_bit,
) -> None:
    work = tmp_path / "night"
    lin = stage_dir(work, "linearize")
    lin.mkdir(parents=True)
    inputs = [
        _input(lin / "bad_skysub.fits", 2000.0, skyok=skyok, fail_bit=fail_bit),
        _input(lin / "good_a_skysub.fits", 12.0, skyok=True),
        _input(lin / "good_b_skysub.fits", 16.0, skyok=True),
    ]
    result = run_stack2d(
        _cfg(work), inputs=inputs, out_dir=stage_dir(work, "stack2d")
    )
    with fits.open(result["stack2d_fits"]) as hdul:
        np.testing.assert_allclose(hdul["SCI"].data, 14.0, atol=1e-4)
        np.testing.assert_array_equal(
            hdul["COV"].data, np.full((8, 16), 2, dtype=np.int16),
        )
    bad_meta = next(
        x for x in result["inputs"] if "bad_skysub" in x["path"]
    )
    assert bad_meta["rejected"] is True
    assert bad_meta["reject_reason"]["science_gate"] == "SKY_MODEL_FAILED"


def test_stack2d_fails_closed_if_every_sky_input_is_passthrough(
    tmp_path: Path,
) -> None:
    work = tmp_path / "night"
    lin = stage_dir(work, "linearize")
    lin.mkdir(parents=True)
    inputs = [
        _input(lin / "bad1_skysub.fits", 100.0, skyok=False),
        _input(lin / "bad2_skysub.fits", 120.0, skyok=True, fail_bit=True),
    ]
    with pytest.raises(RuntimeError, match="no scientifically usable exposure"):
        run_stack2d(
            _cfg(work), inputs=inputs, out_dir=stage_dir(work, "stack2d")
        )
