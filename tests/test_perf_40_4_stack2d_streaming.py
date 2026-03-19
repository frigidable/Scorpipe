from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy.io import fits

from scorpio_pipe.stages import stack2d as st2
from scorpio_pipe.workspace_paths import stage_dir


def _write_linearized_input(path: Path, value: float, *, var_value: float = 4.0) -> None:
    sci = np.full((8, 16), float(value), dtype=np.float32)
    var = np.full((8, 16), float(var_value), dtype=np.float32)
    mask = np.zeros((8, 16), dtype=np.uint16)

    hdr = fits.Header()
    hdr["BUNIT"] = "ADU/s"
    hdr["CRVAL1"] = 5000.0
    hdr["CDELT1"] = 1.0
    hdr["CRPIX1"] = 1.0
    hdr["CUNIT1"] = "Angstrom"
    hdr["NAXIS1"] = sci.shape[1]
    hdr["EXPTIME"] = 100.0

    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(data=sci, header=hdr),
            fits.ImageHDU(data=var, name="VAR"),
            fits.ImageHDU(data=mask, name="MASK"),
        ]
    )
    hdul.writeto(path)


def test_perf_40_4_stack2d_streaming_avoids_full_cube_stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "night"
    lin_dir = stage_dir(work_dir, "linearize")
    out_dir = stage_dir(work_dir, "stack2d")
    lin_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = lin_dir / "obj_a_skysub.fits"
    p2 = lin_dir / "obj_b_skysub.fits"
    _write_linearized_input(p1, 10.0)
    _write_linearized_input(p2, 14.0)

    cfg = {
        "work_dir": str(work_dir),
        "stack2d": {
            "enabled": True,
            "method": "invvar_huber",
            "chunk_rows": 4,
            "robust_iter": 2,
            "save_png": False,
        },
    }

    def _forbid_stack(*args, **kwargs):
        raise AssertionError("stack2d should not build a full (nexp, y, x) cube via np.stack")

    monkeypatch.setattr(st2.np, "stack", _forbid_stack)

    result = st2.run_stack2d(cfg, inputs=[p1, p2], out_dir=out_dir)
    out_fits = Path(result["stack2d_fits"])
    assert out_fits.exists()

    with fits.open(out_fits, memmap=False) as hdul:
        sci = np.asarray(hdul[0].data, dtype=np.float32)
        var = np.asarray(hdul["VAR"].data, dtype=np.float32)
        cov = np.asarray(hdul["COV"].data, dtype=np.int16)
        assert np.allclose(sci, 12.0, atol=1e-5)
        assert np.allclose(var, 2.0, atol=1e-5)
        assert np.all(cov == 2)
