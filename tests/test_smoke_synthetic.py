from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy.io import fits

from scorpio_pipe.work_layout import ensure_work_layout
from scorpio_pipe.stages.linearize import run_linearize
from scorpio_pipe.stages.sky_sub import run_sky_sub
from scorpio_pipe.stages.stack2d import run_stack2d
from scorpio_pipe.stages.extract1d import run_extract1d


def _write(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=data.astype("float32", copy=False)).writeto(path, overwrite=True)


def test_smoke_synthetic_chain():
    with tempfile.TemporaryDirectory(prefix="scorpipe_pytest_smoke_") as td:
        work_dir = Path(td) / "work"
        layout = ensure_work_layout(work_dir)

        ny, nx = 16, 32
        x = np.arange(nx)[None, :]
        lam = 5000.0 + 2.0 * x
        lambda_map = np.repeat(lam, ny, axis=0)
        wavesol_dir = work_dir / "wavesol"
        wavesol_dir.mkdir(parents=True, exist_ok=True)
        _write(wavesol_dir / "lambda_map.fits", lambda_map)

        # one tiny exposure is enough for a smoke chain
        img = 10.0 + 0.1 * np.arange(nx)[None, :] + np.zeros((ny, 1))
        img = np.repeat(img, ny, axis=0)
        raw = layout.raw / "obj_0001.fits"
        _write(raw, img)

        cfg = {
            "work_dir": str(work_dir),
            "frames": {"obj": [str(raw)]},
            "linearize": {"dlambda_A": "auto", "save_per_exposure": True, "save_preview": True},
            "roi": {"obj_y1": 6, "obj_y2": 9, "sky_y1": 1, "sky_y2": 4, "sky2_y1": 12, "sky2_y2": 15, "units": "px"},
            "sky_sub": {"save_per_exp_sky_model": False},
            "stack2d": {},
            "extract1d": {"mode": "boxcar", "aperture_half_width": 2},
        }

        lin = run_linearize(cfg)
        run_sky_sub(cfg)
        sky_fits = sorted((work_dir / "products" / "sky" / "per_exp").glob("*.fits"))
        st = run_stack2d(cfg, inputs=sky_fits)
        ex = run_extract1d(cfg, stacked_fits=st["stacked2d_fits"])

        assert Path(lin["preview_fits"]).exists()
        assert Path(st["stacked2d_fits"]).exists()
        assert Path(ex["spec1d_fits"]).exists()
