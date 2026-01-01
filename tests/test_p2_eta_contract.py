from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy.io import fits  # noqa: E402

from scorpio_pipe.work_layout import ensure_work_layout
from scorpio_pipe.io.mef import WaveGrid, write_sci_var_mask
from scorpio_pipe.stages.extract1d import run_extract1d
from scorpio_pipe.workspace_paths import stage_dir


def _make_stack2d_fits(path: Path, *, etaappl: bool | None) -> None:
    ny, nlam = 24, 64
    wave = WaveGrid(lambda0=5000.0, dlambda=2.0, nlam=nlam, unit="Angstrom")
    y = np.arange(ny)[:, None]
    x = np.arange(nlam)[None, :]

    # Smooth object + background
    sci = 10.0 + 0.2 * x + 50.0 * np.exp(-0.5 * ((y - ny / 2.0) / 2.0) ** 2)
    var = np.full_like(sci, 9.0, dtype=float)
    mask = np.zeros_like(sci, dtype=np.uint16)

    hdr = fits.Header()
    hdr["STKMETH"] = ("mean", "stack2d method")
    if etaappl is not None:
        hdr["ETAAPPL"] = (bool(etaappl), "eta(lambda) applied to VAR")

    write_sci_var_mask(path, sci, var=var, mask=mask, header=hdr, grid=wave)


def test_p2_etaappl_propagates_to_spec1d_header():
    with tempfile.TemporaryDirectory(prefix="scorpipe_pytest_p2_eta_") as td:
        work_dir = Path(td) / "work"
        ensure_work_layout(work_dir)

        sdir = stage_dir(work_dir, "stack2d")
        stack = sdir / "stack2d.fits"
        _make_stack2d_fits(stack, etaappl=True)

        cfg = {
            "work_dir": str(work_dir),
            "roi": {
                "obj_y1": 10,
                "obj_y2": 14,
                "sky_y1": 2,
                "sky_y2": 6,
                "sky2_y1": 17,
                "sky2_y2": 21,
                "units": "px",
            },
            "extract1d": {"mode": "boxcar", "aperture_half_width": 2},
        }

        out = run_extract1d(cfg, in_fits=stack)
        spec = Path(out["spec1d_fits"])
        assert spec.exists()

        hdr = fits.getheader(spec, 0)
        assert bool(hdr.get("ETAAPPL")) is True


def test_p2_eta_missing_stamp_is_error_for_stack2d_like_input():
    with tempfile.TemporaryDirectory(prefix="scorpipe_pytest_p2_eta_neg_") as td:
        work_dir = Path(td) / "work"
        ensure_work_layout(work_dir)

        sdir = stage_dir(work_dir, "stack2d")
        stack = sdir / "stack2d.fits"
        _make_stack2d_fits(stack, etaappl=None)

        cfg = {
            "work_dir": str(work_dir),
            "roi": {
                "obj_y1": 10,
                "obj_y2": 14,
                "sky_y1": 2,
                "sky_y2": 6,
                "sky2_y1": 17,
                "sky2_y2": 21,
                "units": "px",
            },
            "extract1d": {"mode": "boxcar", "aperture_half_width": 2},
        }

        with pytest.raises(ValueError, match="ETAAPPL"):
            run_extract1d(cfg, in_fits=stack)
