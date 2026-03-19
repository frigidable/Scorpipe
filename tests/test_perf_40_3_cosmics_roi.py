from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")
clean_cosmics = pytest.importorskip("scorpio_pipe.stages.cosmics").clean_cosmics


def test_cosmics_roi_limits_processing_to_requested_window(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    work_dir = tmp_path / "work"
    out_dir = work_dir / "05_cosmics"
    data_dir.mkdir(parents=True)
    work_dir.mkdir(parents=True)

    a = np.zeros((10, 10), dtype=np.float32)
    b = np.zeros((10, 10), dtype=np.float32)
    a[2, 2] = 10000.0  # outside ROI -> must remain untouched
    b[7, 7] = 10000.0  # inside ROI -> should be cleaned

    fits.writeto(data_dir / "obj1.fits", a, overwrite=True)
    fits.writeto(data_dir / "obj2.fits", b, overwrite=True)

    cfg = {
        "config_dir": str(tmp_path),
        "data_dir": str(data_dir),
        "work_dir": str(work_dir),
        "frames": {"obj": ["obj1.fits", "obj2.fits"]},
        "cosmics": {
            "enabled": True,
            "method": "two_frame_diff",
            "apply_to": ["obj"],
            "k": 3.0,
            "save_png": False,
            "save_mask_fits": True,
            "roi": {"y0": 6, "y1": 9, "x0": 6, "x1": 9},
        },
    }

    summary_path = clean_cosmics(cfg, out_dir=out_dir)
    assert summary_path.is_file()

    obj1_clean = fits.getdata(out_dir / "obj" / "clean" / "obj1_clean.fits")
    obj2_clean = fits.getdata(out_dir / "obj" / "clean" / "obj2_clean.fits")
    obj1_mask = fits.getdata(out_dir / "obj" / "clean" / "obj1_clean_mask.fits")
    obj2_mask = fits.getdata(out_dir / "obj" / "clean" / "obj2_clean_mask.fits")

    assert float(obj1_clean[2, 2]) == 10000.0
    assert float(obj2_clean[7, 7]) == 0.0
    assert int(obj1_mask[2, 2]) == 0
    assert int(obj2_mask[7, 7]) > 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["roi"] == {"y0": 6, "y1": 9, "x0": 6, "x1": 9}
    assert summary["items"][0]["roi"] == {"y0": 6, "y1": 9, "x0": 6, "x1": 9}
