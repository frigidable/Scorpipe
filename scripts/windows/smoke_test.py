"""
smoke_test.py — synthetic end-to-end smoke run (v5.17)

Goal:
  - Create tiny synthetic long-slit-like frames + a simple 2D lambda_map
  - Run: linearize -> sky_sub -> stack2d -> extract1d
  - Verify that canonical products (MEF with SCI/VAR/MASK) are created.

Usage:
  python smoke_test.py

This does NOT require real FITS from SCORPIO; it is meant to catch obvious regressions.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

from scorpio_pipe.work_layout import ensure_work_layout
from scorpio_pipe.stages.linearize import run_linearize
from scorpio_pipe.stages.sky_sub import run_sky_sub
from scorpio_pipe.stages.stack2d import run_stack2d
from scorpio_pipe.stages.extract1d import run_extract1d


def _make_synthetic_frame(shape=(32, 64), *, seed=0, obj_y=(14, 18)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ny, nx = shape

    # "Sky": smooth background + a few fake OH-like emission lines in x
    x = np.arange(nx)[None, :]
    sky = 50.0 + 0.2 * x + 8.0 * np.exp(-0.5 * ((x - 18.0) / 1.5) ** 2) + 6.0 * np.exp(-0.5 * ((x - 43.0) / 2.0) ** 2)

    # Object: narrow spatial profile + one emission feature
    y = np.arange(ny)[:, None]
    y0 = 0.5 * (obj_y[0] + obj_y[1])
    prof = np.exp(-0.5 * ((y - y0) / 1.2) ** 2)
    obj_line = 30.0 * np.exp(-0.5 * ((x - 32.0) / 2.2) ** 2)
    obj = prof * obj_line

    img = sky + obj
    # Poisson-ish noise + readout-ish noise
    img_noisy = rng.normal(loc=img, scale=np.sqrt(np.maximum(img, 1.0)) + 1.5)
    return img_noisy.astype(np.float32)


def _write_fits(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=data).writeto(path, overwrite=True)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="scorpipe_smoke_") as td:
        work_dir = Path(td) / "work"
        layout = ensure_work_layout(work_dir)

        # A minimal wavesolution product: lambda_map(x,y)
        # linear in x; constant in y
        ny, nx = 32, 64
        x = np.arange(nx)[None, :]
        lam_map = 5000.0 + 1.2 * x + np.zeros((ny, 1), dtype=float)
        wavesol_dir = work_dir / "wavesol"
        wavesol_dir.mkdir(parents=True, exist_ok=True)
        lambda_map_path = wavesol_dir / "lambda_map.fits"
        _write_fits(lambda_map_path, np.repeat(lam_map, ny, axis=0))

        # Two synthetic exposures
        raw1 = layout.raw / "obj_0001.fits"
        raw2 = layout.raw / "obj_0002.fits"
        _write_fits(raw1, _make_synthetic_frame(seed=1))
        _write_fits(raw2, _make_synthetic_frame(seed=2))

        cfg = {
            "work_dir": str(work_dir),
            "frames": {"obj": [str(raw1), str(raw2)]},
            "linearize": {
                "enabled": True,
                "lambda_map_path": str(lambda_map_path),
                # allow auto dlambda selection from data
                "dlambda_A": "auto",
                "grid_mode": "intersection",
                "per_exposure": True,
                "y_crop_top": 0,
                "y_crop_bottom": 0,
                "save_preview": True,
            },
            "sky": {
                "enabled": True,
                "per_exposure": True,
                "stack_after": True,
                "save_sky_model": True,
                "save_per_exp_model": True,
                "roi": {
                    # object zone + two sky zones (pixel units)
                    "obj_y0": 14,
                    "obj_y1": 18,
                    "sky_up_y0": 2,
                    "sky_up_y1": 10,
                    "sky_dn_y0": 22,
                    "sky_dn_y1": 30,
                },
                # keep a small window in the synthetic band as a "critical" zone
                "critical_windows_A": [[5150, 5250]],
            },
            "stack2d": {
                "enabled": True,
                "sigma_clip": 3.5,
                "max_iters": 3,
            },
            "extract1d": {
                "enabled": True,
                "mode": "boxcar",
                "aperture_half_width": 3,
            },
        }

        # Run pipeline stages (canonical order v5.40+):
        #   wavesol -> sky_sub (raw) -> linearize -> stack2d -> extract1d
        _ = run_sky_sub(cfg)
        lin_info = run_linearize(cfg)

        from scorpio_pipe.workspace_paths import stage_dir

        # gather per-exp rectified sky-subtracted frames
        lin_dir = stage_dir(work_dir, "linearize")
        sky_fits = sorted(lin_dir.glob("*_skysub.fits"))
        if not sky_fits:
            raise RuntimeError(f"No rectified sky-subtracted FITS in {lin_dir}")
        st_info = run_stack2d(cfg, inputs=sky_fits)
        ex_info = run_extract1d(cfg, stacked_fits=st_info["stacked2d_fits"])

        # Basic checks
        assert Path(lin_info["preview_fits"]).exists()
        assert Path(st_info["stacked2d_fits"]).exists()
        assert Path(ex_info["spec1d_fits"]).exists()

        print("SMOKE OK")
        print("work_dir:", work_dir)
        print("lin_preview:", lin_info["preview_fits"])
        print("stacked2d:", st_info["stacked2d_fits"])
        print("spec1d:", ex_info["spec1d_fits"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
