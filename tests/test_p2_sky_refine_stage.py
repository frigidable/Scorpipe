from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy.io import fits  # noqa: E402

from scorpio_pipe.io.mef import write_sci_var_mask
from scorpio_pipe.sky_residuals import run_sky_residual_refine
from scorpio_pipe.workspace_paths import stage_dir


def _robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return float(1.4826 * mad)


def _write_primary(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=data.astype("float32", copy=False)).writeto(path, overwrite=True)


def test_p2_sky_refine_reduces_sky_residuals_and_preserves_object() -> None:
    rng = np.random.default_rng(12)

    with tempfile.TemporaryDirectory(prefix="scorpipe_pytest_p2_sky_refine_") as td:
        work_dir = Path(td) / "work"
        sky_dir = stage_dir(work_dir, "sky")
        wavesol_dir = stage_dir(work_dir, "wavesol")
        refine_dir = stage_dir(work_dir, "sky_refine")
        sky_dir.mkdir(parents=True, exist_ok=True)
        wavesol_dir.mkdir(parents=True, exist_ok=True)
        refine_dir.mkdir(parents=True, exist_ok=True)

        ny, nx = 56, 180
        y = np.arange(ny)[:, None]
        x = np.arange(nx)[None, :]
        lam = 5000.0 + 0.9 * x
        lam_map = np.repeat(lam, ny, axis=0)
        _write_primary(wavesol_dir / "lambda_map.fits", lam_map)

        line_centers = np.array([26.0, 64.0, 102.0, 145.0])
        line_sigma = 1.25
        sky_model = np.zeros((ny, nx), dtype=float)
        for c in line_centers:
            sky_model += 80.0 * np.exp(-0.5 * ((x - c) / line_sigma) ** 2)
        sky_model += 8.0

        grad = np.gradient(sky_model, axis=1)

        obj_y0 = ny / 2.0
        obj_sig = 2.2
        obj_prof = np.exp(-0.5 * ((y - obj_y0) / obj_sig) ** 2)
        obj_cont = 18.0 + 0.03 * (x - nx / 2.0)
        obj = 22.0 * obj_prof * obj_cont

        residual = 0.14 * sky_model + 0.35 * grad
        skysub = obj + residual + rng.normal(0.0, 1.6, size=(ny, nx))
        var = np.full((ny, nx), 3.2, dtype=np.float32)
        mask = np.zeros((ny, nx), dtype=np.uint16)

        stem = "obj_0001"
        raw_path = work_dir / "raw" / f"{stem}.fits"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(b"dummy")

        write_sci_var_mask(sky_dir / f"{stem}_skysub_raw.fits", skysub.astype(np.float32), var=var, mask=mask)
        write_sci_var_mask(sky_dir / f"{stem}_skymodel_raw.fits", sky_model.astype(np.float32), var=var, mask=mask)

        cfg = {
            "work_dir": str(work_dir),
            "frames": {"obj": [str(raw_path)]},
            "roi": {
                "obj_y1": int(obj_y0 - 2),
                "obj_y2": int(obj_y0 + 2),
                "sky_y1": 3,
                "sky_y2": 13,
                "sky2_y1": ny - 14,
                "sky2_y2": ny - 4,
                "units": "px",
            },
            "sky_sub": {
                "geometry": {
                    "roi_policy": "prefer_user",
                    "edge_margin_px": 4,
                }
            },
            "sky_refine": {
                "detect_sigma": 1.5,
                "detect_snr": 3.5,
                "strength": 1.0,
                "lsf_strength": 0.6,
                "object_protection": {
                    "continuum_sigma": 2.0,
                    "line_sigma": 3.0,
                    "asym_low_px": 1,
                    "asym_high_px": 3,
                },
            },
        }

        run_sky_residual_refine(cfg, out_dir=refine_dir)

        ref_path = refine_dir / f"{stem}_skysub_refined_raw.fits"
        corr_path = refine_dir / f"{stem}_skycorr_raw.fits"
        assert ref_path.exists()
        assert corr_path.exists()
        assert (refine_dir / "done.json").exists()
        assert (refine_dir / "qc_sky_refine.json").exists()
        assert (refine_dir / "sky_residual_windows.json").exists()
        assert (refine_dir / "sky_observer_pack.md").exists()

        refined = fits.getdata(ref_path, ext=1).astype(float)

        sky_rows = np.r_[3:14, ny - 14 : ny - 3]
        before_sig = _robust_sigma(skysub[sky_rows, 4:-4])
        after_sig = _robust_sigma(refined[sky_rows, 4:-4])
        assert after_sig < 0.90 * before_sig

        obj_rows = slice(int(obj_y0 - 2), int(obj_y0 + 3))
        obj_delta = float(np.nanmedian(refined[obj_rows, 10:-10] - skysub[obj_rows, 10:-10]))
        assert obj_delta > -2.0

        windows = json.loads((refine_dir / "sky_residual_windows.json").read_text(encoding="utf-8"))
        assert windows, "expected problematic skyline windows to be detected"
        assert any(w.get("action") == "apply" for w in windows)
