from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy.io import fits  # noqa: E402

from scorpio_pipe.work_layout import ensure_work_layout
from scorpio_pipe.stages.sky_sub import run_sky_sub
from scorpio_pipe.workspace_paths import stage_dir


def _robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def _write_fits(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=data.astype("float32", copy=False)).writeto(path, overwrite=True)


def test_p2_syn_sky_sub_raw_reduces_sky_residuals_and_keeps_object():
    """Sky subtraction (Kelson RAW) should learn on sky bands and not eat the object."""

    rng = np.random.default_rng(7)

    with tempfile.TemporaryDirectory(prefix="scorpipe_pytest_p2_sky_") as td:
        work_dir = Path(td) / "work"
        layout = ensure_work_layout(work_dir)

        ny, nx = 48, 128
        y = np.arange(ny)[:, None]
        x = np.arange(nx)[None, :]

        # Simple wavelength map for RAW branch (monotonic).
        lam = 5000.0 + 2.0 * x
        lambda_map = np.repeat(lam, ny, axis=0)
        wavesol = stage_dir(work_dir, "wavesol")
        _write_fits(wavesol / "lambda_map.fits", lambda_map)

        # Synthetic scene:
        # - sky: narrow lines + background
        # - object: Gaussian in y, smooth continuum in x
        bg = 20.0
        noise = 5.0

        line_centers = np.array([30.0, 60.0, 92.0])
        line_sigma = 1.0
        sky_lines = np.zeros((1, nx), dtype=float)
        for c in line_centers:
            sky_lines += 80.0 * np.exp(-0.5 * ((x - c) / line_sigma) ** 2)

        obj_y0 = ny / 2.0
        obj_sig_y = 2.0
        obj_profile = np.exp(-0.5 * ((y - obj_y0) / obj_sig_y) ** 2)
        obj_cont = 10.0 + 0.05 * (x - nx / 2.0)
        obj = 35.0 * obj_profile * obj_cont

        raw = bg + sky_lines + obj
        raw += rng.normal(0.0, noise, size=raw.shape)

        raw_path = layout.raw / "obj_0001.fits"
        _write_fits(raw_path, raw)

        cfg = {
            "work_dir": str(work_dir),
            "frames": {"obj": [str(raw_path)]},
            "roi": {
                "obj_y1": int(obj_y0 - 2),
                "obj_y2": int(obj_y0 + 2),
                "sky_y1": 2,
                "sky_y2": 10,
                "sky2_y1": ny - 11,
                "sky2_y2": ny - 3,
                "units": "px",
            },
            "sky_sub": {
                "primary_method": "kelson_raw",
                "save_per_exp_sky_model": True,
                "kelson_raw": {
                    # keep the model simple but non-trivial
                    "knot_step_A": 2.0,
                    "object_eating_warn": -1.0,
                },
            },
        }

        run_sky_sub(cfg)

        sdir = stage_dir(work_dir, "sky")
        out = sorted(sdir.rglob("*_skysub_raw.fits"))
        assert out, "Expected *_skysub_raw.fits output"
        skysub_path = out[0]

        skysub = fits.getdata(skysub_path).astype(float)

        # Sky bands should have fewer systematic residuals after subtraction.
        y1, y2 = cfg["roi"]["sky_y1"], cfg["roi"]["sky_y2"]
        y3, y4 = cfg["roi"]["sky2_y1"], cfg["roi"]["sky2_y2"]

        raw_sky = np.concatenate([raw[y1 : y2 + 1], raw[y3 : y4 + 1]], axis=0)
        sub_sky = np.concatenate([skysub[y1 : y2 + 1], skysub[y3 : y4 + 1]], axis=0)

        # exclude edges (coverage artifacts)
        raw_sky = raw_sky[:, 5:-5].ravel()
        sub_sky = sub_sky[:, 5:-5].ravel()

        s_before = _robust_sigma(raw_sky - np.median(raw_sky))
        s_after = _robust_sigma(sub_sky - np.median(sub_sky))

        assert s_after < 0.85 * s_before

        # Object band median should not be driven negative.
        oy1, oy2 = cfg["roi"]["obj_y1"], cfg["roi"]["obj_y2"]
        obj_med = float(np.median(skysub[oy1 : oy2 + 1, 5:-5]))
        assert obj_med > -0.5 * noise

        # If a risk metric is implemented, we must not raise the object-eating flag.
        done = sdir / "done.json"
        # Stage done.json is optional, but if present it must not claim object eating.
        import json

        if done.exists():
            dd = json.loads(done.read_text(encoding="utf-8"))
            qc = dd.get("qc") if isinstance(dd.get("qc"), dict) else {}
            flags = qc.get("flags") if isinstance(qc.get("flags"), list) else []
            codes = {f.get("code") for f in flags if isinstance(f, dict)}
            assert "OBJECT_EATING_RISK" not in codes
