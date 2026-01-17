from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy.io import fits  # noqa: E402

from scorpio_pipe.maskbits import SKYMODEL_FAIL
from scorpio_pipe.work_layout import ensure_work_layout
from scorpio_pipe.stages.sky_sub import run_sky_sub
from scorpio_pipe.workspace_paths import stage_dir


def _write_fits(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=data.astype("float32", copy=False)).writeto(path, overwrite=True)


def test_p0_j_sky_soft_mode_emits_passthrough_and_does_not_fail_stage():
    """P0-J: In soft mode, sky stage must not abort on missing sky windows.

    We intentionally force the auto-ROI to produce no usable sky windows,
    and assert that the stage:
      - writes a pass-through skysub product,
      - sets QADEGRD=1 and SKYMODEL_FAIL in MASK,
      - records WARN-level QC (not ERROR/FATAL).
    """

    rng = np.random.default_rng(42)

    with tempfile.TemporaryDirectory(prefix="scorpipe_pytest_p0j_sky_") as td:
        work_dir = Path(td) / "work"
        layout = ensure_work_layout(work_dir)

        ny, nx = 48, 96
        x = np.arange(nx, dtype=float)[None, :]

        # lambda_map required by the RAW branch
        lam = 5000.0 + 2.0 * x
        lambda_map = np.repeat(lam, ny, axis=0)
        wavesol = stage_dir(work_dir, "wavesol")
        _write_fits(wavesol / "lambda_map.fits", lambda_map)

        # Synthetic raw frame
        raw = 1000.0 + rng.normal(0.0, 5.0, size=(ny, nx))
        raw_path = layout.raw / "obj_0001.fits"
        _write_fits(raw_path, raw)

        cfg = {
            "work_dir": str(work_dir),
            "frames": {"obj": [str(raw_path)]},
            "sky": {
                "primary_method": "kelson_raw",
                "failure_policy": "soft",
                "save_per_exp_sky_model": True,
                # Force auto-ROI to mark almost everything as object by making the
                # detection threshold very low and dilation very wide.
                "geometry": {
                    "roi_policy": "auto",
                    "thresh_sigma": 0.0,
                    "dilation_px": 25,
                },
                "kelson_raw": {
                    "knot_step_A": 2.0,
                    "object_eating_warn": -1.0,
                },
            },
        }

        # Must not raise.
        outs = run_sky_sub(cfg)
        assert isinstance(outs, dict)

        sdir = stage_dir(work_dir, "sky")
        out_files = sorted(sdir.rglob("*_skysub_raw.fits"))
        assert out_files, "Expected pass-through *_skysub_raw.fits output"
        skysub_path = out_files[0]

        with fits.open(skysub_path) as hdul:
            phdr = hdul[0].header
            sci = hdul["SCI"].data.astype(float)
            var = hdul["VAR"].data.astype(float)
            mask = hdul["MASK"].data

        # pass-through: SCI unchanged.
        # Note: the helper writes the input frame as float32, so compare against the
        # float32-rounded values to avoid platform-specific float64 rounding noise.
        raw_written = raw.astype(np.float32).astype(float)
        assert np.allclose(sci, raw_written, rtol=0.0, atol=1e-6)
        assert np.all(np.isfinite(var))

        # Soft-degrade markers
        assert int(phdr.get("QADEGRD", 0)) == 1
        assert int(phdr.get("SKYOK", 1)) == 0

        # SKYMODEL_FAIL must be set everywhere (mask OR)
        assert int(np.count_nonzero(mask & SKYMODEL_FAIL)) == mask.size

        done = sdir / "done.json"
        assert done.exists(), "Sky stage must always write done.json"
        dd = json.loads(done.read_text(encoding="utf-8"))
        qc = dd.get("qc") if isinstance(dd.get("qc"), dict) else {}

        assert qc.get("max_severity") in {"WARN", "OK", "INFO"}
        flags = qc.get("flags") if isinstance(qc.get("flags"), list) else []
        codes = {f.get("code") for f in flags if isinstance(f, dict)}

        assert "SKY_SUB_PASSTHROUGH" in codes
        # Auto geometry and/or stage should emit NO_SKY_WINDOWS, but it must not be ERROR in soft mode.
        assert "NO_SKY_WINDOWS" in codes

        sev_map = {f.get("code"): f.get("severity") for f in flags if isinstance(f, dict)}
        assert sev_map.get("NO_SKY_WINDOWS") != "ERROR"
