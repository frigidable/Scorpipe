from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy.io import fits  # noqa: E402

from scorpio_pipe.ui.pipeline_runner import TASKS  # noqa: E402
from scorpio_pipe.workspace_paths import stage_dir  # noqa: E402


def _mk_run_root(tmp_path: Path) -> Path:
    # Fully canonical layout is not required for stage-level functions, but we
    # keep the contract form for clarity.
    run_root = tmp_path / "workspace" / "31_12_2025" / "ngc2146_VPHG1200@540_01"
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def _write_fits(path: Path, shape: tuple[int, int] = (10, 20), *, waveunit: str | None = None, waveref: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros(shape, dtype=np.float32)
    hdu = fits.PrimaryHDU(data)
    if waveunit is not None:
        hdu.header["WAVEUNIT"] = str(waveunit)
    if waveref is not None:
        hdu.header["WAVEREF"] = str(waveref)
    hdu.writeto(path, overwrite=True)


def test_stack2d_requires_linearize_skysub_products(tmp_path: Path) -> None:
    run_root = _mk_run_root(tmp_path)
    cfg = {"work_dir": str(run_root), "config_dir": str(tmp_path)}

    # A tempting but forbidden fallback: sky products exist in 09_sky, but stack2d
    # must only use 10_linearize.
    sky_dir = stage_dir(run_root, "sky")
    (sky_dir).mkdir(parents=True, exist_ok=True)
    _write_fits(sky_dir / "obj1_skysub.fits")

    with pytest.raises(FileNotFoundError) as e:
        TASKS["stack2d"](cfg, run_root)
    msg = str(e.value)
    assert "10_linearize" in msg


def test_extract1d_requires_stack2d(tmp_path: Path) -> None:
    run_root = _mk_run_root(tmp_path)
    cfg = {"work_dir": str(run_root), "config_dir": str(tmp_path)}

    with pytest.raises(FileNotFoundError) as e:
        TASKS["extract1d"](cfg, run_root)
    assert "11_stack" in str(e.value)


def test_linearize_requires_sky_and_rectification_model(tmp_path: Path) -> None:
    run_root = _mk_run_root(tmp_path)

    # Minimal raw obj frame list.
    raw = run_root / "raw"
    obj_path = raw / "obj1.fits"
    _write_fits(obj_path, shape=(10, 20))

    # Minimal wavesol products required to reach the sky-product check.
    wsol_dir = stage_dir(run_root, "wavesol")
    _write_fits(wsol_dir / "lambda_map.fits", shape=(10, 20), waveunit="Angstrom", waveref="air")

    # rectification_model.json is required in non-legacy mode.
    rect_model_path = wsol_dir / "rectification_model.json"
    rect_model_path.write_text(
        json.dumps({"schema": 1, "lambda_map": {"path": "lambda_map.fits"}}, indent=2),
        encoding="utf-8",
    )

    cfg = {
        "work_dir": str(run_root),
        "config_dir": str(tmp_path),
        "frames": {"obj": [str(obj_path)]},
    }

    # No 09_sky/<stem>_skysub_raw.fits -> must fail (no silent fallback).
    with pytest.raises(FileNotFoundError) as e:
        TASKS["linearize"](cfg, run_root)
    assert "09_sky" in str(e.value)