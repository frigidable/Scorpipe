from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("astropy")

from scorpio_pipe.work_layout import ensure_work_layout
from scorpio_pipe.workspace_paths import stage_dir
from scorpio_pipe.stages.sky_sub import run_sky_sub
from scorpio_pipe.stages.extract1d import run_extract1d


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def test_sky_sub_writes_done_json_on_fail() -> None:
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td) / "workspace"
        ensure_work_layout(wd)
        cfg = {
            "workspace": {"root": str(wd)},
            "frames": {
                "obj": ["missing_raw.fits"],
            },
            "sky_sub": {"primary_method": "kelson_raw"},
        }

        with pytest.raises(Exception):
            run_sky_sub(cfg)

        dpath = stage_dir(wd, "sky") / "done.json"
        assert dpath.exists(), "done.json must be written even on failure"
        d = _read_json(dpath)
        assert d.get("status") == "fail"
        codes = {f.get("code") for f in d.get("flags", []) if isinstance(f, dict)}
        assert "STAGE_FAILED" in codes


def test_extract1d_writes_done_json_on_fail() -> None:
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td) / "workspace"
        ensure_work_layout(wd)

        cfg = {
            "workspace": {"root": str(wd)},
            "extract1d": {"enabled": True},
        }

        with pytest.raises(Exception):
            run_extract1d(cfg, in_fits=wd / "does_not_exist.fits")

        dpath = stage_dir(wd, "extract") / "done.json"
        assert dpath.exists(), "done.json must be written even on failure"
        d = _read_json(dpath)
        assert d.get("status") == "fail"
        codes = {f.get("code") for f in d.get("flags", []) if isinstance(f, dict)}
        assert "STAGE_FAILED" in codes
