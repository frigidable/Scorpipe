from __future__ import annotations

from pathlib import Path

import pytest

from scorpio_pipe.run_validate import RunLayoutError, validate_run_dir
from scorpio_pipe.work_layout import ensure_work_layout


def _mk_run(tmp_path: Path) -> Path:
    run_root = (
        tmp_path
        / "workspace"
        / "31_12_2025"
        / "ngc2146_VPHG1200@540_01"
    )
    ensure_work_layout(run_root)
    return run_root


def test_validate_run_dir_ok(tmp_path: Path) -> None:
    run_root = _mk_run(tmp_path)
    v = validate_run_dir(run_root, strict=True)
    assert v.ok
    assert v.night_date == "2025-12-31"
    assert v.object_name == "ngc2146"
    assert v.disperser == "VPHG1200@540"
    assert v.run_id == "01"


def test_validate_run_dir_requires_run_json(tmp_path: Path) -> None:
    run_root = _mk_run(tmp_path)
    p = run_root / "run.json"
    assert p.exists()
    p.unlink()
    with pytest.raises(RunLayoutError):
        validate_run_dir(run_root, strict=True)


def test_validate_run_dir_detects_mismatch(tmp_path: Path) -> None:
    run_root = _mk_run(tmp_path)
    p = run_root / "run.json"
    data = p.read_text(encoding="utf-8")
    # crude edit: replace object field
    p.write_text(data.replace('"object": "ngc2146"', '"object": "wrong"'), encoding="utf-8")
    with pytest.raises(RunLayoutError):
        validate_run_dir(run_root, strict=True)
