from __future__ import annotations

import json
from pathlib import Path

import pytest


def _require_astropy() -> None:
    # Many CI/lightweight environments run contract/unit tests without Astropy.
    pytest.importorskip("astropy")


from scorpio_pipe.work_layout import ensure_work_layout
from scorpio_pipe.version import PIPELINE_VERSION


def _write_minimal_manifest(path: Path, *, pipeline_version: str = str(PIPELINE_VERSION)) -> None:
    payload = {
        "schema": "scorpio-pipe.dataset-manifest.v3",
        "schema_version": 3,
        "pipeline_version": pipeline_version,
        "generated_utc": "2026-01-30T00:00:00Z",
        "science_sets": [],
        "calibration_pools": {"bias": [], "flat": [], "arc": []},
        "matches": [],
        "warnings": [],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_linearize_forbids_lambda_map_rglob_when_manifest_present(tmp_path: Path) -> None:
    """P0-B11 regression: manifest present -> forbid legacy rglob discovery."""

    _require_astropy()
    from scorpio_pipe.stages.linearize import run_linearize

    work_dir = tmp_path / "work"
    ensure_work_layout(work_dir)

    # Create a validated dataset manifest.
    _write_minimal_manifest(work_dir / "dataset_manifest.json")

    # Place a "tempting" legacy lambda_map in a directory that would be found by rglob.
    legacy = work_dir / "wavesol" / "legacy"
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / "lambda_map.fits").write_bytes(b"FAKE")

    cfg = {"config_dir": str(tmp_path), "work_dir": str(work_dir)}

    with pytest.raises(FileNotFoundError) as e:
        run_linearize(cfg)

    msg = str(e.value).lower()
    assert "dataset_manifest" in msg and "disabled" in msg, str(e.value)
