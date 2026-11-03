"""Runtime project-manifest metadata must not defeat strict config validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scorpio_pipe.config import load_config
from scorpio_pipe.schema import find_unknown_keys, schema_validate
from scorpio_pipe.typed_config import ConfigValidationError, build_typed_config


def _minimal_config(tmp_path: Path) -> dict:
    return {
        "data_dir": str(tmp_path / "raw"),
        "work_dir": str(tmp_path / "work"),
        "frames": {
            "obj": [],
            "__setup__": {
                "instrument": "SCORPIO2",
                "mode": "Spectra",
                "disperser": "VPHG1200@540",
                "slit": "1.0",
                "binning": "1x1",
                "shape": "64x128",
            },
        },
    }


def test_loaded_config_manifest_runtime_metadata_passes_strict_gate(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(_minimal_config(tmp_path)), encoding="utf-8")

    cfg = load_config(cfg_path)
    assert isinstance(cfg["_project_manifest"], dict)
    assert cfg["_project_manifest"]["has_manifest"] in (True, False)

    report = schema_validate(cfg)
    assert report.ok, report.errors
    assert "_project_manifest" not in find_unknown_keys(cfg).get("top", [])

    typed = build_typed_config(cfg, config_path=cfg_path)
    assert typed.project_manifest_runtime == cfg["_project_manifest"]
    assert "_project_manifest" not in typed.model_dump(by_alias=True)


def test_unknown_top_level_key_still_fails_with_runtime_metadata(tmp_path: Path) -> None:
    cfg = _minimal_config(tmp_path)
    cfg["_project_manifest"] = {"has_manifest": False}
    cfg["invalid_scientific_typo"] = 1

    report = schema_validate(cfg)
    assert not report.ok
    assert "invalid_scientific_typo" in find_unknown_keys(cfg)["top"]
    with pytest.raises(ConfigValidationError, match="UNKNOWN_KEYS"):
        build_typed_config(cfg)
