from pathlib import Path

import pytest

from scorpio_pipe.operational_presets import apply_setup_preset_defaults
from scorpio_pipe.schema import schema_validate
from scorpio_pipe.typed_config import build_typed_config, ConfigValidationError


def make_config(tmp_path: Path):
    return {
        "data_dir": str(tmp_path / "raw"),
        "work_dir": str(tmp_path / "run"),
        "object": "N2146sat",
        "project": {"wave_reference": "air"},
        "instrument": {
            "name": "SCORPIO",
            "mode": "Spectra",
            "disperser": "VPHG1200B",
            "slit": "1.2",
            "binning": "1x2",
        },
        "frames": {
            "obj": [],
            "__setup__": {
                "instrument": "SCORPIO",
                "mode": "Spectra",
                "disperser": "VPHG1200B",
                "slit": "1.2",
                "binning": "1x2",
            },
        },
    }


def test_gui_preset_schema_contract(tmp_path):
    cfg = apply_setup_preset_defaults(make_config(tmp_path))
    assert "_observer_setup_preset" in cfg
    assert "lsf_model" in cfg
    assert "sky_refine" in cfg
    report = schema_validate(cfg)
    assert report.ok, report.errors
    typed = build_typed_config(cfg)
    assert typed.project.wave_reference == "air"
    assert typed.lsf_model == cfg["lsf_model"]
    assert typed.sky_refine == cfg["sky_refine"]
    assert typed.observer_setup_preset == cfg["_observer_setup_preset"]
    assert "_observer_setup_preset" not in typed.model_dump(by_alias=True)


def test_runtime_profiles_and_unknown_keys(tmp_path):
    cfg = apply_setup_preset_defaults(make_config(tmp_path))
    cfg["_profiles_applied"] = ["test_profile"]
    typed = build_typed_config(cfg)
    assert typed.profiles_applied_runtime == ["test_profile"]
    cfg["invalid_scientific_typo"] = True
    with pytest.raises(ConfigValidationError, match="UNKNOWN_KEYS"):
        build_typed_config(cfg)
