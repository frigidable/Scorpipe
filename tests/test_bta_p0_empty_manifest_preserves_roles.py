"""Real BTA 2024-12-06 P0: empty generated manifests must not erase FITS roles."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scorpio_pipe.project_manifest import (
    apply_project_manifest_to_cfg,
    default_project_manifest_dict,
)
from scorpio_pipe.work_layout import ensure_work_layout


def _fixture(tmp_path: Path):
    raw = tmp_path / "s241206"
    run = tmp_path / "06_12_2024" / "N2146sat_VPHG1200B_01"
    raw.mkdir()
    # Only the role-relevant files are created, never modified by the tests.
    names = [
        "s22800604.fts",
        *[f"s228006{n:02d}.fts" for n in range(7, 13)],
        "s22800606.fts",
        "s22801620.fts",
        "s22800408.fts",
    ]
    for name in names:
        (raw / name).write_bytes(b"raw-immutable")
    run.mkdir(parents=True)
    cfg = {
        "data_dir": str(raw),
        "work_dir": str(run),
        "config_dir": str(run),
        "frames": {
            "obj": [str(raw / "s22800604.fts")] +
                   [str(raw / f"s228006{n:02d}.fts") for n in range(7, 13)],
            "neon": [str(raw / "s22800606.fts")],
            "bias": [str(raw / "s22801620.fts")],
            "flat": [str(raw / "s22800408.fts")],
            "sky": [],
            "__setup__": {
                "object": "N2146sat", "disperser": "VPHG1200B",
                "slit": "1.2", "binning": "1x2",
            },
        },
    }
    return raw, run, cfg


def _write_manifest(run: Path, document: dict):
    path = run / "project_manifest.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_generated_empty_manifest_preserves_all_seven_science_and_calibrations(tmp_path):
    raw, run, cfg = _fixture(tmp_path)
    ensure_work_layout(run)  # In production this creates the all-empty template
    before = {key: list(val) for key, val in cfg["frames"].items()
              if isinstance(val, list)}
    apply_project_manifest_to_cfg(cfg)
    assert cfg["frames"]["obj"] == before["obj"]
    assert len(cfg["frames"]["obj"]) == 7
    assert cfg["frames"]["neon"] == before["neon"]
    assert cfg["frames"]["bias"] == before["bias"]
    assert cfg["frames"]["flat"] == before["flat"]
    assert cfg["frames"]["__setup__"]["disperser"] == "VPHG1200B"
    meta = cfg["_project_manifest"]
    assert meta["has_manifest"] is True
    assert meta["role_selection_status"] == "template_unconfigured"
    assert meta["roles_present"] == []
    assert meta["roles_applied"] == []
    assert meta["frames_source"] == "config"
    assert sorted(p.name for p in raw.iterdir()) == sorted(
        ["s22800604.fts"] +
        [f"s228006{n:02d}.fts" for n in range(7, 13)] +
        ["s22800606.fts", "s22801620.fts", "s22800408.fts"]
    )


def test_curated_role_overrides_while_other_present_empty_roles_stay_explicit(tmp_path):
    raw, run, cfg = _fixture(tmp_path)
    man = default_project_manifest_dict()
    man["roles"]["OBJECT_FRAMES"]["files"] = ["s22800607.fts"]
    _write_manifest(run, man)
    apply_project_manifest_to_cfg(cfg)
    assert cfg["frames"]["obj"] == [str(raw / "s22800607.fts")]
    assert cfg["frames"]["neon"] == []  # curated manifest explicitly empties it
    assert cfg["_project_manifest"]["role_selection_status"] == "explicit"
    assert "ARCS" in cfg["_project_manifest"]["roles_present"]
    assert cfg["_project_manifest"]["frames_source"] == "manifest"


def test_global_exclusion_only_preserves_selection_but_removes_excluded_frame(tmp_path):
    raw, run, cfg = _fixture(tmp_path)
    man = default_project_manifest_dict()
    man["exclude"]["files"] = ["s22800604.fts"]
    _write_manifest(run, man)
    apply_project_manifest_to_cfg(cfg)
    assert len(cfg["frames"]["obj"]) == 6
    assert str(raw / "s22800604.fts") not in cfg["frames"]["obj"]
    assert cfg["frames"]["neon"] == [str(raw / "s22800606.fts")]
    assert cfg["_project_manifest"]["frames_source"] == "config+manifest_exclude"
    assert cfg["_project_manifest"]["roles_present"] == []


def test_explicit_all_empty_roles_can_still_fail_closed(tmp_path):
    raw, run, cfg = _fixture(tmp_path)
    man = default_project_manifest_dict()
    man["role_policy"] = "explicit"
    _write_manifest(run, man)
    apply_project_manifest_to_cfg(cfg)
    assert all(cfg["frames"][key] == [] for key in
               ("obj", "neon", "bias", "flat", "sky"))
    assert cfg["_project_manifest"]["frames_source"] == "manifest"
    assert cfg["_project_manifest"]["role_selection_status"] == "explicit"


def test_valid_existing_explicit_manifest_is_not_reinterpreted(tmp_path):
    raw, run, cfg = _fixture(tmp_path)
    man = {"schema": "scorpio-pipe.project-manifest.v1.2",
           "roles": {"ARCS": {"files": ["s22800606.fts"], "globs": []}}}
    _write_manifest(run, man)
    apply_project_manifest_to_cfg(cfg)
    assert cfg["frames"]["neon"] == [str(raw / "s22800606.fts")]
    assert len(cfg["frames"]["obj"]) == 7  # unspecified roles survive
    assert cfg["_project_manifest"]["roles_applied"] == ["ARCS"]


def test_config_source_without_any_manifest_is_still_unmodified(tmp_path):
    raw, run, cfg = _fixture(tmp_path)
    apply_project_manifest_to_cfg(cfg)
    assert len(cfg["frames"]["obj"]) == 7
    assert cfg["_project_manifest"]["has_manifest"] is False
    assert cfg["_project_manifest"]["frames_source"] == "config"


def test_full_load_config_does_not_erase_yaml_frames_after_layout_creates_template(tmp_path):
    from scorpio_pipe.config import load_config

    raw, run, candidate = _fixture(tmp_path)
    path = run / "config.yaml"
    # This reproduces the actual GUI-generated on-disk YAML before the
    # work layout and project manifest are installed.
    path.write_text(yaml.safe_dump({
        "data_dir": str(raw),
        "object": "N2146sat",
        "work_dir": str(run),
        "frames": candidate["frames"],
    }, sort_keys=False), encoding="utf-8")
    assert not (run / "project_manifest.yaml").exists()
    effective = load_config(path)
    assert (run / "project_manifest.yaml").exists()
    assert len(effective["frames"]["obj"]) == 7
    assert len(effective["frames"]["neon"]) == 1
    assert len(effective["frames"]["bias"]) == 1
    assert len(effective["frames"]["flat"]) == 1
    assert effective["_project_manifest"]["frames_source"] == "config"
    # Ensure the raw source and saved YAML were not silently rewritten.
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert len(saved["frames"]["obj"]) == 7
    assert (raw / "s22800607.fts").read_bytes() == b"raw-immutable"
