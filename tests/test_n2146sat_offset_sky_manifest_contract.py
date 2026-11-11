"""Observer-confirmed 2024-12-06 N2146sat OFF/ON role integration regression.

Historical runs selected seven OBJ. The 600-s OBJ04 exposure is dedicated
OFFSET_SKY and must be excluded from the six-object scientific sample.
Tests do not open astronomical data or enable science processing.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scorpio_pipe.project_manifest import apply_project_manifest_to_cfg


def _name_set(paths: list[str]) -> set[str]:
    return {Path(p).name for p in paths}


def _historic_cfg(tmp_path: Path) -> dict:
    raw = tmp_path / "night"
    raw.mkdir()
    return {
        "work_dir": str(tmp_path / "new_isolated_run"),
        "config_dir": str(tmp_path),
        "data_dir": str(raw),
        "frames": {
            "obj": [str(raw / f"s228006{i:02d}.fts") for i in (4, 7, 8, 9, 10, 11, 12)],
            "sky": [],
            "bias": [str(raw / f"s228001{i:02d}.fts") for i in range(1, 21)],
            "neon": [str(raw / f"s228006{i:02d}.fts") for i in (6, 13)],
            "flat": [str(raw / f"s228006{i:02d}.fts") for i in (14, 15, 16)],
        },
    }


def _install_manifest(tmp_path: Path, cfg: dict, *, object_files: list[str], sky_files: list[str]) -> None:
    p = tmp_path / "explicit_roles.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "schema": "scorpio-pipe.project-manifest.v1.2",
                "roles": {
                    "OBJECT_FRAMES": {"files": object_files},
                    "SKY_FRAMES": {"files": sky_files},
                },
                "exclude": {"files": [], "globs": []},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    cfg["project_manifest_path"] = str(p)


def test_observer_confirmed_sky_replaces_historical_seven_obj(tmp_path: Path) -> None:
    cfg = _historic_cfg(tmp_path)
    obj_names = [f"s228006{i:02d}.fts" for i in range(7, 13)]
    _install_manifest(
        tmp_path, cfg, object_files=obj_names, sky_files=["s22800604.fts"]
    )
    result = apply_project_manifest_to_cfg(cfg)
    frames = result["frames"]
    assert _name_set(frames["obj"]) == set(obj_names)
    assert _name_set(frames["sky"]) == {"s22800604.fts"}
    assert not set(frames["obj"]) & set(frames["sky"])
    assert len(frames["bias"]) == 20
    assert _name_set(frames["neon"]) == {"s22800606.fts", "s22800613.fts"}
    assert _name_set(frames["flat"]) == {
        "s22800614.fts", "s22800615.fts", "s22800616.fts"
    }
    assert result["_project_manifest"]["n_sky_frames"] == 1
    assert result["_project_manifest"]["frames_source"] == "manifest"


def test_conflicting_object_and_sky_roles_rejected_fail_closed(tmp_path: Path) -> None:
    cfg = _historic_cfg(tmp_path)
    same_file_absolute = str(tmp_path / "night" / "s22800604.fts")
    _install_manifest(
        tmp_path,
        cfg,
        object_files=["s22800604.fts", "s22800607.fts"],
        sky_files=[same_file_absolute],
    )
    with pytest.raises(ValueError, match="both OBJECT_FRAMES and SKY_FRAMES"):
        apply_project_manifest_to_cfg(cfg)


def test_reassignment_is_not_a_global_exclusion(tmp_path: Path) -> None:
    cfg = _historic_cfg(tmp_path)
    _install_manifest(
        tmp_path, cfg,
        object_files=["s22800607.fts"],
        sky_files=["s22800604.fts"],
    )
    result = apply_project_manifest_to_cfg(cfg)
    assert "s22800604.fts" in _name_set(result["frames"]["sky"])
    assert "s22800604.fts" not in _name_set(result["frames"]["obj"])
    assert not result["_project_manifest"]["exclude"]
