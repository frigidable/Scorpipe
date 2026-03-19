from __future__ import annotations

from pathlib import Path

import yaml

from scorpio_pipe.project_manifest import (
    ProjectManifestModel,
    apply_project_manifest_to_cfg,
    default_project_manifest_dict,
    read_project_manifest,
    write_project_manifest,
)


def test_project_manifest_search_prefers_data_dir(tmp_path: Path) -> None:
    """P0-I: night-level manifest in data_dir has highest priority."""

    data_dir = tmp_path / "night"
    work_dir = tmp_path / "work" / "run1"
    config_dir = tmp_path / "cfg"
    data_dir.mkdir(parents=True)
    work_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)

    # Two manifests: data_dir should win.
    man_data = {
        "schema": "scorpio-pipe.project-manifest.v1.2",
        "exclude": {"files": ["bad1.fits"], "globs": []},
        "roles": {"OBJECT_FRAMES": {"files": ["obj1.fits"], "globs": []}},
    }
    man_work = {
        "schema": "scorpio-pipe.project-manifest.v1.2",
        "roles": {"OBJECT_FRAMES": {"files": ["obj_work.fits"], "globs": []}},
    }
    (data_dir / "project_manifest.yaml").write_text(yaml.safe_dump(man_data, sort_keys=False), encoding="utf-8")
    (work_dir / "project_manifest.yaml").write_text(yaml.safe_dump(man_work, sort_keys=False), encoding="utf-8")

    # Create dummy files so resolution works.
    (data_dir / "obj1.fits").write_text("", encoding="utf-8")
    (data_dir / "bad1.fits").write_text("", encoding="utf-8")
    (work_dir / "obj_work.fits").write_text("", encoding="utf-8")

    cfg = {
        "data_dir": str(data_dir),
        "work_dir": str(work_dir),
        "config_dir": str(config_dir),
        "frames": {"obj": [str(data_dir / "obj1.fits"), str(work_dir / "obj_work.fits")]},
    }

    apply_project_manifest_to_cfg(cfg)
    assert cfg["_project_manifest"]["path"] == str(data_dir / "project_manifest.yaml")
    # Role applied from data_dir manifest
    assert cfg["frames"]["obj"] == [str(data_dir / "obj1.fits")]
    # Global exclude propagated
    assert str(data_dir / "bad1.fits") in (cfg.get("exclude_frames") or [])


def test_global_exclude_removes_from_roles(tmp_path: Path) -> None:
    data_dir = tmp_path / "night"
    work_dir = tmp_path / "work" / "run1"
    data_dir.mkdir(parents=True)
    work_dir.mkdir(parents=True)

    (data_dir / "a.fits").write_text("", encoding="utf-8")
    (data_dir / "b.fits").write_text("", encoding="utf-8")

    man = {
        "schema": "scorpio-pipe.project-manifest.v1.2",
        "exclude": {"files": ["b.fits"], "globs": []},
        "roles": {
            "OBJECT_FRAMES": {"files": ["a.fits", "b.fits"], "globs": []},
            "BIAS": {"files": ["b.fits"], "globs": []},
        },
    }
    (data_dir / "project_manifest.yaml").write_text(yaml.safe_dump(man, sort_keys=False), encoding="utf-8")

    cfg = {
        "data_dir": str(data_dir),
        "work_dir": str(work_dir),
        "frames": {},
    }
    apply_project_manifest_to_cfg(cfg)
    assert cfg["frames"]["obj"] == [str(data_dir / "a.fits")]
    assert cfg["frames"].get("bias", []) == []


def test_p0_1_default_manifest_includes_standard_roles() -> None:
    man = default_project_manifest_dict()
    roles = man["roles"]

    assert man["schema"] == "scorpio-pipe.project-manifest.v1.2"
    assert "STANDARD_FRAMES" in roles
    assert "TELLURIC_STANDARD_FRAMES" in roles
    assert roles["STANDARD_FRAMES"] == {"files": [], "globs": []}
    assert roles["TELLURIC_STANDARD_FRAMES"] == {"files": [], "globs": []}


def test_p0_1_legacy_shape_coerces_standard_roles_and_round_trips(tmp_path: Path) -> None:
    pm = ProjectManifestModel.model_validate(
        {
            "schema": "scorpio-pipe.project-manifest.v1.2",
            "standard_frames": ["std1.fits", "std2.fits"],
            "telluric_standard_frames": ["tell1.fits"],
            "object_frames": ["obj1.fits"],
            "exclude_frames": ["bad1.fits"],
        }
    )

    assert pm.standard_frames == ["std1.fits", "std2.fits"]
    assert pm.telluric_standard_frames == ["tell1.fits"]
    assert pm.object_frames == ["obj1.fits"]
    assert pm.exclude_files == ["bad1.fits"]

    out = pm.to_manifest_dict()
    assert out["roles"]["STANDARD_FRAMES"]["files"] == ["std1.fits", "std2.fits"]
    assert out["roles"]["TELLURIC_STANDARD_FRAMES"]["files"] == ["tell1.fits"]

    path = tmp_path / "project_manifest.yaml"
    write_project_manifest(pm, path)
    pm2 = read_project_manifest(path)
    assert pm2.standard_frames == ["std1.fits", "std2.fits"]
    assert pm2.telluric_standard_frames == ["tell1.fits"]
    assert pm2.exclude_files == ["bad1.fits"]


def test_p0_1_apply_manifest_supports_standard_roles(tmp_path: Path) -> None:
    data_dir = tmp_path / "night"
    work_dir = tmp_path / "work" / "run1"
    data_dir.mkdir(parents=True)
    work_dir.mkdir(parents=True)

    for name in ("std1.fits", "std2.fits", "tell1.fits"):
        (data_dir / name).write_text("", encoding="utf-8")

    man = {
        "schema": "scorpio-pipe.project-manifest.v1.2",
        "roles": {
            "STANDARD_FRAMES": {"files": ["std1.fits", "std2.fits"], "globs": [], "exclude": ["std2.fits"]},
            "TELLURIC_STANDARD_FRAMES": {"files": ["tell1.fits"], "globs": []},
        },
    }
    (data_dir / "project_manifest.yaml").write_text(yaml.safe_dump(man, sort_keys=False), encoding="utf-8")

    cfg = {
        "data_dir": str(data_dir),
        "work_dir": str(work_dir),
        "frames": {},
    }

    apply_project_manifest_to_cfg(cfg)

    assert cfg["frames"]["standard"] == [str(data_dir / "std1.fits")]
    assert cfg["frames"]["telluric_standard"] == [str(data_dir / "tell1.fits")]
    assert "STANDARD_FRAMES" in cfg["_project_manifest"]["roles_present"]
    assert "TELLURIC_STANDARD_FRAMES" in cfg["_project_manifest"]["roles_present"]
