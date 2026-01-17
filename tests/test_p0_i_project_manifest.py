from __future__ import annotations

from pathlib import Path

import yaml

from scorpio_pipe.project_manifest import apply_project_manifest_to_cfg


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
        "schema": "scorpio-pipe.project-manifest.v1.1",
        "exclude": {"files": ["bad1.fits"], "globs": []},
        "roles": {"OBJECT_FRAMES": {"files": ["obj1.fits"], "globs": []}},
    }
    man_work = {
        "schema": "scorpio-pipe.project-manifest.v1.1",
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
        "schema": "scorpio-pipe.project-manifest.v1.1",
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
