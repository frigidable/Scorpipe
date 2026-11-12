"""Observer-confirmed offset sky and six target exposures: isolated role plan."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from scorpio_pipe.config import load_config


FILE = Path(__file__).resolve().parents[1] / "scripts/qa/sc1_offset_sky_role_plan.py"


def _module():
    spec = importlib.util.spec_from_file_location("sc1_offset_sky_role_plan", FILE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fixture(tmp_path):
    root = tmp_path / "old_run"
    raw = tmp_path / "night_raw"
    root.mkdir()
    raw.mkdir()
    old = root / "03_bias"
    old.mkdir()
    old_product = old / "masterbias.fits"
    old_product.write_bytes(b"previous science product - immutable")
    roles = {
        "bias": [f"s228001{i:02d}.fts" for i in range(1, 21)],
        "obj": ["s22800604.fts", *[f"s228006{i:02d}.fts" for i in range(7, 13)]],
        "flat": [f"flat{i:02d}.fts" for i in range(1, 4)],
        "neon": [f"neon{i:02d}.fts" for i in range(1, 3)],
    }
    entries = []
    for role, names in roles.items():
        for name in names:
            path = raw / name
            path.write_bytes(f"{role}/{name}/synthetic".encode("ascii"))
            entries.append(
                {"role": role, "filename": name, "source_path": str(path.resolve()),
                 "size_bytes": path.stat().st_size,
                 "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            )
    cfg = {
        "work_dir": str(root.resolve()),
        "data_dir": str(raw.resolve()),
        "project": {"wave_reference": "air"},
        "calib": {"bias_policy": "strict"},
        "flatfield": {"enabled": False},
        "superneon": {"bias_sub": False},
        "frames": {**roles, "sky": [], "standard": []},
    }
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    canonical = "\n".join(
        f"{x['role']}\t{x['filename']}\t{x['source_sha256']}"
        for x in entries
    )
    manifest_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    inventory = tmp_path / "sc1_inventory.json"
    inventory.write_text(json.dumps({
        "schema": "SC1_RAW_PIXEL_READABILITY_V1",
        "status": "READ_ONLY_PIXEL_READABILITY_NOT_SCIENCE_QUALIFICATION",
        "all_selected_primary_arrays_readable_and_finite": True,
        "source_config": str(config.resolve()),
        "input_manifest_sha256": manifest_hash,
        "input_files": entries,
    }), encoding="utf-8")
    return config, inventory, roles, old_product, raw


def test_actual_config_loader_resolves_six_obj_one_sky_without_original_writes(tmp_path):
    mod = _module()
    config, inventory, roles, product, raw = _fixture(tmp_path)
    baseline = {p: p.read_bytes() for p in [config, product, *raw.iterdir()]}
    # A pre-existing night-level manifest must NOT win over explicit new path.
    (raw / "project_manifest.yaml").write_text(yaml.safe_dump({
        "roles": {"OBJECT_FRAMES": {"files": ["s22800604.fts"]}},
    }), encoding="utf-8")
    baseline[raw / "project_manifest.yaml"] = (raw / "project_manifest.yaml").read_bytes()

    output = tmp_path / "external_qa" / "isolated_role_plan"
    result = mod.prepare(config, inventory, output)
    assert result["status"] == "ROLE_CONTRACT_RESOLVED_NO_SCIENCE_EXECUTION"
    assert result["stage_execution_authorized"] is False
    assert result["sky_subtraction_approved"] is False
    assert result["stack_pointing_approved"] is False
    assert result["source_raw_manifest_sha256"]
    assert [Path(p).name for p in result["resolved_roles"]["obj"]] == [
        f"s228006{i:02d}.fts" for i in range(7, 13)
    ]
    assert [Path(p).name for p in result["resolved_roles"]["sky"]] == [
        "s22800604.fts"
    ]
    assert set(result["resolved_roles"]["obj"]).isdisjoint(
        result["resolved_roles"]["sky"]
    )
    man = yaml.safe_load((output / "project_manifest.yaml").read_text())
    assert set(man["roles"]) == {"OBJECT_FRAMES", "SKY_FRAMES"}
    assert man["role_policy"] == "explicit"
    assert man["exclude"] == {"files": [], "globs": []}

    actual = load_config(output / "config.yaml")
    meta = actual["_project_manifest"]
    assert meta["frames_source"] == "manifest"
    assert meta["n_sky_frames"] == 1
    assert Path(meta["path"]) == output / "project_manifest.yaml"
    assert len(actual["frames"]["obj"]) == 6
    assert len(actual["frames"]["sky"]) == 1
    assert len(actual["frames"]["bias"]) == 20
    assert len(actual["frames"]["flat"]) == 3
    assert len(actual["frames"]["neon"]) == 2
    assert all(p.read_bytes() == content for p, content in baseline.items())
    assert (root := config.parent / "03_bias").exists()
    assert product.parent == root

    with pytest.raises(FileExistsError, match="Never overwrite"):
        mod.prepare(config, inventory, output)


def test_input_content_lock_rejects_changed_bytes_before_creating_output(tmp_path):
    mod = _module()
    config, inventory, roles, product, raw = _fixture(tmp_path)
    (raw / roles["obj"][0]).write_bytes(b"changed same-sized raw pixels")
    out = tmp_path / "external" / "isolated"
    with pytest.raises(ValueError, match="RAW bytes changed|RAW inventory selection"):
        mod.prepare(config, inventory, out)
    assert not out.exists()


def test_explicit_wrong_or_overlapping_obj_sky_selection_is_rejected(tmp_path):
    mod = _module()
    config, inventory, roles, product, raw = _fixture(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    cfg["frames"]["obj"][-1] = "s22800601.fts"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="not exactly OFF04"):
        mod.prepare(config, inventory, tmp_path / "external_wrong")
    assert not (tmp_path / "external_wrong").exists()

    cfg["frames"]["obj"][-1] = "s22800612.fts"
    cfg["frames"]["sky"] = ["s22800604.fts"]
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="Saved SKY must be empty"):
        mod.prepare(config, inventory, tmp_path / "external_overlap")


def test_refuse_plan_location_inside_existing_raw_or_work(tmp_path):
    mod = _module()
    config, inventory, _, _, raw = _fixture(tmp_path)
    with pytest.raises(ValueError, match="outside RAW"):
        mod.prepare(config, inventory, raw / "new_plan")
    with pytest.raises(ValueError, match="outside RAW"):
        mod.prepare(config, inventory, config.parent / "new_plan")
    assert not (raw / "new_plan").exists()
    assert not (config.parent / "new_plan").exists()


def test_bad_inventory_manifest_is_rejected_before_creating_plan(tmp_path):
    mod = _module()
    config, inventory, _, _, _ = _fixture(tmp_path)
    payload = json.loads(inventory.read_text())
    payload["input_manifest_sha256"] = "0" * 64
    inventory.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "new"
    with pytest.raises(ValueError, match="manifest digest"):
        mod.prepare(config, inventory, out)
    assert not out.exists()
