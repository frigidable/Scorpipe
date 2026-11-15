"""Synthetic-only, nonqualifying pilot evidence concordance tests."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

FILE = Path(__file__).resolve().parents[1] / "scripts/qa/sc1_pilot_evidence_gate.py"


def _mod():
    s = importlib.util.spec_from_file_location("sc1_pilot_evidence_gate", FILE)
    assert s and s.loader
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _fixture(tmp_path):
    m = _mod()
    rawdir = tmp_path / "raw"
    rawdir.mkdir()
    olddir = tmp_path / "old_work"
    olddir.mkdir()
    qa = tmp_path / "qa"
    qa.mkdir()
    isolated = qa / "isolated_plan"
    isolated.mkdir()
    saved_path = olddir / "config.yaml"
    inv_path = qa / "original_raw_inventory.json"
    plan_path = isolated / "role_contract.json"

    names = {
        "bias": [f"s228001{i:02d}.fts" for i in range(1, 21)],
        "obj": ["s22800604.fts", *[f"s228006{i:02d}.fts" for i in range(7, 13)]],
        "flat": [f"s228007{i:02d}.fts" for i in range(1, 4)],
        "neon": ["s22800606.fts", "s22800613.fts"],
    }
    for group, groupnames in names.items():
        for name in groupnames:
            (rawdir / name).write_bytes(f"{group} {name} synthetic".encode())
    saved = {"work_dir": str(olddir), "data_dir": str(rawdir),
             "frames": {**{k: [str(rawdir/n) for n in v] for k,v in names.items()}, "sky": []}}
    saved_path.write_text(yaml.safe_dump(saved), encoding="utf-8")
    orig_sha = _sha(saved_path)
    entries = []
    for role, groupnames in names.items():
        for name in groupnames:
            p = rawdir / name
            entries.append({"role": role, "filename": name,
                            "size_bytes": p.stat().st_size,
                            "source_path": str(p), "source_sha256": _sha(p)})
    canonical = "\n".join(f"{e['role']}\t{e['filename']}\t{e['source_sha256']}" for e in entries)
    raw = {
        "schema": m.RAW_SCHEMA, "status": m.RAW_STATUS,
        "header_audit_mode": "audit_existing_readonly",
        "all_selected_primary_arrays_readable_and_finite": True,
        "stage_execution_authorized": False, "source_config": str(saved_path),
        "input_counts": m.ORIGINAL_COUNTS,
        "work_dir": str(olddir), "raw_data_dir": str(rawdir),
        "input_files": entries, "input_manifest_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }
    _save_json(inv_path, raw)
    selected = {k:[str(rawdir/n) for n in names[k]] for k in ("bias","flat","neon")}
    selected["obj"] = [str(rawdir/n) for n in m.ON]
    selected["sky"] = [str(rawdir/m.OFF)]
    isolated_config = isolated / "config.yaml"
    manifest_file = isolated / "project_manifest.yaml"
    iso = copy.deepcopy(saved)
    iso["work_dir"] = str(isolated)
    iso["project_manifest_path"] = str(manifest_file)
    iso["frames"] = {**selected}
    isolated_config.write_text(yaml.safe_dump(iso), encoding="utf-8")
    manifest = {"schema": "scorpio-pipe.project-manifest.v1.2", "role_policy": "explicit",
                "exclude": {"files": [], "globs": []},
                "roles": {"OBJECT_FRAMES": {"files": selected["obj"], "globs": [], "exclude": []},
                          "SKY_FRAMES": {"files": selected["sky"], "globs": [], "exclude": []}}}
    manifest_file.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    role = {
        "schema": m.ROLE_SCHEMA, "status": m.ROLE_STATUS,
        "source_config": str(saved_path), "source_config_sha256": orig_sha,
        "source_raw_inventory": str(inv_path), "source_raw_manifest_sha256": raw["input_manifest_sha256"],
        "isolated_config": str(isolated_config), "isolated_manifest": str(manifest_file),
        "resolved_manifest_path": str(manifest_file), "resolved_frames_source": "manifest",
        "resolved_roles": selected,
        "observational_roles": {m.OFF:"OFFSET_SKY_OBSERVER_CONFIRMED",
                                **{p:"ON_TARGET_SCIENCE" for p in m.ON}},
        "sky_spatial_cleanliness_verified": False, "sky_temporal_transfer_verified": False,
        "sky_subtraction_approved": False, "stack_pointing_approved": False,
        "wavelength_lsf_approved": False, "science_e2e_approved": False,
        "stage_execution_authorized": False,
    }
    _save_json(plan_path, role)
    return m, saved_path, inv_path, plan_path, qa / "pilot.json", raw, role


def test_consistent_historical_evidence_is_only_a_hold(tmp_path):
    m, saved, inv, role, out, _, _ = _fixture(tmp_path)
    original_sha = _sha(saved)
    record = m.prepare(saved, inv, role, out)
    assert record["status"] == "EVIDENCE_CONSISTENT_SCIENCE_HOLD"
    assert record["science_qualified"] is False
    assert record["sky_only_rows_certified"] == 0
    assert not record["raw_fits_pixels_read"]
    assert not record["raw_fits_source_hashes_recomputed"]
    assert record["stage_execution_authorized"] is False
    assert len(record["unresolved_science_gates"]) == 6
    assert set(record["unresolved_science_gates"].values()) == {"HOLD"}
    assert record["observational_selection"]["offset_sky"] == ["s22800604.fts"]
    assert _sha(saved) == original_sha
    assert json.loads(out.read_text())["observatory_release_approved"] is False


def test_saved_config_mutation_fail_closed(tmp_path):
    m, saved, inv, role, out, _, _ = _fixture(tmp_path)
    saved.write_bytes(saved.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="Saved config bytes changed"):
        m.prepare(saved, inv, role, out)
    assert not out.exists()


def test_mismatched_raw_manifest_fail_closed(tmp_path):
    m, saved, inv, role, out, raw, _ = _fixture(tmp_path)
    raw["input_files"][3]["source_sha256"] = "a" * 64
    _save_json(inv, raw)
    with pytest.raises(ValueError, match="pinned manifest digest"):
        m.prepare(saved, inv, role, out)
    assert not out.exists()


def test_swapped_off_sky_role_rejected(tmp_path):
    m, saved, inv, role_path, out, _, role = _fixture(tmp_path)
    role["resolved_roles"]["sky"] = role["resolved_roles"]["obj"][:1]
    _save_json(role_path, role)
    with pytest.raises(ValueError, match="dedicated OFF exposure"):
        m.prepare(saved, inv, role_path, out)
    assert not out.exists()


def test_isolated_manifest_edit_rejected(tmp_path):
    m, saved, inv, role, out, _, _ = _fixture(tmp_path)
    isolated_manifest = out.parent / "isolated_plan" / "project_manifest.yaml"
    doc = yaml.safe_load(isolated_manifest.read_text())
    doc["roles"]["SKY_FRAMES"]["files"] = doc["roles"]["OBJECT_FRAMES"]["files"][:1]
    isolated_manifest.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="Manifest SKY_FRAMES source selection changed"):
        m.prepare(saved, inv, role, out)
    assert not out.exists()


def test_unsafe_approval_in_role_rejected(tmp_path):
    m, saved, inv, role_file, out, _, role = _fixture(tmp_path)
    role["sky_subtraction_approved"] = True
    _save_json(role_file, role)
    with pytest.raises(ValueError, match="sky_subtraction_approved"):
        m.prepare(saved, inv, role_file, out)
    assert not out.exists()


def test_protected_outputs_and_no_overwrite(tmp_path):
    m, saved, inv, role, out, _, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="protected"):
        m.prepare(saved, inv, role, saved.parent / "extra.json")
    with pytest.raises(ValueError, match="protected"):
        m.prepare(saved, inv, role, out.parent / "isolated_plan" / "report.json")
    out.write_text("unchanged", encoding="utf-8")
    with pytest.raises(FileExistsError):
        m.prepare(saved, inv, role, out)
    assert out.read_text() == "unchanged"


def test_missing_raw_fits_prevents_false_inventory_success(tmp_path):
    m, saved, inv, role, out, _, _ = _fixture(tmp_path)
    (saved.parent.parent / "raw" / "s22800607.fts").unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        m.prepare(saved, inv, role, out)
    assert not out.exists()
