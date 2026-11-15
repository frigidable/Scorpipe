"""Synthetic Gate B0: detector header-section evidence, NOT overscan science."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qa/sc1_ccd_section_header_audit.py"
PILOT_TEST = Path(__file__).with_name("test_sc1_pilot_evidence_gate.py")


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _setup(tmp_path, monkeypatch):
    module = _module(SCRIPT, "sc1_ccd_header")
    pilot_fixture = _module(PILOT_TEST, "sc1_pilot_fixture_helper")
    pilot, saved, inv, role, report, _, _ = pilot_fixture._fixture(tmp_path)
    pilot.prepare(saved, inv, role, report)

    def synthetic_header(path):
        s = module.parse_section("[2049:2080,1:1032]")
        return {
            "cards": {"NAXIS1": "2080", "NAXIS2": "1032",
                      "NODE": "A", "RATE": "145", "GAIN": "0.9",
                      "BINNING": "1x2"},
            "declared_sections": {"BIASSEC": s},
            "section_conflicts": [], "unparseable_sections": [],
        }
    monkeypatch.setattr(module, "inspect_header", synthetic_header)
    return module, report, tmp_path / "qa" / "ccd_geometry.json", inv


def test_section_indexing_and_reversal():
    module = _module(SCRIPT, "sections_parser")
    good = module.parse_section("[2049:2080,1:1032]")
    assert good["parse_ok"]
    assert good["python_x_zero_based_half_open"] == [2048, 2080]
    assert good["python_y_zero_based_half_open"] == [0, 1032]
    reverse = module.parse_section("[2080:2049,1032:1]")
    assert reverse["parse_ok"]
    assert reverse["reversed_x"] and reverse["reversed_y"]
    assert reverse["python_x_zero_based_half_open"] == [2048, 2080]


def test_invalid_section_and_overlap():
    module = _module(SCRIPT, "sections_invalid")
    assert not module.parse_section("[0:32,1:1032]")["parse_ok"]
    assert not module.parse_section("[2070:2090,1:1032]")["parse_ok"]
    assert not module.parse_section("unknown")["parse_ok"]
    assert module.overlap(module.parse_section("[1:2048,1:1032]"),
                          module.parse_section("[2040:2080,1:1032]"))


def test_32_original_shas_are_verified_but_hold_remains(tmp_path, monkeypatch):
    module, pilot, out, _ = _setup(tmp_path, monkeypatch)
    result = module.prepare(pilot, out)
    assert result["raw_frames_sha256_reverified"] == 32
    assert result["original_raw_fits_headers_examined"] == 32
    assert result["declared_section_keyword_counts"] == {"BIASSEC": 32}
    assert result["science_qualified"] is False
    assert result["physical_overscan_region_certified"] is False
    assert result["overscan_correction_approved"] is False
    assert result["raw_fits_pixels_read"] is False
    assert result["stage_execution_authorized"] is False
    assert result["science_hold"] is True
    assert json.loads(out.read_text())["raw_frames_sha256_reverified"] == 32


def test_same_size_raw_byte_mutation_fails_closed(tmp_path, monkeypatch):
    module, pilot, out, _ = _setup(tmp_path, monkeypatch)
    p = tmp_path / "raw" / "s22800604.fts"
    p.write_bytes(b"X" * p.stat().st_size)
    with pytest.raises(ValueError, match="Original FITS changed"):
        module.prepare(pilot, out)
    assert not out.exists()


def test_changed_raw_inventory_fails_closed(tmp_path, monkeypatch):
    module, pilot, out, raw = _setup(tmp_path, monkeypatch)
    raw.write_bytes(raw.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="raw_inventory_json_sha256"):
        module.prepare(pilot, out)
    assert not out.exists()


def test_unsafe_science_approval_cannot_enter_gate_b(tmp_path, monkeypatch):
    module, pilot, out, _ = _setup(tmp_path, monkeypatch)
    record = json.loads(pilot.read_text())
    record["science_qualified"] = True
    pilot.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Pilot P0 evidence"):
        module.prepare(pilot, out)
    assert not out.exists()


def test_protected_outputs_and_existing_file_are_rejected(tmp_path, monkeypatch):
    module, pilot, out, _ = _setup(tmp_path, monkeypatch)
    for target in (
        tmp_path / "raw" / "audit.json",
        tmp_path / "old_work" / "audit.json",
        tmp_path / "qa" / "isolated_plan" / "audit.json",
    ):
        with pytest.raises(ValueError, match="outside source"):
            module.prepare(pilot, target)
        assert not target.exists()
    out.write_text("preserve original")
    with pytest.raises(FileExistsError):
        module.prepare(pilot, out)
    assert out.read_text() == "preserve original"


def test_conflicting_declared_sections_never_approve_overscan(tmp_path, monkeypatch):
    module, pilot, out, _ = _setup(tmp_path, monkeypatch)

    def conflicting_header(path):
        return {
            "cards": {},
            "declared_sections": {
                "DATASEC": module.parse_section("[1:2048,1:1032]"),
                "BIASSEC": module.parse_section("[2040:2080,1:1032]"),
            },
            "section_conflicts": ["DECLARED_DATASEC_BIASSEC_OVERLAP"],
            "unparseable_sections": [],
        }
    monkeypatch.setattr(module, "inspect_header", conflicting_header)
    result = module.prepare(pilot, out)
    assert result["any_section_overlaps"]
    assert result["physical_overscan_region_certified"] is False
    assert result["science_hold"] is True
