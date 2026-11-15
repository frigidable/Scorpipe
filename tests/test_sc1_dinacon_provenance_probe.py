"""Synthetic-only DINACON provenance intake tests. Never certify overscan."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "qa"
PILOT_FIXTURE = Path(__file__).with_name("test_sc1_pilot_evidence_gate.py")


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup(tmp_path, monkeypatch):
    helper = _module(PILOT_FIXTURE, "sc1_pilot_fixture")
    p0_mod, saved, inv, role, p0_path, _, _ = helper._fixture(tmp_path)
    p0_mod.prepare(saved, inv, role, p0_path)
    b0_mod = _module(SCRIPTS / "sc1_ccd_section_header_audit.py", "sc1_b0")
    monkeypatch.setattr(
        b0_mod, "inspect_header",
        lambda path: {
            "cards": {"NODE": "A", "RATE": "145", "GAIN": "0.9",
                      "BINNING": "1x2", "NAXIS1": "2080", "NAXIS2": "1032"},
            "declared_sections": {}, "section_conflicts": [],
            "unparseable_sections": [],
        }
    )
    b0_path = tmp_path / "qa" / "b0.json"
    b0_mod.prepare(p0_path, b0_path)
    mod = _module(SCRIPTS / "sc1_dinacon_provenance_probe.py", "sc1_dinacon")
    monkeypatch.setattr(
        mod, "probe_header",
        lambda path: {"cards": {"ORIGIN": "SYNTHETIC"},
                      "comment_history_sample": []}
    )
    scan = tmp_path / "technical"
    scan.mkdir()
    out = tmp_path / "qa" / "dinacon_discovery.json"
    return mod, p0_path, b0_path, scan, out


def test_scoped_finder_and_five_header_anchors_remain_hold(tmp_path, monkeypatch):
    mod, p0, b0, scan, out = _setup(tmp_path, monkeypatch)
    (scan / "chip.cfg").write_text("synthetic, not authentic", encoding="utf-8")
    (scan / "controller.ngr").write_bytes(b"synthetic diagram")
    (scan / "irrelevant.txt").write_text("ignore", encoding="utf-8")
    before = (scan / "chip.cfg").read_bytes()
    record = mod.prepare(p0, b0, [scan], out)
    assert record["status"] == "TECHNICAL_PROVENANCE_DISCOVERY_ONLY_SCIENCE_HOLD"
    assert record["raw_anchor_headers_read"] == 5
    assert record["raw_anchor_source_sha256_reverified"] == 5
    assert record["raw_fits_pixel_arrays_read"] is False
    assert len(record["technical_file_candidates"]) == 2
    assert record["candidate_counts"]["DINASYSTEM_NAMED_CONFIGURATION"] == 1
    assert all(not x["2024_night_origin_authenticated"]
               for x in record["technical_file_candidates"])
    assert record["control_2024_readout_configuration_found_and_authenticated"] is False
    assert record["physical_overscan_region_certified"] is False
    assert record["stage_execution_authorized"] is False
    assert record["science_qualified"] is False
    assert record["science_hold"] is True
    assert (scan / "chip.cfg").read_bytes() == before
    assert json.loads(out.read_text())["source_controller_firmware_authenticated"] is False


def test_empty_search_does_not_prove_configuration_absent(tmp_path, monkeypatch):
    mod, p0, b0, scan, out = _setup(tmp_path, monkeypatch)
    record = mod.prepare(p0, b0, [scan], out)
    assert record["technical_file_candidates"] == []
    assert record["control_2024_readout_configuration_found_and_authenticated"] is False
    assert record["science_hold"]


def test_source_kind_is_narrow_and_does_not_run_files(tmp_path):
    mod = _module(SCRIPTS / "sc1_dinacon_provenance_probe.py", "sc1_kind")
    for x in ("chip.cfg", "fits_.hdr", "init_ccd.cfg", "tele_foo.cfg",
              "clock.ngr", "controller.ldr", "dina_run.cmd"):
        assert mod.source_kind(x)
    for x in ("config.yaml", "data.fts", "image.png", "make.bat", "hello.py"):
        assert mod.source_kind(x) is None


def test_mutated_raw_anchor_is_rejected(tmp_path, monkeypatch):
    mod, p0, b0, scan, out = _setup(tmp_path, monkeypatch)
    path = tmp_path / "raw" / "s22800604.fts"
    path.write_bytes(b"X" * path.stat().st_size)
    with pytest.raises(ValueError, match="Anchor RAW source changed"):
        mod.prepare(p0, b0, [scan], out)
    assert not out.exists()


def test_wrong_p0_b0_chain_rejected(tmp_path, monkeypatch):
    mod, p0, b0, scan, out = _setup(tmp_path, monkeypatch)
    p0.write_bytes(p0.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="B0 source evidence"):
        mod.prepare(p0, b0, [scan], out)
    assert not out.exists()


def test_unsafe_b0_flags_rejected(tmp_path, monkeypatch):
    mod, p0, b0, scan, out = _setup(tmp_path, monkeypatch)
    obj = json.loads(b0.read_text())
    obj["physical_overscan_region_certified"] = True
    b0.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="B0 source evidence"):
        mod.prepare(p0, b0, [scan], out)
    assert not out.exists()


def test_existing_and_protected_outputs_rejected(tmp_path, monkeypatch):
    mod, p0, b0, scan, out = _setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="protected"):
        mod.prepare(p0, b0, [scan], tmp_path / "raw" / "result.json")
    out.write_text("unchanged", encoding="utf-8")
    with pytest.raises(ValueError, match="nonexisting"):
        mod.prepare(p0, b0, [scan], out)
    assert out.read_text() == "unchanged"


def test_symlinks_are_not_followed_and_scan_limits_are_bounded(tmp_path, monkeypatch):
    mod, p0, b0, scan, out = _setup(tmp_path, monkeypatch)
    (scan / ".git").mkdir()
    (scan / ".git" / "chip.cfg").write_text("do-not-read")
    (scan / "source.cfg").write_text("not a known technical file")
    rec = mod.prepare(p0, b0, [scan], out)
    assert rec["technical_file_candidates"] == []
    assert rec["science_hold"] is True


def test_symlinked_pilot_evidence_path_is_refused(tmp_path, monkeypatch):
    mod, p0, b0, scan, out = _setup(tmp_path, monkeypatch)
    link = tmp_path / "qa" / "linked_p0.json"
    try:
        link.symlink_to(p0)
    except (OSError, NotImplementedError):
        pytest.skip("Platform does not allow symlinks")
    with pytest.raises(ValueError, match="Symlinked evidence"):
        mod.prepare(link, b0, [scan], out)
    assert not out.exists()
