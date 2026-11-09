"""Fail-closed test contracts for the isolated engineering E2E pilot."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1] /
    "scripts/qa/sc1_engineering_pilot.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("sc1_engineering_pilot", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pilot_selftest():
    load_module().selftest()


def test_pilot_must_be_new_sibling_and_canonical_run(tmp_path):
    m = load_module()
    orig = tmp_path / "source_01"
    orig.mkdir()
    cfg = orig / "config.yaml"
    cfg.write_text("work_dir: .")
    m.check_pilot_destination(cfg, tmp_path / "source_91")
    with pytest.raises(ValueError, match="SAME night"):
        m.check_pilot_destination(cfg, tmp_path / "different" / "source_91")
    with pytest.raises(ValueError, match="canonical"):
        m.check_pilot_destination(cfg, tmp_path / "pilot_name")
    with pytest.raises(FileExistsError):
        m.check_pilot_destination(cfg, orig)


def test_no_duplicate_or_missing_selected_raw_frames(tmp_path):
    m = load_module()
    paths = {}
    for key, count in m.EXPECTED.items():
        vals = []
        for i in range(1, count+1):
            name = (
                f"s228001{i:02d}.fts" if key=="bias"
                else f"{key}_{i}.fts"
            )
            p = tmp_path / name
            p.write_bytes(b"test")
            vals.append(str(p))
        paths[key] = vals
    m.check_expected_frames({"frames": paths})
    paths["obj"][1] = paths["obj"][0]
    with pytest.raises(RuntimeError, match="Duplicate"):
        m.check_expected_frames({"frames": paths})


def test_bias_cohort_must_be_exact_20_audited_files(tmp_path):
    m = load_module()
    paths = {}
    for key, count in m.EXPECTED.items():
        names = (
            [f"s228001{i:02d}.fts" for i in range(1,21)]
            if key == "bias" else
            [f"{key}_{i}.fts" for i in range(count)]
        )
        selected = []
        for name in names:
            p = tmp_path / name
            p.write_bytes(b"test")
            selected.append(str(p))
        paths[key] = selected
    outsider = tmp_path / "s22809999.fts"
    outsider.write_bytes(b"test")
    paths["bias"][-1] = str(outsider)
    with pytest.raises(RuntimeError, match="audited early20"):
        m.check_expected_frames({"frames": paths})


def test_orig_source_fingerprint_detects_source_config_change(tmp_path):
    m = load_module()
    folder = tmp_path / "test_01"
    folder.mkdir()
    cfg = folder / "config.yaml"
    cfg.write_text("calib: {}\n")
    baseline = m.fingerprint_sources(cfg)
    assert baseline[str(cfg)] == m.checksum(cfg)
    assert all(baseline[str(folder / f)] is None for f in m.ARTIFACTS)
    cfg.write_text("calib: {bias_policy: strict}\n")
    assert m.fingerprint_sources(cfg) != baseline


def test_pilot_does_not_use_engine_overrides_or_modify_original_calibration():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "resume=True, force=False" in source
    assert "qc_override=False, preflight_override=False" in source
    assert "source_work_string" in source
    assert "original_files_unchanged" in source
    assert "SCIENCE E2E: HOLD" in source
    assert "bias_gross_reject_adu" in source
    assert 'out_path.open("w"' not in source
