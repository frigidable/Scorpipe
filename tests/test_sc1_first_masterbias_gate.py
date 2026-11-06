"""No production writes: guard exact first SCORPIO-1 MasterBias selection."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.io import fits

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qa" / "sc1_first_masterbias_gate.py"


def module():
    spec = importlib.util.spec_from_file_location("sc1_first_bias_gate", SCRIPT)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.EXPECTED_SHAPE = (6, 8)
    return m


def case(tmp_path):
    run = tmp_path / "run"
    data = tmp_path / "raw"
    run.mkdir(parents=True)
    data.mkdir(parents=True)
    selected = {}
    for role, count in (("bias", 20), ("obj", 7), ("flat", 3), ("neon", 2)):
        selected[role] = []
        for j in range(1, count+1):
            name = (f"s228001{j:02d}.fts" if role == "bias"
                    else f"{role}_{j:02d}.fts")
            p = data / name
            hdu = fits.PrimaryHDU(np.full((6, 8), 999, dtype=np.float32))
            hdu.header["RATE"] = 145 if role in ("bias", "obj") else 225
            hdu.header["GAIN"] = 0.9 if role in ("bias", "obj") else 3.5
            hdu.header["NODE"] = "A"
            hdu.header["BINNING"] = "1x2"
            hdu.writeto(p)
            selected[role].append(name)
    c = {
        "work_dir": str(run),
        "data_dir": str(data),
        "project": {"wave_reference": "air"},
        "calib": {"bias_policy": "strict"},
        "flatfield": {"enabled": False},
        "superneon": {"bias_sub": False},
        "frames": selected,
    }
    config = run / "config.yaml"
    config.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
    return config, c, data


def test_gate_selftest():
    module().selftest()


def test_valid_exact_night_header_gate_is_readonly(tmp_path):
    m = module()
    conf, cfg, data = case(tmp_path)
    all_files = [conf] + list(data.iterdir())
    baseline = {p: p.read_bytes() for p in all_files}
    report, paths = m.collect(conf)
    assert len(paths) == 32
    assert report["input_counts"] == {"bias": 20, "obj": 7, "flat": 3, "neon": 2}
    assert report["audited_early20_bias_confirmed"] is True
    assert report["status"] == "PRECHECK_PASSED_NOT_SCIENCE_VALIDATION"
    assert "RATE" not in report["readout_modes"]["bias"]  # tokenized readout
    out = tmp_path / "qa" / "precheck.json"
    m.run(conf, out, execute=False)
    assert out.exists()
    for p in all_files:
        assert p.read_bytes() == baseline[p]
    assert len(list(conf.parent.glob("??_*"))) == 0


def test_invalid_science_bias_mode_fails_closed(tmp_path):
    m = module()
    conf, cfg, data = case(tmp_path)
    target = data / cfg["frames"]["obj"][0]
    with fits.open(target, mode="update") as hdu:
        hdu[0].header["GAIN"] = 0.92
    with pytest.raises(RuntimeError, match="Unexpected readout"):
        m.collect(conf)


def test_missing_frame_fails_closed(tmp_path):
    m = module()
    conf, cfg, data = case(tmp_path)
    (data / cfg["frames"]["neon"][0]).unlink()
    with pytest.raises(FileNotFoundError):
        m.collect(conf)


def test_duplicate_bias_selection_fails_closed(tmp_path):
    m = module()
    conf, cfg, data = case(tmp_path)
    cfg["frames"]["bias"][-1] = cfg["frames"]["bias"][0]
    conf.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Duplicate FITS selection"):
        m.collect(conf)


def test_distinct_but_wrong_bias_cohort_fails_closed(tmp_path):
    m = module()
    conf, cfg, data = case(tmp_path)
    old_name = cfg["frames"]["bias"][-1]
    replacement_name = "s22800121.fts"
    (data / replacement_name).write_bytes((data / old_name).read_bytes())
    cfg["frames"]["bias"][-1] = replacement_name
    conf.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(RuntimeError, match="audited early20 cohort"):
        m.collect(conf)


def test_fatal_policy_and_prior_calibration_products(tmp_path):
    m = module()
    conf, cfg, data = case(tmp_path)
    cfg["calib"]["bias_policy"] = "degraded"
    conf.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(RuntimeError, match="strict"):
        m.collect(conf)
    cfg["calib"]["bias_policy"] = "strict"
    conf.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (conf.parent / "02_superbias").mkdir()
    with pytest.raises(RuntimeError, match="Prior stage products"):
        m.collect(conf)


def test_air_convention_is_explicit_and_wrong_value_fails_before_build(tmp_path):
    m = module()
    conf, cfg, data = case(tmp_path)
    assert m.collect(conf)[0]["status"] == "PRECHECK_PASSED_NOT_SCIENCE_VALIDATION"
    for value in ("UNKNOWN", "vacuum", ""):
        cfg["project"]["wave_reference"] = value
        conf.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        with pytest.raises(RuntimeError, match="project.wave_reference=air"):
            m.collect(conf)


def test_autoconfig_emits_air_for_scorpio_instead_of_unknown():
    from scorpio_pipe.autocfg import AutoConfig
    for instrument in ("SCORPIO", "SCORPIO-2"):
        c = AutoConfig(
            data_dir="D:/synthetic",
            object_name="Test",
            frames={"__setup__": {"instrument": instrument, "mode": "Spectra",
                                    "disperser": "VPHG1200", "slit": "1.0",
                                    "binning": "1x2", "window": "", "shape": "1032x2080"}},
            work_dir=".",
        )
        data = yaml.safe_load(c.to_yaml_text())
        assert data["project"]["wave_reference"] == "air"
