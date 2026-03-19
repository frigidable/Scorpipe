from __future__ import annotations

import json
from pathlib import Path

from scorpio_pipe.benchmark.certification import build_release_certification, write_release_certification
from scorpio_pipe.benchmark.long_compare import build_scorpipe_observer_bundle, write_observer_bundle


def test_build_scorpipe_observer_bundle_exports_canonical_metrics(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    wavesol = run_root / "08_wavesol"
    sky = run_root / "09_sky"
    extract = run_root / "12_extract"
    line_measure = run_root / "line_measure"
    wavesol.mkdir(parents=True, exist_ok=True)
    sky.mkdir(parents=True, exist_ok=True)
    extract.mkdir(parents=True, exist_ok=True)
    line_measure.mkdir(parents=True, exist_ok=True)

    (wavesol / "wavesolution_1d.json").write_text(json.dumps({"rms_A": 0.04, "dispersion_A_per_px": 0.87, "n_pairs": 18}), encoding="utf-8")
    (wavesol / "wavesolution_2d.json").write_text(json.dumps({"rms_A": 0.11, "n_used": 15}), encoding="utf-8")
    (wavesol / "wave_done.json").write_text(json.dumps({"metrics": {"rms2d_A": 0.11, "wave_trust_score": 0.79, "wave_line_density": 1.4}}), encoding="utf-8")
    (sky / "sky_done.json").write_text(json.dumps({"status": "ok", "metrics": {"robust_sigma_r_after_median": 2.8, "sky_coverage_frac_median": 0.82}, "extra": {"flexure": {"delta_A_median": 0.13, "flexure_score_median": 1.2}}}), encoding="utf-8")
    (sky / "qc_sky.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    sky_refine = run_root / "sky_refine"
    sky_refine.mkdir(parents=True, exist_ok=True)
    (sky_refine / "qc_sky_refine.json").write_text(json.dumps({"n_windows_total": 3, "aggregate_metrics": {"sky_sigma_gain_median": 0.22, "object_delta_median": -0.1}}), encoding="utf-8")
    (extract / "extract_done.json").write_text(json.dumps({"metrics": {"trace_stability_px": 0.08, "contamination_risk": 0.12}, "extra": {"extract": {"snr_trace_median": 14.5, "snr_fixed_median": 12.0}}}), encoding="utf-8")
    (extract / "extraction_qc.json").write_text(json.dumps({"diagnostics": {"trace_stability_px": 0.08, "contamination_risk": 0.12}}), encoding="utf-8")
    (line_measure / "science_lines.json").write_text(json.dumps({"lines": [{"detected": True}, {"detected": False}, {"detected": True}]}), encoding="utf-8")

    cfg = {"work_dir": str(run_root)}
    bundle = build_scorpipe_observer_bundle(cfg, run_root=run_root)
    payload = bundle.as_dict()

    assert payload["wavelength_solution"]["rms"] == 0.11
    assert payload["sky_subtraction"]["sky_rms"] == 2.8
    assert payload["extraction"]["snr"] == 14.5
    assert payload["science_outputs"]["n_detected_lines"] == 2

    out = write_observer_bundle(tmp_path / "bundle.json", bundle)
    assert out.exists()


def test_release_certification_marks_public_registry_incomplete(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    fixtures_dir = tmp_path / "fixtures"
    reports_dir = tmp_path / "reports"
    cases_dir.mkdir(parents=True, exist_ok=True)
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    sp_bundle = fixtures_dir / "real_sp.json"
    lg_bundle = fixtures_dir / "real_long.json"
    sp_bundle.write_text(json.dumps({
        "source": "sp",
        "wavelength_solution": {"rms": 0.10},
        "sky_subtraction": {"sky_rms": 1.0},
        "extraction": {"snr": 12.0},
        "response_calibration": {},
        "science_outputs": {"n_detected_lines": 3},
    }), encoding="utf-8")
    lg_bundle.write_text(json.dumps({
        "source": "long",
        "wavelength_solution": {"rms": 0.15},
        "sky_subtraction": {"sky_rms": 1.2},
        "extraction": {"snr": 10.0},
        "response_calibration": {},
        "science_outputs": {"n_detected_lines": 2},
    }), encoding="utf-8")

    (cases_dir / "real_case.case.json").write_text(json.dumps({
        "case_id": "real_case",
        "title": "Real case",
        "category": "reference",
        "source_type": "private",
        "description": "",
        "observer_notes": "",
        "scorpipe_bundle": "../fixtures/real_sp.json",
        "long_bundle": "../fixtures/real_long.json",
        "metrics": {
            "wavelength_solution.rms": {"prefer": "lower"},
            "sky_subtraction.sky_rms": {"prefer": "lower"},
            "extraction.snr": {"prefer": "higher"}
        }
    }), encoding="utf-8")

    registry = {
        "schema": "scorpio_pipe.long_parity_registry",
        "schema_version": 1,
        "minimum_real_cases_required": 2,
        "entries": [
            {"case_id": "mini_night_synthetic", "status": "fixture", "manifest": None, "category": "ci-fixture"},
            {"case_id": "real_case", "status": "ready", "manifest": "real_case.case.json", "category": "reference"},
            {"case_id": "pending_case", "status": "pending-private-data", "manifest": None, "category": "sky"},
        ],
    }
    registry_path = cases_dir / "registry.json"
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    golden = tmp_path / "golden_compare.json"
    golden.write_text(json.dumps({"status": "PASS", "regressions": [], "missing_in_current": []}), encoding="utf-8")
    limitations = tmp_path / "known_limitations.md"
    limitations.write_text("- Manual sign-off still pending.\n", encoding="utf-8")

    payload = build_release_certification(
        registry_path=registry_path,
        reports_root=reports_dir,
        golden_compare=golden,
        known_limitations_path=limitations,
        observer_signoff=None,
    )

    summary = payload["summary"]
    assert summary["status"] == "INCOMPLETE"
    assert summary["observatory_ready"] is False
    assert summary["populated_real_cases"] == 1
    assert summary["passed_real_cases"] == 1
    assert summary["pending_private_cases"] == 1
    assert summary["blocking_requirements"]
    assert any("observer sign-off" in item.lower() for item in summary["blocking_requirements"])
    assert any("Observer sign-off is missing" in item for item in payload["known_limitations"])
    assert (reports_dir / "real_case" / "long_parity_report.json").exists()

    out = write_release_certification(tmp_path / "cert", payload)
    assert out["json"].exists()
    assert out["markdown"].exists()
    assert out["html"].exists()
