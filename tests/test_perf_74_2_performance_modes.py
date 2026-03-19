import json
from pathlib import Path

from scorpio_pipe.perf.profiles import apply_performance_profile
from scorpio_pipe.prov_capture import ensure_run_provenance
from scorpio_pipe.qc_report import build_qc_report
from scorpio_pipe.work_layout import ensure_work_layout


def test_fast_profile_applies_and_records_overrides() -> None:
    cfg = {
        "work_dir": ".",
        "data_dir": ".",
        "runtime": {"performance_mode": "fast"},
        "wavesol": {"trace_y_step": 1, "poly_maxiter": 10, "geometry_prior": {"block_height": 16}},
        "sky": {
            "kelson_raw": {"max_iter": 4, "flexure_n_samples": 200000, "delta_step_A": 0.2},
            "sky_scale_raw": {"max_iter": 4},
        },
        "linearize": {"stack_maxiters": 2},
        "stack2d": {"robust_iter": 3, "var_floor_sample_rows": 96, "var_floor_sample_cols": 256},
        "cosmics": {"la_niter": 4, "lap_niter": 2},
        "sky_refine": {"detect_snr": 4.0, "top_n_summary": 10},
        "slit_sanity": {"max_candidate_frames": 12},
    }

    out = apply_performance_profile(cfg)
    prof = out["runtime"]["performance_profile"]

    assert out["runtime"]["performance_mode"] == "fast"
    assert prof["applied_mode"] == "fast"
    assert out["wavesol"]["trace_y_step"] == 2
    assert out["wavesol"]["geometry_prior"]["block_height"] == 32
    assert out["sky"]["kelson_raw"]["max_iter"] == 2
    assert out["sky"]["kelson_raw"]["flexure_n_samples"] == 50000
    assert out["sky"]["kelson_raw"]["delta_step_A"] == 0.4
    assert out["linearize"]["stack_maxiters"] == 1
    assert out["stack2d"]["robust_iter"] == 1
    assert out["cosmics"]["la_niter"] == 2
    assert out["sky_refine"]["detect_snr"] == 5.0
    assert out["sky_refine"]["top_n_summary"] == 6
    assert out["slit_sanity"]["max_candidate_frames"] == 4
    assert prof["n_overrides"] >= 6
    assert any(it.get("path") == "wavesol.trace_y_step" for it in prof.get("overrides", []))


def test_provenance_and_qc_report_surface_fast_mode(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    ensure_work_layout(work_dir)
    cfg = apply_performance_profile({
        "work_dir": str(work_dir),
        "data_dir": str(tmp_path),
        "runtime": {"performance_mode": "fast"},
        "frames": {},
    })

    prov = ensure_run_provenance(work_dir, cfg)
    sig = json.loads(Path(prov["run_signature_json"]).read_text(encoding="utf-8"))
    eff = Path(prov["effective_config_yaml"]).read_text(encoding="utf-8")

    assert sig["performance_mode"] == "fast"
    assert sig["performance_profile"]["n_overrides"] >= 1
    assert "performance_mode: fast" in eff
    assert "performance_profile:" in eff

    qc = build_qc_report(cfg)
    payload = json.loads(qc["json"].read_text(encoding="utf-8"))
    assert payload["runtime"]["performance_mode"] == "fast"
    alerts = payload.get("qc", {}).get("alerts", [])
    assert any(str(a.get("code") or "") == "QC_FAST_MODE_RUN" for a in alerts)
