"""Unqualified Stack2D covariance must not masquerade as line detections.

A line-window S/N based on fixed-weight conditional VAR is a diagnostic
candidate. It is not a validated detection or upper limit until the full
statistical uncertainty model is qualified.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.stages import line_measure as lm


def _candidate():
    return {
        "name": "Hbeta",
        "flux": 5.0,
        "err_stat": 0.5,
        "snr_stat": 10.0,
        "detected": True,
        "upper_limit_3sigma": 1.5,
        "reason": None,
    }


@pytest.mark.parametrize("header,expected_cause", [
    ({"STKVARQ": "COND_ONLY"}, "STACK_FIXED_WEIGHT_CONDITIONAL_VAR"),
    ({"STKXCVQ": "UNMODEL"}, "STACK_SHARED_COVARIANCE_UNMODELLED"),
    ({"STKYCVQ": "UNMODEL"}, "STACK_SPATIAL_COVARIANCE_UNMODELLED"),
    ({"SBCOVQ": "UNMODEL"}, "MASTERBIAS_COVARIANCE_UNMODELLED"),
])
def test_any_known_unmodelled_variance_blocks_formal_detection(
    header, expected_cause,
):
    rows, is_shared = lm._qualify_lines_for_shared_bias_covariance(
        [_candidate()], fits.Header(header)
    )
    assert is_shared is False
    row = rows[0]
    assert row["detected"] is False
    assert row["conditional_detection_candidate"] is True
    assert row["uncertainty_gate_triggered"] is True
    assert row["uncertainty_qualified"] is False
    assert expected_cause in row["unqualified_uncertainty_causes"]
    assert row["upper_limit_3sigma"] is None
    assert row["flux"] == 5.0
    assert row["snr_stat"] == 10.0
    assert row["reason"] == "uncertified_stack_or_calibration_uncertainty"


def test_shared_masterbias_retains_existing_specific_gate():
    rows, is_shared = lm._qualify_lines_for_shared_bias_covariance(
        [_candidate()],
        fits.Header({"SBCOVQ": "UNMODEL", "SBGSTAT": "SHARED"}),
    )
    assert is_shared is True
    row = rows[0]
    assert row["shared_bias_detection_gate_triggered"] is True
    assert row["conditional_detection_candidate"] is True
    assert row["detected"] is False
    assert row["upper_limit_3sigma"] is None
    assert row["uncertainty_status"] == "SHARED_MASTERBIAS_COVARIANCE_UNMODELLED"
    assert row["reason"] == "uncertified_shared_masterbias_uncertainty"


def test_partial_line_reason_stays_instead_of_being_overwritten():
    partial = _candidate()
    partial["detected"] = False
    partial["reason"] = "partial_aperture_or_spectral_coverage"
    rows, _ = lm._qualify_lines_for_shared_bias_covariance(
        [partial], fits.Header({"STKVARQ": "COND_ONLY"})
    )
    row = rows[0]
    assert row["conditional_detection_candidate"] is False
    assert row["reason"] == "partial_aperture_or_spectral_coverage"
    assert row["upper_limit_3sigma"] is None


def test_no_negative_status_is_not_by_itself_a_science_qualification():
    rows, is_shared = lm._qualify_lines_for_shared_bias_covariance(
        [_candidate()], fits.Header()
    )
    assert is_shared is False
    row = rows[0]
    assert row["detected"] is True
    assert row["uncertainty_qualified"] is None
    assert row["uncertainty_gate_triggered"] is False
    assert row["unqualified_uncertainty_causes"] == []


def test_line_measure_stage_reports_candidates_but_not_detections(
    tmp_path, monkeypatch,
):
    run_root = tmp_path / "run"
    header = fits.Header({
        "WAVEREF": "air", "WAVEUNIT": "Angstrom",
        "REBNORM": "COUNT_V1", "STKVARQ": "COND_ONLY",
        "STKXCVQ": "UNMODEL", "STKYCVQ": "UNMODEL",
    })
    wave = 4950.0 + np.arange(101, dtype=float)
    spectrum = {
        "wave": wave,
        "flux": np.ones_like(wave),
        "var": np.ones_like(wave),
        "mask": np.zeros_like(wave, dtype=np.uint16),
        "header": header,
        "dq1d_version": None,
    }
    paths = {
        "fluxcal_spec1d_fits": run_root / "science.fits",
        "calibration_systematics_fits": run_root / "calsys.fits",
        "fluxcal_qc_json": run_root / "fluxcal_qc.json",
    }
    monkeypatch.setattr(
        lm, "PRODUCT_CATALOG",
        SimpleNamespace(best_path=lambda _cfg, key, **_kw: paths[key]),
    )
    monkeypatch.setattr(lm, "read_spec1d", lambda *_a, **_kw: spectrum)
    z = np.zeros_like(wave)
    monkeypatch.setattr(
        lm, "_read_calsys", lambda *_a, **_kw: {"rel": z, "tell": z, "abs": z},
    )
    monkeypatch.setattr(lm, "measure_lines", lambda *_a, **_kw: [_candidate()])
    monkeypatch.setattr(lm, "balmer_decrement_summary", lambda _rows: None)

    def fake_csv(path, _rows):
        p = Path(path)
        p.write_text("name\n", encoding="utf-8")
        return p

    def fake_png(path, *_args):
        p = Path(path)
        p.write_bytes(b"synthetic")
        return p

    monkeypatch.setattr(lm, "write_lines_csv", fake_csv)
    monkeypatch.setattr(lm, "plot_line_measurements_png", fake_png)
    monkeypatch.setattr(lm, "write_done_json", lambda **_kw: None)

    stage_path = run_root / "line_measure"
    result = lm.run_line_measure({}, run_root=run_root, stage_dir_path=stage_path)
    assert result is not None
    data = json.loads((stage_path / "science_lines.json").read_text(encoding="utf-8"))
    qc = json.loads((stage_path / "line_measure_plan.json").read_text(encoding="utf-8"))["qc"]
    assert data["n_conditional_detection_candidates"] == 1
    assert data["uncertainty_detection_gate_triggered"] is True
    assert data["lines"][0]["detected"] is False
    assert data["lines"][0]["conditional_detection_candidate"] is True
    assert qc["n_conditional_detection_candidates"] == 1
    assert qc["n_lines_detected"] == 0
