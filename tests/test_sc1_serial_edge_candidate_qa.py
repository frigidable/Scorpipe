"""Regression tests for read-only serial-edge candidate diagnostics."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load():
    p = (Path(__file__).resolve().parents[1] /
         "scripts/qa/sc1_serial_edge_candidate_qa.py")
    spec = importlib.util.spec_from_file_location("sc1_edge_candidate", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_edge_geometry_exactly_16_plus_2048_plus_16():
    m = load()
    assert m.EDGE == {"left": (0, 16), "right": (2064, 2080)}
    assert m.INNER == {"left": (16, 32), "right": (2048, 2064)}
    assert sum(b-a for a,b in m.EDGE.values())+2048 == 2080
    assert m.Y_START == 32 and m.Y_STOP == 992


def test_trimmed_candidate_distributions_preserve_real_read_noise():
    m = load()
    a = np.repeat(np.arange(994,1005), 100).astype(np.int16)
    a = np.append(a, [3000])
    r = m.robust_sample_metrics(a)
    assert r["median_adu"] == 999
    assert r["unique_integer_level_count"] == 12
    assert r["trimmed_pixel_count"] == 1100
    assert r["fraction_rejected_at_30adu"] == pytest.approx(1/1101)
    assert r["trimmed_std_adu"] > 0
    assert r["modal_value_fraction"] < .1


def test_candidate_profiles_show_optical_nonresponse_but_no_certification():
    m = load()
    a = np.full(m.EXPECTED_SHAPE, 10000, dtype=np.int16)
    a[:, :16] = 999
    a[:, 2064:] = 1000
    r = m.characterize(a)
    assert r["candidate_overscan_certified"] is False
    assert r["x_illuminated_section_certified"] is False
    assert r["right_minus_left_candidate_trimmed_mean_adu"] == 1
    assert r["neighbor_minus_candidate_adu"] == {"left": 9001, "right": 9000}
    assert len(r["candidate"]["left"]["column_medians_adu"]) == 16
    assert len(r["candidate"]["right"]["y_segment_trimmed_mean_adu"]) == 3


def test_input_guards_nonfinite_and_wrong_shape():
    m = load()
    with pytest.raises(ValueError, match="Unexpected CCD shape"):
        m.characterize(np.zeros((1046,2068)))
    with pytest.raises(TypeError, match="numerical"):
        m.characterize(np.zeros(m.EXPECTED_SHAPE, dtype="U1"))
    with pytest.raises(ValueError, match="Invalid candidate stripe"):
        m.robust_sample_metrics([1,np.inf,3])


def test_reject_if_edge_pixel_outlier_fraction_too_high():
    m = load()
    vals = np.concatenate((np.zeros(900), np.ones(100)*1000))
    with pytest.raises(ValueError, match="Excessive candidate edge"):
        m.robust_sample_metrics(vals)


def test_same_mode_bias_obj_gate_rejects_other_gain():
    m = load()
    def rec(group, gain):
        return {
            "filename": group+".fts", "header": {
                "NODE":"A", "RATE": "145.0" if gain==0.9 else "225.0",
                "GAIN": str(gain), "BINNING":"1x2", "DATE":"2024-12-06"
            },
            "statistics": {"candidate": {
                "left":{"trimmed_mean_adu":999.0},
                "right":{"trimmed_mean_adu":1000.0},
            }},
        }
    rs = {k: [] for k in m.REQUIRED_COUNTS}
    rs["bias_early20"] = [rec("bias",.9)]
    rs["obj_selected7"] = [rec("obj",.9)]
    rs["flat_selected3"] = [rec("flat",3.5)]
    rs["neon_selected2"] = [rec("neon",3.5)]
    result = m.summarize_modes(rs)
    assert result["science_epoch_offset_established"] is False
    assert result["physical_overscan_certified"] is False
    assert result["cross_mode_zero_subtraction_performed"] is False
    assert result["same_readout_temporal_candidate_edge_diagnostics"][1][
        "left_minus_right_delta_disagreement_adu"] == 0
    rs["obj_selected7"][0]["header"]["GAIN"] = "3.5"
    with pytest.raises(RuntimeError, match="Incompatible BIAS / OBJ"):
        m.summarize_modes(rs)


def test_script_emits_only_external_report_and_never_applies_bias():
    m = load()
    source = Path(m.__file__).read_text(encoding="utf8")
    assert 'out_path.open("x"' in source
    assert "OVERSCAN_UNVERIFIED_SCIENCE_HOLD" in source
    assert "SCIENCE E2E HOLD" in source
    assert "from sc1_bias_night_timeline_qa import" in source
    assert "corrected.fits" not in source
