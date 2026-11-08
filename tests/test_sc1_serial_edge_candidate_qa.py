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
    assert r["passes_provisional_candidate_quality_screen"] is True
    assert r["modal_value_fraction"] < .1


def test_candidate_profiles_show_optical_nonresponse_but_no_certification():
    m = load()
    a = np.full(m.EXPECTED_SHAPE, 10000, dtype=np.int16)
    a[:, :16] = 999
    a[:, 2064:] = 1000
    r = m.characterize(a)
    assert r["candidate_overscan_certified"] is False
    assert r["x_illuminated_section_certified"] is False
    # Perfectly constant edges could be software padding, not real overscan.
    assert r["right_minus_left_candidate_trimmed_mean_adu"] is None
    assert r["candidate"]["left"]["shows_minimum_pixel_variability"] is False
    assert r["candidate"]["right"]["passes_provisional_candidate_quality_screen"] is False
    assert r["neighbor_minus_candidate_adu"] == {"left": 9001, "right": 9000}
    assert len(r["candidate"]["left"]["column_medians_adu"]) == 16
    assert len(r["candidate"]["right"]["y_segment_trimmed_mean_adu"]) == 3
    assert r["vertical_extreme_row_profile_unverified"]["top_0_23"][
        "physical_parallel_overscan_certified"] is False
    assert len(r["vertical_extreme_row_profile_unverified"][
        "bottom_1008_1031"]["row_median_adu"]) == 24


def test_input_guards_nonfinite_and_wrong_shape():
    m = load()
    with pytest.raises(ValueError, match="Unexpected CCD shape"):
        m.characterize(np.zeros((1046,2068)))
    with pytest.raises(TypeError, match="numerical"):
        m.characterize(np.zeros(m.EXPECTED_SHAPE, dtype="U1"))
    with pytest.raises(ValueError, match="Invalid candidate stripe"):
        m.robust_sample_metrics([1,np.inf,3])


def test_high_rejection_fraction_reports_failed_provisional_screen_not_error():
    m = load()
    vals = np.concatenate((np.zeros(900), np.ones(100)*1000))
    result = m.robust_sample_metrics(vals)
    assert result["fraction_rejected_at_30adu"] == pytest.approx(0.1)
    assert result["passes_provisional_30adu_1pct_quality_screen"] is False
    assert result["trimmed_pixel_count"] == 900
    assert result["untrimmed_mean_adu"] == pytest.approx(100.0)
    assert result["trimmed_mean_adu"] == pytest.approx(0.0)


def test_realistic_illuminated_neighbor_gradient_is_descriptive_not_fatal():
    m = load()
    arr = np.full(m.EXPECTED_SHAPE, 999, dtype=np.int16)
    # During a science exposure the band bordering a non-illuminated
    # candidate strip may contain a strong illumination gradient.
    y = np.arange(m.EXPECTED_SHAPE[0], dtype=np.int16)
    arr[:, 16:32] = 1100 + (y // 2)[:, None]
    arr[:, 2048:2064] = 1200 + (y // 2)[:, None]
    result = m.characterize(arr)
    for side in ("left", "right"):
        assert result["neighbor"][side][
            "passes_provisional_30adu_1pct_quality_screen"] is False
        assert result["neighbor"][side]["fraction_rejected_at_30adu"] > 0.5
        assert result["candidate"][side][
            "passes_provisional_30adu_1pct_quality_screen"] is True
    # Both fixed 999-ADU bands are intentionally disqualified.
    assert result["candidate_both_sides_pass_provisional_quality_screen"] is False
    assert result["candidate"]["left"]["shows_minimum_pixel_variability"] is False
    assert result["neighbor_minus_candidate_adu"]["left"] > 100


def test_same_mode_bias_obj_gate_rejects_other_gain():
    m = load()
    def rec(group, gain):
        return {
            "filename": group+".fts", "header": {
                "NODE":"A", "RATE": "145.0" if gain==0.9 else "225.0",
                "GAIN": str(gain), "BINNING":"1x2", "DATE":"2024-12-06"
            },
            "statistics": {"candidate": {
                "left":{"trimmed_mean_adu":999.0,
                        "passes_provisional_candidate_quality_screen":True},
                "right":{"trimmed_mean_adu":1000.0,
                         "passes_provisional_candidate_quality_screen":True},
            }},
        }
    rs = {k: [] for k in m.REQUIRED_COUNTS}
    rs["bias_early20"] = [rec("bias",.9) for _ in range(20)]
    rs["obj_selected7"] = [rec("obj",.9)]
    rs["flat_selected3"] = [rec("flat",3.5)]
    rs["neon_selected2"] = [rec("neon",3.5)]
    result = m.summarize_modes(rs)
    assert result["science_epoch_offset_established"] is False
    assert result["physical_overscan_certified"] is False
    assert result["cross_mode_zero_subtraction_performed"] is False
    assert result["reference_quality_by_side"]["left"]["qualifying_frames"] == 20
    assert result["reference_quality_by_side"]["right"]["reference_available"]
    assert result["same_readout_temporal_candidate_edge_diagnostics"][-1][
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


def test_unqualified_candidate_is_retained_but_offset_withheld():
    m = load()
    def frame(name, quality_left, quality_right):
        return {
            "filename": name,
            "header":{"NODE":"A","RATE":"145","GAIN":"0.9",
                      "BINNING":"1x2","DATE":"2024-12-06"},
            "statistics":{"candidate":{
                "left":{"trimmed_mean_adu":999.2,
                        "passes_provisional_candidate_quality_screen":quality_left},
                "right":{"trimmed_mean_adu":999.5,
                         "passes_provisional_candidate_quality_screen":quality_right},
            }},
        }
    r={k:[] for k in m.REQUIRED_COUNTS}
    r["bias_early20"]=[frame("bias",True,True) for _ in range(20)]
    r["obj_selected7"]=[frame("obj",False,True)]
    s=m.summarize_modes(r)
    obs=s["same_readout_temporal_candidate_edge_diagnostics"][-1]
    assert obs["provisional_quality_left"] is False
    assert obs["provisional_quality_right"] is True
    assert obs["delta_left_against_early20_median_adu"] is None
    assert obs["delta_right_against_early20_median_adu"] == pytest.approx(0.0)
    assert obs["left_minus_right_delta_disagreement_adu"] is None
    assert obs["frame_candidate_status"] == "ONE_OR_BOTH_EDGES_UNQUALIFIED"


def test_too_few_quality_early_biases_withhold_reference_and_all_offsets():
    m = load()
    def frame(ok):
        return {
            "filename":"test.fts",
            "header":{"NODE":"A","RATE":"145","GAIN":"0.9",
                      "BINNING":"1x2","DATE":"2024-12-06"},
            "statistics":{"candidate":{
                side:{"trimmed_mean_adu":999.0,
                      "passes_provisional_candidate_quality_screen":ok}
                for side in ("left","right")
            }}
        }
    r={k:[] for k in m.REQUIRED_COUNTS}
    r["bias_early20"]=[frame(i < 15) for i in range(20)]
    r["obj_selected7"]=[frame(True)]
    s=m.summarize_modes(r)
    assert s["reference_quality_by_side"]["left"]["qualifying_frames"] == 15
    assert s["reference_early20_edge_trimmed_means_adu"]["left"] is None
    assert s["reference_early20_edge_trimmed_means_adu"]["right"] is None
    assert s["same_readout_temporal_candidate_edge_diagnostics"][-1][
        "delta_left_against_early20_median_adu"] is None


def test_variable_readout_like_candidate_passes_provisional_screen():
    m = load()
    rng = np.random.default_rng(173)
    a = np.full(m.EXPECTED_SHAPE, 2400, dtype=np.int16)
    for x0,x1 in m.EDGE.values():
        a[:,x0:x1] = np.rint(rng.normal(999, 3.0, (1032,16))).astype(np.int16)
    stats = m.characterize(a)
    assert stats["candidate_both_sides_pass_provisional_quality_screen"] is True
    assert stats["candidate"]["left"]["trimmed_std_adu"] > 2.0
    assert stats["candidate"]["right"]["trimmed_std_adu"] > 2.0
    assert stats["right_minus_left_candidate_trimmed_mean_adu"] is not None
    assert stats["candidate_overscan_certified"] is False
