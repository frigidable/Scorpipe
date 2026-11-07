"""Controlled synthetic variance coverage for 20-sample quantized-Gaussian BIAS."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

P=Path(__file__).resolve().parents[1]/"scripts"/"qa"/"sc1_masterbias_variance_coverage_sim.py"


def load():
    spec=importlib.util.spec_from_file_location("sc1_mbias_coverage",P)
    assert spec is not None and spec.loader is not None
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_coverage_sim_selftest():
    load().selftest()


def test_quantized_normal_clean_and_row_glitch_reproducible():
    m=load()
    a=m.simulate(3.145,n_replicates=3000,seed=45)
    b=m.simulate(3.145,n_replicates=3000,seed=45)
    assert a==b
    clean=a["conditional_scenarios"]["clean"]
    glitch=a["conditional_scenarios"]["row_559"]
    hit=a["conditional_scenarios"]["transient_17166"]
    assert clean["support_counts"]=={"20":3000}
    assert glitch["support_counts"]=={"19":3000}
    assert hit["support_counts"]=={"19":3000}
    assert glitch["old_estimator"]["mean_reported_variance_adu2"]>1000
    assert glitch["candidate_clipped_estimator"]["mean_reported_variance_adu2"]<2
    assert hit["old_estimator"]["mean_reported_variance_adu2"]>100000
    assert hit["candidate_clipped_estimator"]["mean_reported_variance_adu2"]<2
    for scenario in (clean,glitch,hit):
        for name in ("old_estimator","candidate_clipped_estimator"):
            v=scenario[name]
            assert 0<=v["coverage_abs_err_le_reported_1sigma"]<=1
            assert 0<=v["coverage_abs_err_le_reported_1p96sigma"]<=1


@pytest.mark.parametrize("rn,reps,threshold",[
    (0,1000,50),(-3,1000,50),(3.145,99,50),(3.145,100,0),
])
def test_invalid_simulation_parameters_fail_closed(rn,reps,threshold):
    with pytest.raises(ValueError,match="Invalid"):
        load().simulate(rn,n_replicates=reps,threshold_adu=threshold)


def test_coverage_depends_on_sub_adu_bias_phase():
    m=load()
    s0=m.simulate(3.145,n_replicates=6000,seed=71,bias_offset_adu=999.0)
    s5=m.simulate(3.145,n_replicates=6000,seed=71,bias_offset_adu=999.5)
    c0=s0["conditional_scenarios"]["clean"]["candidate_clipped_estimator"]
    c5=s5["conditional_scenarios"]["clean"]["candidate_clipped_estimator"]
    assert abs(c0["coverage_abs_err_le_reported_1sigma"]
               -c5["coverage_abs_err_le_reported_1sigma"]) > 0.03
    assert s5["parameters"]["true_bias_offset_adu"]==999.5
