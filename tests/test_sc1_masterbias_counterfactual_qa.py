"""Numerical regression checks for hypothetical gross-outlier MasterBias models.

This only validates diagnostic arithmetic; it does NOT validate clipped-median
uncertainty coverage under non-Gaussian detector contamination.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]/"scripts"/"qa"
sys.path.insert(0,str(ROOT))


def module():
    spec=importlib.util.spec_from_file_location("sc1_mbias_counterfactual",ROOT/"sc1_masterbias_counterfactual_qa.py")
    assert spec and spec.loader
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_counterfactual_selftest():
    module().selftest()


def test_clean_inputs_have_no_rejection_and_reproduce_old_median_var():
    m=module()
    rng=np.random.default_rng(1200540)
    a=1000+rng.normal(0,3,(20,6,8))
    original=a.copy()
    for thr in (30,50,100):
        out=m.candidate(a,thr)
        assert np.all(out["n"]==20)
        assert np.all(out["valid"])
        assert np.array_equal(out["old_med"],out["clean_med"])
        np.testing.assert_allclose(out["old_var"],out["plugin_var"],rtol=1e-12)
    assert np.array_equal(a,original)


def test_one_row_glitch_only_affects_one_input_sample_per_pixel():
    m=module()
    rng=np.random.default_rng(90210)
    a=1000+rng.normal(0,3,(20,4,8))
    a[0,0,:]+=559
    before=a.copy()
    out=m.candidate(a,100)
    assert np.all(out["n"][0,:]==19)
    assert np.all(out["n"][1:,:]==20)
    assert np.sum(~out["keep"])==8
    assert np.all(out["old_var"][0,:]>100)
    assert np.max(out["plugin_var"][0,:])<10
    assert np.array_equal(a,before)


def test_independent_extreme_hit_changes_variance_not_median_much():
    m=module()
    rng=np.random.default_rng(760)
    a=1000+rng.normal(0,3,(20,3,4))
    a[11,1,2]+=17000
    out=m.candidate(a,100)
    assert out["n"][1,2]==19
    assert out["old_var"][1,2]>100000
    assert out["plugin_var"][1,2]<10
    assert abs(out["old_med"][1,2]-out["clean_med"][1,2])<3
    c19=m.normal_median_variance_factor(19)
    expected=np.var(np.delete(a[:,1,2],11),ddof=1)*c19
    assert out["plugin_var"][1,2]==pytest.approx(expected)


def test_threshold_sensitivity_and_invalid_support_not_zeroed():
    m=module()
    a=np.full((20,2,2),1000.0)
    a[0,0,0]=1050
    low=m.candidate(a,30)
    high=m.candidate(a,100)
    assert low["n"][0,0]==19 and high["n"][0,0]==20
    a[:,1,1]=np.arange(20)*1000.0
    a[9,1,1]=9500.0
    a[10,1,1]=9500.0
    # all but the two central measurements are outliers at 100 ADU
    out=m.candidate(a,100)
    assert out["n"][1,1]==2
    assert out["valid"][1,1]
    with pytest.raises(ValueError,match="Threshold"):
        m.candidate(a,0)
    with pytest.raises(ValueError,match="Expected 20 finite"):
        m.candidate(a[:19],100)
