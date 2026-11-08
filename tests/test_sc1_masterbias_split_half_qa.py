"""Numerical regression tests for real-night 10+10 MasterBias split diagnostics."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

QA=Path(__file__).resolve().parents[1]/"scripts"/"qa"
if str(QA) not in sys.path:
    sys.path.insert(0,str(QA))


def load():
    spec=importlib.util.spec_from_file_location("sc1_bias_split",QA/"sc1_masterbias_split_half_qa.py")
    assert spec and spec.loader
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selftest():
    load().selftest()


def test_partitions_are_balanced_disjoint_and_cover_all_20():
    p=load().splits()
    assert len(p)==3
    for a,b in p.values():
        assert len(a)==len(b)==10
        assert not set(a)&set(b)
        assert sorted(list(a)+list(b))==list(range(20))


def test_clean_half_unchanged_by_extreme_threshold():
    m=load()
    rng=np.random.default_rng(2026)
    stack=999.+rng.normal(0,2.8,size=(10,5,7))
    original=m.estimate_half(stack)
    candidate=m.estimate_half(stack,threshold=50.)
    assert np.array_equal(original[0],candidate[0])
    np.testing.assert_allclose(original[1],candidate[1],rtol=1e-12)
    assert np.all(candidate[2]==10)
    assert candidate[3]==0


def test_half_single_row_corruption_is_one_sample_not_one_deleted_frame():
    m=load()
    rng=np.random.default_rng(20261107)
    stack=999.+rng.normal(0,3.0,size=(10,4,10))
    stack[0,0,:]+=559
    orig=m.estimate_half(stack)
    clipped=m.estimate_half(stack,threshold=50)
    assert np.all(clipped[2][0,:]==9)
    assert np.all(clipped[2][1:,:]==10)
    assert clipped[3]==10
    assert float(np.median(orig[1][0]))>1000
    assert float(np.median(clipped[1][0]))<10


def test_nonfinite_too_few_frames_and_invalid_threshold_fail():
    m=load()
    stack=np.ones((10,4,5))
    for arr in (stack[:9],np.full((10,4,5),np.nan)):
        with pytest.raises(ValueError,match="Expected ten finite"):
            m.estimate_half(arr)
    with pytest.raises(ValueError,match="Invalid rejection"):
        m.estimate_half(stack,threshold=0)
    with pytest.raises(ValueError,match="Invalid rejection"):
        m.estimate_half(stack,threshold=-1)


def test_roi_64px_tiles_and_split_half_coverage_accumulator():
    m=load()
    a=m._stat_accum()
    rng=np.random.default_rng(441)
    delta=rng.normal(0,1.0,size=(32,2080))
    variance=np.ones(delta.shape)*2.0
    count=np.full(delta.shape,10,dtype=np.int16)
    # Start at y32 within candidate ROI; 32 of 64 rows for first tile row.
    m._add(a,delta,variance,32,count,count,0)
    assert a["n"]==32*1984
    assert a["excluded_samples"]==0
    assert a["valid_n"]==a["n"]
    assert a["tile_counts"][0,0]==2048
    assert a["tile_counts"][0,30]==2048
    assert np.all(a["tile_counts"][1:,:]==0)
    delta2=rng.normal(0,1.0,size=(32,2080))
    m._add(a,delta2,variance,64,count,count,0)
    assert a["tile_counts"][0,0]==4096
    assert a["tile_counts"][1,0]==0
    # The same first 64-row tile must be completed by y[32:64] AND
    # y[64:96], so *both* chunks belong to tile row 0.
    result=m._finish(a)
    assert result["n_complete_tiles_64x64"]==31
    assert result["minimum_accepted_in_either_half"]==10
    assert result["frac_abs_difference_le_1sigma"]>0.8


def test_zero_prediction_fails_closed():
    m=load()
    a=m._stat_accum()
    d=np.ones((32,2080))
    v=np.zeros_like(d)
    n=np.ones_like(d,dtype=np.int16)*10
    m._add(a,d,v,32,n,n,0)
    m._add(a,d,v,64,n,n,0)
    with pytest.raises(RuntimeError,match="Insufficient valid"):
        m._finish(a)
