"""Regression checks for read-only epoch-vs-placebo spatial bias diagnostics."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1]/"scripts"/"qa"/"sc1_bias_epoch_spatial_placebo_qa.py"


def load():
    spec=importlib.util.spec_from_file_location("sc1_bias_spatial_placebo", SCRIPT)
    assert spec and spec.loader
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_selftest():
    load().selftest()


def test_partition_sizes_disjoint_memberships_and_reproducibility():
    m=load()
    a,b=m.partitions(),m.partitions()
    assert set(a)==set(b)
    for name,(first,second) in a.items():
        assert list(map(int,first))==list(map(int,b[name][0]))
        assert list(map(int,second))==list(map(int,b[name][1]))
        assert set(first).isdisjoint(second)
        if name=="between_early20_late19":
            assert len(first)==19 and len(second)==20
            assert min(first)>=20 and max(second)<20
        else:
            assert (len(first),len(second)) in ((10,10),(10,9),(9,10))


def test_constant_epoch_difference_with_single_noisy_pixel():
    m=load()
    a=m.empty_stats()
    rng=np.random.default_rng(717)
    for y in range(0,1032,32):
        rows=min(32,1032-y)
        block=np.zeros((rows,2080),dtype=np.float64)
        block[:,32:2016]=-0.11+rng.normal(0,1.4,size=(rows,1984))
        m.add_block(a,block,y)
    result=m.finish(a)
    assert result["roi_n_pixels"]==960*1984
    assert result["difference_pixel_mean_adu"]==pytest.approx(-0.11,abs=0.005)
    assert result["independent_pixel_reference_tile_std_adu"]==pytest.approx(1.4/64,rel=.025)
    assert 0.65<result["normalized_tile_variance_factor"]<1.40
    assert len(result["all_tile_means_adu_15_by_31"])==15
    assert len(result["all_tile_means_adu_15_by_31"][0])==31


def test_tile_spatial_pattern_distinct_from_scalar_baseline():
    m=load()
    a=m.empty_stats()
    for y in range(0,1032,32):
        rows=min(32,1032-y)
        block=np.zeros((rows,2080))
        if y<992 and y+rows>32:
            for j in range(rows):
                fy=y+j
                if 32<=fy<992:
                    block[j,32:2016] = (
                        0.5 + (1 if fy<96 else -0.1) +
                        np.arange(1984,dtype=float)*0.001
                    )
        m.add_block(a,block,y)
    result=m.finish(a)
    assert result["spatial_tile_rms_after_scalar_removal_adu"]>0.05
    assert result["normalized_tile_variance_factor"]>1
    assert result["tile_max_mean_adu"]>result["tile_min_mean_adu"]


def test_accumulator_rejects_nonfinite_and_wrong_geometry():
    m=load()
    a=m.empty_stats()
    with pytest.raises(ValueError,match="Expected"):
        m.add_block(a,np.zeros((32,20)),0)
    wrong=np.zeros((32,2080));wrong[4,9]=np.nan
    with pytest.raises(ValueError,match="Nonfinite"):
        m.add_block(a,wrong,0)


def test_partial_roi_coverage_fails_closed():
    m=load()
    a=m.empty_stats()
    m.add_block(a,np.zeros((32,2080)),32)
    with pytest.raises(RuntimeError,match="Not all"):
        m.finish(a)
