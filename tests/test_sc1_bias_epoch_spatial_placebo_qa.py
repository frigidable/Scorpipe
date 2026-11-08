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


def test_disjoint_early_late_cohorts_and_fixed_partition_order():
    m=load()
    p=m.partitions()
    a_late,a_early=p["between_epoch_disjoint_A"]
    b_late,b_early=p["between_epoch_disjoint_B"]
    assert len(a_late)==10 and len(b_late)==9
    assert len(a_early)==len(b_early)==10
    assert not set(a_late)&set(b_late)
    assert not set(a_early)&set(b_early)
    assert set(a_late)|set(b_late)==set(range(20,39))
    assert set(a_early)|set(b_early)==set(range(20))


def test_disjoint_epoch_tile_maps_identical_pattern_differing_global_zero():
    m=load()
    y,x=np.indices((15,31))
    map_a=0.02*y - 0.005*x
    map_b=map_a + 3.0
    a=m.independent_epoch_spatial_repeatability(map_a,map_b)
    assert a["n_tiles"]==465
    assert a["centered_spatial_correlation"]==pytest.approx(1.0,abs=1e-12)
    # At the exact mean, centered product can be zero for several tiles.
    assert a["same_sign_centered_tiles_fraction"]>0.98
    assert a["mean_map_B_adu"]-a["mean_map_A_adu"]==pytest.approx(3.0)
    assert a["planar_slopes"]["map_A"]["slope_per_64px_tile_y_adu"]==pytest.approx(0.02)
    assert a["planar_slopes"]["map_B"]["slope_per_64px_tile_x_adu"]==pytest.approx(-0.005)
    assert a["calibrated_significance_test"] is False


def test_uncorrelated_spatial_maps_do_not_produce_spurious_strong_correlation():
    m=load()
    rng=np.random.default_rng(20261108)
    map_a=rng.normal(0,.05,size=(15,31))
    map_b=rng.normal(0,.05,size=(15,31))
    a=m.independent_epoch_spatial_repeatability(map_a,map_b)
    assert abs(a["centered_spatial_correlation"])<0.15
    assert a["n_tiles"]==465


def test_split_epoch_repeatability_invalid_geometry_fails_closed():
    m=load()
    a=np.zeros((15,31))
    b=np.ones((15,30))
    with pytest.raises(ValueError,match="Expected two"):
        m.independent_epoch_spatial_repeatability(a,b)
    b=np.ones((15,31));b[0,0]=np.nan
    with pytest.raises(ValueError,match="Non-finite"):
        m.independent_epoch_spatial_repeatability(a,b)
    with pytest.raises(ValueError,match="constant tile"):
        m.independent_epoch_spatial_repeatability(a,np.ones((15,31)))
