"""Read-only common-mode BIAS drift diagnostic regression contracts."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

QA = Path(__file__).resolve().parents[1] / "scripts" / "qa"
if str(QA) not in sys.path:
    sys.path.insert(0,str(QA))


def load():
    p=QA/"sc1_masterbias_frame_baseline_qa.py"
    spec=importlib.util.spec_from_file_location("sc1_baseline",p)
    assert spec and spec.loader
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_selftest():
    load().selftest()


def test_tile_averages_are_correct_on_constant_adu_offset():
    m=load()
    ref=np.full((960,1984),999.0,dtype=float)
    raw=ref+0.5
    tiles,stats=m.tile_stats(raw,ref)
    assert tiles.shape==(15,31)
    assert stats["n_discarded"]==0
    assert stats["n_pixels"]==960*1984
    assert stats["mean_residual_adu"]==0.5
    assert np.all(tiles==0.5)


def test_gross_transient_is_excluded_without_biasing_reference_level():
    m=load()
    ref=np.full((960,1984),999.0,dtype=float)
    raw=ref+0.25
    raw[10,15]+=17000
    tile,stats=m.tile_stats(raw,ref)
    assert stats["n_discarded"]==1
    assert stats["mean_residual_adu"]==pytest.approx(0.25,abs=1e-10)
    assert np.allclose(tile,0.25,rtol=0,atol=1e-10)


def test_tile_geometry_and_threshold_fail_closed():
    m=load()
    small=np.ones((64,64))
    with pytest.raises(ValueError,match="Expected"):
        m.tile_stats(small,small)
    full=np.ones((960,1984))
    with pytest.raises(ValueError,match="Invalid diagnostic"):
        m.tile_stats(full,full,threshold_adu=0)
    with pytest.raises(RuntimeError,match="Excessively contaminated"):
        m.tile_stats(full*1000,full,threshold_adu=50)


def test_partition_metrics_are_tile_grounded():
    m=load()
    cube=np.zeros((20,15,31))
    cube[:10,:,:]=-0.3
    cube[10:,:,:]=0.1
    a=m.partition_stats(cube,np.arange(10),np.arange(10,20))
    assert a["mean_difference_adu"]==pytest.approx(-0.4,abs=1e-12)
    assert a["median_tile_difference_adu"]==pytest.approx(-0.4,abs=1e-12)
    assert a["fraction_tiles_difference_negative"]==1
    assert a["fraction_tiles_difference_positive"]==0


def test_duplicate_partition_is_rejected():
    m=load()
    cube=np.zeros((20,15,31))
    with pytest.raises(ValueError,match="Invalid 10\\+10 partition"):
        m.partition_stats(cube,np.arange(10),np.arange(9,19))


def test_ordered_baseline_labels_are_nonexchangeable_in_strong_toy_example():
    m=load()
    offsets=np.r_[np.full(10,-0.3),np.zeros(10)]
    p=m.permutation_test(offsets,n_permutations=4000,seed=33)
    assert p["observed_first10_minus_last10_adu"]==pytest.approx(-0.3)
    assert p["two_sided_permutation_p_under_exchangeability"]<0.005
    assert p["null_requires_exchangeable_independent_frame_baselines"] is True
    with pytest.raises(ValueError):
        m.permutation_test(np.zeros(19))
