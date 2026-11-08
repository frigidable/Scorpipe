"""Geometry and scientific interpretation guards for epoch BIAS median comparison."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

QA=Path(__file__).resolve().parents[1]/"scripts"/"qa"
if str(QA) not in sys.path:
    sys.path.insert(0,str(QA))


def module():
    p=QA/"sc1_bias_epoch_spatial_compare_qa.py"
    spec=importlib.util.spec_from_file_location("sc1_bias_epoch_compare",p)
    assert spec and spec.loader
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def arrays():
    return (
        np.zeros((960,1984),dtype=np.float32),
        np.zeros((960,12),dtype=np.float32),
        np.zeros((960,12),dtype=np.float32),
    )


def test_selftest():
    module().selftest()


def test_uniform_scalar_offset_produces_zero_tile_residual_variance():
    m=module()
    roi,l,r=arrays()
    roi[:]=-0.11
    l[:]=-0.05
    r[:]=-0.03
    report=m.summarize_epoch_difference(roi,l,r)
    assert report["tile_count_64x64"]==465
    assert report["roi_pixel_count"]==960*1984
    assert report["tile_mean_residual_after_subtracting_scalar"]["std_adu"]<1e-6
    assert report["sidebands_uncertified"]["physical_overscan_certified"] is False
    assert report["sidebands_uncertified"]["left_minus_roi_mean_adu"]==pytest.approx(0.06,abs=1e-6)
    assert report["sidebands_uncertified"]["right_minus_roi_mean_adu"]==pytest.approx(0.08,abs=1e-6)


def test_one_tile_spatial_pattern_survives_subtracting_global_zero():
    m=module()
    roi,l,r=arrays()
    roi[0:64,0:64]=1.0
    report=m.summarize_epoch_difference(roi,l,r)
    assert report["tile_count_64x64"]==465
    assert report["tile_mean_residual_after_subtracting_scalar"]["std_adu"]>0
    assert report["tile_mean_difference"]["max_adu"]==pytest.approx(1.0)
    assert report["tile_mean_difference"]["min_adu"]==pytest.approx(0.0)
    assert report["tile_mean_fraction_positive"]==pytest.approx(1/465)


def test_gradient_measured_separately_from_uniform_shift():
    m=module()
    roi,l,r=arrays()
    roi[:]=np.linspace(-0.1,0.1,960,dtype=np.float32)[:,None]
    report=m.summarize_epoch_difference(roi,l,r)
    assert report["roi_difference"]["mean_adu"]==pytest.approx(0,abs=1e-7)
    assert report["per_row_mean_difference"]["std_adu"]>0.05
    assert report["per_column_mean_difference"]["std_adu"]<1e-6
    assert report["tile_mean_residual_after_subtracting_scalar"]["std_adu"]>0.05


def test_invalid_shapes_fail_before_statistics():
    m=module()
    roi,l,r=arrays()
    with pytest.raises(ValueError,match="Expected exact"):
        m.summarize_epoch_difference(roi[:959],l,r)
    with pytest.raises(ValueError,match="Expected exact"):
        m.summarize_epoch_difference(roi,l[:,:11],r)
    with pytest.raises(ValueError,match="Expected exact"):
        m.summarize_epoch_difference(roi,l,r[:,:11])


def test_nonfinite_epoch_differences_fail_closed():
    m=module()
    roi,l,r=arrays()
    roi[150,225]=np.nan
    with pytest.raises(ValueError,match="Nonfinite"):
        m.summarize_epoch_difference(roi,l,r)
