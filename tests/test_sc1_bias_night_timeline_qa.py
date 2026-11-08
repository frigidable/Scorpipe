"""Synthetic regression checks for SCORPIO-1 early/late BIAS timeline audit."""
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
    spec=importlib.util.spec_from_file_location(
        "sc1_night_timeline",QA/"sc1_bias_night_timeline_qa.py")
    assert spec and spec.loader
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_selftest():
    load().selftest()


def test_exact_41_bias_inventory_and_nonoverlap():
    m=load()
    assert len(m.EARLY_NAMES)==20
    assert len(m.LATE_NAMES)==19
    assert len(m.FLAGGED_NAMES)==2
    assert len(m.ALL_NAMES)==41
    assert set(m.EARLY_NAMES).isdisjoint(m.LATE_NAMES)
    assert set(m.FLAGGED_NAMES).isdisjoint(m.LATE_NAMES)
    assert "s22800120.fts" in m.EARLY_NAMES
    assert "s22801702.fts" in m.LATE_NAMES


def test_roi_gross_transient_rejection_and_uncertified_strips():
    m=load()
    science=np.full(m.EXPECTED_SHAPE,999.,dtype=float)
    bias=science+0.3
    bias[32+40,32+80]+=9000.
    summary=m.roi_summary(bias,science)
    assert summary["roi_outlier_count"]==1
    assert summary["mean_roi_residual_adu"]==pytest.approx(0.3,abs=1e-12)
    assert summary["median_tile_residual_adu"]==pytest.approx(0.3,abs=1e-12)
    assert all(not v["physical_overscan_certified"]
               for v in summary["candidate_side_strips"].values())


def test_roi_side_strips_are_separate_from_central_pixels():
    m=load()
    master=np.full(m.EXPECTED_SHAPE,999.,dtype=float)
    bias=master.copy()
    bias[32:992,32:2016]+=0.2
    bias[32:992,2:14]+=0.5
    bias[32:992,2066:2078]-=0.1
    summary=m.roi_summary(bias,master)
    assert summary["mean_roi_residual_adu"]==pytest.approx(0.2,abs=1e-12)
    assert summary["candidate_side_strips"]["left"]["mean_residual_adu"]==pytest.approx(0.5,abs=1e-12)
    assert summary["candidate_side_strips"]["right"]["mean_residual_adu"]==pytest.approx(-0.1,abs=1e-12)


def test_nearly_whole_region_outliers_cause_aborted_diagnostic():
    m=load()
    raw=np.full(m.EXPECTED_SHAPE,999.,dtype=float)
    reference=raw.copy()
    raw[32:992,32:2016]+=100
    with pytest.raises(RuntimeError,match="Excessive transient"):
        m.roi_summary(raw,reference)


def test_linear_summary_recovers_drift_without_assuming_physical_cause():
    m=load()
    records=[{"mean_roi_residual_adu":-0.3+0.02*i,
              "std_tile_residual_adu":0.05} for i in range(20)]
    s=m.linear_summary(records)
    assert s["n_frames"]==20
    assert s["monotonically_increasing"] is True
    assert s["slope_adu_per_frame_index"]==pytest.approx(0.02,abs=1e-12)
    assert s["end_minus_start_adu"]==pytest.approx(0.38,abs=1e-12)


def test_missing_late_frame_inventory_fails_closed(tmp_path):
    m=load()
    first20=[]
    for name in sorted(m.EARLY_NAMES):
        p=tmp_path/name
        p.write_bytes(b"test")
        first20.append(p)
    with pytest.raises(FileNotFoundError,match="missing"):
        m.locate_frames(tmp_path,first20)


def test_ambiguous_late_frame_name_fails_closed(tmp_path):
    m=load()
    early=[]
    for name in sorted(m.EARLY_NAMES):
        p=tmp_path/name
        p.write_bytes(b"test")
        early.append(p)
    for name in m.LATE_NAMES|m.FLAGGED_NAMES:
        p=tmp_path/name
        p.write_bytes(b"test")
    dup=tmp_path/"s228017"
    dup.mkdir()
    (dup/"s22801702.fts").write_bytes(b"dup")
    with pytest.raises(RuntimeError,match="Ambiguous source FITS"):
        m.locate_frames(tmp_path,early)


def test_unverified_ambiguous_date_obs_not_reinterpreted():
    m=load()
    assert m.date_value("2024/06/12") is None
    assert m.date_value("2024-12-06T12:29:19").year==2024
    assert m.date_value("wrong") is None
