"""Invariant checks for read-only SCORPIO-1 FLAT shift triangle closure."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts" / "qa" / "sc1_fast_flat_shift_closure_qa.py"


def module():
    spec = importlib.util.spec_from_file_location("sc1_flat_closure", PATH)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rows():
    m = module()
    result = []
    for stripe in range(12):
        for window in range(4):
            for a, b, delta in (
                (m.A, m.B, -0.15),
                (m.A, m.C, -0.10),
                (m.B, m.C, +0.05),
            ):
                result.append(
                    dict(flat_a=a, flat_b=b, spatial_stripe=stripe,
                         xwindow=window, eligible="True", best_delta_px=delta)
                )
    return result


def test_synthetic_closure_selftest():
    module().selftest()


def test_exact_closure_cannot_be_rejected():
    summary, data = module().calculate(rows())
    assert summary["n_eligible_triangles"] == 48
    assert summary["gate"] == "HOLD_UNCERTAINTY_UNCALIBRATED"
    assert summary["median_absolute_closure_px"] == pytest.approx(0, abs=1e-12)
    assert len(data) == 48


def test_intentionally_wrong_triangle_is_flagged():
    data = rows()
    for row in data:
        if row["flat_a"] == module().A and row["flat_b"] == module().C:
            row["best_delta_px"] = -0.25
    summary, _ = module().calculate(data)
    assert summary["gate"] == "HOLD_INCONSISTENT_TRIANGLES"
    assert summary["median_absolute_closure_px"] == pytest.approx(0.15)


def test_duplicate_missing_or_unexpected_window_rejected():
    m = module()
    data = rows()
    with pytest.raises(ValueError, match="Duplicate"):
        m.calculate(data + [dict(data[0])])
    with pytest.raises(ValueError, match="144"):
        m.calculate(data[:-1])
    data[0]["xwindow"] = 4
    with pytest.raises(ValueError, match="Unexpected row key"):
        m.calculate(data)


def test_no_jointly_eligible_triangles_is_hold_not_false_zero():
    m = module()
    data = rows()
    for row in data:
        if row["flat_a"] == m.A and row["flat_b"] == m.C:
            row["eligible"] = "False"
            row["best_delta_px"] = ""
    summary, _ = m.calculate(data)
    assert summary["n_eligible_triangles"] == 0
    assert summary["median_absolute_closure_px"] is None
    assert summary["gate"] == "HOLD_INSUFFICIENT_TRIANGLES"
