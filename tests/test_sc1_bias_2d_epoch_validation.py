"""Synthetic regressions for independent, read-only SCORPIO-1 bias diagnostics."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qa" / "sc1_bias_2d_epoch_validation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("sc1_bias_2d_qa", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selftest():
    load_module().selftest()


def test_one_each_trimmed_mean_rejects_single_high_low_outlier():
    m = load_module().OneEachTrimmedMean()
    base = np.full((64, 64), 2.0, np.float32)
    for n in range(10):
        a = base.copy()
        if n == 0:
            a[0, 0] = 6000
        elif n == 1:
            a[0, 0] = -6000
        m.add(a)
    assert m.n == 10
    assert np.max(np.abs(m.result() - base)) < 1e-6


def test_bias_inventory_exactly_41_and_flags_two_anomalies():
    m = load_module()
    early = [f"s228001{i:02d}.fts" for i in range(1, 21)]
    late = [f"s228017{i:02d}.fts" for i in range(1, 21)]
    names = early + ["s22801620.fts"] + late
    rows = [{"role": "bias", "file": "C:/raw/" + p} for p in names]
    inv, group_early, group_late, flagged = m.expected_inventory(rows)
    assert len(inv) == 41
    assert len(group_early) == 20
    assert len(group_late) == 19
    assert flagged == ["s22801620.fts", "s22801701.fts"]
    with pytest.raises(RuntimeError, match="inventory"):
        m.expected_inventory(rows[:-1])


def test_binned_spatial_map_and_centering():
    m = load_module()
    image = np.full((128, 128), 2.0)
    image[:, 64:] += 0.2
    value = m.blocks(m.centered(image))
    assert value.shape == (2, 2)
    assert np.allclose(value[:, 0], -0.1)
    assert np.allclose(value[:, 1], +0.1)
