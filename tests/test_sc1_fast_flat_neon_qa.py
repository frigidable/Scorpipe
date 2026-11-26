"""Synthetic invariants for read-only SCORPIO-1 fast-readout FLAT/NEON QA."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qa" / "sc1_fast_flat_neon_qa.py"


def load():
    spec = importlib.util.spec_from_file_location("sc1_fast_flat_qa", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selftest():
    load().selftest()


def test_robust_side_mean_removes_extreme_pixel():
    rng = np.random.default_rng(23)
    a = rng.normal(1000.35, 2, (100, 500))
    original = load().clipped_mean(a)[0]
    a[1, 1] = 65000
    assert abs(load().clipped_mean(a)[0] - original) < 0.01


def test_flat_pair_rejects_cosmic_without_shifting_scale():
    m = load()
    rng = np.random.default_rng(4)
    yy, xx = np.mgrid[:128, :128]
    basis = 12000 + xx * 12 + yy * 7
    b = basis + rng.normal(0, 1, basis.shape)
    a = 1.002 * basis + rng.normal(0, 1, basis.shape)
    a[1, 1] += 15000
    result = m.pair_metrics(a, b, 3.5, 2.0)
    assert result["relative_scale_a_over_b"] == pytest.approx(1.002, abs=1e-5)
    assert result["n_rejected_pair_outliers"] > 0


def test_pair_rejects_mismatched_shape_and_nonphysical_gain():
    m = load()
    a = np.ones((64, 64))
    with pytest.raises(ValueError, match="Incompatible"):
        m.pair_metrics(a, a[:32], 3.5, 2)
    with pytest.raises(ValueError, match="Invalid"):
        m.pair_metrics(a, a, -3.5, 2)
