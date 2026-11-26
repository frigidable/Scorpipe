"""Non-production SCORPIO-1 spectral FLAT shift and config route checks."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "scripts" / "qa"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shift_qa_selftest_with_known_positive_shift_and_featureless_continuum():
    load("sc1_fast_flat_shift_qa").selftest()


def test_negative_known_subpixel_shift():
    m = load("sc1_fast_flat_shift_qa")
    rng = np.random.default_rng(83)
    x = np.arange(2016.0)
    cont = 13000 * (1 + .05 * x / 2016)
    p = np.zeros_like(x)
    for k in [200, 265, 325, 373, 615, 710, 802, 904, 1015, 1123, 1270, 1355, 1512, 1618, 1690]:
        p -= .04 * np.exp(-.5 * ((x-k)/5)**2)
    b = cont * (1 + p) + rng.normal(0, 5, len(x))
    a = cont * (1 + np.interp(x + .65, x, p)) + rng.normal(0, 5, len(x))
    for left, right in m.WINDOWS:
        r = m.shift_window(a, b, x1=left, x2=right)
        assert r["eligible"] is True
        assert r["best_delta_px"] == pytest.approx(-.65, abs=.10)


def test_ambiguous_or_unilluminated_windows_not_reported_as_measurement():
    m = load("sc1_fast_flat_shift_qa")
    x = np.arange(2016.0)
    a = np.full(x.shape, 1200.0)
    b = np.full(x.shape, 1200.0)
    r = m.shift_window(a, b)
    assert not r["eligible"]
    assert r["reason"] == "LOW_ILLUMINATION"


def test_route_qa_selftest_and_manifest_switch():
    load("sc1_pipeline_flat_route_qa").selftest()
