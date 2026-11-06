"""Synthetic regressions for fast SCORPIO-1 spatial FLAT diagnostic."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qa" / "sc1_fast_flat_spatial_qa.py"


def load():
    spec = importlib.util.spec_from_file_location("sc1_fast_flat_spatial", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_photon_noise_plus_smooth_illumination():
    load().selftest()


def test_fully_identical_flats_have_zero_spatial_shift():
    mod = load()
    y, x = np.mgrid[:128, :128]
    a = (10000 + x * 20 + y * 10).astype(float)
    r = mod.spatial_pair(a, a, tile=64)
    assert r["tile_ratio_std_percent"] == pytest.approx(0, abs=1e-10)
    assert r["highpass_residual_normalized_mad"] == pytest.approx(0, abs=1e-10)


def test_slow_spectral_gradient_attributes_variance_to_dispersion_x():
    mod = load()
    y, x = np.mgrid[:256, :256]
    b = (11000 + x * 10 + y * 5).astype(float)
    a = b * (1 + 0.004 * (x/255 - 0.5))
    r = mod.spatial_pair(a, b, tile=64)
    assert r["axis_fraction"]["dispersion_x"] > 0.99
    assert r["axis_fraction"]["slit_y"] < 0.01


def test_reject_shape_and_gain():
    mod = load()
    a = np.ones((128, 128))
    with pytest.raises(ValueError, match="dimensions"):
        mod.spatial_pair(a, a[:64])
    with pytest.raises(ValueError, match="gain"):
        mod.spatial_pair(a, a, gain=-3)
