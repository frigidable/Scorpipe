"""Exact arithmetic tests for read-only MasterBias raw-pixel reconstruction."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qa" / "sc1_masterbias_raw_sample_qa.py"


def mod():
    spec = importlib.util.spec_from_file_location("sc1_raw_bias_audit", SCRIPT)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_raw_sample_selftest():
    mod().selftest()


def test_pixelwise_exact_median_and_estimator_var():
    rng = np.random.default_rng(2026)
    a = 1000 + rng.normal(0, 3, (20, 120))
    factor = 0.07343702770505486
    sci = np.median(a, axis=0)
    var = np.var(a, axis=0, ddof=1) * factor
    summary, median, residual, spikes, robust = mod().sample_metrics(
        a, sci, var, factor
    )
    assert summary["n_pixels"] == 120
    assert summary["max_abs_sci_difference_adu"] == 0
    assert summary["max_rel_var_difference"] == 0
    assert np.array_equal(median, sci)
    assert not np.any(spikes)
    assert np.isfinite(robust).all()


def test_single_cosmic_outlier_inflates_var_but_preserves_median():
    a = np.full((20, 4), 1000., dtype=float)
    a[9, 0] = 2000.
    a[7, 1] = 1250.
    factor = 0.07343702770505486
    sci = np.median(a, axis=0)
    var = np.var(a, axis=0, ddof=1) * factor
    summary, med, residual, spikes, robust = mod().sample_metrics(
        a, sci, var, factor
    )
    assert np.array_equal(med, [1000, 1000, 1000, 1000])
    assert var[0] > 1000
    assert spikes.tolist() == [1, 1, 0, 0]
    assert summary["n_pixels_raw_deviation_over_100adu"] == 2
    assert summary["n_pixels_with_multiple_deviations_over_100adu"] == 0
    assert summary["median_robust_sigma_adu"] == 0


@pytest.mark.parametrize("shape", [(19, 5), (20, 3, 5), (20,)])
def test_incorrect_raw_stack_fails_closed(shape):
    a = np.zeros(shape)
    with pytest.raises(ValueError):
        mod().sample_metrics(a, np.zeros(5), np.ones(5), 0.073)


def test_wrong_factor_or_expected_shape_fails():
    a = np.ones((20, 5))
    with pytest.raises(ValueError, match="Invalid"):
        mod().sample_metrics(a, np.ones(5), np.ones(5), 0)
    with pytest.raises(ValueError, match="Sample sizes"):
        mod().sample_metrics(a, np.ones(4), np.ones(4), .07)
