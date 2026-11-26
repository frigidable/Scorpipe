"""Independent synthetic Gaussian coverage for grouped frozen Kelson SkySub."""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.calib.sky_mc_coverage import synthetic_grouped_sky_coverage


@pytest.mark.parametrize("y_order", [0, 1])
def test_synthetic_shared_bias_2d_qr_monte_carlo_has_correct_coverage(y_order):
    # The Monte Carlo oracle draws CCD pixel-level independent and common
    # MasterBias errors, then RE-FITS each realization with a dense QR solver.
    # It never uses the analytic WLS adjoint to construct each realization.
    report = synthetic_grouped_sky_coverage(
        group_ids=("A", "A"), y_order=y_order,
        n_realizations=10000, seed=1544 + y_order,
    )
    assert report["status"] == "MC_CONSISTENT", report
    assert report["scope"] == "SYNTHETIC_GAUSSIAN_FIXED_KELSON_WLS"
    assert report["science_uncertainty_qualified"] is False
    pred = np.asarray(report["analytic_covariance"])
    empirical = np.asarray(report["empirical_covariance"])
    independent = np.asarray(report["analytic_independent"])
    shared = np.asarray(report["analytic_shared_masterbias"])
    assert pred.shape == (2, 2)
    np.testing.assert_allclose(pred, independent + shared, rtol=1e-12)
    assert np.min(np.linalg.eigvalsh(pred)) > 0
    assert np.min(np.linalg.eigvalsh(empirical)) > 0
    assert np.all(np.diag(pred) > np.diag(independent))
    assert np.max(report["variance_relative_error"]) < .07
    assert report["cross_covariance_scaled_error"] < .07
    assert np.max(np.abs(np.asarray(report["coverage_1sigma"]) - .68268949)) < .03
    assert np.max(np.abs(np.asarray(report["coverage_95percent"]) - .95)) < .025


def test_two_distinct_masterbias_groups_reduce_shared_covariance():
    shared = synthetic_grouped_sky_coverage(
        group_ids=("A", "A"), n_realizations=8000, seed=1881,
    )
    separate = synthetic_grouped_sky_coverage(
        group_ids=("A", "B"), n_realizations=8000, seed=1881,
    )
    assert shared["status"] == "MC_CONSISTENT", shared
    assert separate["status"] == "MC_CONSISTENT", separate
    p_shared = np.asarray(shared["analytic_covariance"])
    p_separate = np.asarray(separate["analytic_covariance"])
    assert p_shared[0, 0] > 1.10 * p_separate[0, 0]
    assert (
        np.asarray(shared["analytic_independent"])[0, 0] ==
        pytest.approx(np.asarray(separate["analytic_independent"])[0, 0])
    )
    empirical_shared = np.asarray(shared["empirical_covariance"])[0, 0]
    empirical_separate = np.asarray(separate["empirical_covariance"])[0, 0]
    assert empirical_shared > 1.08 * empirical_separate


def test_ignoring_shared_masterbias_fails_variance_and_coverage_oracle():
    # A deliberately wrong independent-only variance must be distinguishable
    # from the full grouped covariance on the SAME Monte Carlo samples.
    report = synthetic_grouped_sky_coverage(
        group_ids=("A", "A"), n_realizations=10000, seed=455,
    )
    predicted = np.asarray(report["analytic_covariance"])
    independent = np.asarray(report["analytic_independent"])
    empirical = np.asarray(report["empirical_covariance"])
    assert report["status"] == "MC_CONSISTENT"
    assert empirical[0, 0] > 1.25 * independent[0, 0]
    assert predicted[0, 0] > 1.25 * independent[0, 0]
    assert min(report["coverage_95percent_independent_only"]) < .93
    assert min(report["coverage_95percent"]) > .93


def test_monte_carlo_seed_is_exactly_reproducible():
    kw = dict(n_realizations=1000, seed=33, group_ids=("C", "C"))
    a = synthetic_grouped_sky_coverage(**kw)
    b = synthetic_grouped_sky_coverage(**kw)
    assert a == b
    assert not a["science_uncertainty_qualified"]


@pytest.mark.parametrize("kwargs", [
    {"group_ids": ("", "A")},
    {"group_ids": ("A",)},
    {"group_ids": ("A", " A")},
    {"group_ids": ("A", 17)},
    {"group_ids": ("A", "B", "C")},
    {"y_order": 2},
    {"y_order": True},
    {"n_realizations": 999},
    {"n_realizations": 20001},
    {"n_realizations": 2.1},
    {"n_realizations": True},
    {"seed": -1},
    {"seed": 1.5},
])
def test_monte_carlo_rejects_invalid_experiment_specifications(kwargs):
    with pytest.raises(ValueError):
        synthetic_grouped_sky_coverage(**kwargs)
