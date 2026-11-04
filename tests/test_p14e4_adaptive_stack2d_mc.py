"""Real streaming Stack2D Huber and shared-MasterBias Monte Carlo contracts."""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.calib.adaptive_stack_mc import (
    synthetic_adaptive_stack2d_coverage,
)


def test_huber_stacking_independent_calibration_groups_is_mc_consistent():
    report = synthetic_adaptive_stack2d_coverage(
        group_ids=("A", "B", "C"), method="invvar_huber",
        n_realizations=600, seed=20261104,
        robust_iter=2, huber_c=1.,
    )
    assert report["status"] == "MC_CONSISTENT", report
    assert report["statistically_consistent_with_published_conditional_var"]
    assert report["mean_fraction_of_pixels_with_robust_suppression"] > .1
    assert report["known_fixed_invvar_grouped_variance"] == pytest.approx(2./3.)
    assert report["n_realizations"] == 600
    assert not report["science_uncertainty_qualified"]
    assert not report["production_variance_release_allowed"]


def test_same_masterbias_group_produces_measurable_diagonal_var_undercoverage():
    """Shared errors must not be counted as three independent calibrations."""
    report = synthetic_adaptive_stack2d_coverage(
        group_ids=("A", "A", "A"), method="invvar_huber",
        n_realizations=600, seed=20261104,
        robust_iter=2, huber_c=1.,
    )
    assert report["status"] == "CONDITIONAL_VAR_UNDERCOVERAGE", report
    assert report["undercoverage_detected"] is True
    assert report["known_fixed_invvar_grouped_variance"] == pytest.approx(4./3.)
    assert min(report["empirical_to_conditional_variance_ratios"]) > 1.5
    assert min(report["coverage_95percent"]) < .9
    assert report["science_uncertainty_qualified"] is False
    assert report["production_variance_release_allowed"] is False


@pytest.mark.parametrize("groups,correct", [
    (("A", "B", "C"), 2./3.),
    (("A", "A", "A"), 4./3.),
])
def test_plain_invvar_exact_grouped_variance_has_independent_mc_oracle(groups, correct):
    result = synthetic_adaptive_stack2d_coverage(
        group_ids=groups, method="invvar",
        n_realizations=800, seed=1091,
    )
    assert result["known_fixed_invvar_grouped_variance"] == pytest.approx(
        correct, rel=0, abs=1e-12
    )
    np.testing.assert_allclose(
        result["mean_reported_conditional_variance"],
        [2./3., 2./3.], atol=2e-7, rtol=0,
    )
    actual = np.diag(result["empirical_covariance"])
    assert np.all(np.abs(actual/correct - 1.) < .2), result
    if groups[0] == groups[1]:
        assert result["status"] == "CONDITIONAL_VAR_UNDERCOVERAGE", result
    else:
        assert result["status"] == "MC_CONSISTENT", result
    assert result["mean_fraction_of_pixels_with_robust_suppression"] == 0.


def test_grouped_huber_seeded_report_is_identical_and_never_releases_var():
    args = dict(group_ids=("A", "B", "C"), n_realizations=200,
                seed=11, robust_iter=1, huber_c=.8)
    first = synthetic_adaptive_stack2d_coverage(**args)
    second = synthetic_adaptive_stack2d_coverage(**args)
    assert first == second
    assert first["method"] == "invvar_huber"
    assert first["scope"] == "SYNTHETIC_PRODUCTION_ADAPTIVE_STACK2D_SHARED_BIAS_MC"
    assert first["science_uncertainty_qualified"] is False
    assert first["production_variance_release_allowed"] is False


@pytest.mark.parametrize("bad", [
    {"group_ids": ("A", "B")},
    {"group_ids": ("A", "", "C")},
    {"group_ids": ("A", " B", "C")},
    {"group_ids": ("A", 5, "C")},
    {"method": "sigma_clip"},
    {"n_realizations": 199},
    {"n_realizations": 2001},
    {"n_realizations": True},
    {"seed": -1},
    {"seed": 1.1},
    {"robust_iter": -1},
    {"robust_iter": True},
    {"robust_iter": 6},
    {"huber_c": 0.},
    {"huber_c": float("nan")},
    {"huber_c": True},
    {"true_flux": float("nan")},
    {"true_flux": 1001.},
    {"true_flux": True},
])
def test_invalid_adaptive_stack_mc_controls_fail_closed(bad):
    with pytest.raises(ValueError):
        synthetic_adaptive_stack2d_coverage(**bad)
