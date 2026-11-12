"""P1.4e3: same-data adaptive optimal extraction synthetic MC oracles."""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.calib.optimal_profile_mc import synthetic_optimal_profile_coverage


def test_noise_only_same_data_positive_profile_causes_bias_and_undercoverage():
    """A profile fitted to the same positive-clipped noise is endogenous."""
    report = synthetic_optimal_profile_coverage(
        true_flux=0., smooth_lambda_sigma=0.,
        n_realizations=400, seed=20261104,
    )
    assert report["status"] == "ADAPTIVE_PROFILE_DISCREPANCY", report
    assert report["fixed_reference_consistent"], report
    assert abs(report["fixed_mean_error"]) < report["reference_tolerances"]["mean_absolute"]
    assert report["fixed_coverage_95percent"] > .88
    assert report["adaptive_mean_error"] > 1.
    assert report["adaptive_coverage_95percent"] < .85
    assert report["adaptive_bias_detected"]
    assert report["adaptive_undercoverage_detected"]
    assert report["profile_reestimated_per_draw"] is True
    assert report["science_uncertainty_qualified"] is False
    assert report["production_variance_release_allowed"] is False


def test_wavelength_smoothing_mitigates_but_does_not_claim_full_science_validity():
    raw = synthetic_optimal_profile_coverage(
        true_flux=0., n_realizations=400, seed=20261104,
        smooth_lambda_sigma=0.,
    )
    smooth = synthetic_optimal_profile_coverage(
        true_flux=0., n_realizations=400, seed=20261104,
        smooth_lambda_sigma=3.,
    )
    assert raw["status"] == "ADAPTIVE_PROFILE_DISCREPANCY"
    assert smooth["fixed_reference_consistent"]
    assert smooth["adaptive_mean_error"] < raw["adaptive_mean_error"] - .4
    assert smooth["science_uncertainty_qualified"] is False
    # The same independent known-profile oracle must generate identical
    # results when only the adaptive profile-smoothing parameter changes.
    for field in ("known_fixed_profile_variance", "fixed_empirical_variance",
                  "fixed_mean_error", "fixed_coverage_95percent",
                  "fixed_adjacent_wavelength_covariance"):
        assert raw[field] == smooth[field], field


def test_bright_source_profile_results_remain_diagnostic_only():
    report = synthetic_optimal_profile_coverage(
        true_flux=40., n_realizations=400, seed=150,
        smooth_lambda_sigma=3.,
    )
    assert report["fixed_reference_consistent"], report
    assert report["status"] in {
        "ADAPTIVE_PROFILE_DISCREPANCY",
        "NO_DETECTED_ADAPTIVE_DISCREPANCY",
    }
    assert np.isfinite(report["adaptive_empirical_variance"])
    assert report["adaptive_empirical_variance"] > 0
    assert report["adaptive_mean_reported_conditional_variance"] > 0
    assert 0 <= report["adaptive_coverage_1sigma"] <= 1
    assert 0 <= report["adaptive_coverage_95percent"] <= 1
    assert report["science_uncertainty_qualified"] is False


def test_seeded_optimal_profile_mc_is_exactly_reproducible():
    args = dict(true_flux=0., n_realizations=200, seed=777,
                smooth_lambda_sigma=1.5)
    a = synthetic_optimal_profile_coverage(**args)
    b = synthetic_optimal_profile_coverage(**args)
    assert a == b
    assert a["scope"] == "SYNTHETIC_SAME_DATA_OPTIMAL_PROFILE_MC"


@pytest.mark.parametrize("bad", [
    {"true_flux": float("nan")},
    {"true_flux": -1.},
    {"true_flux": 1001.},
    {"true_flux": True},
    {"n_realizations": 199},
    {"n_realizations": 1501},
    {"n_realizations": True},
    {"seed": -1},
    {"seed": 1.2},
    {"smooth_lambda_sigma": -1.},
    {"smooth_lambda_sigma": 11.},
    {"smooth_lambda_sigma": float("nan")},
    {"smooth_lambda_sigma": True},
])
def test_invalid_optimal_mc_experiment_controls_fail_closed(bad):
    with pytest.raises(ValueError):
        synthetic_optimal_profile_coverage(**bad)
