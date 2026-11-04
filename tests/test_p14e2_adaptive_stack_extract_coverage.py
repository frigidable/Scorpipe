"""Independent statistics for actual Stack2D and Extract1D adaptive kernels.

All inputs are bounded synthetic CCD arrays; this file does not open any
observational science FITS and does not modify production FITS variance.
"""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.calib.post_sky_adaptive_coverage import (
    synthetic_adaptive_stack_extract_coverage,
)


@pytest.fixture(scope="module")
def noise_only_ablation():
    return synthetic_adaptive_stack_extract_coverage(
        n_realizations=360, seed=20261104,
        object_flux=18., detector_sigma=5.,
        huber_c=1.2, optimal_smooth_lambda_sigma=1.4,
    )


def test_independent_inverse_variance_boxcar_reference_has_correct_coverage(
    noise_only_ablation,
):
    result = noise_only_ablation
    assert result["status"] == "REFERENCE_CONSISTENT", result
    assert result["scope"] == "SYNTHETIC_POST_SKYSUB_ADAPTIVE_STACK_EXTRACT"
    assert result["n_realizations"] == 360
    assert result["n_rows"] == 5
    assert result["n_exposures"] == 3
    assert result["n_wavelength_bins"] == 9
    assert result["science_uncertainty_qualified"] is False
    assert result["production_variance_release_allowed"] is False
    expected_var = 5.*25./3.
    assert result["analytic_invvar_boxcar_variance"] == pytest.approx(expected_var)
    reference = result["modes"]["invvar_boxcar"]
    np.testing.assert_allclose(
        reference["mean_reported_conditional_variance"],
        expected_var, rtol=2e-5,
    )
    coverage = np.asarray(reference["coverage95"])
    assert np.max(np.abs(coverage-.95)) <= result["reference_boxcar_coverage_tolerance"]
    assert np.max(np.abs(
        np.asarray(reference["empirical_variance"])/expected_var-1.
    )) <= result["reference_boxcar_variance_tolerance"]


def test_actual_huber_and_optimal_variants_are_measured_not_qualified(
    noise_only_ablation,
):
    result = noise_only_ablation
    assert result["fraction_huber_reweighted_realizations"] > 0.1
    assert result["adaptive_intervals_are_descriptive_only"]
    assert set(result["modes"]) == {
        "invvar_boxcar", "invvar_optimal", "huber_boxcar", "huber_optimal",
    }
    for name, mode in result["modes"].items():
        for key in (
            "empirical_variance", "mean_reported_conditional_variance",
            "variance_ratio_empirical_to_reported",
            "coverage95", "mean_error",
        ):
            values = np.asarray(mode[key])
            assert values.shape == (9,), (name, key)
            assert np.isfinite(values).all(), (name, key)
        assert np.all(np.asarray(mode["empirical_variance"]) > 0)
        assert np.all(np.asarray(mode["mean_reported_conditional_variance"]) > 0)
        assert mode["adaptive_stack"] is name.startswith("huber")
        assert mode["data_derived_extraction_profile"] is name.endswith("optimal")
        assert np.isfinite(mode["neighbor_wavelength_covariance_3_4"])
        if name.endswith("optimal"):
            assert 0 <= mode["fraction_profile_fallback"] <= 1
        else:
            assert mode["fraction_profile_fallback"] is None
    # The actual self-profile extraction and Huber reweighting should
    # activate on independent Gaussian draws; these are different estimators.
    base = np.asarray(result["modes"]["invvar_boxcar"]["empirical_variance"])
    robust = np.asarray(result["modes"]["huber_optimal"]["empirical_variance"])
    assert not np.allclose(base, robust, rtol=1e-4, atol=1e-4)
    assert not result["science_uncertainty_qualified"]


def test_explicit_nonGaussian_synthetic_outlier_not_published_as_valid_coverage():
    report = synthetic_adaptive_stack_extract_coverage(
        n_realizations=220, seed=1872, outlier_sigma=10.,
    )
    assert report["status"] == "CONTAMINATED_REFERENCE", report
    assert report["outlier_sigma"] == 10.
    assert report["modes"]["invvar_boxcar"]["mean_error"][4] > 5.
    assert report["science_uncertainty_qualified"] is False
    assert report["production_variance_release_allowed"] is False


def test_repeated_small_experiment_is_reproducible():
    settings = dict(n_realizations=200, seed=73)
    a = synthetic_adaptive_stack_extract_coverage(**settings)
    b = synthetic_adaptive_stack_extract_coverage(**settings)
    assert a == b
    assert not a["science_uncertainty_qualified"]


@pytest.mark.parametrize("settings", [
    {"n_realizations": 199},
    {"n_realizations": 1201},
    {"n_realizations": True},
    {"n_realizations": 200.5},
    {"seed": -1},
    {"seed": 2.3},
    {"object_flux": -1.},
    {"object_flux": float("nan")},
    {"detector_sigma": 0.},
    {"detector_sigma": float("inf")},
    {"huber_c": 0.},
    {"huber_c": -1.},
    {"optimal_smooth_lambda_sigma": -1.},
    {"outlier_sigma": -1.},
    {"outlier_sigma": float("nan")},
])
def test_invalid_controls_fail_before_any_synthetic_fits(settings):
    with pytest.raises(ValueError):
        synthetic_adaptive_stack_extract_coverage(**settings)
