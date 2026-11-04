"""Full synthetic SkySub refit → COUNT_V1 → fixed 1D estimator coverage."""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.calib.sky_chain_mc_reclipping import (
    synthetic_reclipped_sky_to_1d_coverage,
)


@pytest.mark.parametrize("y_order", [0, 1])
def test_smooth_sky_reclipping_full_chain_matches_conditional_covariance(y_order):
    result = synthetic_reclipped_sky_to_1d_coverage(
        y_order=y_order,
        n_realizations=250, seed=2411+y_order,
        group_ids=("A", "A"), clip_sigma=7., contamination_sigma=0.,
    )
    assert result["status"] == "MC_CONSISTENT", result
    assert result["n_realizations"] == 250
    assert result["selection_changed_per_exposure"] == [0, 0]
    assert result["nonconverged_per_exposure"] == [0, 0]
    assert result["selection_change_fraction_any_exposure"] == 0.
    assert result["conditional_mc_consistent"]
    assert result["scope"] == "SYNTHETIC_KELSON_RECLIPPED_FIXED_2D_TO_1D"
    assert result["science_uncertainty_qualified"] is False
    assert result["production_variance_release_allowed"] is False
    predicted = np.asarray(result["analytic_covariance"])
    observed = np.asarray(result["empirical_covariance"])
    assert predicted.shape == (2, 2)
    assert np.min(np.linalg.eigvalsh(predicted)) > 0
    assert np.min(np.linalg.eigvalsh(observed)) > 0
    np.testing.assert_allclose(
        predicted[0, 1],
        (result["analytic_cross_independent"]
         + result["analytic_cross_shared_bias"]),
        atol=1e-11, rtol=1e-11,
    )
    assert np.max(result["relative_variance_errors"]) < result["tolerances"]["relative_variance"]
    assert max(np.abs(np.asarray(result["coverage_95percent"])-.95)) < result["tolerances"]["coverage_95"]


def test_full_chain_reclipping_detects_synthetic_sky_selection_changes():
    report = synthetic_reclipped_sky_to_1d_coverage(
        n_realizations=250, seed=1904, group_ids=("A", "A"),
        clip_sigma=3.2, contamination_sigma=4.1,
    )
    assert report["status"] == "SELECTION_VARIABLE", report
    assert report["selection_change_fraction_any_exposure"] > 0
    assert report["selection_changed_per_exposure"][0] > 0
    assert report["science_uncertainty_qualified"] is False
    assert report["production_variance_release_allowed"] is False
    # Even a nonstationary estimator must report the coverage, not
    # silently treat this failure of frozen-sample assumptions as zero noise.
    assert len(report["coverage_1sigma"]) == 2
    assert len(report["coverage_95percent"]) == 2


def test_cross_exposure_bias_sharing_changes_predicted_extracted_covariance():
    settings = dict(
        n_realizations=200, seed=101,
        clip_sigma=7., contamination_sigma=0.,
    )
    shared = synthetic_reclipped_sky_to_1d_coverage(
        group_ids=("A", "A"), **settings,
    )
    split = synthetic_reclipped_sky_to_1d_coverage(
        group_ids=("A", "B"), **settings,
    )
    for result in (shared, split):
        assert result["status"] == "MC_CONSISTENT", result
        assert result["nonconverged_per_exposure"] == [0, 0]
    c_shared = np.asarray(shared["analytic_covariance"])
    c_split = np.asarray(split["analytic_covariance"])
    assert not np.allclose(c_shared, c_split, rtol=1e-8, atol=1e-12)
    assert shared["analytic_cross_independent"] == pytest.approx(
        split["analytic_cross_independent"], abs=0., rel=0.,
    )


def test_seed_reproducibility_of_full_chain_raw_reclipping_diagnostic():
    settings = dict(n_realizations=200, seed=99017)
    first = synthetic_reclipped_sky_to_1d_coverage(**settings)
    second = synthetic_reclipped_sky_to_1d_coverage(**settings)
    assert first == second
    assert first["science_uncertainty_qualified"] is False


@pytest.mark.parametrize("bad", [
    {"group_ids": ("A",)},
    {"group_ids": ("A", " B")},
    {"group_ids": ("", "A")},
    {"group_ids": ("A", 7)},
    {"y_order": 2},
    {"y_order": True},
    {"n_realizations": 199},
    {"n_realizations": 1001},
    {"n_realizations": True},
    {"seed": -1},
    {"seed": 1.5},
    {"clip_sigma": 0.5},
    {"clip_sigma": float("nan")},
    {"contamination_sigma": -1.},
    {"contamination_sigma": float("nan")},
])
def test_invalid_reclipping_full_chain_specs_fail_before_simulation(bad):
    with pytest.raises(ValueError):
        synthetic_reclipped_sky_to_1d_coverage(**bad)
