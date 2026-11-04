"""Controlled synthetic LU/SVD normal-equation conditioning vs MC coverage."""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.calib.sky_conditioning_coverage_sweep import (
    synthetic_wls_conditioning_coverage_sweep,
)


def test_condition_sweep_separates_numeric_gate_from_empirical_coverage():
    # Independent SVD-solved Monte Carlo can show nominal coverage for
    # selected apertures while LU remains unsafe on highly collinear fits.
    result = synthetic_wls_conditioning_coverage_sweep(
        n_realizations=3000, seed=113,
    )
    assert result["status"] == "REPORTED", result
    assert result["scope"] == "SYNTHETIC_NEAR_COLLINEAR_WLS_SVD_MC"
    assert result["science_uncertainty_qualified"] is False
    trials = result["trials"]
    assert len(trials) == 4
    assert [x["status"] for x in trials] == [
        "NUMERIC_SAFE", "NUMERIC_SAFE",
        "NUMERIC_BLOCKED", "NUMERIC_BLOCKED",
    ]
    for t in trials:
        assert t["reference_mc_consistent"], t
        assert np.max(np.abs(np.asarray(t["reference_coverage95"])-.95)) < .04
        assert np.min(np.linalg.eigvalsh(t["reference_covariance"])) > 0
        assert np.min(np.linalg.eigvalsh(t["empirical_covariance"])) > 0
        assert t["lu_predicted_covariance"] is not None
        assert np.isfinite(t["lu_to_svd_variance_ratios"]).all()
    assert max(trials[0]["weighted_design_conditions"]) < 1e5
    assert min(trials[-1]["weighted_design_conditions"]) > 1e6
    # Detect the threshold crossing from the SVD spectrum, not an
    # arbitrarily failing Monte Carlo test on a single observable.
    assert all(
        "ILL_CONDITIONED" in x["svd_gate_statuses"] for x in trials[2:]
    )


def test_shared_vs_independent_calibration_groups_change_sampling_covariance():
    args = dict(perturbations=(.1, .001), n_realizations=2000, seed=99)
    same = synthetic_wls_conditioning_coverage_sweep(
        **args, group_ids=("A", "A"),
    )
    separate = synthetic_wls_conditioning_coverage_sweep(
        **args, group_ids=("A", "B"),
    )
    assert same["status"] == "REPORTED"
    assert separate["status"] == "REPORTED"
    for sc, dc in zip(same["trials"], separate["trials"]):
        assert sc["status"] == "NUMERIC_SAFE"
        assert dc["status"] == "NUMERIC_SAFE"
        assert sc["reference_covariance"][0][0] > (
            1.05 * dc["reference_covariance"][0][0]
        )
        assert sc["reference_mc_consistent"]
        assert dc["reference_mc_consistent"]


def test_exact_seed_reproducibility_and_no_science_qualification():
    settings = dict(
        perturbations=(.1, .001), n_realizations=2000, seed=11,
    )
    a = synthetic_wls_conditioning_coverage_sweep(**settings)
    b = synthetic_wls_conditioning_coverage_sweep(**settings)
    assert a == b
    assert not a["science_uncertainty_qualified"]


@pytest.mark.parametrize("params", [
    {"perturbations": []},
    {"perturbations": [1e-1]},
    {"perturbations": [1e-3, 1e-2]},
    {"perturbations": [1e-2, 1e-2]},
    {"perturbations": [1e-2, float("nan")]},
    {"perturbations": [1e-1, -1.]},
    {"perturbations": [True, .1]},
    {"group_ids": ("A",)},
    {"group_ids": ("A", " B")},
    {"group_ids": ("", "A")},
    {"n_realizations": 1999},
    {"n_realizations": 20001},
    {"n_realizations": True},
    {"seed": -1},
    {"seed": 2.5},
    {"max_condition": 1},
    {"max_condition": float("nan")},
    {"max_numerical_error": -1.},
    {"max_numerical_error": float("inf")},
])
def test_invalid_condition_sweep_spec_rejected_before_monte_carlo(params):
    with pytest.raises(ValueError):
        synthetic_wls_conditioning_coverage_sweep(**params)
