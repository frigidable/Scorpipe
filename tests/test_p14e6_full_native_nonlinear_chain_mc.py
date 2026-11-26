"""P1.4e6: native CCD Kelson→COUNT_V1→adaptive Stack2D/Extract1D MC."""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.calib.full_nonlinear_chain_mc import (
    synthetic_full_nonlinear_chain_coverage,
)


@pytest.mark.parametrize("y_order", [0, 1])
def test_full_native_chain_has_consistent_fixed_grouped_covariance(y_order):
    result = synthetic_full_nonlinear_chain_coverage(
        group_ids=("A", "B", "C"), y_order=y_order,
        n_realizations=250, seed=2718+y_order,
        true_flux=0., clip_sigma=7., contamination_sigma=0.,
    )
    assert result["status"] == "JOINT_ADAPTIVE_MC_REPORTED", result
    assert result["frozen_reference_mc_consistent"], result
    assert result["baseline_lsqr_stop"] in (1, 2)
    assert result["per_exposure_nonconverged"] == [0, 0, 0]
    assert result["per_exposure_sky_reselection"] == [0, 0, 0]
    assert result["any_reselection_fraction"] == 0.
    assert result["scope"] == "SYNTHETIC_NATIVE_KELSON_TO_ADAPTIVE_STACK_EXTRACT"
    assert result["science_uncertainty_qualified"] is False
    assert result["production_variance_release_allowed"] is False
    assert result["frozen_reference_variance"] == pytest.approx(
        result["frozen_reference_independent"]
        + result["frozen_reference_shared_masterbias"], rel=1e-12,
    )
    assert result["frozen_reference_variance"] > 0.
    assert abs(result["frozen_reference_coverage95"]-.95) <= (
        result["tolerances"]["reference_coverage95"]
    )
    assert set(result["estimators"]) == {
        "invvar_known", "invvar_adaptive", "huber_known", "huber_adaptive",
    }
    assert result["mean_suppression_fraction"]["invvar"] == 0.
    assert result["mean_suppression_fraction"]["huber"] > 0.


def test_explicit_shared_masterbias_changes_full_native_reference_variance():
    args = dict(y_order=0, true_flux=0., n_realizations=250,
                clip_sigma=7., contamination_sigma=0., seed=1519)
    split = synthetic_full_nonlinear_chain_coverage(
        group_ids=("A", "B", "C"), **args,
    )
    shared = synthetic_full_nonlinear_chain_coverage(
        group_ids=("A", "A", "A"), **args,
    )
    assert split["status"] == "JOINT_ADAPTIVE_MC_REPORTED", split
    assert shared["status"] == "JOINT_ADAPTIVE_MC_REPORTED", shared
    assert shared["frozen_reference_variance"] > split["frozen_reference_variance"]
    assert shared["frozen_reference_independent"] == pytest.approx(
        split["frozen_reference_independent"], rel=0, abs=0,
    )
    assert shared["frozen_reference_shared_masterbias"] > (
        split["frozen_reference_shared_masterbias"]
    )
    assert shared["frozen_reference_mc_consistent"]
    assert split["frozen_reference_mc_consistent"]
    assert shared["production_variance_release_allowed"] is False


def test_complete_nonlinear_chain_reports_pairwise_interaction_not_assumed_additive():
    report = synthetic_full_nonlinear_chain_coverage(
        n_realizations=200, seed=17, true_flux=25.,
        smooth_lambda_sigma=3., y_order=0,
    )
    assert report["status"] == "JOINT_ADAPTIVE_MC_REPORTED", report
    p = report["paired_effects"]
    assert len(p) == 5
    for detail in p.values():
        assert np.isfinite(detail["mean"])
        assert np.isfinite(detail["standard_error"])
        assert detail["standard_error"] >= 0
        assert isinstance(detail["detected_at_5sigma"], bool)
    np.testing.assert_allclose(
        p["interaction"]["mean"],
        p["profile_huber"]["mean"]-p["profile_invvar"]["mean"],
        atol=1e-11, rtol=1e-11,
    )
    assert np.isfinite(report["known_transformed_true_flux"])
    assert report["science_uncertainty_qualified"] is False


def test_threshold_crossing_sky_reselection_is_explicitly_unqualified():
    report = synthetic_full_nonlinear_chain_coverage(
        group_ids=("A", "B", "C"), n_realizations=200,
        seed=1904, clip_sigma=3.2, contamination_sigma=4.1,
        true_flux=0.,
    )
    assert report["status"] in {
        "SKY_SELECTION_VARIABLE", "NUMERIC_UNSTABLE",
    }, report
    assert report["any_reselection_fraction"] > 0.
    assert sum(report["per_exposure_sky_reselection"]) > 0
    assert report["science_uncertainty_qualified"] is False
    assert report["production_variance_release_allowed"] is False
    for name in ("invvar_known", "invvar_adaptive",
                 "huber_known", "huber_adaptive"):
        assert 0 <= report["estimators"][name]["coverage_95percent"] <= 1


def test_same_seed_returns_exact_same_raw_to_extracted_report():
    opts = dict(
        group_ids=("A", "B", "C"), true_flux=0.,
        n_realizations=200, seed=677,
        huber_c=1.5, robust_iter=1,
    )
    a = synthetic_full_nonlinear_chain_coverage(**opts)
    b = synthetic_full_nonlinear_chain_coverage(**opts)
    assert a == b
    assert not a["science_uncertainty_qualified"]


@pytest.mark.parametrize("bad", [
    {"group_ids": ("A", "B")},
    {"group_ids": ("A", "", "C")},
    {"group_ids": ("A", " B", "C")},
    {"group_ids": ("A", 4, "C")},
    {"y_order": 2},
    {"y_order": True},
    {"true_flux": float("nan")},
    {"true_flux": -1.},
    {"true_flux": 1001.},
    {"true_flux": True},
    {"n_realizations": 199},
    {"n_realizations": 801},
    {"n_realizations": True},
    {"seed": -1},
    {"seed": 1.5},
    {"clip_sigma": float("nan")},
    {"clip_sigma": .5},
    {"clip_sigma": True},
    {"contamination_sigma": -1.},
    {"contamination_sigma": float("nan")},
    {"contamination_sigma": True},
    {"huber_c": 0.},
    {"huber_c": float("nan")},
    {"huber_c": True},
    {"robust_iter": -1},
    {"robust_iter": 6},
    {"robust_iter": True},
    {"smooth_lambda_sigma": -1.},
    {"smooth_lambda_sigma": 11.},
    {"smooth_lambda_sigma": float("nan")},
    {"smooth_lambda_sigma": True},
])
def test_invalid_full_native_nonlinear_mc_controls_fail_closed(bad):
    with pytest.raises(ValueError):
        synthetic_full_nonlinear_chain_coverage(**bad)
