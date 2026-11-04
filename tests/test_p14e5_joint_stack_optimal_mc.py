"""Paired production adaptive Stack2D and Extract1D Monte Carlo validation."""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.calib.joint_adaptive_stack_optimal_mc import (
    synthetic_joint_stack_optimal_coverage,
)


def test_four_paired_estimators_detect_same_data_profile_bias():
    report = synthetic_joint_stack_optimal_coverage(
        group_ids=("A", "B", "C"), true_flux=0.,
        n_realizations=400, seed=20261104, huber_c=1.,
        robust_iter=2, smooth_lambda_sigma=0.,
    )
    assert report["status"] == "PAIRED_MC_REPORTED", report
    assert report["known_grouped_reference_consistent"], report
    assert report["exact_invvar_known_profile_grouped_variance"] == pytest.approx(
        report["naive_invvar_known_profile_variance"], rel=0, abs=1e-12,
    )
    assert report["science_uncertainty_qualified"] is False
    assert report["production_variance_release_allowed"] is False
    assert report["adaptive_profile_relearned_per_draw"] is True
    assert report["huber_weights_relearned_per_draw"] is True

    e = report["estimators"]
    assert set(e) == {
        "invvar_known", "invvar_adaptive", "huber_known", "huber_adaptive"
    }
    assert abs(e["invvar_known"]["mean_error"]) < (
        report["reference_tolerances"]["mean_absolute"]
    )
    assert abs(e["invvar_known"]["coverage_95percent"]-.95) < (
        report["reference_tolerances"]["reference_coverage95"]
    )
    assert e["invvar_adaptive"]["mean_error"] > 1.
    assert e["invvar_adaptive"]["coverage_95percent"] < .9
    assert report["paired_effects"]["adaptive_profile_after_invvar"][
        "detected_at_5sigma"
    ]
    assert report["paired_effects"]["adaptive_profile_after_invvar"]["mean"] > 1.
    assert report["mean_suppression_fraction"]["invvar"] == 0.
    assert report["mean_suppression_fraction"]["huber"] > 0.


def test_common_masterbias_breaks_independent_only_final_interval():
    same = synthetic_joint_stack_optimal_coverage(
        group_ids=("A", "A", "A"), true_flux=0.,
        n_realizations=400, seed=303, huber_c=1.,
    )
    assert same["status"] == "PAIRED_MC_REPORTED", same
    expected_naive = same["naive_invvar_known_profile_variance"]
    expected_grouped = same["exact_invvar_known_profile_grouped_variance"]
    assert expected_grouped / expected_naive == pytest.approx(2., rel=0, abs=1e-12)
    assert same["known_grouped_reference_consistent"]
    assert abs(same["known_grouped_reference_coverage95"]-.95) < (
        same["reference_tolerances"]["reference_coverage95"]
    )
    baseline = same["estimators"]["invvar_known"]
    assert baseline["variance_ratio"] > 1.5
    assert baseline["coverage_95percent"] < .9
    assert same["science_uncertainty_qualified"] is False
    assert same["production_variance_release_allowed"] is False


def test_interaction_recorded_without_assuming_additivity_or_sign():
    report = synthetic_joint_stack_optimal_coverage(
        group_ids=("A", "B", "C"), true_flux=25.,
        n_realizations=250, seed=1512,
        smooth_lambda_sigma=3.,
    )
    assert report["status"] == "PAIRED_MC_REPORTED", report
    effects = report["paired_effects"]
    assert len(effects) == 5
    for entry in effects.values():
        assert np.isfinite(entry["mean"])
        assert np.isfinite(entry["standard_error"])
        assert entry["standard_error"] >= 0
        assert isinstance(entry["detected_at_5sigma"], bool)
    np.testing.assert_allclose(
        effects["huber_profile_interaction"]["mean"],
        effects["adaptive_profile_after_huber"]["mean"]
        - effects["adaptive_profile_after_invvar"]["mean"],
        atol=1e-12, rtol=1e-12,
    )
    assert report["science_uncertainty_qualified"] is False


def test_fixed_seed_joint_adaptive_qa_reproducible():
    params = dict(
        group_ids=("A", "B", "C"), n_realizations=250,
        seed=5194, true_flux=0., huber_c=1.5,
    )
    first = synthetic_joint_stack_optimal_coverage(**params)
    second = synthetic_joint_stack_optimal_coverage(**params)
    assert first == second
    assert first["scope"] == "SYNTHETIC_ACTUAL_STACK2D_AND_OPTIMAL_EXTRACT_JOINT_MC"


@pytest.mark.parametrize("bad", [
    {"group_ids": ("A", "B")},
    {"group_ids": ("A", "", "C")},
    {"group_ids": ("A", " B", "C")},
    {"group_ids": ("A", 4, "C")},
    {"true_flux": float("nan")},
    {"true_flux": -1.},
    {"true_flux": True},
    {"true_flux": 1001.},
    {"n_realizations": 249},
    {"n_realizations": 1501},
    {"n_realizations": True},
    {"seed": -1},
    {"seed": 1.5},
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
def test_joint_adaptive_mc_invalid_controls_fail_closed(bad):
    with pytest.raises(ValueError):
        synthetic_joint_stack_optimal_coverage(**bad)
