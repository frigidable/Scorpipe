"""Synthetic repeated production Kelson clipping vs conditional covariance."""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.calib.sky_mc_reclipping_coverage import (
    synthetic_reclipped_sky_coverage,
)


@pytest.mark.parametrize("y_order", [0, 1])
def test_smooth_noise_large_clip_keeps_conditional_coverage(y_order):
    report = synthetic_reclipped_sky_coverage(
        y_order=y_order, group_ids=("A", "A"),
        clip_sigma=7.0, contamination_sigma=0.0,
        n_realizations=350, seed=1651 + y_order,
    )
    assert report["status"] == "MC_CONSISTENT", report
    assert report["n_realizations"] == 350
    assert report["selection_change_fraction_any_exposure"] == 0.
    assert report["n_selection_changed_per_exposure"] == [0, 0]
    assert report["n_nonconverged_per_exposure"] == [0, 0]
    assert report["conditional_mc_consistent"]
    assert report["science_uncertainty_qualified"] is False
    pred = np.asarray(report["analytic_covariance"])
    empirical = np.asarray(report["empirical_covariance"])
    np.testing.assert_allclose(
        pred,
        np.asarray(report["analytic_independent"])
        + np.asarray(report["analytic_shared_masterbias"]),
        rtol=1e-12, atol=1e-12,
    )
    assert np.min(np.linalg.eigvalsh(pred)) > 0.
    assert np.min(np.linalg.eigvalsh(empirical)) > 0.


def test_single_synthetic_sky_outlier_changes_full_refit_selection():
    report = synthetic_reclipped_sky_coverage(
        group_ids=("A", "A"), y_order=0,
        clip_sigma=3.2, contamination_sigma=4.1,
        n_realizations=300, seed=1904,
    )
    assert report["status"] == "SELECTION_VARIABLE", report
    assert report["selection_change_fraction_any_exposure"] > 0.01
    assert report["n_selection_changed_per_exposure"][0] > 0
    assert not report["science_uncertainty_qualified"]
    # Coverage and variance are *still measured*, even if the conditional
    # fixed-sample model is out of scope for some realizations.
    assert len(report["coverage_95percent"]) == 2
    assert len(report["variance_relative_error"]) == 2


def test_shared_and_independent_bias_groups_change_covariance_prediction():
    same = synthetic_reclipped_sky_coverage(
        group_ids=("A", "A"), n_realizations=220, seed=419,
    )
    split = synthetic_reclipped_sky_coverage(
        group_ids=("A", "B"), n_realizations=220, seed=419,
    )
    assert same["status"] == "MC_CONSISTENT", same
    assert split["status"] == "MC_CONSISTENT", split
    s = np.asarray(same["analytic_covariance"])
    d = np.asarray(split["analytic_covariance"])
    ind = np.asarray(same["analytic_independent"])
    np.testing.assert_allclose(
        ind, np.asarray(split["analytic_independent"]),
        rtol=0, atol=0,
    )
    assert s[0, 0] > d[0, 0]
    assert np.all(np.diag(s) > np.diag(ind))


def test_synthetic_production_reclipping_report_seed_exactly_reproducible():
    kw = dict(n_realizations=200, seed=9)
    a = synthetic_reclipped_sky_coverage(**kw)
    b = synthetic_reclipped_sky_coverage(**kw)
    assert a == b
    assert a["scope"] == "SYNTHETIC_PRODUCTION_KELSON_RECLIPPED"


@pytest.mark.parametrize("bad", [
    {"group_ids": ("A",)},
    {"group_ids": ("A", " B")},
    {"group_ids": ("", "B")},
    {"group_ids": ("A", 5)},
    {"y_order": 2},
    {"y_order": True},
    {"n_realizations": 199},
    {"n_realizations": 3001},
    {"n_realizations": True},
    {"seed": -1},
    {"seed": 1.2},
    {"clip_sigma": 0.5},
    {"clip_sigma": float("nan")},
    {"contamination_sigma": -1.},
    {"contamination_sigma": float("nan")},
])
def test_reclipping_mc_rejects_bad_experiment_controls(bad):
    with pytest.raises(ValueError):
        synthetic_reclipped_sky_coverage(**bad)
