"""Independent SVD oracle for fixed SkySub WLS normal-equation numerics."""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.calib.sky_bias_covariance import (
    FrozenSkyWLS, frozen_kelson_bspline_operator,
)
from scorpio_pipe.calib.sky_wls_conditioning import audit_frozen_wls_numerics
from scorpio_pipe.stages.sky_sub import _make_bspline_knot_vector


def _well_conditioned_op():
    n = 13
    x = np.linspace(-1., 1., n)
    design = np.column_stack([np.ones(n), x, x*x])
    idx = np.array([0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12])
    weights = np.linspace(.6, 1.7, len(idx))
    return FrozenSkyWLS(design, idx, weights)


def test_independent_svd_agrees_with_wls_forward_model_and_adjoint():
    report = audit_frozen_wls_numerics(_well_conditioned_op(), n_probes=4)
    assert report["status"] == "STABLE", report
    assert report["n_probes"] == 4
    assert report["effective_rank"] == 3
    assert report["n_pixels"] == 13
    assert report["condition_weighted_design"] < 1e5
    assert report["max_forward_relative_error"] < 1e-10
    assert report["max_model_relative_error"] < 1e-10
    assert report["max_adjoint_relative_error"] < 1e-10
    assert report["max_duality_relative_error"] < 1e-10
    assert not report["science_uncertainty_qualified"]


def test_ill_conditioned_full_rank_wls_cannot_claim_stability():
    n = 16
    v = np.linspace(-1., 1., n)
    # Two almost-collinear fit columns. Full rank but a small singular
    # value: normal equations square the already large condition number.
    design = np.column_stack([
        np.ones(n), v, v + 1e-4*np.sin(np.arange(n)*1.31),
    ])
    op = FrozenSkyWLS(design, np.arange(n), np.ones(n))
    report = audit_frozen_wls_numerics(
        op, n_probes=2, max_condition=1000.,
    )
    assert report["status"] == "ILL_CONDITIONED", report
    assert report["effective_rank"] == 3
    assert report["condition_weighted_design"] > 1000.
    assert report["normal_equation_condition_estimate"] > 1e6
    assert not report["science_uncertainty_qualified"]


@pytest.mark.parametrize("order", [0, 1])
def test_actual_kelson_bspline_sparse_basis_matches_independent_svd(order):
    rows, cols = 8, 20
    yy, xx = np.indices((rows, cols))
    wave = 5000. + 1.5*xx + .03*yy
    mask = np.ones((rows, cols), dtype=bool)
    mask[3:5] = False
    var = (2. + .02*xx + .03*yy)
    knot = _make_bspline_knot_vector(
        float(np.min(wave)), float(np.max(wave)),
        knot_step_A=5., degree=2,
    )
    op = frozen_kelson_bspline_operator(
        wave, np.linspace(-1., 1., rows),
        knots=knot, degree=2, y_order=order,
        final_fit_mask=mask, variance=var,
    )
    report = audit_frozen_wls_numerics(
        op, n_probes=2, max_condition=1e6,
        max_relative_error=1e-7,
    )
    assert report["status"] == "STABLE", report
    assert report["effective_rank"] == op.design.shape[1]
    assert report["n_fit_samples"] == mask.sum()
    assert report["max_model_relative_error"] < 1e-7


def test_zero_numeric_tolerance_detects_roundoff():
    report = audit_frozen_wls_numerics(
        _well_conditioned_op(), n_probes=2,
        max_relative_error=0.,
    )
    assert report["status"] == "NUMERIC_MISMATCH"
    assert not report["science_uncertainty_qualified"]


def test_seeded_svd_diagnostic_is_reproducible():
    op = _well_conditioned_op()
    a = audit_frozen_wls_numerics(op, n_probes=3, seed=177)
    b = audit_frozen_wls_numerics(op, n_probes=3, seed=177)
    assert a == b


@pytest.mark.parametrize("settings", [
    {"n_probes": 0},
    {"n_probes": 9},
    {"n_probes": True},
    {"seed": -1},
    {"seed": 1.25},
    {"max_condition": 0.},
    {"max_condition": np.nan},
    {"max_relative_error": -1.},
    {"max_relative_error": np.inf},
    {"max_fit_entries": 2},
    {"max_fit_entries": True},
    {"max_native_pixels": 3},
    {"max_native_pixels": False},
])
def test_svd_diagnostic_rejects_invalid_or_expensive_config(settings):
    with pytest.raises(ValueError):
        audit_frozen_wls_numerics(_well_conditioned_op(), **settings)


def test_ridge_fit_refuses_unregularized_svd_interpretation():
    n = 9
    x = np.column_stack([np.ones(n), np.linspace(-1., 1., n)])
    op = FrozenSkyWLS(x, np.arange(n), np.ones(n), ridge=.001)
    with pytest.raises(ValueError, match="unregularized"):
        audit_frozen_wls_numerics(op)
