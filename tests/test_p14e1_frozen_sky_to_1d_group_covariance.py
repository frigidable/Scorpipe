"""Independent dense QR and pixel-level Monte Carlo oracle for SkySub -> 1D."""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.calib.sky_bias_covariance import FrozenSkyWLS
from scorpio_pipe.calib.sky_chain_grouped_covariance import (
    covariance_of_frozen_sky_chain_grouped_pair,
)
from scorpio_pipe.covariance_chain import FrozenCovarianceExposure
from tests.test_p14d1_frozen_chain_covariance import (
    _dense_output_coefficient_matrices,
    _fixtures,
)


def _model(group_ids=("A", "A")):
    exposures, Rs, Etrace, Efixed, d, u, v = _fixtures()
    ny, nx = exposures[0].input_variance.shape
    yy, xx = np.indices((ny, nx), dtype=float)
    x = 2. * xx.ravel() / (nx - 1.) - 1.
    y = 2. * yy.ravel() / (ny - 1.) - 1.
    design = np.column_stack((np.ones(ny*nx), x, y, x*y))
    # Sky windows only in first and last native slit rows.
    idx = np.r_[np.arange(nx), np.arange((ny-1)*nx, ny*nx)]
    bvar = np.linspace(.55, 1.25, ny*nx)
    vars_pre = []
    sensitivities = []
    ops = []
    qr = []
    for e, ex in enumerate(exposures):
        var = np.where(ex.valid_mask, ex.input_variance, 1.7 + .2*e)
        sens = -(0.7 + .08*e + .025*np.arange(ny*nx)).reshape(ny, nx)
        weight = 1. / (var.ravel()[idx] + sens.ravel()[idx]**2*bvar[idx])
        op = FrozenSkyWLS(design, idx, weight)
        root = np.sqrt(weight)
        Q, R = np.linalg.qr(design[idx]*root[:, None], mode="reduced")
        assert np.linalg.cond(R) < 1e6
        vars_pre.append(var)
        sensitivities.append(sens)
        ops.append(op)
        qr.append((Q, R, root))
    return {
        "exposures": exposures, "Rs": Rs, "left_E": Etrace, "right_E": Efixed,
        "transfer": d, "u": u, "v": v, "ops": ops, "var": vars_pre,
        "sensitivity": sensitivities, "bvar": bvar, "groups": group_ids,
        "design": design, "idx": idx, "qr": qr,
    }


def _kwargs(s):
    return dict(
        exposures=s["exposures"], sky_operators=s["ops"],
        left_extraction_weights=s["left_E"],
        right_extraction_weights=s["right_E"],
        fluxcal_transfer=s["transfer"],
        left_weights=s["u"], right_weights=s["v"],
        independent_variances=s["var"],
        bias_sensitivities=s["sensitivity"],
        group_ids=s["groups"],
        masterbias_variances={g: s["bvar"] for g in s["groups"]},
    )


def _reference(s):
    # Independent dense forward model. No calls to the function under
    # test or to FrozenSkyWLS.apply/adjoint.
    L = _dense_output_coefficient_matrices(
        s["exposures"], s["Rs"], s["left_E"], s["transfer"],
    )
    Q = _dense_output_coefficient_matrices(
        s["exposures"], s["Rs"], s["right_E"], s["transfer"],
    )
    projected = []
    for e in range(len(L)):
        A = s["design"][s["idx"]]
        root = s["qr"][e][2]
        fit = A * root[:, None]
        select = np.eye(s["design"].shape[0])[s["idx"]]
        coefficients = np.linalg.lstsq(
            fit, select * root[:, None], rcond=None,
        )[0]
        sky_residual = np.eye(s["design"].shape[0]) - s["design"]@coefficients
        projected.append((
            s["u"] @ L[e] @ sky_residual,
            s["v"] @ Q[e] @ sky_residual,
        ))
    independent = sum(
        float(np.dot(x.ravel(), a*b))
        for x, (a, b) in zip(s["var"], projected)
    )
    shared = 0.
    for gid in set(s["groups"]):
        x = sum(
            (s["sensitivity"][e].ravel() * p[0]
             for e, p in enumerate(projected) if s["groups"][e] == gid),
            np.zeros(s["design"].shape[0]),
        )
        y = sum(
            (s["sensitivity"][e].ravel() * p[1]
             for e, p in enumerate(projected) if s["groups"][e] == gid),
            np.zeros(s["design"].shape[0]),
        )
        shared += float(np.dot(s["bvar"], x*y))
    return independent, shared, independent+shared


@pytest.mark.parametrize("groups", [("A", "A"), ("A", "B")])
def test_grouped_cross_stage_covariance_matches_independent_dense_qr(groups):
    s = _model(groups)
    before = [v.copy() for v in s["var"]]
    predicted = covariance_of_frozen_sky_chain_grouped_pair(**_kwargs(s))
    expected = _reference(s)
    np.testing.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)
    for a, b in zip(s["var"], before):
        np.testing.assert_array_equal(a, b)


def test_shared_bias_groups_propagate_across_sky_model_and_both_exposures():
    same = _model(("A", "A"))
    distinct = _model(("A", "B"))
    ind_same, shared_same, total_same = covariance_of_frozen_sky_chain_grouped_pair(
        **_kwargs(same)
    )
    ind_distinct, shared_distinct, total_distinct = covariance_of_frozen_sky_chain_grouped_pair(
        **_kwargs(distinct)
    )
    assert ind_same == pytest.approx(ind_distinct, rel=0, abs=0)
    assert total_same == pytest.approx(ind_same+shared_same)
    assert total_distinct == pytest.approx(ind_distinct+shared_distinct)
    assert shared_same != pytest.approx(shared_distinct, rel=1e-4)
    np.testing.assert_allclose(
        [ind_same, shared_same, total_same], _reference(same), rtol=1e-10,
    )


def test_independent_noisy_sky_qr_refits_match_full_chain_covariance():
    s = _model(("A", "A"))
    expected = np.asarray(_reference(s))
    rng = np.random.default_rng(20261104)
    n_realizations = 28000
    bias = (rng.standard_normal((n_realizations, s["design"].shape[0]))
            * np.sqrt(s["bvar"])[None, :])
    measured = np.zeros((n_realizations, 2))
    L = _dense_output_coefficient_matrices(
        s["exposures"], s["Rs"], s["left_E"], s["transfer"],
    )
    Q = _dense_output_coefficient_matrices(
        s["exposures"], s["Rs"], s["right_E"], s["transfer"],
    )
    for e, (q, r, root) in enumerate(s["qr"]):
        independent = (
            rng.standard_normal((n_realizations, s["design"].shape[0]))
            * np.sqrt(s["var"][e].ravel())[None, :]
        )
        true_sky = s["design"] @ np.array([13., -2., 1., .25])
        obj = np.zeros(s["design"].shape[0])
        obj[4:12] = 2. + .2*e  # Outside the sky training rows.
        signal = (true_sky[None, :] + obj[None, :] + independent
                  + s["sensitivity"][e].ravel()[None, :]*bias)
        # Full per-realization independent QR sky fit and subtraction.
        coefficients = np.linalg.solve(
            r, ((signal[:, s["idx"]]*root[None, :]) @ q).T,
        ).T
        sky_sub = signal - coefficients@s["design"].T - obj[None, :]
        measured[:, 0] += sky_sub @ (s["u"] @ L[e])
        measured[:, 1] += sky_sub @ (s["v"] @ Q[e])
    empirical = np.cov(measured, rowvar=False, ddof=1)
    kw = _kwargs(s)
    _, _, pred_ab = covariance_of_frozen_sky_chain_grouped_pair(**kw)
    kw["right_extraction_weights"] = s["left_E"]
    kw["right_weights"] = s["u"]
    _, _, pred_aa = covariance_of_frozen_sky_chain_grouped_pair(**kw)
    kw["left_extraction_weights"] = s["right_E"]
    kw["left_weights"] = s["v"]
    kw["right_extraction_weights"] = s["right_E"]
    kw["right_weights"] = s["v"]
    _, _, pred_bb = covariance_of_frozen_sky_chain_grouped_pair(**kw)
    predicted = np.array([[pred_aa, pred_ab], [pred_ab, pred_bb]])
    assert np.linalg.eigvalsh(predicted)[0] > 0
    # 28k independent MC draws, with normalized bounds appropriate to
    # both diagonal variances and the signed off-diagonal covariance.
    normalized = (
        np.abs(empirical-predicted) /
        np.sqrt(np.outer(np.diag(predicted), np.diag(predicted)))
    )
    assert np.max(normalized) < .045, (empirical, predicted, normalized)
    assert expected[2] == pytest.approx(pred_ab, rel=1e-10)


@pytest.mark.parametrize("damage", [
    "missing_group", "wrong_group", "wrong_sky_pixels", "wrong_bias",
    "wrong_sensitivity", "nan_pre_sky_var", "wrong_extraction",
    "bad_stack", "uncovered_bin", "missing_row_plan",
])
def test_unknown_upstream_lineage_and_unsupported_geometry_fail_closed(damage):
    s = _model()
    args = _kwargs(s)
    if damage == "missing_group":
        args["group_ids"] = ("", "A")
    elif damage == "wrong_group":
        args["masterbias_variances"] = {"C": s["bvar"]}
    elif damage == "wrong_sky_pixels":
        args["sky_operators"] = [FrozenSkyWLS(
            np.ones((20, 1)), np.arange(4), np.ones(4)
        ), s["ops"][1]]
    elif damage == "wrong_bias":
        args["masterbias_variances"] = {"A": np.ones(12)}
    elif damage == "wrong_sensitivity":
        args["bias_sensitivities"] = [np.ones((2, 4)), s["sensitivity"][1]]
    elif damage == "nan_pre_sky_var":
        args["independent_variances"] = [s["var"][0].copy(), s["var"][1]]
        args["independent_variances"][0][2, 0] = np.nan
    elif damage == "wrong_extraction":
        args["left_extraction_weights"] = np.ones((3, 3))
    elif damage == "bad_stack":
        ex = s["exposures"][0]
        w = ex.stack_weights.copy()
        w[1, 1] = .8
        args["exposures"] = [
            FrozenCovarianceExposure(ex.row_plans, ex.input_variance,
                                     ex.valid_mask, w, ex.y_shift, ex.output_scale),
            s["exposures"][1],
        ]
    elif damage == "uncovered_bin":
        a = s["exposures"][0]
        b = s["exposures"][1]
        wa = a.stack_weights.copy()
        wb = b.stack_weights.copy()
        wa[:, 0] = 0.
        wb[:, 0] = 0.
        args["exposures"] = [
            FrozenCovarianceExposure(a.row_plans, a.input_variance,
                                     a.valid_mask, wa, a.y_shift, a.output_scale),
            FrozenCovarianceExposure(b.row_plans, b.input_variance,
                                     b.valid_mask, wb, b.y_shift, b.output_scale),
        ]
    elif damage == "missing_row_plan":
        a = s["exposures"][0]
        p = list(a.row_plans)
        p[1] = None
        args["exposures"] = [
            FrozenCovarianceExposure(p, a.input_variance, a.valid_mask,
                                     a.stack_weights, a.y_shift, a.output_scale),
            s["exposures"][1],
        ]
    with pytest.raises(ValueError):
        covariance_of_frozen_sky_chain_grouped_pair(**args)
