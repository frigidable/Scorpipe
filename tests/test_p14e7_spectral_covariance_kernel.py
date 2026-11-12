"""P1.4e7: 1-D spectral covariance and integrated line uncertainty oracles."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from scorpio_pipe.calib.spectral_covariance_kernel import (
    frozen_sky_1d_spectral_covariance_kernel,
)
from tests.test_p14e1_frozen_sky_to_1d_group_covariance import (
    _model, _kwargs,
)
from tests.test_p14d1_frozen_chain_covariance import (
    _dense_output_coefficient_matrices,
)


def _call(model, *, bins=(0, 1, 2), weights=(1., 1., 1.)):
    d = _kwargs(model)
    return frozen_sky_1d_spectral_covariance_kernel(
        exposures=d["exposures"],
        sky_operators=d["sky_operators"],
        extraction_weights=model["left_E"],
        fluxcal_transfer=d["fluxcal_transfer"],
        independent_variances=d["independent_variances"],
        bias_sensitivities=d["bias_sensitivities"],
        group_ids=d["group_ids"],
        masterbias_variances=d["masterbias_variances"],
        wavelength_indices=bins,
        integrated_line_weights=weights,
    )


def _independent_dense_reference(model):
    """Dense forward geometry and weighted QR: no tested covariance adjoint."""
    M = _dense_output_coefficient_matrices(
        model["exposures"], model["Rs"],
        model["left_E"], model["transfer"],
    )
    npx = model["design"].shape[0]
    projectors = []
    for e in range(2):
        root = model["qr"][e][2]
        select = np.eye(npx)[model["idx"]]
        weighted_fit = model["design"][model["idx"]] * root[:, None]
        coeff_map = np.linalg.lstsq(
            weighted_fit, select * root[:, None], rcond=None,
        )[0]
        residual_map = np.eye(npx) - model["design"] @ coeff_map
        projectors.append(M[e] @ residual_map)
    ind = np.zeros((3, 3))
    for e, matrix in enumerate(projectors):
        ind += (
            matrix * model["var"][e].ravel()[None, :]
        ) @ matrix.T
    shared = np.zeros((3, 3))
    for g in set(model["groups"]):
        gradient = sum((
            model["sensitivity"][e].ravel()[None, :]*matrix
            for e, matrix in enumerate(projectors)
            if model["groups"][e] == g
        ), np.zeros((3, npx)))
        shared += (
            gradient * model["bvar"][None, :]
        ) @ gradient.T
    return ind, shared, projectors, M


@pytest.mark.parametrize("groups", [("A", "A"), ("A", "B")])
def test_three_bin_kernel_matches_independent_dense_forward_qr(groups):
    s = _model(groups)
    report = _call(s)
    independent, shared, _, _ = _independent_dense_reference(s)
    total = independent+shared
    np.testing.assert_allclose(report["independent_covariance"], independent,
                               rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(report["shared_masterbias_covariance"], shared,
                               rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(report["total_covariance"], total,
                               rtol=1e-10, atol=1e-10)
    assert np.min(np.linalg.eigvalsh(total)) > 0
    assert report["has_nonzero_interchannel_covariance"]
    assert report["integrated_line_variance_full"] == pytest.approx(
        np.ones(3) @ total @ np.ones(3), rel=1e-12
    )
    assert report["integrated_line_variance_diagonal_only"] == pytest.approx(
        np.trace(total), rel=1e-12
    )
    # In THIS explicit algebraic geometry, nearby-bin covariance causes
    # a meaningful positive missed term; no general sign is asserted.
    assert report["line_variance_ratio_full_to_diagonal"] > 1.1
    assert report["line_sigma_ratio_full_to_diagonal"] > 1.05
    assert not report["science_uncertainty_qualified"]
    assert not report["production_variance_release_allowed"]


def test_grouping_changes_spectral_covariance_without_changing_independent_noise():
    same = _call(_model(("A", "A")))
    split = _call(_model(("A", "B")))
    np.testing.assert_allclose(
        same["independent_covariance"], split["independent_covariance"],
        atol=1e-12, rtol=0.,
    )
    assert (
        same["integrated_line_variance_masterbias"]
        > split["integrated_line_variance_masterbias"]
    )
    assert (
        same["integrated_line_variance_full"]
        > split["integrated_line_variance_full"]
    )


def test_true_full_line_variance_matches_20000_independent_qr_refits():
    s = _model(("A", "A"))
    result = _call(s)
    rng = np.random.default_rng(20261104)
    n = 20000
    npx = s["design"].shape[0]
    shared_noise = rng.standard_normal((n, npx))*np.sqrt(s["bvar"])[None, :]
    observed = np.zeros((n, 3), dtype=np.float64)
    _, _, _, M = _independent_dense_reference(s)
    for e, (q, r, root) in enumerate(s["qr"]):
        independent_noise = (
            rng.standard_normal((n, npx))
            * np.sqrt(s["var"][e].ravel())[None, :]
        )
        true_sky = s["design"] @ np.array([13., -2., 1., .25])
        obj = np.zeros(npx)
        obj[4:12] = 2. + .2*e  # Not included in sky training.
        frames = (true_sky[None, :] + obj[None, :]
                  + independent_noise
                  + s["sensitivity"][e].ravel()[None, :]*shared_noise)
        coefficients = np.linalg.solve(
            r,
            ((frames[:, s["idx"]]*root[None, :]) @ q).T,
        ).T
        sky_subtracted_errors = frames - coefficients @ s["design"].T - obj
        observed += sky_subtracted_errors @ M[e].T
    empirical = np.cov(observed, rowvar=False, ddof=1)
    predicted = np.asarray(result["total_covariance"])
    error = np.abs(empirical-predicted) / np.sqrt(np.outer(
        np.diag(predicted), np.diag(predicted)
    ))
    assert np.max(error) < .065, (predicted, empirical, error)
    weights = np.ones(3)
    empiric_line = float(np.var(observed @ weights, ddof=1))
    assert empiric_line == pytest.approx(
        result["integrated_line_variance_full"], rel=.06
    )
    assert empiric_line > 1.1*result["integrated_line_variance_diagonal_only"]


def test_signed_line_continuum_subtraction_is_quadratic_not_diagonal():
    s = _model(("A", "B"))
    weights = np.array([1., -.5, 1.])
    result = _call(s, weights=weights)
    C = np.asarray(result["total_covariance"])
    assert result["integrated_line_variance_full"] == pytest.approx(
        float(weights @ C @ weights), rel=1e-12
    )
    assert result["integrated_line_variance_diagonal_only"] == pytest.approx(
        float(np.dot(weights**2, np.diag(C))), rel=1e-12
    )
    assert result["scope"] == "FROZEN_SKYSUB_COUNT_V1_STACK_EXTRACT_FLUXCAL_GROUPED"
    assert result["status"] == "CONDITIONAL_SPECTRAL_KERNEL_ONLY"
    assert result["selection_masks_and_downstream_weights_frozen"] is True
    assert result == _call(s, weights=weights)


@pytest.mark.parametrize("bins,weights", [
    ((), ()),
    ((0,), (1.,)),
    ((0, 0), (1., 1.)),
    ((1, 0), (1., 1.)),
    ((-1, 1), (1., 1.)),
    ((0, 3), (1., 1.)),
    ((False, 1), (1., 1.)),
    ((0., 1), (1., 1.)),
    (tuple(range(13)), tuple([1.]*13)),
    ((0, 1), (1.,)),
    ((0, 1), (1., float("nan"))),
    ((0, 1), (1., float("inf"))),
    ((0, 1), (True, 1.)),
    ((0, 1), (0., 0.)),
])
def test_invalid_spectral_index_and_line_weight_requests_rejected(bins, weights):
    with pytest.raises(ValueError):
        _call(_model(), bins=bins, weights=weights)


def test_missing_masterbias_group_or_nonfinite_native_var_fails_closed():
    s = _model()
    args = _kwargs(s)
    args["group_ids"] = ["A", "B"]
    with pytest.raises(ValueError):
        frozen_sky_1d_spectral_covariance_kernel(
            exposures=args["exposures"], sky_operators=args["sky_operators"],
            extraction_weights=s["left_E"],
            fluxcal_transfer=args["fluxcal_transfer"],
            independent_variances=args["independent_variances"],
            bias_sensitivities=args["bias_sensitivities"],
            group_ids=args["group_ids"],
            masterbias_variances=args["masterbias_variances"],
            wavelength_indices=[0, 1],
            integrated_line_weights=[1., 1.],
        )
    s = _model()
    s["var"][0][0, 0] = np.nan
    with pytest.raises(ValueError):
        _call(s)
