"""Captured actual SkySub fit + source MasterBias conditional math checks.

Numerical tests are intentionally detector-grid only. They do not validate
post-Sky covariance, upstream shared MasterFlat or selection randomness.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.calib.captured_sky_bias_conditional import (
    evaluate_captured_sky_bias_conditional_pair,
    load_captured_sky_bias_conditional_basis,
)
from tests.test_sky_masterbias_source_audit import _night


def _one(tmp_path):
    source, manifest, residual = _night(tmp_path / "science", gid="B1", bvar=4.0)
    return source, manifest, residual


def test_captured_actual_sky_wls_and_local_bias_maps_are_bound(tmp_path):
    source, manifest, residual = _one(tmp_path)
    b = load_captured_sky_bias_conditional_basis([manifest])
    assert b.status == "HOLD_CONDITIONAL_SOURCE_NOISE_NOT_FULL_COVARIANCE"
    assert b.science_uncertainty_qualified is False
    assert b.production_variance_release_allowed is False
    assert b.group_ids == ("B1",)
    assert b.ccd_shapes == ((48, 128),)
    with fits.open(source) as hdul:
        np.testing.assert_allclose(
            b.vrest_if_independent[0].reshape(b.ccd_shapes[0]),
            hdul["VREST"].data,
        )
        np.testing.assert_allclose(
            b.signed_bias_sensitivities[0].reshape(b.ccd_shapes[0]),
            hdul["BSENS"].data,
        )
    with fits.open(residual) as hdul:
        assert hdul[0].header["VCMPQ"] == "SKY_LOST"
    u = np.zeros((48, 128), dtype=float)
    u[24, 65] = 1.
    ans = evaluate_captured_sky_bias_conditional_pair(b, [u], [u])
    assert ans["status"] == "HOLD_CONDITIONAL_SOURCE_NOISE_NOT_FULL_COVARIANCE"
    assert ans["vrest_as_if_independent_term"] > 0.
    assert ans["signed_shared_MasterBias_term"] >= 0.
    assert ans["conditional_sum_under_unverified_independence"] == pytest.approx(
        ans["vrest_as_if_independent_term"] + ans["signed_shared_MasterBias_term"]
    )
    assert ans["downstream_COUNT_V1_Stack2D_Extract1D"] == "NOT_BOUND"
    assert ans["science_uncertainty_qualified"] is False


def test_grouped_frozen_bias_adjoint_matches_direct_matrix_free_oracle(tmp_path):
    _, first, _ = _night(tmp_path / "first", gid="SAME", bvar=4.0)
    _, second, _ = _night(tmp_path / "second", gid="SAME", bvar=4.0)
    b = load_captured_sky_bias_conditional_basis([first, second])
    assert b.group_ids == ("SAME", "SAME")
    u = np.zeros((48, 128), dtype=float)
    v = np.zeros_like(u)
    u[24, 65] = 1.
    v[24, 66] = 1.
    ans = evaluate_captured_sky_bias_conditional_pair(b, [u, u], [v, v])
    aa = [op.adjoint(u.ravel()) for op in b.operators]
    bb = [op.adjoint(v.ravel()) for op in b.operators]
    as_if_independent = sum(float(np.dot(w, a*z)) for w, a, z in zip(
        b.vrest_if_independent, aa, bb,
    ))
    bias_left = sum((s*a for s, a in zip(b.signed_bias_sensitivities, aa)),
                    np.zeros(u.size))
    bias_right = sum((s*z for s, z in zip(b.signed_bias_sensitivities, bb)),
                     np.zeros(u.size))
    shared = float(np.dot(b.masterbias_variances["SAME"], bias_left*bias_right))
    np.testing.assert_allclose(
        [ans["vrest_as_if_independent_term"],
         ans["signed_shared_MasterBias_term"],
         ans["conditional_sum_under_unverified_independence"]],
        [as_if_independent, shared, as_if_independent+shared],
        atol=1e-9, rtol=1e-9,
    )
    assert ans["shared_MasterFlat_and_VREST_off_diagonal"] == "NOT_MODELLED"


def test_grouped_bias_injected_noise_matches_conditional_variance(tmp_path):
    _, first, _ = _night(tmp_path / "first", gid="SAME", bvar=4.)
    _, second, _ = _night(tmp_path / "second", gid="SAME", bvar=4.)
    basis = load_captured_sky_bias_conditional_basis([first, second])
    u = np.zeros((48, 128), dtype=float)
    u[24, 65] = 1.
    res = evaluate_captured_sky_bias_conditional_pair(basis, [u, u], [u, u])
    rng = np.random.default_rng(20261112)
    n = 1000
    # Independent toy VREST noise is an explicit synthetic assumption,
    # not evidence of real FlatField independence.
    common = rng.standard_normal((n, u.size))
    shared_coeff = sum(
        (s*op.adjoint(u.ravel()) for s, op in zip(
            basis.signed_bias_sensitivities, basis.operators,
        )), np.zeros(u.size),
    )
    measured = (common * np.sqrt(basis.masterbias_variances["SAME"])[None, :]) @ shared_coeff
    for op, vr in zip(basis.operators, basis.vrest_if_independent):
        ind = rng.standard_normal((n, u.size))
        measured += (ind * np.sqrt(vr)[None, :]) @ op.adjoint(u.ravel())
    actual = np.var(measured, ddof=1)
    expected = res["conditional_sum_under_unverified_independence"]
    # 1000 Gaussian draws: five standard errors ~ 22% on variance
    assert actual == pytest.approx(expected, rel=.24)


def test_rehashed_source_var_change_without_fit_update_must_be_rejected(tmp_path):
    source, manifest, _ = _one(tmp_path)
    with fits.open(source, memmap=False) as hdul:
        # Preserve physical VBIAS+VREST closure, but invalidate the
        # actual Kelson fitted source variance at all selected pixels.
        hdul["VAR"].data += 1.
        hdul["VREST"].data += 1.
        hdul.writeto(source, overwrite=True)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    from scorpio_pipe.covariance_operator_capture import sha256_file
    payload["records"]["selected_sky_input"]["sha256"] = sha256_file(source)
    payload["records"]["selected_sky_input"]["size_bytes"] = source.stat().st_size
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="final fit variance differs"):
        load_captured_sky_bias_conditional_basis([manifest])


def test_missing_components_or_wrong_budget_fail_closed(tmp_path):
    _, manifest, _ = _one(tmp_path)
    with pytest.raises(ValueError, match="resource budgets"):
        load_captured_sky_bias_conditional_basis([manifest], max_pixels_per_exposure=0)
    with pytest.raises(ValueError, match="pixel limit"):
        load_captured_sky_bias_conditional_basis([manifest], max_pixels_per_exposure=100)
    with pytest.raises(ValueError, match="resource budgets"):
        load_captured_sky_bias_conditional_basis([manifest], max_exposures=0)


def test_no_stage_functionals_with_wrong_shape_or_nonfinite_are_accepted(tmp_path):
    _, manifest, _ = _one(tmp_path)
    b = load_captured_sky_bias_conditional_basis([manifest])
    zeros = np.zeros((48, 128))
    with pytest.raises(ValueError, match="functional geometry"):
        evaluate_captured_sky_bias_conditional_pair(
            b, [np.zeros((128, 48))], [zeros]
        )
    zeros[4, 4] = np.nan
    with pytest.raises(ValueError, match="functional geometry"):
        evaluate_captured_sky_bias_conditional_pair(b, [zeros], [zeros])
