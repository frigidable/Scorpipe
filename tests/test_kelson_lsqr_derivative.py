"""Real SkySub LSQR finite-difference and frozen WLS operator fidelity."""
from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.calib.sky_lsqr_derivative import (
    assess_kelson_lsqr_directional_derivatives,
)
from tests.test_kelson_frozen_capture import _science_night


def _actual_fit_paths(tmp_path):
    stage, residual_path = _science_night(tmp_path / "science", capture=True)
    model_path = stage / "obj_0001_skymodel_raw.fits"
    with fits.open(residual_path) as hdul:
        sidecar_path = stage / hdul[0].header["SKYFRZP"]
    assert sidecar_path.is_file() and model_path.is_file()
    return sidecar_path, model_path, residual_path


def test_actual_science_sky_fit_lsqr_directional_derivative_diagnostic(tmp_path):
    sidecar, model, residual = _actual_fit_paths(tmp_path)
    files = [sidecar, model, residual]
    original = [p.read_bytes() for p in files]
    report = assess_kelson_lsqr_directional_derivatives(
        sidecar, model, residual, seed=20261104, n_probes=3,
        epsilon_sigma=.15, relative_tolerance=.05,
        replay_tolerance=.01,
    )
    assert report["scope"] == "CONDITIONAL_FINAL_SKY_MASK_ONLY"
    assert report["fidelity_audit"] == "MATCH"
    assert report["n_probes"] == 3
    assert len(report["probes"]) == 3
    assert all(np.isfinite(q["relative_l2_difference"]) for q in report["probes"])
    assert all(np.isfinite(q["model_response_relative_l2_difference"])
               for q in report["probes"])
    assert report["status"] == "MATCH", report
    assert report["max_model_response_relative_l2_difference"] <= .05
    assert report["replay_model_relative_difference"] <= .01
    assert report["all_lsqr_stops_nominal"]
    assert report["science_uncertainty_qualified"] is False
    assert [p.read_bytes() for p in files] == original


def test_unacceptable_recorded_lsqr_stop_fails_numerical_qualification(tmp_path):
    sidecar, model, residual = _actual_fit_paths(tmp_path)
    with np.load(sidecar, allow_pickle=False) as arr:
        archive = {key: np.asarray(arr[key]).copy() for key in arr.files}
    archive["lsqr_stop"] = np.array(7, dtype=np.int16)
    np.savez_compressed(sidecar, **archive)
    report = assess_kelson_lsqr_directional_derivatives(
        sidecar, model, residual, n_probes=1,
    )
    assert report["status"] == "UNSTABLE"
    assert not report["all_lsqr_stops_nominal"]
    assert not report["science_uncertainty_qualified"]


def test_wrong_fits_model_aborts_prior_to_derivative_simulation(tmp_path):
    sidecar, model, residual = _actual_fit_paths(tmp_path)
    with fits.open(model, mode="update", memmap=False) as h:
        h["SCI"].data[0, 0] += 10
    with pytest.raises(ValueError, match="do not reproduce"):
        assess_kelson_lsqr_directional_derivatives(
            sidecar, model, residual,
        )


@pytest.mark.parametrize("params", [
    {"n_probes": 0},
    {"n_probes": 21},
    {"epsilon_sigma": 0.0},
    {"epsilon_sigma": float("nan")},
    {"relative_tolerance": -.1},
    {"replay_tolerance": -1.},
])
def test_invalid_directional_probe_controls_fail_before_io(tmp_path, params):
    with pytest.raises(ValueError):
        assess_kelson_lsqr_directional_derivatives(
            tmp_path / "missing.npz",
            tmp_path / "missing_model.fits",
            tmp_path / "missing_residual.fits",
            **params,
        )


def test_fixed_seed_is_reproducible_for_actual_sky_frozen_fit(tmp_path):
    sidecar, model, residual = _actual_fit_paths(tmp_path)
    left = assess_kelson_lsqr_directional_derivatives(
        sidecar, model, residual, n_probes=2, seed=123, relative_tolerance=.2,
    )
    right = assess_kelson_lsqr_directional_derivatives(
        sidecar, model, residual, n_probes=2, seed=123, relative_tolerance=.2,
    )
    np.testing.assert_allclose(
        [p["model_response_relative_l2_difference"] for p in left["probes"]],
        [p["model_response_relative_l2_difference"] for p in right["probes"]],
        rtol=0., atol=0.,
    )
    assert left["status"] == right["status"]


def test_early_stopped_lsqr_jacobian_differs_from_converged_wls():
    # This is an independent algebraic counterexample, not a replayed
    # astronomy observation. It proves why an LSQR iteration cap cannot
    # be silently replaced by the converged fixed-weight WLS derivative.
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import lsqr

    X = np.asarray([
        [1., 0., 1.],
        [1., 1., 0.],
        [0., 1., 2.],
        [1., -1., 1.],
        [0., 1., 1.],
        [2., 0., 1.],
    ])
    A = csr_matrix(X)
    base = np.array([3., 2., 8., 1., 4., -2.])
    direction = np.array([1., -1., .5, 2., 0., -.4])
    eps = .1

    def derivative(itlim):
        cp = lsqr(A, base + eps*direction, atol=1e-12,
                  btol=1e-12, iter_lim=itlim)[0]
        cm = lsqr(A, base - eps*direction, atol=1e-12,
                  btol=1e-12, iter_lim=itlim)[0]
        return (cp - cm) / (2.*eps)

    exact = np.linalg.lstsq(X, direction, rcond=None)[0]
    short_error = np.linalg.norm(X @ (derivative(1) - exact))
    converged_error = np.linalg.norm(X @ (derivative(20) - exact))
    assert short_error > .2
    assert converged_error < 1e-10
