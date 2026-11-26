"""Synthetic full Kelson re-clipping versus the fixed-mask sky covariance oracle."""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.calib.sky_clipping_sensitivity import (
    audit_synthetic_kelson_clipping,
)


def _synthetic_sky():
    rng = np.random.default_rng(154)
    ny, nx = 10, 48
    x = np.arange(nx)[None, :]
    y = np.arange(ny)[:, None]
    wave = 5000. + 1.3 * x + .024 * y
    sky = (
        20. + .025 * x
        + 8. * np.exp(-.5 * ((x - 19.) / 2.) ** 2)
        + rng.normal(0., .5, size=(ny, nx))
    )
    var = np.ones_like(sky)
    mask = np.zeros((ny, nx), dtype=bool)
    mask[[0, 1, 2, 3, 6, 7, 8, 9], :] = True
    return sky, var, wave, mask


def _call(sky, var, wave, mask, **kwargs):
    settings = dict(
        knot_step_A=8., degree=3, y_order=0,
        min_pixels=64, clip_sigma=4.5, max_iter=4,
        seed=20261104, n_probes=2, perturbation_sigma=.15,
        residual_tolerance=.05, model_tolerance=.05,
    )
    settings.update(kwargs)
    return audit_synthetic_kelson_clipping(
        sky, var, wave, mask, **settings,
    )


def test_smooth_synthetic_noise_preserves_selection_and_derivative():
    sky, var, wave, mask = _synthetic_sky()
    original = [x.copy() for x in (sky, var, wave, mask)]
    result = _call(sky, var, wave, mask)
    assert result["status"] == "FROZEN_STABLE", result
    assert result["n_candidates"] == 8 * 48
    assert result["n_baseline_rejected"] == 0
    assert result["n_probes"] == 2
    assert result["scope"] == "SYNTHETIC_FIXED_GEOMETRY_REFIT_CLIPPING"
    assert not result["science_uncertainty_qualified"]
    assert result["baseline_lsqr_stop"] in (1, 2)
    for rec in result["probes"]:
        assert rec["status"] == "FROZEN_STABLE"
        assert rec["plus_selection_changes"] == 0
        assert rec["minus_selection_changes"] == 0
        assert rec["model_derivative_relative_error"] < .05
        assert rec["residual_derivative_relative_error"] < .05
        assert set(rec["lsqr_stop_codes"]).issubset({1, 2})
    for a, b in zip((sky, var, wave, mask), original):
        np.testing.assert_array_equal(a, b)


def test_outlier_crosses_clipping_boundary_and_frozen_operator_rejected():
    sky, var, wave, mask = _synthetic_sky()
    sky[3, 34] += 15.  # A single synthetic cosmic contamination pixel.
    # Positive perturbation removes the injected excess; negative doubles
    # it. The physical data-dependent clipped pixel set must change.
    v = np.zeros((1, *sky.shape), dtype=float)
    v[0, 3, 34] = -1.
    result = _call(
        sky, var, wave, mask, n_probes=1, perturbation_sigma=15.,
        probe_vectors=v,
    )
    assert result["n_baseline_rejected"] >= 1
    assert result["status"] == "SELECTION_CHANGED", result
    assert result["probes"][0]["status"] == "SELECTION_CHANGED"
    assert (result["probes"][0]["plus_selection_changes"]
            + result["probes"][0]["minus_selection_changes"]) > 0
    assert result["science_uncertainty_qualified"] is False


def test_tight_zero_error_tolerances_fail_even_with_fixed_sky_selection():
    sky, var, wave, mask = _synthetic_sky()
    result = _call(
        sky, var, wave, mask, n_probes=1,
        residual_tolerance=0., model_tolerance=0.,
    )
    assert result["n_baseline_rejected"] == 0
    assert result["status"] == "NUMERIC_UNSTABLE", result
    assert result["probes"][0]["plus_selection_changes"] == 0
    assert result["probes"][0]["minus_selection_changes"] == 0


def test_clipping_sensitivity_same_seed_exactly_reproducible():
    args = _synthetic_sky()
    a = _call(*args, seed=123)
    b = _call(*args, seed=123)
    assert a == b


@pytest.mark.parametrize("damage", [
    "wrong_shape", "nonfinite_wavelength", "negative_variance",
    "float_mask", "empty_mask", "bad_probe_shape", "zero_probe",
    "invalid_sigma", "invalid_pixel_budget",
])
def test_invalid_synthetic_inputs_rejected_before_publication(damage):
    sky, var, wave, mask = _synthetic_sky()
    args = {"n_probes": 1}
    if damage == "wrong_shape":
        wave = wave[:-1, :]
    elif damage == "nonfinite_wavelength":
        wave[0, 0] = np.nan
    elif damage == "negative_variance":
        var[0, 0] = -1.
    elif damage == "float_mask":
        mask = mask.astype(float)
    elif damage == "empty_mask":
        mask[:] = False
    elif damage == "bad_probe_shape":
        args["probe_vectors"] = np.ones((1, 2, 2))
    elif damage == "zero_probe":
        args["probe_vectors"] = np.zeros((1, *sky.shape))
    elif damage == "invalid_sigma":
        args["perturbation_sigma"] = 0.
    elif damage == "invalid_pixel_budget":
        args["max_pixels"] = 10
    with pytest.raises(ValueError):
        _call(sky, var, wave, mask, **args)
