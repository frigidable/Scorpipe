"""Independent P1.4c diagnostic line-flux/continuum covariance tests."""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.science.line_covariance import measure_line_flux_linear_continuum
from scorpio_pipe.science.lines import LineSpec, measure_line
from scorpio_pipe.covariance_linearized import covariance_of_linearized_functionals
from scorpio_pipe.stages.linearize import _build_linearize_row_plan


def _demo():
    wave = np.arange(80., 121., 1.)
    spec = LineSpec("TEST", 100., 4., (-12., -7.), (7., 12.))
    continuum = 3.0 + 0.035 * (wave - 100.)
    line = np.zeros(wave.size)
    line[(wave >= 99.) & (wave <= 101.)] = [2., 3., 2.]
    return wave, continuum + line, np.full(wave.size, 0.04), spec


def test_linear_continuum_exact_flux_and_sideband_noise():
    wave, f, var, spec = _demo()
    d = measure_line_flux_linear_continuum(wave, f, var, spec=spec)
    assert d.flux == pytest.approx(7., abs=1e-11)
    assert d.continuum_at_center == pytest.approx(3., abs=1e-12)
    assert d.covariance_status == "DIAG_ONLY"
    assert d.error_conditional is None
    assert d.wavelength_coverage == pytest.approx(1.)
    assert d.sideband_pixel_count == 12

    # An independent WLS normal-equation oracle for the *total* estimator.
    sb = ((wave >= 88) & (wave <= 93)) | ((wave >= 107) & (wave <= 112))
    X = np.c_[np.ones(sb.sum()), wave[sb] - 100]
    W = np.diag(1. / var[sb])
    H = np.linalg.solve(X.T @ W @ X, X.T @ W)
    integral = np.clip(
        np.minimum(wave + .5, 104.) - np.maximum(wave - .5, 96.), 0., None
    )
    u = integral.copy()
    u[sb] -= np.array([integral.sum(), integral @ (wave - 100)]) @ H
    np.testing.assert_allclose(u, d.functional_weights, atol=1e-12)
    assert d.error_diag**2 == pytest.approx(float(u @ np.diag(var) @ u))
    assert d.error_diag**2 > float(np.sum(integral**2 * var))


def test_constant_continuum_and_outside_nan_are_safe():
    wave, f, var, spec = _demo()
    f[0] = np.nan
    var[1] = np.nan
    d = measure_line_flux_linear_continuum(wave, f, var, spec=spec, continuum_order=0)
    assert np.isfinite(d.flux)
    assert d.continuum_order == 0
    valid = np.ones(wave.size, dtype=bool)
    valid[np.where(wave == 100.)[0][0]] = False
    with pytest.raises(ValueError, match="line window"):
        measure_line_flux_linear_continuum(wave, f, var, spec=spec, valid_mask=valid)


def test_one_invalid_sideband_fails_closed():
    wave, f, var, spec = _demo()
    valid = np.ones(wave.size, dtype=bool)
    valid[(wave >= 88.) & (wave <= 93.)] = False
    with pytest.raises(ValueError, match="sideband"):
        measure_line_flux_linear_continuum(wave, f, var, spec=spec, valid_mask=valid)


def test_boundaries_and_nonmonotonic_grid_rejected():
    wave, f, var, spec = _demo()
    with pytest.raises(ValueError, match="strictly increasing"):
        measure_line_flux_linear_continuum(wave[::-1], f, var, spec=spec)
    other = LineSpec("edge", 81, 4, (-12., -7.), (7., 12.))
    with pytest.raises(ValueError, match="beyond"):
        measure_line_flux_linear_continuum(wave, f, var, spec=other)
    bad = LineSpec("overlap", 100, 4, (-5., -2.), (7., 12.))
    with pytest.raises(ValueError, match="sidebands"):
        measure_line_flux_linear_continuum(wave, f, var, spec=bad)
    with pytest.raises(ValueError, match="continuum_order"):
        measure_line_flux_linear_continuum(wave, f, var, spec=spec, continuum_order=2)
    with pytest.raises(ValueError, match="Boolean"):
        measure_line_flux_linear_continuum(
            wave, f, var, spec=spec,
            valid_mask=np.ones(wave.size, dtype=np.uint16),
        )


def test_conditional_rectification_covariance_and_monte_carlo():
    # 16 independent 2-A CCD pixels are mapped onto 32 1-A output bins.
    centers = np.arange(1., 32., 2.)
    wave = np.arange(.5, 32., 1.)
    edges_in = np.arange(0., 33., 2.)
    edges_out = np.arange(0., 33., 1.)
    plan = _build_linearize_row_plan(centers, edges_out)
    assert plan is not None
    N = centers.size
    V = np.linspace(.5, 2.5, N)
    valid = np.ones(N, dtype=bool)
    A = np.maximum(
        0.,
        np.minimum(edges_out[1:, None], edges_in[None, 1:])
        - np.maximum(edges_out[:-1, None], edges_in[None, :-1]),
    ) / 2.
    C = (A * V[None, :]) @ A.T
    flux = 2.0 + 1.5 * np.exp(-.5 * ((wave - 16.) / 1.5)**2)
    vout = np.diag(C)
    spec = LineSpec("test", 16., 3., (-10., -6.), (6., 10.))
    callback = lambda u, v: covariance_of_linearized_functionals(
        plan, V, valid, u, v
    )
    d = measure_line_flux_linear_continuum(
        wave, flux, vout, spec=spec, covariance_operator=callback
    )
    u = d.functional_weights
    assert d.covariance_status == "CONDITIONAL_USER_OPERATOR"
    assert d.error_conditional**2 == pytest.approx(float(u @ C @ u), rel=1e-12)
    assert d.error_conditional > d.error_diag

    rng = np.random.default_rng(20261031)
    draws = rng.normal(size=(8000, N)) * np.sqrt(V)[None, :]
    noise_line = (draws @ A.T) @ u
    assert np.var(noise_line, ddof=1) == pytest.approx(
        d.error_conditional**2, rel=.045
    )


def test_invalid_covariance_callback_is_rejected():
    wave, f, var, spec = _demo()
    with pytest.raises(ValueError, match="Covariance operator"):
        measure_line_flux_linear_continuum(
            wave, f, var, spec=spec,
            covariance_operator=lambda _u, _v: float("nan"),
        )
    with pytest.raises(ValueError, match="Covariance operator"):
        measure_line_flux_linear_continuum(
            wave, f, var, spec=spec,
            covariance_operator=lambda _u, _v: -100.,
        )


def test_legacy_median_continuum_uncertainty_is_not_in_err_stat():
    # This discrepancy exists even before resampling correlations.
    wave = np.arange(65., 136., 1.)
    spec = LineSpec("demo", 100., 5., (-20., -11.), (11., 20.))
    var = np.full(wave.size, 0.04)
    rng = np.random.default_rng(4401)
    obs = []
    for _ in range(1500):
        f = 3.0 + rng.normal(0., np.sqrt(var))
        m = measure_line(
            wave, f, var, calsys_rel=None, calsys_tell=None,
            calsys_abs=None, spec=spec,
        )
        obs.append(m["flux"])
    assert np.std(obs, ddof=1) > 1.15 * m["err_stat"]


def test_partial_boundary_bins_are_integrated_without_gradient_bias():
    wave = np.arange(80., 121., 1.)
    spec = LineSpec("partial", 100.25, 3.25, (-12., -7.), (7., 12.))
    flux = 2. + 0.1 * (wave - spec.rest_wavelength)
    var = np.full(wave.size, .09)
    d = measure_line_flux_linear_continuum(wave, flux, var, spec=spec)
    assert d.flux == pytest.approx(0., abs=1e-12)
    assert d.wavelength_coverage == pytest.approx(1.)
    assert d.line_pixel_count >= 7
    assert d.error_diag > 0


def test_nonuniform_grid_with_fixed_linear_continuum():
    wave = 70. + np.cumsum(np.linspace(.75, 1.35, 65))
    center = wave[32]
    spec = LineSpec("nonuniform", float(center), 4., (-12., -7.), (7., 12.))
    flux = 5.0 + 0.03 * (wave - center)
    var = np.linspace(.01, .1, wave.size)
    d = measure_line_flux_linear_continuum(wave, flux, var, spec=spec)
    assert d.flux == pytest.approx(0., abs=1e-11)
    assert d.error_diag > 0
    assert d.wavelength_coverage == pytest.approx(1.)
