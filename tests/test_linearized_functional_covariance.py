"""Independent numerical oracles for on-demand COUNT_V1 covariance (P1.4b)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scorpio_pipe.covariance_linearized import (
    covariance_of_linearized_functionals as measure_cov,
)
from scorpio_pipe.stages.linearize import _build_linearize_row_plan


def _example():
    edges = np.arange(0.0, 26.0, 2.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = np.arange(0.0, 25.0)
    plan = _build_linearize_row_plan(centers, out)
    assert plan is not None
    variance = np.full(12, 16.0)
    valid = np.ones(12, dtype=bool)
    return plan, variance, valid


def _dense_oracle(edges_in, edges_out, var, left, right, valid, scale=1.0):
    lo = np.maximum(edges_out[:-1, None], edges_in[None, :-1])
    hi = np.minimum(edges_out[1:, None], edges_in[None, 1:])
    A = np.maximum(hi - lo, 0.0) / np.diff(edges_in)[None, :]
    A[:, ~valid] = 0.0
    A = np.asarray(scale) * A if np.ndim(scale) == 0 else scale[:, None] * A
    C = (A * var[None, :]) @ A.T
    return float(left @ C @ right)


def test_splitting_single_noisy_pixel_requires_offdiagonal_covariance():
    plan, var, valid = _example()
    u0 = np.zeros(24); u0[0] = 1.0
    u1 = np.zeros(24); u1[1] = 1.0
    assert measure_cov(plan, var, valid, u0) == pytest.approx(4.0)
    assert measure_cov(plan, var, valid, u1) == pytest.approx(4.0)
    assert measure_cov(plan, var, valid, u0, u1) == pytest.approx(4.0)
    assert measure_cov(plan, var, valid, u0 + u1) == pytest.approx(16.0)
    assert measure_cov(plan, var, valid, u0 - u1) == pytest.approx(0.0)


def test_full_sum_recovers_original_uncertainty_and_excludes_masked_pixels():
    plan, var, valid = _example()
    weights = np.ones(24)
    assert measure_cov(plan, var, valid, weights) == pytest.approx(12 * 16.0)
    valid[3] = False
    assert measure_cov(plan, var, valid, weights) == pytest.approx(11 * 16.0)
    head = np.zeros(24); head[:8] = 1.0
    assert measure_cov(plan, var, valid, head) == pytest.approx(3 * 16.0)


@pytest.mark.parametrize("seed", [20261031, 111222])
def test_nonuniform_reversed_dispersion_independent_dense_oracle(seed):
    rng = np.random.default_rng(seed)
    edges = np.r_[0.0, np.cumsum(np.linspace(0.7, 2.2, 12))]
    new = np.linspace(0.0, edges[-1], 27)
    centers = (edges[:-1] + edges[1:]) / 2
    var = rng.uniform(0.1, 20.0, size=12)
    valid = rng.random(12) > 0.25
    l = rng.normal(size=new.size - 1)
    r = rng.normal(size=new.size - 1)
    scale = rng.uniform(0.25, 3.0, size=l.size)
    # Production COUNT_V1 infers pixel borders from center midpoints, rather
    # than trusting these original nonuniform edges. The reference operator
    # independently reconstructs those inferred physical intervals.
    inferred = np.r_[
        centers[0] - 0.5 * (centers[1] - centers[0]),
        0.5 * (centers[:-1] + centers[1:]),
        centers[-1] + 0.5 * (centers[-1] - centers[-2]),
    ]
    expected = _dense_oracle(inferred, new, var, l, r, valid, scale)
    plan = _build_linearize_row_plan(centers, new)
    assert plan is not None
    assert measure_cov(plan, var, valid, l, r, output_scale=scale) == pytest.approx(
        expected, rel=1e-11, abs=1e-10
    )
    # Wavelength decreases in detector-storage order; physics is invariant.
    flipped = _build_linearize_row_plan(centers[::-1], new)
    assert flipped is not None
    assert measure_cov(
        flipped, var[::-1], valid[::-1], l, r, output_scale=scale
    ) == pytest.approx(expected, rel=1e-11, abs=1e-10)


def test_scaling_density_and_exposure_time_is_quadratic():
    plan, var, valid = _example()
    u = np.zeros(24); u[:2] = 1.0
    base = measure_cov(plan, var, valid, u)
    # For a 1-A output bin, a 10-s exposure gives rate measurements.
    assert measure_cov(plan, var, valid, u, output_scale=0.1) == pytest.approx(base / 100)
    # Any deterministic wavelength-bin scaling belongs in the same operator.
    scale = np.ones(24); scale[:2] = 0.5
    assert measure_cov(plan, var, valid, u, output_scale=scale) == pytest.approx(base / 4)
    with pytest.raises(ValueError, match="Output scale"):
        measure_cov(plan, var, valid, u, output_scale=0.0)


def test_masked_nonfinite_var_is_allowed_but_valid_nonfinite_var_fails():
    plan, var, valid = _example()
    u = np.ones(24)
    var[0] = np.nan
    with pytest.raises(ValueError, match="Valid CCD"):
        measure_cov(plan, var, valid, u)
    valid[0] = False
    assert measure_cov(plan, var, valid, u) == pytest.approx(11 * 16.0)
    var[1] = -1
    with pytest.raises(ValueError, match="Valid CCD"):
        measure_cov(plan, var, valid, u)


def test_wrong_shapes_or_integer_dq_cannot_implicitly_define_validity():
    plan, var, valid = _example()
    u = np.ones(24)
    with pytest.raises(ValueError, match="Boolean valid_mask"):
        measure_cov(plan, var, np.ones(12, dtype=np.uint16), u)
    with pytest.raises(ValueError, match="Output functional"):
        measure_cov(plan, var, valid, np.ones(23))
    with pytest.raises(ValueError, match="Invalid COUNT_V1"):
        broken = replace(plan, starts=np.ones_like(plan.starts))
        measure_cov(broken, var, valid, u)
    with pytest.raises(ValueError, match="Invalid COUNT_V1"):
        broken = replace(plan, weights=np.full_like(plan.weights, -0.1))
        measure_cov(broken, var, valid, u)


def test_seeded_monte_carlo_recovers_integrated_line_variance():
    plan, var, valid = _example()
    u = np.zeros(24); u[0:8] = 1.0
    var_exact = measure_cov(plan, var, valid, u)
    rng = np.random.default_rng(20261031)
    draws = rng.normal(size=(18000, 12)) * np.sqrt(var)
    # Completely independent manual oracle: first eight output bins cover
    # the first four detector pixels, so their sum equals the four inputs.
    measured = np.sum(draws[:, :4], axis=1)
    assert np.var(measured, ddof=1) == pytest.approx(var_exact, rel=0.04)


def test_long_wavelength_grid_uses_no_dense_covariance_matrix():
    # 3600 A / 0.04 A gives 90k output bins, as in long SCORPIO previews.
    centers = np.linspace(3650.0, 7250.0, 4096)
    edges_out = np.linspace(3650.0, 7250.0, 90001)
    plan = _build_linearize_row_plan(centers, edges_out)
    assert plan is not None
    var = np.ones(centers.size)
    valid = np.ones(centers.size, dtype=bool)
    line_u = np.zeros(90000)
    line_u[40000:40200] = 1.0
    value = measure_cov(plan, var, valid, line_u)
    assert np.isfinite(value) and value > 0.0
    # A full 90k x 90k float64 covariance would exceed 60 GiB.
    assert plan.weights.nbytes < 15_000_000
