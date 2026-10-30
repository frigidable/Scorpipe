"""Independent analytic oracles for Issue #32: integrated detector-pixel counts.

This suite separates physical requirements from production characterization.
Strict xfails document *known* discrepancies in the current linearize code:
their xfail status is NOT evidence that the pipeline is scientifically correct.
No observation files or external numerical references are needed.
"""

from __future__ import annotations

import numpy as np
import pytest


def _overlap_operator(edges_in: np.ndarray, edges_out: np.ndarray) -> np.ndarray:
    """Output integrated counts per input integrated count; independent oracle.

    Assumption: within each input pixel, signal density is constant over lambda.
    A[k,i] = overlap(input_i, output_k) / width(input_i).
    """
    left = np.maximum(edges_out[:-1, None], edges_in[None, :-1])
    right = np.minimum(edges_out[1:, None], edges_in[None, 1:])
    overlap = np.maximum(right - left, 0.0)
    return overlap / np.diff(edges_in)[None, :]


def _oracle(
    counts: np.ndarray,
    variance: np.ndarray,
    edges_in: np.ndarray,
    edges_out: np.ndarray,
    *,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Independent flux, diagonal variance, covariance, and valid coverage."""
    A = _overlap_operator(edges_in, edges_out)
    if valid is not None:
        A = A * np.asarray(valid, dtype=float)[None, :]
    counts = np.asarray(counts, dtype=float)
    variance = np.asarray(variance, dtype=float)
    flux = A @ counts
    C = (A * variance[None, :]) @ A.T
    overlap = _overlap_operator(edges_in, edges_out) * np.diff(edges_in)[None, :]
    if valid is not None:
        overlap = overlap * np.asarray(valid, dtype=float)[None, :]
    covfrac = np.sum(overlap, axis=1) / np.diff(edges_out)
    return flux, np.diag(C), C, covfrac


def _uniform_case():
    edges_in = np.arange(0.0, 26.0, 2.0)  # 12 detector pixels, 2 A each
    edges_out = np.arange(0.0, 25.0, 1.0)  # 24 output bins, 1 A each
    counts = np.full(12, 20.0)  # e- per detector pixel
    variance = np.full(12, 16.0)  # (e-)^2 per detector pixel
    centers = 0.5 * (edges_in[:-1] + edges_in[1:])
    return centers, edges_in, edges_out, counts, variance


def test_oracle_conserves_integrated_counts_and_units() -> None:
    _centers, edges_in, edges_out, counts, var = _uniform_case()
    flux, out_var, C, cov = _oracle(counts, var, edges_in, edges_out)
    np.testing.assert_allclose(flux, 10.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out_var, 4.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(cov, 1.0, rtol=0, atol=1e-12)
    assert np.isclose(flux.sum(), counts.sum())
    # Two adjacent output bins from one input pixel share the same noise.
    assert C[0, 1] == pytest.approx(4.0)
    # Summing two bins requires their covariance, not just their diagonal VAR.
    assert C[:2, :2].sum() == pytest.approx(16.0)
    assert out_var[:2].sum() == pytest.approx(8.0)


def test_oracle_nonuniform_dispersion_and_reversed_storage() -> None:
    edges_in = np.array([0., 1., 3., 6., 10., 15., 21., 28., 36., 45., 55., 66., 78.])
    edges_out = np.arange(0.0, 79.0)
    widths = np.diff(edges_in)
    counts = 3.0 * widths  # exactly constant source density, 3 e-/A
    variance = np.full(widths.shape, 9.0)
    flux, _v, _C, cov = _oracle(counts, variance, edges_in, edges_out)
    np.testing.assert_allclose(flux, 3.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(cov, 1.0, rtol=0, atol=1e-12)
    assert flux.sum() == pytest.approx(counts.sum())
    # Explicitly permuting the physical detector-pixel columns leaves the
    # output invariant when the corresponding input counts are reordered.
    matrix = _overlap_operator(edges_in, edges_out)
    np.testing.assert_allclose(
        matrix[:, ::-1] @ counts[::-1], flux, rtol=0, atol=1e-12
    )


def test_oracle_masked_input_has_partial_coverage_without_renormalization() -> None:
    _centers, edges_in, edges_out, counts, var = _uniform_case()
    valid = np.ones(12, dtype=bool)
    valid[5] = False
    flux, out_var, _C, coverage = _oracle(
        counts, var, edges_in, edges_out, valid=valid
    )
    np.testing.assert_allclose(flux[10:12], 0.0)
    np.testing.assert_allclose(out_var[10:12], 0.0)
    np.testing.assert_allclose(coverage[10:12], 0.0)
    assert flux.sum() == pytest.approx(counts[valid].sum())
    assert coverage[0] == pytest.approx(1.0)


def test_oracle_monte_carlo_recovers_diagonal_and_adjacent_covariance() -> None:
    _centers, edges_in, edges_out, counts, var = _uniform_case()
    A = _overlap_operator(edges_in, edges_out)
    rng = np.random.default_rng(20261031)
    draws = rng.normal(size=(15000, counts.size)) * np.sqrt(var)[None, :]
    output_noise = draws @ A.T
    empirical = np.cov(output_noise[:, :2], rowvar=False, ddof=1)
    theoretical = (A * var[None, :]) @ A.T
    np.testing.assert_allclose(
        empirical, theoretical[:2, :2], rtol=0.05, atol=0.1
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Issue #32: production operator treats integrated e-/pixel as density",
)
def test_production_integrated_counts_conserve_flux() -> None:
    from scorpio_pipe.stages.linearize import (
        _apply_linearize_row_plan, _build_linearize_row_plan,
    )

    centers, _edges_in, edges_out, counts, _var = _uniform_case()
    plan = _build_linearize_row_plan(centers, edges_out)
    assert plan is not None
    density, cov = _apply_linearize_row_plan(counts, plan)
    # Current default integrated-bin path scales the returned density by dw.
    actual_integrated = density * np.diff(edges_out)
    np.testing.assert_allclose(actual_integrated, 10.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(cov, 1.0, rtol=0, atol=1e-12)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Issue #32: production VAR uses density normalization for integrated e-2/pixel",
)
def test_production_integrated_counts_propagate_variance() -> None:
    from scorpio_pipe.stages.linearize import (
        _apply_linearize_row_plan_var, _build_linearize_row_plan,
    )

    centers, _edges_in, edges_out, _counts, var = _uniform_case()
    plan = _build_linearize_row_plan(centers, edges_out)
    assert plan is not None
    density_variance = _apply_linearize_row_plan_var(var, plan)
    actual_integrated_var = density_variance * np.diff(edges_out) ** 2
    np.testing.assert_allclose(actual_integrated_var, 4.0, rtol=0, atol=1e-12)
