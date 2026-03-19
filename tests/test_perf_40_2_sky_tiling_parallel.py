from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("astropy")

from scorpio_pipe.stages.sky_sub import (  # noqa: E402
    _eval_linear_spline_map,
    _eval_model_map,
    _make_bspline_knot_vector,
)


def test_perf_40_2_bspline_tiled_parallel_matches_serial() -> None:
    rng = np.random.default_rng(123)
    ny, nx = 24, 48
    lam = np.repeat((5000.0 + 1.2 * np.arange(nx, dtype=float))[None, :], ny, axis=0)
    y_rows = np.linspace(-1.0, 1.0, ny, dtype=float)

    t = _make_bspline_knot_vector(float(np.nanmin(lam)), float(np.nanmax(lam)), knot_step_A=3.0, degree=3)
    n_basis = int(len(t) - 3 - 1)
    c0 = rng.normal(0.0, 1.0, size=n_basis)
    c1 = rng.normal(0.0, 0.2, size=n_basis)

    serial = _eval_model_map(
        lam,
        t=t,
        degree=3,
        c0=c0,
        c1=c1,
        y_norm_rows=y_rows,
        delta_A=0.15,
        strip_rows=6,
        n_workers=1,
        parallel_min_pixels=1,
    )
    parallel = _eval_model_map(
        lam,
        t=t,
        degree=3,
        c0=c0,
        c1=c1,
        y_norm_rows=y_rows,
        delta_A=0.15,
        strip_rows=6,
        n_workers=2,
        parallel_min_pixels=1,
    )

    assert np.allclose(serial, parallel, rtol=0.0, atol=1e-6)


def test_perf_40_2_linear_tiled_parallel_matches_serial() -> None:
    ny, nx = 20, 40
    lam = np.repeat((5100.0 + 0.8 * np.arange(nx, dtype=float))[None, :], ny, axis=0)
    a_nodes = np.linspace(0.7, 1.3, 32, dtype=float)

    serial = _eval_linear_spline_map(
        lam,
        lam0=5100.0,
        step_A=1.0,
        a_nodes=a_nodes,
        delta_A=0.1,
        strip_rows=5,
        n_workers=1,
        parallel_min_pixels=1,
    )
    parallel = _eval_linear_spline_map(
        lam,
        lam0=5100.0,
        step_A=1.0,
        a_nodes=a_nodes,
        delta_A=0.1,
        strip_rows=5,
        n_workers=2,
        parallel_min_pixels=1,
    )

    assert np.allclose(serial, parallel, rtol=0.0, atol=1e-7)
