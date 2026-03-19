import numpy as np


def _import_helpers():
    import pytest

    pytest.importorskip("astropy")
    from scorpio_pipe.stages.linearize import (
        _apply_linearize_row_plan,
        _apply_linearize_row_plan_var,
        _build_linearize_row_plan,
        _get_linearize_grid_plan,
        _rebin_row_cumulative,
        _rebin_row_var_weightsquared,
        clear_linearize_grid_cache,
    )

    return {
        "apply": _apply_linearize_row_plan,
        "apply_var": _apply_linearize_row_plan_var,
        "build": _build_linearize_row_plan,
        "get": _get_linearize_grid_plan,
        "rebin": _rebin_row_cumulative,
        "rebin_var": _rebin_row_var_weightsquared,
        "clear": clear_linearize_grid_cache,
    }


def test_linearize_row_plan_matches_reference_rebin():
    h = _import_helpers()

    nx = 128
    x = np.arange(nx, dtype=float)
    lam_row = 5000.0 + 1.7 * x + 0.001 * (x - 50.0) ** 2
    edges_out = np.linspace(lam_row[8], lam_row[-9], 96)

    values = np.sin(x / 11.0) + 10.0
    var = np.full(nx, 2.5, dtype=float)
    valid = np.ones(nx, dtype=bool)
    valid[17:21] = False
    valid[70] = False

    row_plan = h["build"](lam_row, edges_out)
    assert row_plan is not None

    out_ref, cov_ref = h["rebin"](values, lam_row, edges_out, valid_mask=valid)
    var_ref = h["rebin_var"](var, lam_row, edges_out, valid_mask=valid)

    out_new, cov_new = h["apply"](values, row_plan, valid_mask=valid)
    var_new = h["apply_var"](var, row_plan, valid_mask=valid)

    assert np.allclose(out_new, out_ref, atol=1e-12)
    assert np.allclose(cov_new, cov_ref, atol=1e-12)
    assert np.allclose(var_new, var_ref, atol=1e-12)


def test_linearize_grid_plan_cache_hits_on_same_signature():
    h = _import_helpers()
    h["clear"]()

    ny, nx = 6, 64
    x = np.arange(nx, dtype=float)[None, :]
    y = np.arange(ny, dtype=float)[:, None]
    lam_map = 5000.0 + 1.25 * x + 0.02 * y + 0.0005 * (x - 20.0) ** 2
    edges_out = np.linspace(float(np.nanmax(np.nanmin(lam_map, axis=1))), float(np.nanmin(np.nanmax(lam_map, axis=1))), 80)

    plan1, hit1 = h["get"](
        lam_map,
        edges_out=edges_out,
        frame_signature_key="shape=6x64, bin=1x1, roi=full, readout=fast",
        lambda_map_key="sha256:abc",
        wave0=float(edges_out[0]),
        dw=float(edges_out[1] - edges_out[0]),
        nlam=int(edges_out.size - 1),
    )
    plan2, hit2 = h["get"](
        lam_map,
        edges_out=edges_out,
        frame_signature_key="shape=6x64, bin=1x1, roi=full, readout=fast",
        lambda_map_key="sha256:abc",
        wave0=float(edges_out[0]),
        dw=float(edges_out[1] - edges_out[0]),
        nlam=int(edges_out.size - 1),
    )

    assert hit1 is False
    assert hit2 is True
    assert plan1 is plan2
