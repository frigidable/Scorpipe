"""Actual Kelson final-fit snapshot: strict archive and stage-level evidence."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.calib.sky_fit_capture import (
    capture_kelson_fit, load_frozen_kelson_fit,
)
from scorpio_pipe.stages.sky_sub import _fit_kelson_bspline, run_sky_sub
from scorpio_pipe.work_layout import ensure_work_layout
from scorpio_pipe.workspace_paths import stage_dir


def _direct_fit_example():
    rng = np.random.default_rng(1501)
    ny, nx = 8, 32
    x = np.arange(nx)[None, :]
    y = np.arange(ny)[:, None]
    lam = np.broadcast_to(5000. + 1.5*x + .03*y, (ny, nx)).copy()
    y_rows = np.linspace(-1., 1., ny)
    accepted_rows = np.array([0, 1, 2, 5, 6, 7], dtype=int)
    idx = np.concatenate([j*nx+np.arange(nx) for j in accepted_rows])
    values = (22. + 7.*np.sin((lam - 5000.) / 7.) +
              .5*y_rows[:, None] +
              rng.normal(0., .5, size=(ny, nx)))
    variance = np.full(lam.shape, 3.0)
    vals = values.ravel()[idx]
    lams = lam.ravel()[idx]
    yy = np.repeat(y_rows, nx)[idx]
    vv = variance.ravel()[idx]
    fit = _fit_kelson_bspline(
        lams, yy, vals, vv,
        knot_step_A=8., y_order=1, clip_sigma=4.5,
        max_iter=3, min_pixels=50, degree=2,
        rng=np.random.default_rng(10),
    )
    return dict(fit=fit, lambda_map=lam, y_norm_rows=y_rows,
                candidate_flat_indices=idx, candidate_lambda=lams,
                candidate_y=yy, candidate_data=vals, candidate_variance=vv)


def test_actual_fit_capture_roundtrips_postclip_selection_and_basis(tmp_path):
    args = _direct_fit_example()
    path = tmp_path / "sky_frozen.npz"
    result = capture_kelson_fit(path, **args)
    assert result["status"] == "FROZEN_CONDITIONAL"
    assert not result["covariance_qualified"]
    assert result["n_final_fit_pixels"] == int(args["fit"]["_cache"]["keep"].sum())
    op, (c0, c1) = load_frozen_kelson_fit(path)
    assert op.n_pixels == args["lambda_map"].size
    assert c1.size > 0
    np.testing.assert_array_equal(
        op.fit_indices,
        args["candidate_flat_indices"][args["fit"]["_cache"]["keep"]],
    )
    with np.load(path, allow_pickle=False) as payload:
        assert int(payload["version"]) == 1
        assert payload["lambda_map"].shape == args["lambda_map"].shape
        assert payload["y_order"] == 1
        assert payload["fit_variance"].shape == op.fit_indices.shape
    # Fitted spline basis and coefficients reproduce the production model.
    coeff = np.r_[c0, c1]
    reconstructed = np.asarray(op.design @ coeff).ravel()
    knot = args["fit"]["basis"]["t"]
    from scorpio_pipe.stages.sky_sub import _eval_bspline
    production = _eval_bspline(knot, c0, 2, args["lambda_map"].ravel()) + (
        np.repeat(args["y_norm_rows"], args["lambda_map"].shape[1]) *
        _eval_bspline(knot, c1, 2, args["lambda_map"].ravel())
    )
    np.testing.assert_allclose(reconstructed, production, rtol=1e-6, atol=1e-4)


def test_capture_refuses_inconsistent_actual_fit_cache(tmp_path):
    args = _direct_fit_example()
    args["fit"]["_cache"]["keep"] = np.zeros(2, dtype=bool)
    path = tmp_path / "bad.npz"
    with pytest.raises(ValueError, match="selection"):
        capture_kelson_fit(path, **args)
    assert not path.exists()


def test_loader_rejects_incomplete_versioned_snapshot(tmp_path):
    path = tmp_path / "incomplete.npz"
    np.savez_compressed(path, version=np.int16(1))
    with pytest.raises(ValueError, match="schema"):
        load_frozen_kelson_fit(path)


def _science_night(work: Path, capture: bool):
    rng = np.random.default_rng(27)
    layout = ensure_work_layout(work)
    ny, nx = 48, 128
    y = np.arange(ny)[:, None]
    x = np.arange(nx)[None, :]
    lam = np.repeat((5000. + 2.*x), ny, axis=0)
    wavesol = stage_dir(work, "wavesol")
    fits.writeto(wavesol / "lambda_map.fits", lam.astype("f4"), overwrite=True)
    skyline = sum(65. * np.exp(-.5*((x-c)/1.3)**2) for c in (30., 60., 92.))
    obj = 35. * np.exp(-.5*((y-24.)/2.0)**2) * (10. + .03*x)
    raw = 20. + skyline + obj + rng.normal(0., 3., size=(ny, nx))
    path = layout.raw / "obj_0001.fits"
    fits.writeto(path, raw.astype("f4"), overwrite=True)
    config = {
        "work_dir": str(work),
        "frames": {"obj": [str(path)]},
        "roi": {
            "obj_y1": 22, "obj_y2": 26,
            "sky_y1": 2, "sky_y2": 10,
            "sky2_y1": 37, "sky2_y2": 45,
            "units": "px",
        },
        "sky_sub": {
            "primary_method": "kelson_raw",
            "save_png": False,
            "covariance_capture": {"enabled": capture},
            "kelson_raw": {"knot_step_A": 2., "object_eating_warn": -1.},
        },
    }
    run_sky_sub(config)
    stage = stage_dir(work, "sky")
    science = stage / "obj_0001_skysub_raw.fits"
    assert science.exists()
    return stage, science


def test_actual_sky_stage_capture_is_opt_in_and_preserves_science(tmp_path):
    uncaptured_dir, uncaptured = _science_night(tmp_path / "control", capture=False)
    captured_dir, captured = _science_night(tmp_path / "with_frozen", capture=True)
    with fits.open(uncaptured) as baseline, fits.open(captured) as selected:
        for name in ("SCI", "VAR", "MASK"):
            np.testing.assert_array_equal(selected[name].data, baseline[name].data)
        assert "SKYFRZV" not in baseline[0].header
        header = selected[0].header
        assert header["SKYFRZV"] == 1
        assert header["SKYFRZQ"] == "FIT_ONLY"
        saved = captured_dir / header["SKYFRZP"]
        assert saved.exists()
    op, coefficients = load_frozen_kelson_fit(saved)
    assert op.n_pixels == 48 * 128
    assert op.fit_indices.size >= 64
    assert coefficients[0].size >= 5
    done = json.loads((captured_dir / "done.json").read_text(encoding="utf-8"))
    assert "FROZEN_CONDITIONAL" in json.dumps(done)


def test_capture_failure_preserves_valid_science_product(tmp_path, monkeypatch):
    import scorpio_pipe.calib.sky_fit_capture as capture_module
    def fail_capture(*args, **kwargs):
        raise ValueError("controlled ancillary capture failure")
    monkeypatch.setattr(capture_module, "capture_kelson_fit", fail_capture)
    stage, path = _science_night(tmp_path / "failed_capture", capture=True)
    with fits.open(path) as hdul:
        assert "SKYFRZV" not in hdul[0].header
        assert bool(hdul[0].header.get("SKYOK", True))
    done = json.loads((stage / "done.json").read_text(encoding="utf-8"))
    assert "SKY_FROZEN_CAPTURE_FAILED" in json.dumps(done)
    assert not (stage / "obj_0001_kelson_frozen_fit_v1.npz").exists()
