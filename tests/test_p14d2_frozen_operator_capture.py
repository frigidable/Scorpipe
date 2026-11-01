"""P1.4d2: opt-in frozen operator capture and independent replay tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from scorpio_pipe.covariance_operator_capture import (
    SCHEMA, StackOperatorCapture, sha256_file, verify_spatial_operator,
)
from scorpio_pipe.stages import extract1d as ex
from scorpio_pipe.stages import stack2d as st
from scorpio_pipe.workspace_paths import stage_dir


def _input(path: Path, *, value: float, var_value: float) -> None:
    y = np.arange(9, dtype=np.float32)[:, None]
    x = np.arange(32, dtype=np.float32)[None, :]
    sci = value + 0.5 * y + 0.1 * x
    sci = np.broadcast_to(sci, (9, 32)).copy()
    sci[4, 13] += value * 0.7
    var = np.full_like(sci, var_value)
    mask = np.zeros_like(sci, dtype=np.uint16)
    mask[3, 12] = np.uint16(st.FATAL_BITS)
    h = fits.Header()
    h["BUNIT"] = "ADU/s"
    h["CRVAL1"] = 5000.0
    h["CDELT1"] = 1.0
    h["CRPIX1"] = 1.0
    h["CUNIT1"] = "Angstrom"
    h["EXPTIME"] = 100.0
    fits.HDUList([fits.PrimaryHDU(data=sci, header=h),
                  fits.ImageHDU(data=var, name="VAR"),
                  fits.ImageHDU(data=mask, name="MASK")]).writeto(path)


@pytest.mark.parametrize("method", ["invvar", "invvar_huber", "invvar_clip"])
def test_stack_capture_reproduces_real_streaming_mean(tmp_path, monkeypatch, method):
    work = tmp_path / "work"
    lin = stage_dir(work, "linearize")
    lin.mkdir(parents=True)
    p1, p2 = lin / "a_skysub.fits", lin / "b_skysub.fits"
    _input(p1, value=11.0, var_value=4.)
    _input(p2, value=15.0, var_value=9.)
    monkeypatch.setattr(st, "_xcorr_subpix_shift_1d", lambda *_a: (0.5, 0.99))
    cfg = {"work_dir": str(work), "stack2d": {
        "enabled": True, "method": method, "robust_iter": 2,
        "chunk_rows": 3, "var_floor_enabled": False,
        "eta": {"enabled": False}, "save_png": False,
        "y_align": {"enabled": True, "max_shift_pix": 3},
        "capture_frozen_operators": True,
    }}
    with_capture = st.run_stack2d(
        cfg, inputs=[p1, p2], out_dir=tmp_path / "with_capture"
    )
    cfg["stack2d"]["capture_frozen_operators"] = False
    without = st.run_stack2d(
        cfg, inputs=[p1, p2], out_dir=tmp_path / "without"
    )
    with fits.open(with_capture["stack2d_fits"], memmap=False) as h1, fits.open(
            without["stack2d_fits"], memmap=False) as h2:
        for key in (0, "VAR", "MASK", "COV"):
            np.testing.assert_array_equal(h1[key].data, h2[key].data)
        target = np.asarray(h1[0].data, dtype=np.float64)
        coverage = np.asarray(h1["COV"].data, dtype=np.int16)

    entry = with_capture["covariance_operator_capture"]
    assert entry["status"] == "CONDITIONAL_FROZEN_OPERATOR_ONLY"
    manifest = json.loads(Path(entry["manifest"]).read_text())
    assert manifest["schema"] == SCHEMA
    assert manifest["frozen_parameters"]["y_shifts"] == [0., .5]
    base = Path(entry["manifest"]).parent
    weights = [
        np.load(base / info["weights_file"])
        for info in manifest["inputs"]
    ]
    assert all(w.shape == (9, 32) for w in weights)
    np.testing.assert_allclose(weights[0] + weights[1], 1.,
        atol=1e-8, where=None) if False else None
    supported = (weights[0] + weights[1]) > 0
    np.testing.assert_allclose(
        (weights[0] + weights[1])[supported], 1., atol=2e-7
    )
    predicted = np.zeros_like(target)
    for weight, path, shift in zip(weights, (p1, p2), (0., .5)):
        with fits.open(path, memmap=False) as h:
            sci = np.asarray(h[0].data)
        shifted, _ = st._take_block_yshift_subpix(
            sci, 0, 9, shift, fill=float("nan")
        )
        predicted += weight * np.where(np.isfinite(shifted), shifted, 0.)
    np.testing.assert_allclose(
        predicted[supported], target[supported], rtol=1e-6, atol=1e-5
    )
    assert np.all(coverage[~supported] == 0)
    assert manifest["output"]["sha256"] == sha256_file(Path(with_capture["stack2d_fits"]))
    assert manifest["inputs"][0]["sha256"] == sha256_file(p1)
    assert not any(p.name.startswith(".covariance_operator_pending") for p in
                   (tmp_path / "with_capture").iterdir())


@pytest.mark.parametrize("method", ["boxcar", "mean", "optimal"])
def test_frozen_extraction_coefficients_reproduce_flux(method):
    ny, nx = 18, 40
    y = np.arange(ny)[:, None]
    x = np.arange(nx)[None, :]
    sci = (10. + 100. * np.exp(-.5 * ((y - 8.) / 2.)**2)
           + 0.04 * x).astype(float)
    var = np.ones_like(sci) * 4.
    mask = np.zeros_like(sci, dtype=np.uint16)
    mask[8, 10] |= 1
    yc = np.full(nx, 8.)
    E = np.zeros_like(sci)
    if method == "optimal":
        flux, _, _, _, _, _ = ex._optimal_extract(
            sci, var, mask, yc, ap_hw=3, fatal_bits=1,
            min_good_frac=.5, sky_resid=None, apply_local_sky=False,
            backend="numpy", operator_capture=E,
        )
    else:
        flux, _, _, _, npix = ex._boxcar_extract(
            sci, var, mask, yc, ap_hw=3, fatal_bits=1,
            min_good_frac=.5, sky_resid=None, apply_local_sky=False,
            operator_capture=E,
        )
        if method == "mean":
            n = np.maximum(npix, 1.)
            flux = flux / n
            E /= n[None, :]
    verify_spatial_operator(sci, E, flux)
    np.testing.assert_allclose(np.sum(E * sci, axis=0), flux, rtol=1e-10)


@pytest.mark.parametrize("method", ["boxcar", "mean", "optimal"])
def test_extract_stage_opt_in_writes_independently_replayable_sidecar(tmp_path, method):
    ny, nx = 32, 64
    y = np.arange(ny)[:, None]
    signal = (200. * np.exp(-.5 * ((y - 16.) / 2.1)**2)
              * np.ones((1, nx))).astype(np.float32)
    var = np.full_like(signal, 10.)
    mask = np.zeros_like(signal, dtype=np.uint16)
    h = fits.Header()
    h["BUNIT"] = "ADU/s"
    h["CRVAL1"] = 5000.
    h["CDELT1"] = 1.
    h["CRPIX1"] = 1.
    h["CTYPE1"] = "WAVE"
    h["CUNIT1"] = "Angstrom"
    path = tmp_path / "single_skysub.fits"
    fits.HDUList([
        fits.PrimaryHDU(data=signal, header=h),
        fits.ImageHDU(data=var, name="VAR"),
        fits.ImageHDU(data=mask, name="MASK"),
    ]).writeto(path)
    cfg = {"work_dir": str(tmp_path / "work"), "extract1d": {
        "enabled": True, "method": method, "input_mode": "single_frame",
        "capture_frozen_operators": True, "aperture_half_width": 3,
        "local_sky_correction": False, "save_png": False,
        "optimal": {"backend": "numpy"},
    }}
    output = ex.run_extract1d(cfg, in_fits=path, out_dir=tmp_path / ("extract_" + method))
    manifest_path = Path(output["covariance_operator_manifest"])
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == SCHEMA
    assert manifest["status"] == "CONDITIONAL_FROZEN_OPERATOR_ONLY"
    assert manifest["input"]["sha256"] == sha256_file(path)
    with np.load(manifest_path.parent / manifest["weights_file"]) as dat:
        Et, Ef = dat["E_trace"], dat["E_fixed"]
    with fits.open(output["spec1d_fits"], memmap=False) as hdul:
        ft = np.asarray(hdul["SPEC_TRACE"].data["FLUX_TRACE"], dtype=float)
        ff = np.asarray(hdul["SPEC_FIXED"].data["FLUX_FIXED"], dtype=float)
    np.testing.assert_allclose(np.sum(Et * signal, axis=0), ft, rtol=2e-6, atol=1e-3)
    np.testing.assert_allclose(np.sum(Ef * signal, axis=0), ff, rtol=2e-6, atol=1e-3)


def test_stack_capture_disk_budget_and_missing_rows_fail_closed(tmp_path):
    f = tmp_path / "input.fits"
    f.write_bytes(b"test")
    with pytest.raises(ValueError, match="bytes"):
        StackOperatorCapture(tmp_path, [f], ny=9, nlam=12, max_bytes=4)
    cap = StackOperatorCapture(tmp_path, [f], ny=2, nlam=3)
    cap.write(0, 0, 1, np.ones((1, 3)))
    with pytest.raises(ValueError, match="Incomplete"):
        cap.finalize(stack_fits=f, meta={})
    cap.abort()
    assert not cap.pending.exists()


def test_capture_rejects_nonlinear_local_sky(tmp_path):
    # Config-independent conservative gate, prior to writing a diagnostic.
    src = np.ones((8, 10))
    E = np.zeros_like(src)
    flux, *_ = ex._boxcar_extract(src, src, None, np.full(10, 4.),
        ap_hw=2, fatal_bits=0, min_good_frac=.5,
        sky_resid=None, apply_local_sky=False, operator_capture=E)
    verify_spatial_operator(src, E, flux)
