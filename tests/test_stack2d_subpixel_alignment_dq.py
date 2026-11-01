"""P1.4d prerequisite: subpixel Stack2D row masks and spatial DQ geometry."""
from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.stages import stack2d as st
from scorpio_pipe.workspace_paths import stage_dir


def test_noninteger_subpixel_shift_sci_var_and_dq_have_exact_weights():
    ny, nx = 6, 11
    sci = np.arange(ny, dtype=float)[:, None] * 10. + np.arange(nx)[None, :]
    var = np.arange(1., ny + 1.)[:, None] * np.ones((1, nx))
    mask = np.array([1, 2, 4, 8, 16, 32], dtype=np.uint16)[:, None] * np.ones((1, nx), dtype=np.uint16)
    # out y=1,2,3 -> input y=.75,1.75,2.75.
    y0, y1, shift = 1, 4, .25
    got, filled = st._take_block_yshift_subpix(sci, y0, y1, shift, fill=np.nan)
    v, vfilled = st._take_block_yshift_subpix_var(var, y0, y1, shift, fill=np.nan)
    m = st._take_block_yshift_subpix_mask(mask, y0, y1, shift)
    expected = .25 * sci[0:3, :] + .75 * sci[1:4, :]
    expected_var = (.25**2) * var[0:3, :] + (.75**2) * var[1:4, :]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(v, expected_var, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(m, mask[0:3, :] | mask[1:4, :])
    assert not filled.any()
    assert not vfilled.any()


def test_negative_shift_and_missing_edges_preserve_no_coverage():
    ny, nx = 5, 9
    sci = np.arange(ny * nx, dtype=float).reshape(ny, nx)
    var = np.full((ny, nx), 8., dtype=float)
    mask = np.zeros((ny, nx), dtype=np.uint16)
    got, filled = st._take_block_yshift_subpix(sci, 0, ny, -.5, fill=np.nan)
    vv, vf = st._take_block_yshift_subpix_var(var, 0, ny, -.5, fill=np.nan)
    mm = st._take_block_yshift_subpix_mask(mask, 0, ny, -.5)
    np.testing.assert_allclose(got[:ny-1], .5*sci[:ny-1]+.5*sci[1:])
    np.testing.assert_allclose(vv[:ny-1], 4.)
    assert not filled[:ny-1].any()
    assert not vf[:ny-1].any()
    assert np.isnan(got[-1]).all()
    assert np.isnan(vv[-1]).all()
    assert filled[-1].all()
    assert vf[-1].all()
    np.testing.assert_array_equal(
        mm[-1], np.full(nx, st.MASK_NO_COVERAGE, dtype=np.uint16)
    )


@pytest.mark.parametrize("shift", [-1.0, 1.0])
def test_integral_nonzero_y_shift_needs_only_actual_source_row(shift):
    ny, nx = 5, 8
    sci = np.arange(ny, dtype=float)[:, None] + np.zeros((ny,nx))
    var = 2. + sci
    mask = (np.arange(ny, dtype=np.uint16)[:, None] + 1) * np.ones((1,nx), dtype=np.uint16)
    # An unused neighbor is nonfinite. Integer alignment must ignore it.
    sci[3, :] = np.nan
    var[3, :] = np.nan
    got, filled = st._take_block_yshift_subpix(sci, 0, ny, shift, fill=np.nan)
    vv, vf = st._take_block_yshift_subpix_var(var, 0, ny, shift, fill=np.nan)
    mm = st._take_block_yshift_subpix_mask(mask, 0, ny, shift)
    for y in range(ny):
        source = int(y - shift)
        if 0 <= source < ny:
            np.testing.assert_array_equal(got[y], sci[source])
            np.testing.assert_array_equal(vv[y], var[source])
            np.testing.assert_array_equal(mm[y], mask[source])
            assert not filled[y].any()
            assert not vf[y].any()
        else:
            assert np.isnan(got[y]).all()
            assert np.isnan(vv[y]).all()
            assert filled[y].all()
            assert vf[y].all()
            assert np.all(mm[y] == st.MASK_NO_COVERAGE)


def test_spatial_covariance_is_missing_from_marginal_var_even_with_correct_interpolation():
    # Adjacent output y rows sample common input rows. This is a concrete
    # negative oracle for treating shifted spatial outputs as independent.
    rng = np.random.default_rng(20261031)
    n = 12000
    raw = rng.normal(size=(n, 4))
    out_a = .25 * raw[:, 0] + .75 * raw[:, 1]
    out_b = .25 * raw[:, 1] + .75 * raw[:, 2]
    expected_cov = .75 * .25
    assert np.cov(out_a, out_b, ddof=1)[0,1] == pytest.approx(expected_cov, rel=.08)
    assert np.var(out_a,ddof=1) == pytest.approx(.25**2+.75**2,rel=.06)
    sci = np.zeros((4, 7), dtype=np.float32)
    var = np.ones_like(sci)
    out, _ = st._take_block_yshift_subpix_var(var, 1, 3, .25, fill=np.nan)
    np.testing.assert_allclose(out, .25**2 + .75**2)


def _write_input(path, value):
    sci = np.full((8, 16), value, dtype=np.float32)
    var = np.full_like(sci, 4.)
    mask = np.zeros_like(sci, dtype=np.uint16)
    hdr = fits.Header()
    hdr["BUNIT"] = "ADU/s"
    hdr["CRVAL1"] = 5000.
    hdr["CDELT1"] = 1.
    hdr["CRPIX1"] = 1.
    hdr["CUNIT1"] = "Angstrom"
    hdr["EXPTIME"] = 100.
    fits.HDUList([
        fits.PrimaryHDU(data=sci,header=hdr),
        fits.ImageHDU(data=var,name="VAR"),
        fits.ImageHDU(data=mask,name="MASK"),
    ]).writeto(path)


def test_full_stack2d_with_nonzero_subpix_alignment_no_longer_crashes(tmp_path,monkeypatch):
    work = tmp_path / "night"
    lin = stage_dir(work, "linearize")
    out = stage_dir(work, "stack2d")
    lin.mkdir(parents=True,exist_ok=True)
    out.mkdir(parents=True,exist_ok=True)
    a,b=lin/"obj_a_skysub.fits",lin/"obj_b_skysub.fits"
    _write_input(a,10.)
    _write_input(b,14.)
    monkeypatch.setattr(st,"_xcorr_subpix_shift_1d",lambda *_args:(.5,1.))
    cfg = {"work_dir":str(work),"stack2d":{
        "enabled":True,"method":"invvar","chunk_rows":3,
        "y_align":{"enabled":True,"max_shift_pix":3},
        "eta":{"enabled":False},"var_floor_enabled":False,
        "save_png":False,
    }}
    res=st.run_stack2d(cfg, inputs=[a,b], out_dir=out)
    with fits.open(res["stack2d_fits"],memmap=False) as hdul:
        data=np.asarray(hdul[0].data,dtype=float)
        var=np.asarray(hdul["VAR"].data,dtype=float)
        cov=np.asarray(hdul["COV"].data)
        assert data.shape==(8,16)
        assert np.isfinite(data[1:7]).all()
        assert np.isfinite(var[1:7]).all()
        assert np.all(cov[1:7]==2)
        np.testing.assert_allclose(data[1:7], (10/4+14/2)/(.25+.5), atol=2e-4)
        np.testing.assert_allclose(var[1:7], 1/(.25+.5), atol=2e-4)


def test_huber_does_not_poison_single_exposure_edge_after_subpixel_shift(tmp_path, monkeypatch):
    work = tmp_path / "night"
    lin = stage_dir(work, "linearize")
    out = stage_dir(work, "stack2d")
    lin.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    a, b = lin / "a_skysub.fits", lin / "b_skysub.fits"
    _write_input(a, 10.)
    _write_input(b, 14.)
    monkeypatch.setattr(st, "_xcorr_subpix_shift_1d", lambda *_args: (.5, 1.))
    result = st.run_stack2d({
        "work_dir": str(work),
        "stack2d": {
            "enabled": True, "method": "invvar_huber", "robust_iter": 2,
            "chunk_rows": 3, "var_floor_enabled": False, "save_png": False,
            "eta": {"enabled": False},
            "y_align": {"enabled": True, "max_shift_pix": 3},
        },
    }, inputs=[a, b], out_dir=out)
    with fits.open(result["stack2d_fits"], memmap=False) as hdul:
        sci = np.asarray(hdul[0].data, dtype=float)
        var = np.asarray(hdul["VAR"].data, dtype=float)
        cov = np.asarray(hdul["COV"].data)
        covered = cov > 0
        assert covered.any()
        assert np.all(np.isfinite(sci[covered]))
        assert np.all(np.isfinite(var[covered]) & (var[covered] > 0))
        assert np.all(cov[0] == 1)
        np.testing.assert_allclose(sci[0], 10., atol=1e-5)
        np.testing.assert_allclose(var[0], 4., atol=1e-5)


def test_accumulator_ignores_nonfinite_zero_support_weight():
    acc = st._WeightedMeanAccumulator((1, 2))
    acc.update(np.array([[5., np.nan]]), np.array([[1., np.nan]]))
    acc.update(np.array([[7., 9.]]), np.array([[1., 2.]]))
    mean, var = acc.finalize()
    np.testing.assert_allclose(mean, [[6., 9.]])
    np.testing.assert_allclose(var, [[.5, .5]])
