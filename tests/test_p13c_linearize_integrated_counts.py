"""Physical-contract regressions for P1.3c count-conserving linearization.

All expectations follow independent detector-pixel overlap geometry.
No real FITS observations are read or modified.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.stages.linearize import (
    _apply_linearize_row_plan,
    _apply_linearize_row_plan_var,
    _build_linearize_grid_plan,
    _build_linearize_row_plan,
    _process_linearize_frame,
    _rebin_row_cumulative,
    _rebin_row_var_weightsquared,
    _write_mef,
)
from scorpio_pipe.contracts.units import var_bunit_from_sci_bunit


def _fixture(tmp_path, *, bunit="e-", gain=2.0, dw=1.0,
             density=False, normalize=False):
    ny, nx = 4, 12
    centers = 5000.0 + 2.0 * np.arange(nx, dtype=float)
    edge_out = 4999.0 + dw * np.arange(int(round(24.0 / dw)) + 1)
    lam_map = np.broadcast_to(centers, (ny, nx)).copy()
    in_adu = bunit.strip().lower().startswith("adu")
    sci = np.full((ny, nx), 10.0 if in_adu else 20.0, dtype="f4")
    var = np.full((ny, nx), 4.0 if in_adu else 16.0, dtype="f4")
    mask = np.zeros((ny, nx), dtype="u2")
    hdr = fits.Header()
    hdr["BUNIT"] = bunit
    hdr["GAIN"] = gain
    hdr["RDNOISE"] = 3.0
    hdr["EXPTIME"] = 100.0
    path = tmp_path / ("adu.fits" if in_adu else "electron.fits")
    fits.HDUList([
        fits.PrimaryHDU(header=hdr),
        fits.ImageHDU(data=sci, name="SCI"),
        fits.ImageHDU(data=var, name="VAR"),
        fits.ImageHDU(data=mask, name="MASK"),
    ]).writeto(path, overwrite=True)

    task = dict(
        path=str(path), lam_map=lam_map,
        linearize_grid_plan=_build_linearize_grid_plan(lam_map, edge_out),
        ny=ny, nlam=len(edge_out)-1, y0=0, y1=ny,
        mask_dir=str(tmp_path / "masks_absent"),
        lcfg={"post_sky_cleanup": "off", "mask_saturation": False},
        cfg={}, gain_override_f=None, rdnoise_override_f=None,
        scale_to_bin=not density, normalize_exptime=normalize,
        wave0=float(edge_out[0]), dw=float(dw), unit="Angstrom",
    )
    return _process_linearize_frame(task)


@pytest.mark.parametrize("bunit", ["e-", "ADU"])
@pytest.mark.parametrize("dw", [1.0, 0.5])
@pytest.mark.parametrize("density", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_linearize_frame_integrated_and_density_units(
        tmp_path, bunit, dw, density, normalize):
    output = _fixture(
        tmp_path, bunit=bunit, dw=dw, density=density,
        normalize=normalize
    )
    # Input 20 electrons per 2 Angstrom detector pixel (10 e-/Angstrom).
    # Input independent variance is 16 electron^2 per detector pixel.
    expected_counts = 10.0 * dw
    expected_var = 16.0 * (dw / 2.0) ** 2
    if density:
        expected_counts /= dw
        expected_var /= dw**2
    if normalize:
        expected_counts /= 100.0
        expected_var /= 100.0**2

    np.testing.assert_allclose(output["rect"], expected_counts, atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(output["rect_var"], expected_var, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(output["rect_cov"], 1.0, atol=1e-9, rtol=0)
    expected_bunit = "e-/Angstrom" if density else "e-"
    if normalize:
        expected_bunit += "/s"
    assert output["hdr"]["BUNIT"] == expected_bunit
    assert output["hdr"]["SCORPUM"] == "ELECTRON"
    assert output["hdr"]["NORMEXP"] == normalize
    assert output["hdr"]["REBNORM"] == "COUNT_V1"


def test_rebin_reference_and_cached_plan_agree_on_nonuniform_dispersion():
    centers = np.array([0., 1.1, 2.4, 4., 6., 8.7, 12., 16., 20.9, 27.])
    edges_in = np.empty(centers.size + 1)
    edges_in[1:-1] = (centers[1:] + centers[:-1]) / 2.0
    edges_in[0] = 2 * centers[0] - edges_in[1]
    edges_in[-1] = 2 * centers[-1] - edges_in[-2]
    widths = np.diff(edges_in)
    counts = 7.0 * widths
    variance = np.full(centers.size, 5.0)
    edges_out = np.linspace(edges_in[0], edges_in[-1], 109)
    width_out = np.diff(edges_out)

    plan = _build_linearize_row_plan(centers, edges_out)
    assert plan is not None
    flux, coverage = _apply_linearize_row_plan(counts, plan)
    vout = _apply_linearize_row_plan_var(variance, plan)
    np.testing.assert_allclose(flux, 7.0 * width_out, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(coverage, 1.0, atol=1e-11)
    assert flux.sum() == pytest.approx(counts.sum(), abs=1e-11)

    # Independent O(nout * nin) overlap matrix.
    overlap = np.maximum(
        np.minimum(edges_out[1:, None], edges_in[None, 1:])
        - np.maximum(edges_out[:-1, None], edges_in[None, :-1]), 0.
    )
    A = overlap / widths[None, :]
    np.testing.assert_allclose(flux, A @ counts, atol=1e-11)
    np.testing.assert_allclose(vout, (A**2) @ variance, atol=1e-11)

    rf, rc = _rebin_row_cumulative(counts, centers, edges_out)
    rv = _rebin_row_var_weightsquared(variance, centers, edges_out)
    np.testing.assert_allclose(rf, flux, atol=1e-10)
    np.testing.assert_allclose(rv, vout, atol=1e-11)
    np.testing.assert_allclose(rc, coverage, atol=1e-10)

    rev_plan = _build_linearize_row_plan(centers[::-1], edges_out)
    assert rev_plan is not None
    rev_flux, rev_cov = _apply_linearize_row_plan(counts[::-1], rev_plan)
    np.testing.assert_allclose(rev_flux, flux, atol=1e-11)
    np.testing.assert_allclose(rev_cov, coverage, atol=1e-11)
    np.testing.assert_allclose(
        _apply_linearize_row_plan_var(variance[::-1], rev_plan), vout,
        atol=1e-11
    )


def test_coverage_uses_output_bin_width_even_when_flux_uses_input_width():
    centers = 5000.0 + 2.0 * np.arange(12, dtype=float)
    edges_out = 4999.0 + np.arange(25, dtype=float)
    plan = _build_linearize_row_plan(centers, edges_out)
    assert plan is not None
    valid = np.ones(12, dtype=bool)
    valid[5] = False
    flux, cov = _apply_linearize_row_plan(
        np.full(12, 20.0), plan, valid_mask=valid
    )
    np.testing.assert_allclose(cov[[10, 11]], 0)
    np.testing.assert_allclose(flux[[10, 11]], 0)
    np.testing.assert_allclose(cov[:10], 1.0)
    np.testing.assert_allclose(flux[:10], 10.0)


def test_reject_pre_normalized_input_to_avoid_double_exposure_correction(tmp_path):
    with pytest.raises(ValueError, match="expects integrated detector counts"):
        _fixture(tmp_path, bunit="e-/s", dw=1.0, density=False, normalize=True)


@pytest.mark.parametrize("bunit, expected", [
    ("e-", "electron2"),
    ("e-/s", "electron2/s2"),
    ("e-/Angstrom", "(e-/Angstrom)^2"),
    ("e-/Angstrom/s", "(e-/Angstrom/s)^2"),
])
def test_variance_unit_stamping(bunit, expected):
    assert var_bunit_from_sci_bunit(bunit) == expected


def test_fits_writer_preserves_density_variance_dimensions(tmp_path):
    output = _fixture(tmp_path, dw=0.5, density=True, normalize=True)
    path = tmp_path / "rectified.fits"
    _write_mef(
        path, output["rect"], output["hdr"], output["rect_var"],
        output["rect_mask"], output["rect_cov"],
        wave0=4999.0, dw=0.5, unit="Angstrom", wave_ref="air"
    )
    with fits.open(path) as hdul:
        assert hdul["SCI"].header["BUNIT"] == "e-/Angstrom/s"
        assert hdul["VAR"].header["BUNIT"] == "(e-/Angstrom/s)^2"
        assert hdul[0].header["REBNORM"] == "COUNT_V1"


def test_stack_input_unit_parser_keeps_density_dimension():
    from scorpio_pipe.stages.stack2d import _split_rate_unit
    assert _split_rate_unit("e-") == ("e-", False)
    assert _split_rate_unit("e-/s") == ("e-", True)
    assert _split_rate_unit("e-/Angstrom") == ("e-/angstrom", False)
    assert _split_rate_unit("e-/Angstrom/s") == ("e-/angstrom", True)
