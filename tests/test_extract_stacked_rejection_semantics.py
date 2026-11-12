"""Science P0: Stack2D rejected-input provenance is distinct from 1D usability.

Inspired by standalone draft PR #111, integrated with verified Stack2D COV/metadata
requirements. Numerical extraction must not discard accepted coadded samples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.maskbits import NO_COVERAGE, REJECTED
from scorpio_pipe.stages.extract1d import (
    _boxcar_extract,
    _extraction_mask_policy,
    _optimal_extract,
    _strip_informational_1d_bits,
    _verified_stack_rejection_provenance,
    run_extract1d,
)


def _synthetic_rejected_contributor():
    ny, nlam = 7, 21
    sci = np.full((ny, nlam), 10.0, dtype=float)
    var = np.full_like(sci, 4.0)
    mask = np.zeros_like(sci, dtype=np.uint16)
    mask[3, 10] = np.uint16(REJECTED)
    yc = np.full(nlam, 3.0)
    return sci, var, mask, yc


def test_stack_rejected_contributor_is_informational_for_boxcar():
    fatal, info = _extraction_mask_policy(is_stack=True)
    assert not (fatal & REJECTED)
    assert info == REJECTED
    sci, var, mask, yc = _synthetic_rejected_contributor()
    flux, variance, provenance, _, npix = _boxcar_extract(
        sci, var, mask, yc, ap_hw=2, fatal_bits=fatal,
        min_good_frac=0.6, sky_resid=None, apply_local_sky=False,
    )
    np.testing.assert_allclose(flux, 50.0)
    np.testing.assert_allclose(variance, 20.0)
    assert np.all(npix == 5)
    assert int(provenance[10]) & REJECTED
    final = _strip_informational_1d_bits(provenance, bits=info)
    assert not (int(final[10]) & REJECTED)
    assert int(mask[3, 10]) & REJECTED


@pytest.mark.parametrize("backend", ["numpy", "auto"])
def test_stack_rejected_contributor_usable_for_optimal(backend):
    fatal, info = _extraction_mask_policy(is_stack=True)
    sci, var, mask, yc = _synthetic_rejected_contributor()
    flux, variance, provenance, _, npix, meta = _optimal_extract(
        sci, var, mask, yc, ap_hw=2, fatal_bits=fatal,
        min_good_frac=0.6, sky_resid=None, apply_local_sky=False,
        backend=backend,
    )
    assert np.isfinite(flux[10]) and variance[10] > 0
    assert npix[10] == 5
    final = _strip_informational_1d_bits(provenance, bits=info)
    assert (int(final[10]) & REJECTED) == 0
    assert meta["backend_used"] in {"numpy", "numba"}


def test_single_frame_rejected_pixel_still_fatal():
    fatal, info = _extraction_mask_policy(is_stack=False)
    assert fatal & REJECTED
    assert info == 0
    sci, var, mask, yc = _synthetic_rejected_contributor()
    flux, variance, provenance, _, npix = _boxcar_extract(
        sci, var, mask, yc, ap_hw=2, fatal_bits=fatal,
        min_good_frac=0.6, sky_resid=None, apply_local_sky=False,
    )
    assert npix[10] == 4
    assert flux[10] == pytest.approx(40.)
    assert variance[10] == pytest.approx(16.)
    assert int(provenance[10]) & REJECTED


def test_no_coverage_remains_fatal_and_input_provenance_is_immutable():
    flags = np.asarray([REJECTED | NO_COVERAGE, REJECTED, NO_COVERAGE], dtype=np.uint16)
    before = flags.copy()
    output = _strip_informational_1d_bits(flags, bits=REJECTED)
    np.testing.assert_array_equal(output, [NO_COVERAGE, 0, NO_COVERAGE])
    np.testing.assert_array_equal(flags, before)


def _write_stack(path: Path, *, verified: bool, cov: bool = True):
    ny, nlam = 32, 64
    y = np.arange(ny)[:, None]
    profile = np.exp(-0.5 * ((y - 16.) / 2.) ** 2)
    sci = (500. * profile) * np.ones((1, nlam))
    var = np.full((ny, nlam), 10., dtype=np.float32)
    mask = np.full((ny, nlam), REJECTED, dtype=np.uint16)
    # An entirely uncovered wavelength channel has no scientifically usable
    # contributors and must remain fatal even if REJECTED is informational.
    mask[:, 20] |= NO_COVERAGE
    coverage = np.full((ny, nlam), 2, dtype=np.int16)
    coverage[:, 20] = 0
    h = fits.Header()
    h["BUNIT"] = "ADU"
    h["ETAAPPL"] = False
    if verified:
        h["STKMETH"] = "invvar_huber"
    sh = fits.Header()
    sh["CRVAL1"] = 5000.
    sh["CDELT1"] = 1.
    sh["CRPIX1"] = 1.
    sh["CTYPE1"] = "WAVE"
    sh["CUNIT1"] = "Angstrom"
    hs = [
        fits.PrimaryHDU(header=h),
        fits.ImageHDU(sci.astype("f4"), header=sh, name="SCI"),
        fits.ImageHDU(var, header=sh, name="VAR"),
        fits.ImageHDU(mask, header=sh, name="MASK"),
    ]
    if cov:
        hs.append(fits.ImageHDU(coverage, header=sh, name="COV"))
    fits.HDUList(hs).writeto(path, overwrite=True)
    return mask


@pytest.mark.parametrize("qualified,cov", [(True, True), (False, True), (True, False)])
def test_rejection_provenance_requires_verified_stacked_metadata(tmp_path, qualified, cov):
    path = tmp_path / "stacked2d.fits"
    input_mask = _write_stack(path, verified=qualified, cov=cov)
    assert _verified_stack_rejection_provenance(path, fits.getheader(path), input_mask) is (qualified and cov)


def test_missing_no_coverage_on_empty_stack_is_not_verified(tmp_path):
    path = tmp_path / "stacked2d.fits"
    input_mask = _write_stack(path, verified=True)
    with fits.open(path, mode="update") as hdul:
        hdul["MASK"].data[:, 20] &= np.uint16(~int(NO_COVERAGE) & 0xFFFF)
        hdul.flush()
    changed = fits.getdata(path, "MASK")
    assert not _verified_stack_rejection_provenance(path, fits.getheader(path), changed)


def test_real_extract_stage_keeps_stacked_signal_but_respects_uncovered_bins(tmp_path):
    work = tmp_path / "work"
    out = work / "extract_output"
    out.mkdir(parents=True)
    stack = tmp_path / "stacked2d.fits"
    raw_mask = _write_stack(stack, verified=True)
    cfg = {
        "work_dir": str(work),
        "frames": {"__setup__": {"instrument": "scorpio", "mode": "longslit"}},
        "extract1d": {"input_mode": "stack2d", "method": "boxcar"},
    }
    results = run_extract1d(cfg, in_fits=stack, out_dir=out)
    with fits.open(results["spec1d_fits"], memmap=False) as hdul:
        tab = hdul["SPEC_TRACE"].data
        mask1d = np.asarray(tab["MASK_TRACE"], dtype=np.uint16)
        flux1d = np.asarray(tab["FLUX_TRACE"], dtype=float)
        assert not (int(mask1d[10]) & REJECTED)
        assert (int(mask1d[20]) & NO_COVERAGE)
        assert np.isfinite(flux1d[10]) and flux1d[10] > 0
    with fits.open(stack) as hdul:
        np.testing.assert_array_equal(hdul["MASK"].data, raw_mask)
