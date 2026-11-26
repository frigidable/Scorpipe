"""Golden CCD noise is a genuine Poisson + read-noise realization.

Synthetic input data are zero-bias-pedestal ADU. This is deliberate: the Sky
stage's no-VAR fallback estimates photon variance directly from input counts,
so adding an unremoved electronic bias offset would be unphysical.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from tests.golden.mini_night import make_mini_night


def _mad_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - median)))


def test_synthetic_bias_read_noise_matches_ccd_header(tmp_path: Path) -> None:
    mn = make_mini_night(tmp_path)
    bias = np.asarray(fits.getdata(mn.cfg["frames"]["bias"][0]), dtype=float)
    hdr = fits.getheader(mn.cfg["frames"]["bias"][0])
    gain = float(hdr["GAIN"])
    rn_e = float(hdr["RDNOISE"])
    assert gain == pytest.approx(0.62)
    assert rn_e == pytest.approx(3.0)
    assert abs(float(np.median(bias))) < 1.0
    assert 0.8 <= _mad_sigma(bias) * gain / rn_e <= 1.2


def test_synthetic_science_shot_noise_matches_physical_var_model(
    tmp_path: Path,
) -> None:
    mn = make_mini_night(tmp_path)
    header = fits.getheader(mn.cfg["frames"]["obj"][0])
    img = np.asarray(fits.getdata(mn.cfg["frames"]["obj"][0]), dtype=float)
    gain = float(header["GAIN"])
    rn_e = float(header["RDNOISE"])
    assert float(header["EXPTIME"]) == 300.0

    # Independent sky windows outside the seven wavelength lamp/sky lines
    # and the object aperture have a known expectation of 120 ADU.
    rows = np.r_[5:21, 44:60]
    cols = np.r_[0:10, 118:128]
    pixel_sample = img[np.ix_(rows, cols)]
    mean_adu = 120.0
    model_sigma_adu = np.sqrt(mean_adu * gain + rn_e**2) / gain
    assert abs(float(np.median(pixel_sample)) - mean_adu) < 5.0
    ratio = _mad_sigma(pixel_sample) / model_sigma_adu
    assert 0.75 < ratio < 1.25, (
        "Synthetic CCD noise is inconsistent with the shot-noise + read-noise "
        f"variance model: observed/model sigma={ratio:.3f}"
    )


def test_stack2d_never_reintroduces_function_local_fits_import() -> None:
    from scorpio_pipe.stages.stack2d import run_stack2d

    code = run_stack2d.__code__
    assert "fits" not in code.co_varnames, (
        "A function-level FITS import shadows module-level astropy.io.fits "
        "and breaks eta_lambda.fits creation before QC diagnostics"
    )
    assert "fits" in code.co_names
