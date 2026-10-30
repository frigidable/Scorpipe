"""Synthetic regression: RAW linearization must retain all canonical mask bits.

Tests exercise the production _process_linearize_frame path on an identity-like
wavelength map, not just the internal bit-rebinning helper.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from scorpio_pipe.maskbits import (
    EDGE,
    INVALID_WAVELENGTH,
    NO_COVERAGE,
    OUTSIDE_SLIT,
    SKYMODEL_FAIL,
)
from scorpio_pipe.stages.linearize import (
    _build_linearize_grid_plan,
    _process_linearize_frame,
)


def _identity_case(tmp_path):
    ny, nx = 4, 12
    wave = 5000.0 + np.arange(nx, dtype=float)
    edges_out = 4999.5 + np.arange(nx + 1, dtype=float)
    lam_map = np.broadcast_to(wave, (ny, nx)).copy()

    sci = np.full((ny, nx), 7.0, dtype=np.float32)
    var = np.full((ny, nx), 4.0, dtype=np.float32)
    mask = np.zeros((ny, nx), dtype=np.uint16)
    # The sky stage's soft-mode pass-through uses SKYMODEL_FAIL to warn
    # downstream stages that the sky was not subtracted.
    mask[0, 2] = SKYMODEL_FAIL
    mask[1, 3] = EDGE | OUTSIDE_SLIT
    mask[2, 4] = INVALID_WAVELENGTH
    mask[3, 5] = NO_COVERAGE

    header = fits.Header()
    header["BUNIT"] = "e-"
    header["GAIN"] = 2.0
    header["RDNOISE"] = 3.0
    header["EXPTIME"] = 600.0
    header["SKYMD"] = "none"
    header["QADEGRD"] = 1
    path = tmp_path / "synthetic_skysub_raw.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(header=header),
            fits.ImageHDU(data=sci, name="SCI"),
            fits.ImageHDU(data=var, name="VAR"),
            fits.ImageHDU(data=mask, name="MASK"),
        ]
    ).writeto(path)

    task = {
        "path": str(path),
        "lam_map": lam_map,
        "linearize_grid_plan": _build_linearize_grid_plan(lam_map, edges_out),
        "ny": ny,
        "nlam": nx,
        "y0": 0,
        "y1": ny,
        "mask_dir": str(tmp_path / "no_additional_cosmics"),
        "lcfg": {"post_sky_cleanup": "off", "mask_saturation": False},
        "cfg": {},
        "gain_override_f": None,
        "rdnoise_override_f": None,
        "scale_to_bin": False,
        "normalize_exptime": False,
        "wave0": 4999.5,
        "dw": 1.0,
        "unit": "Angstrom",
    }
    return _process_linearize_frame(task)


def test_linearize_preserves_sky_failure_and_other_quality_bits(tmp_path) -> None:
    output = _identity_case(tmp_path)
    mask = output["rect_mask"]
    assert mask[0, 2] & SKYMODEL_FAIL
    assert mask[1, 3] & EDGE
    assert mask[1, 3] & OUTSIDE_SLIT
    assert mask[2, 4] & INVALID_WAVELENGTH
    # A degraded sky product must not silently regain a clean quality mask.
    assert output["hdr"]["QADEGRD"] == 1


def test_input_no_coverage_does_not_contribute_flux_or_variance(tmp_path) -> None:
    output = _identity_case(tmp_path)
    mask = output["rect_mask"]
    assert mask[3, 5] & NO_COVERAGE
    assert output["rect_cov"][3, 5] == 0
    assert output["rect"][3, 5] == 0
    assert output["rect_var"][3, 5] == 0
    # Other wavelength samples keep their original electron units.
    np.testing.assert_allclose(output["rect"][0, 8], 7.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(output["rect_var"][0, 8], 4.0, rtol=0, atol=1e-6)
