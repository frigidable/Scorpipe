"""SCORPIO Sky must not infer Poisson CCD noise from unknown raw pedestal.

Instrument-labelled science requires BIASSUB=True before the Sky stage.
Synthetic zero-pedestal CCD frames must explicitly declare that state;
generic legacy low-level unit tests remain available for numerical exercises.
"""
from __future__ import annotations

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from scorpio_pipe.stages.sky_sub import _prepare_sky_input_planes


@pytest.mark.parametrize("state", [None, False])
@pytest.mark.parametrize("provide_var", [False, True])
def test_scorpio_sky_refuses_unknown_or_unremoved_bias(
    state: bool | None, provide_var: bool,
) -> None:
    hdr = fits.Header()
    hdr["INSTRUME"] = "SCORPIO-2"
    hdr["BUNIT"] = "ADU"
    hdr["GAIN"] = 0.62
    hdr["RDNOISE"] = 3.0
    if state is not None:
        hdr["BIASSUB"] = state
    raw = np.full((12, 18), 1120.0, dtype="f4")
    var = np.full_like(raw, 10.0) if provide_var else None
    with pytest.raises(ValueError, match="BIASSUB=True"):
        _prepare_sky_input_planes(
            raw, var, None, None, hdr, cfg={}, sky_cfg={},
        )


def test_scorpio_sky_accepts_declared_zero_pedestal_and_correct_noise() -> None:
    hdr = fits.Header()
    hdr["INSTRUME"] = "SCORPIO-2"
    hdr["BUNIT"] = "ADU"
    hdr["BIASSUB"] = True
    hdr["GAIN"] = 0.62
    hdr["RDNOISE"] = 3.0
    science = np.full((12, 18), 120.0, dtype="f4")
    sci_e, var_e2, mask, new_header = _prepare_sky_input_planes(
        science, None, None, None, hdr, cfg={}, sky_cfg={},
    )
    assert np.allclose(sci_e, 120 * 0.62)
    assert np.allclose(var_e2, 120 * 0.62 + 3**2)
    assert new_header["SKYIVAR"] == "ESTIM"
    assert mask.dtype == np.uint16


def test_generic_nonscorpio_unit_fixture_keeps_legacy_noise_path() -> None:
    hdr = fits.Header({"BUNIT": "ADU", "GAIN": 2.0, "RDNOISE": 3.0})
    sci = np.full((8, 12), 10.0, dtype="f4")
    out, variance, _, _ = _prepare_sky_input_planes(
        sci, None, None, None, hdr, cfg={}, sky_cfg={},
    )
    assert np.allclose(out, 20.0)
    assert np.allclose(variance, 29.0)
