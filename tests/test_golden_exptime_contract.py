"""Synthetic exposure metadata must support physically defined SCI rate units."""

from __future__ import annotations

from pathlib import Path

import pytest
from astropy.io import fits

from scorpio_pipe.stages.linearize import _get_exptime_seconds
from scorpio_pipe.stages.sky_sub import _read_sky_input_planes
from tests.golden.mini_night import make_mini_night


@pytest.mark.parametrize(
    ("role", "expected_seconds"),
    [("bias", 0.0), ("flat", 10.0), ("neon", 30.0), ("obj", 300.0)],
)
def test_fixture_exposure_times_are_explicit_by_role(
    tmp_path: Path, role: str, expected_seconds: float,
) -> None:
    mn = make_mini_night(tmp_path)
    for filename in mn.cfg["frames"][role]:
        hdr = fits.getheader(filename)
        assert float(hdr["EXPTIME"]) == expected_seconds
        if role == "obj":
            sci, var, mask, sky_hdr = _read_sky_input_planes(Path(filename))
            assert sci.shape == (64, 128)
            assert float(sky_hdr["EXPTIME"]) == expected_seconds
            assert _get_exptime_seconds(sky_hdr, require_valid=True) == expected_seconds
        else:
            assert expected_seconds == 0 or _get_exptime_seconds(
                hdr, require_valid=True
            ) == expected_seconds


def test_rate_normalization_fails_closed_on_missing_or_zero_exptime() -> None:
    with pytest.raises(ValueError, match="missing/invalid exposure time"):
        _get_exptime_seconds(fits.Header(), require_valid=True)
    with pytest.raises(ValueError, match="missing/invalid exposure time"):
        _get_exptime_seconds(fits.Header({"EXPTIME": 0.0}), require_valid=True)
