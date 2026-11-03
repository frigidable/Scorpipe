"""Science setup incompatibility must be fatal, never a warning.

Previously a RuntimeError raised for mixed dispersers/slit widths was caught
by the later broad "except Exception" block, downgrading a science mismatch
to ARC_COMPAT_CHECK_FAILED and continuing to fit a wavelength map.
"""
from __future__ import annotations

from pathlib import Path

import pytest

fits = pytest.importorskip("astropy.io.fits")

from scorpio_pipe.stages.superneon import build_superneon
from scorpio_pipe.stages.wavesolution import (
    ScienceSetupMismatchError,
    build_wavesolution,
)
from tests.golden.mini_night import make_mini_night


@pytest.mark.parametrize(
    ("keyword", "incompatible_value"),
    [
        ("DISPERSE", "VPHG1800@590"),
        ("SLITWID", 2.0),
    ],
)
def test_wavesolution_refuses_two_incompatible_science_setups(
    tmp_path: Path, keyword: str, incompatible_value,
) -> None:
    mn = make_mini_night(tmp_path)
    cfg = {
        **mn.cfg,
        "frames": {**mn.cfg["frames"]},
        "superneon": {**mn.cfg["superneon"], "bias_sub": False},
    }
    # A compatible ARC master is available. The mixed SCI input set, not
    # a missing lamp product, must cause the failure.
    result = build_superneon(cfg, run_root=mn.work_dir)
    assert result.superneon_fits.is_file()

    science_path = Path(cfg["frames"]["obj"][1])
    with fits.open(science_path, mode="update", memmap=False) as hdul:
        hdul[0].header[keyword] = incompatible_value
        hdul.flush()

    with pytest.raises(
        ScienceSetupMismatchError,
        match="Multiple distinct science setups detected",
    ):
        build_wavesolution(cfg, run_root=mn.work_dir)
