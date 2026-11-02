from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from scorpio_pipe.stages.calib import _build_superflat_core


def _frame(path: Path, gain: float, rate: float, signal: float) -> Path:
    image = fits.PrimaryHDU(np.full((12, 20), signal, dtype=np.float32))
    image.header["BINNING"] = "1x2"
    image.header["GAIN"] = gain
    image.header["RATE"] = rate
    image.header["NODE"] = "A"
    image.writeto(path)
    return path


def test_flat_and_master_bias_must_match_readout(tmp_path):
    flat = _frame(tmp_path / "flat.fits", 3.5, 225, 13361)
    bias = _frame(tmp_path / "bias.fits", 0.9, 145, 999)
    with pytest.raises(RuntimeError, match="incompatible"):
        _build_superflat_core(
            flat_paths=[flat], superbias_path=bias,
            combine="median", sigma_clip=0, allow_readout_diff=False,
        )


def test_compatible_readout_preserves_superflat_creation(tmp_path):
    flat = _frame(tmp_path / "flat.fits", 3.5, 225, 13361)
    bias = _frame(tmp_path / "bias.fits", 3.5, 225, 999)
    result, header, stats = _build_superflat_core(
        flat_paths=[flat], superbias_path=bias,
        combine="median", sigma_clip=0, allow_readout_diff=False,
    )
    assert result.shape == (12, 20)
    assert np.allclose(result, 1.0)
    assert header["BIASSUB"] is True
    assert stats["n_used"] == 1
