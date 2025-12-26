from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

pytest.importorskip("astropy")
from scorpio_pipe.fits_utils import read_image_smart


def test_read_image_smart_handles_scaled_int(tmp_path: Path):
    # Create an integer image with BZERO/BSCALE to emulate common instrument files.
    data = (np.arange(32 * 16).reshape(16, 32) % 1000).astype(np.int16)

    hdu = fits.PrimaryHDU(data=data)
    hdu.header["BZERO"] = 32768
    hdu.header["BSCALE"] = 1

    p = tmp_path / "scaled.fits"
    hdu.writeto(p, overwrite=True)

    img, hdr, info = read_image_smart(p, memmap="auto", dtype=np.float32)
    assert img.dtype == np.float32
    assert img.shape == (16, 32)
    assert info.get("has_scaling") is True
    # Under memmap="auto" we expect memmap to be disabled when scaling keys are present.
    assert info.get("memmap_used") in (False, None)


def test_read_image_smart_does_not_raise_numpy2_copy(tmp_path: Path):
    # This specifically guards against NumPy 2.0 strict copy semantics.
    data = (np.random.default_rng(0).normal(size=(10, 12)) * 100).astype(np.float64)
    p = tmp_path / "float64.fits"
    fits.PrimaryHDU(data=data).writeto(p, overwrite=True)

    img, _hdr, _info = read_image_smart(p, memmap="auto", dtype=np.float32)
    assert img.dtype == np.float32