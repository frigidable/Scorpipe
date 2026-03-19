from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")


def _write_img(p: Path, data: np.ndarray, header: dict | None = None) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    h = fits.Header()
    if header:
        for k, v in header.items():
            h[k] = v
    fits.writeto(p, data, header=h, overwrite=True)
    return p


def test_calib_bias_triplet_detects_shape_mismatch(tmp_path: Path) -> None:
    from scorpio_pipe.contracts.validators import ProductContractError, validate_calib_bias_triplet

    sd = tmp_path / "03_superbias"
    _write_img(sd / "master_bias.fits", np.zeros((10, 10), dtype=np.float32))
    _write_img(sd / "master_bias_var.fits", np.zeros((9, 10), dtype=np.float32))
    _write_img(sd / "master_bias_dq.fits", np.zeros((10, 10), dtype=np.uint16))

    with pytest.raises(ProductContractError) as e:
        validate_calib_bias_triplet(sd, stage="superbias")
    assert e.value.code == "bias_var_shape_mismatch"


def test_calib_flat_detects_nan_before_provenance(tmp_path: Path) -> None:
    from scorpio_pipe.contracts.validators import ProductContractError, validate_calib_flat

    sd = tmp_path / "04_superflat"
    arr = np.ones((8, 8), dtype=np.float32)
    arr[0, 0] = np.nan
    _write_img(sd / "superflat.fits", arr)

    with pytest.raises(ProductContractError) as e:
        validate_calib_flat(sd, stage="superflat")
    assert e.value.code == "flat_has_nan"


def test_calib_neon_detects_empty_constant(tmp_path: Path) -> None:
    from scorpio_pipe.contracts.validators import ProductContractError, validate_calib_neon

    sd = tmp_path / "06_superneon"
    _write_img(sd / "superneon.fits", np.zeros((16, 32), dtype=np.float32))

    with pytest.raises(ProductContractError) as e:
        validate_calib_neon(sd, stage="superneon")
    assert e.value.code == "neon_empty"


def test_calib_contracts_happy_path(tmp_path: Path) -> None:
    from scorpio_pipe.contracts.validators import (
        validate_calib_bias_triplet,
        validate_calib_flat,
        validate_calib_neon,
    )

    # superbias triplet
    sd_bias = tmp_path / "03_superbias"
    hdr_bias = {"NBIAS": 5, "SB_GID": "gid", "SB_METH": "median"}
    _write_img(sd_bias / "master_bias.fits", np.ones((10, 10), dtype=np.float32), header=hdr_bias)
    _write_img(sd_bias / "master_bias_var.fits", np.ones((10, 10), dtype=np.float32) * 0.1)
    _write_img(sd_bias / "master_bias_dq.fits", np.zeros((10, 10), dtype=np.uint16))
    validate_calib_bias_triplet(sd_bias, stage="superbias")

    # superflat
    sd_flat = tmp_path / "04_superflat"
    hdr_flat = {"SB_FULL": "/tmp/master_bias.fits"}
    img = np.ones((8, 8), dtype=np.float32)
    img[0, 0] = 0.9
    img[0, 1] = 1.1
    _write_img(sd_flat / "superflat.fits", img, header=hdr_flat)
    validate_calib_flat(sd_flat, stage="superflat")

    # superneon
    sd_neon = tmp_path / "06_superneon"
    hdr_neon = {"NPK": 10}
    rng = np.random.default_rng(0)
    neon = rng.normal(size=(16, 32)).astype(np.float32)
    _write_img(sd_neon / "superneon.fits", neon, header=hdr_neon)
    validate_calib_neon(sd_neon, stage="superneon")

