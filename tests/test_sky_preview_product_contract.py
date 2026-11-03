"""Sky quicklook is a valid SCI-only FITS, not an uncertainty-bearing MEF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from scorpio_pipe.catalog.products import CATALOG as PRODUCTS
from scorpio_pipe.contracts.validators import ProductContractError
from scorpio_pipe.io.mef import write_sci_var_mask
from scorpio_pipe.pipeline.engine import _validate_task_products
from scorpio_pipe.workspace_paths import stage_dir


def _preview(path: Path, *, bunit: str = "electron", nexp: int = 2) -> None:
    hdr = fits.Header()
    hdr["BUNIT"] = bunit
    hdr["NEXP"] = nexp
    hdr["SCORPPRV"] = True
    write_sci_var_mask(path, np.ones((8, 16), dtype=np.float32), header=hdr)


def test_sky_preview_has_separate_strict_image_contract(tmp_path: Path) -> None:
    cfg = {"work_dir": str(tmp_path), "frames": {}}
    spec = PRODUCTS.get_spec("sky_preview_fits")
    assert spec.required_headers_contract == "preview2d"
    path = stage_dir(tmp_path, "sky") / "preview.fits"
    _preview(path)

    # The full engine boundary must accept this deliberately SCI-only preview.
    _validate_task_products(cfg, "sky", tmp_path)
    with fits.open(path) as hdul:
        assert "SCI" in hdul
        assert "VAR" not in hdul
        assert "MASK" not in hdul


@pytest.mark.parametrize(
    ("bunit", "nexp", "code"),
    [
        ("Angstrom", 2, "PREVIEW_BUNIT"),
        ("electron", 0, "PREVIEW_NEXP"),
    ],
)
def test_sky_preview_rejects_bad_metadata(
    tmp_path: Path, bunit: str, nexp: int, code: str,
) -> None:
    path = stage_dir(tmp_path, "sky") / "preview.fits"
    _preview(path, bunit=bunit, nexp=nexp)
    with pytest.raises(ProductContractError) as exc:
        PRODUCTS.validate_products(
            {"work_dir": str(tmp_path)},
            stage_key="sky", product_keys=("sky_preview_fits",),
            run_root=tmp_path,
        )
    assert exc.value.code == code


def test_sky_science_product_still_requires_variance_and_mask(tmp_path: Path) -> None:
    cfg = {"work_dir": str(tmp_path), "frames": {}}
    sd = stage_dir(tmp_path, "sky")
    _preview(sd / "preview.fits")
    # A SCI-only *science* product may not borrow the preview contract.
    _preview(sd / "obj_01_skysub_raw.fits")
    with pytest.raises(ProductContractError) as exc:
        _validate_task_products(cfg, "sky", tmp_path)
    assert exc.value.code == "VAR_MISSING"


def test_sky_preview_rejects_corrupt_image_shape(tmp_path: Path) -> None:
    path = stage_dir(tmp_path, "sky") / "preview.fits"
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = fits.Header()
    hdr["BUNIT"] = "electron"
    hdr["NEXP"] = 2
    write_sci_var_mask(path, np.ones(16, dtype=float), header=hdr)
    with pytest.raises(ProductContractError) as exc:
        PRODUCTS.validate_products(
            {"work_dir": str(tmp_path)},
            stage_key="sky", product_keys=("sky_preview_fits",),
            run_root=tmp_path,
        )
    assert exc.value.code == "PREVIEW_SHAPE"
