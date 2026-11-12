"""A Stack2D eta(lambda) sidecar is a calibrated 1D variance multiplier.

The scientific stack remains a strict 2D SCI/VAR/MASK MEF. The eta file uses
a single primary-HDU float array and may never borrow the MEF contract.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from scorpio_pipe.catalog.products import CATALOG as PRODUCTS
from scorpio_pipe.contracts.validators import (
    ProductContractError,
    validate_eta_lambda_product,
    validate_mef2d_product,
)
from scorpio_pipe.pipeline.engine import _validate_task_products
from scorpio_pipe.workspace_paths import stage_dir


def _write_eta(
    path: Path,
    values: np.ndarray | None = None,
    *,
    unit: str = "dimensionless",
    method: str = "sky_MAD/var_pred",
    window: int = 31,
    lower: float = 1.0,
    upper: float = 5.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = fits.Header()
    header["BUNIT"] = unit
    header["ETAMETH"] = method
    header["ETASMO"] = window
    header["ETAMIN"] = lower
    header["ETAMAX"] = upper
    data = np.array([1.0, 1.2, 2.0, 1.8, 1.1], dtype=np.float32) if values is None else values
    fits.PrimaryHDU(data=data, header=header).writeto(path, overwrite=True)


def test_stack_eta_sidecar_passes_real_catalog_and_engine_boundary(tmp_path: Path) -> None:
    path = stage_dir(tmp_path, "stack2d") / "eta_lambda.fits"
    _write_eta(path)
    assert PRODUCTS.get_spec("eta_lambda_fits").required_headers_contract == "eta1d"
    _validate_task_products({"work_dir": str(tmp_path)}, "stack2d", tmp_path)
    with fits.open(path) as hdul:
        assert len(hdul) == 1
        assert hdul[0].data.ndim == 1
        assert "SCI" not in hdul
        assert "VAR" not in hdul


@pytest.mark.parametrize(
    ("options", "code"),
    [
        ({"unit": "Angstrom"}, "ETA_BUNIT"),
        ({"method": "unknown"}, "ETA_METHOD"),
        ({"window": 2}, "ETA_PARAMS"),
        ({"window": 4}, "ETA_PARAMS"),
        ({"lower": 0}, "ETA_PARAMS"),
        ({"upper": 0.5}, "ETA_PARAMS"),
        ({"values": np.array([1.0, np.nan], dtype=np.float32)}, "ETA_VALUES"),
        ({"values": np.array([1.0, 0.0], dtype=np.float32)}, "ETA_VALUES"),
        ({"values": np.array([1.0, 5.5], dtype=np.float32)}, "ETA_BOUNDS"),
        ({"values": np.ones((2, 4), dtype=np.float32)}, "ETA_SHAPE"),
        ({"values": np.ones(5, dtype=np.int16)}, "ETA_DTYPE"),
    ],
)
def test_eta_contract_rejects_corrupt_or_unphysical_sidecars(
    tmp_path: Path, options: dict, code: str,
) -> None:
    path = tmp_path / "eta.fits"
    _write_eta(path, **options)
    with pytest.raises(ProductContractError) as exc:
        validate_eta_lambda_product(path, stage="stack2d")
    assert exc.value.code == code


def test_eta_contract_rejects_missing_primary_array(tmp_path: Path) -> None:
    path = tmp_path / "no-data.fits"
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU().writeto(path)
    with pytest.raises(ProductContractError) as exc:
        validate_eta_lambda_product(path, stage="stack2d")
    assert exc.value.code == "ETA_SHAPE"


def test_eta_contract_rejects_extra_mef_extensions(tmp_path: Path) -> None:
    path = tmp_path / "not-sidecar.fits"
    _write_eta(path)
    with fits.open(path, mode="update") as hdul:
        hdul.append(fits.ImageHDU(data=np.ones((3, 3)), name="SCI"))
        hdul.flush()
    with pytest.raises(ProductContractError) as exc:
        validate_eta_lambda_product(path, stage="stack2d")
    assert exc.value.code == "ETA_LAYOUT"


def test_science_mef_remains_strict_and_does_not_accept_eta_as_stack(tmp_path: Path) -> None:
    path = stage_dir(tmp_path, "stack2d") / "stack2d.fits"
    _write_eta(path)
    with pytest.raises(ProductContractError) as exc:
        _validate_task_products({"work_dir": str(tmp_path)}, "stack2d", tmp_path)
    assert exc.value.code == "SCI_MISSING"
    # This same file would satisfy an eta contract, but can never masquerade as
    # the scientific variance-bearing stack.
    validate_eta_lambda_product(path, stage="stack2d")
