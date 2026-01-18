"""Backward-compatible boundary contract shim.

The contract "law" now lives in :mod:`scorpio_pipe.contracts.validators`.
This module remains for older code and tests that import
``scorpio_pipe.boundary_contract``.

This shim intentionally exposes a small stable API surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from scorpio_pipe.contracts.validators import (
    ProductContractError,
    validate_lambda_map_product,
    validate_mef2d_product as validate_mef_product,
    validate_product,
    validate_spec1d_product,
)

__all__ = [
    "ProductContractError",
    "validate_mef_product",
    "validate_spec1d_product",
    "validate_lambda_map_product",
    "validate_product",
    "validate_products",
]


def validate_products(paths: Iterable[str | Path], *, stage: str) -> None:
    """Validate a list of products (auto-detect format)."""

    for p in paths:
        validate_product(p, stage=stage)
