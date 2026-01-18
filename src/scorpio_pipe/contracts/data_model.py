"""Data model contract (P0-B1).

This module defines the *versioned* schema for pipeline products.

Design goals
------------
- Make product types explicit (kind enum).
- Stamp a data-model version into every produced FITS product.
- Keep the contract stable and testable.

Notes
-----
FITS keywords are limited to 8 characters. We use a compact primary-header card:

- ``SCORPDMV``: Scorpio Pipe **Data Model Version**.

Changing the meaning of existing fields is a breaking change.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


# Increment only with a deliberate migration plan.
DATA_MODEL_VERSION: int = 1


class ProductKind(str, Enum):
    """Canonical product kinds produced by the pipeline."""

    RAW_FRAME = "RAW_FRAME"
    CALIBRATED_FRAME = "CALIBRATED_FRAME"
    RECTIFIED_2D = "RECTIFIED_2D"
    STACKED_2D = "STACKED_2D"
    ARC_FRAME = "ARC_FRAME"
    WAVESOLUTION = "WAVESOLUTION"
    SPECTRUM_1D = "SPECTRUM_1D"
    QA_REPORT = "QA_REPORT"


def stamp_data_model(hdr: Any, *, version: int = DATA_MODEL_VERSION) -> Any:
    """Stamp the data model version into a FITS header-like mapping.

    The function is intentionally permissive in input type: it supports
    ``astropy.io.fits.Header`` as well as dict-like mappings.
    """

    try:
        hdr["SCORPDMV"] = (int(version), "Scorpio Pipe data model version")
    except Exception:
        try:
            hdr["SCORPDMV"] = int(version)
        except Exception:
            pass
    return hdr


def get_data_model_version(hdr: Any) -> int | None:
    """Return SCORPDMV as int if present, else None."""

    try:
        v = hdr.get("SCORPDMV", None)
    except Exception:
        try:
            v = hdr["SCORPDMV"]
        except Exception:
            v = None
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None
