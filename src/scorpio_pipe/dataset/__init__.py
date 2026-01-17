"""Dataset helpers.

This package hosts the **P0** dataset discovery layer.

Implemented pieces:

* **P0-A2**: deterministic frame classification + long-slit (Spectra) guardrail.
* **P0-B**: dataset night *manifest* builder and strict calibration matching.

The dataset-manifest builder is designed to be importable in lightweight
environments (e.g. CI) without Astropy installed; FITS I/O is therefore
performed via *lazy imports* inside CLI/scan functions.
"""

from __future__ import annotations

from .classify import FrameClass, classify_frame, is_longslit_mode
from .manifest import DatasetManifest

# Builder/matcher are lightweight on import (no Astropy import at module load).
from .builder import build_dataset_manifest, build_dataset_manifest_from_records

__all__ = [
    "FrameClass",
    "classify_frame",
    "is_longslit_mode",
    "DatasetManifest",
    "build_dataset_manifest",
    "build_dataset_manifest_from_records",
]
