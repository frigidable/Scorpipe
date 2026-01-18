"""Calibration association rules (stable import path).

Implementation currently lives in :mod:`scorpio_pipe.dataset.builder`.
"""

from __future__ import annotations

from scorpio_pipe.dataset.builder import (
    build_dataset_manifest,
    build_dataset_manifest_from_records,
)

__all__ = ["build_dataset_manifest", "build_dataset_manifest_from_records"]
