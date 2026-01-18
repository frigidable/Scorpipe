"""Calibration compatibility contract.

This is the **stable import path** for must-match / QC-only compatibility checks
used when associating calibrations to science frames.

Implementation lives in :mod:`scorpio_pipe.calib_compat`.
"""

from __future__ import annotations

from scorpio_pipe.calib_compat import (  # noqa: F401
    CalibMustKey,
    CalibQCKey,
    CalibrationMismatchError,
    compare_compat_headers,
    ensure_compatible_calib,
)

# Backward/compat aliases (older internal names).
CalibCompatError = CalibrationMismatchError
CompatKeys = CalibMustKey

__all__ = [
    "CalibMustKey",
    "CalibQCKey",
    "CalibrationMismatchError",
    "CalibCompatError",
    "CompatKeys",
    "compare_compat_headers",
    "ensure_compatible_calib",
]
