"""Calibration association / compatibility contracts.

This package provides stable entry points for calibration matching rules.

Current implementation reuses existing modules:
- :mod:`scorpio_pipe.calib_compat`
- :mod:`scorpio_pipe.dataset.builder`
"""

from __future__ import annotations

__all__ = ["association", "compat", "manifest_schema"]
