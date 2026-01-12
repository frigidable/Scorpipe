"""Instrument-specific header parsing.

This package implements the **Header Contract** for SCORPIO-1 and SCORPIO-2.

Main entrypoint: :func:`scorpio_pipe.instruments.parse_frame_meta`.
"""

from __future__ import annotations

from .meta import FrameMeta, HeaderContractError, ReadoutKey, parse_frame_meta

__all__ = [
    "FrameMeta",
    "HeaderContractError",
    "ReadoutKey",
    "parse_frame_meta",
]
