"""Metadata normalization layer.

This package provides a stable import path for header normalization and policies.

The implementation currently reuses the instrument-specific parsers in
:mod:`scorpio_pipe.instruments`.
"""

from __future__ import annotations

from .frame_meta import FrameMeta, HeaderContractError, parse_frame_meta

__all__ = ["FrameMeta", "HeaderContractError", "parse_frame_meta"]
