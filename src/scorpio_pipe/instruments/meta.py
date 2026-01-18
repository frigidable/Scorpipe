"""Instrument header parsing (deprecated import path).

Historically, the pipeline implemented normalized metadata in
:mod:`scorpio_pipe.instruments.meta`. Starting from P0-B2, the **single source
of truth** for normalized metadata lives in :mod:`scorpio_pipe.metadata`.

This module is kept as a thin compatibility layer so older imports keep working
without duplicating logic.
"""

from __future__ import annotations

from scorpio_pipe.metadata.frame_meta import (
    FrameMeta,
    HeaderContractError,
    ReadoutKey,
    parse_frame_meta,
)

__all__ = ["FrameMeta", "HeaderContractError", "ReadoutKey", "parse_frame_meta"]
