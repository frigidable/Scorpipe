"""Backward-compatible shim for mask bit definitions.

The authoritative mask-bit contract lives in :mod:`scorpio_pipe.contracts.maskbits`.
This module remains importable because a lot of pipeline code historically
imported :mod:`scorpio_pipe.maskbits`.
"""

from __future__ import annotations

from scorpio_pipe.contracts.maskbits import *  # noqa: F401,F403
