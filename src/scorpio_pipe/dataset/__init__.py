"""Dataset helpers.

P0 tasks introduce a strict, deterministic *classification* layer that decides
what each FITS frame represents (bias/flat/arc/science) and whether it belongs
to the long-slit (Spectra) branch.
"""

from __future__ import annotations

from .classify import FrameClass, classify_frame, is_longslit_mode

__all__ = [
    "FrameClass",
    "classify_frame",
    "is_longslit_mode",
]
