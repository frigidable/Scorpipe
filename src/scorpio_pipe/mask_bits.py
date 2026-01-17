"""Backward-compatible mask bit definitions.

Historically the project carried two modules:

- :mod:`scorpio_pipe.maskbits` (the authoritative uint16 bitmask constants)
- :mod:`scorpio_pipe.mask_bits` (an older experimental enum)

The pipeline now standardizes on :mod:`scorpio_pipe.maskbits`. This module is
kept as a thin shim so older code can continue importing ``MaskBits``.
"""

from __future__ import annotations

from enum import IntEnum

from scorpio_pipe import maskbits as _mb


class MaskBits(IntEnum):
    NO_COVERAGE = int(_mb.NO_COVERAGE)
    BADPIX = int(_mb.BADPIX)
    COSMIC = int(_mb.COSMIC)
    SATURATED = int(_mb.SATURATED)
    OUTSIDE_SLIT = int(getattr(_mb, "OUTSIDE_SLIT", 1 << 7))
    INVALID_WAVELENGTH = int(getattr(_mb, "INVALID_WAVELENGTH", 1 << 8))
    SKYMODEL_FAIL = int(getattr(_mb, "SKYMODEL_FAIL", 1 << 9))
    USER = int(_mb.USER)
    REJECTED = int(_mb.REJECTED)
    # Backward-compat shim: older builds used a SKY bit.
    # Keep it importable, but prefer the new explicit bits.
    SKY = int(getattr(_mb, "SKY", int(getattr(_mb, "OUTSIDE_SLIT", 1 << 7))))


MASK_SCHEMA_VERSION = _mb.MASK_SCHEMA_VERSION


def header_cards(prefix: str = "SCORP") -> dict[str, str]:
    return _mb.header_cards(prefix=prefix)
