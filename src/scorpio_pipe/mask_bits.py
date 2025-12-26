from __future__ import annotations

from enum import IntEnum


class MaskBits(IntEnum):
    """Default uint16 bitmask schema used across the pipeline.

    This schema is intentionally simple and forward-compatible.
    Downstream stages should treat unknown bits as "bad/flagged" unless they
    explicitly handle them.

    Bits
    ----
    0: BADPIX         (static detector bad pixel)
    1: COSMIC         (cosmic ray / transient outlier)
    2: SATURATED      (saturated / non-linear)
    3: EDGE           (edge / extrapolated / invalid resampling region)
    4: SKY_REGION     (pixel belongs to user-defined sky ROI)
    5: OBJ_REGION     (pixel belongs to user-defined object ROI)
    6: CLIPPED        (rejected by sigma-clip in stacking)
    7: RESERVED       (future)
    """

    BADPIX = 1 << 0
    COSMIC = 1 << 1
    SATURATED = 1 << 2
    EDGE = 1 << 3
    SKY_REGION = 1 << 4
    OBJ_REGION = 1 << 5
    CLIPPED = 1 << 6
    RESERVED = 1 << 7


def is_flagged(mask: int, bit: MaskBits) -> bool:
    return (int(mask) & int(bit)) != 0


def add_flag(mask: int, bit: MaskBits) -> int:
    return int(mask) | int(bit)


def remove_flag(mask: int, bit: MaskBits) -> int:
    return int(mask) & (~int(bit))


def mask_any(mask: int, bits: list[MaskBits]) -> bool:
    v = int(mask)
    return any((v & int(b)) != 0 for b in bits)
