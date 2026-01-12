"""Deterministic frame classification.

Task: **P0-A2 — Классификатор кадров (тип кадра и режим long-slit)**.

We must decide:

1) what a frame *is* (bias/flat/arc/science/other)
2) whether it belongs to the long-slit branch (``MODE == Spectra``)

The classifier is deliberately conservative: it prefers explicit header fields
(``IMAGETYP``) and uses minimal heuristics for the object field.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from scorpio_pipe.instruments.meta import FrameMeta


class FrameClass(str, Enum):
    """Canonical frame class."""

    BIAS = "bias"
    FLAT = "flat"
    ARC = "arc"
    SCIENCE = "science"
    OTHER = "other"


def is_longslit_mode(meta: FrameMeta) -> bool:
    """Return True if the frame belongs to the long-slit (Spectra) pipeline."""

    return str(meta.mode).strip().lower() == "spectra"


def _norm_token(s: str) -> str:
    return str(s or "").strip().lower()


def classify_frame(meta: FrameMeta) -> FrameClass:
    """Classify a frame using normalized :class:`~scorpio_pipe.instruments.meta.FrameMeta`.

    Rules (deterministic):

    - Primary signal: ``meta.imagetyp``.
    - ``MODE`` is *not* used to decide BIAS/FLAT/ARC/SCIENCE, but is checked
      separately via :func:`is_longslit_mode` by the long-slit pipeline.
    - For ``obj`` frames we classify as SCIENCE. Standard-star vs science
      target separation is handled later (P1) via OBJECT patterns and/or
      dataset manifest overrides.
    """

    it = _norm_token(meta.imagetyp)

    if it in {"bias", "zero"}:
        return FrameClass.BIAS
    if it in {"flat", "flatfield", "ff"}:
        return FrameClass.FLAT
    if it in {"neon", "arc", "comp", "comparison", "lamp"}:
        return FrameClass.ARC
    if it in {"obj", "object", "science"}:
        return FrameClass.SCIENCE

    # Some archives encode "flat"/"bias" in OBJECT while leaving IMAGETYP vague.
    # We only use this as a *fallback* to avoid misclassifying science frames.
    obj = _norm_token(meta.object_name)
    if obj.startswith("bias"):
        return FrameClass.BIAS
    if obj.startswith("flat"):
        return FrameClass.FLAT
    if obj.startswith("neon") or obj.startswith("arc"):
        return FrameClass.ARC

    return FrameClass.OTHER


@dataclass(frozen=True)
class ClassifiedFrame:
    """Small helper bundle used by dataset scanners."""

    meta: FrameMeta
    frame_class: FrameClass

    @property
    def is_longslit(self) -> bool:
        return is_longslit_mode(self.meta)
