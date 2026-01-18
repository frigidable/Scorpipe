"""Fallback sources for header normalization (P0-B2).

Fallback is allowed only when explicitly permitted by policy for a given field.
Any fallback must be recorded in :attr:`scorpio_pipe.metadata.frame_meta.FrameMeta.meta_provenance`
(and reflected in QA).

This module provides a minimal, serializable fallback container. It is designed
so that association rules can embed the exact fallback inputs into the dataset/
run manifests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class FallbackSource(str, Enum):
    HEADER = "HEADER"
    CONFIG = "CONFIG"
    TABLE = "TABLE"
    DEFAULT = "DEFAULT"


@dataclass(frozen=True)
class FallbackSources:
    """A simple collection of allowed fallbacks.

    The pipeline may pass fallbacks loaded from configuration files (night-level
    config, instrument tables, etc.).

    Structure:

    - ``global_values``: used for all instruments.
    - ``per_instrument``: instrument-specific overrides.
    """

    global_values: Mapping[str, Any] = field(default_factory=dict)
    per_instrument: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    source: FallbackSource = FallbackSource.CONFIG

    def resolve(self, field: str, *, instrument: str | None = None) -> tuple[Any, FallbackSource] | None:
        if instrument:
            vals = self.per_instrument.get(str(instrument), None)
            if vals and field in vals:
                return vals[field], self.source
        if field in self.global_values:
            return self.global_values[field], self.source
        return None
