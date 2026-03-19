from __future__ import annotations

"""InstrumentAdapter port (P0-ARCH-05.1).

This port defines the minimal interface the core pipeline uses to interact with
instrument-specific behavior (header parsing, mode mapping, defaults).

Design
------
- Import-light: core code should not import SCORPIO-specific modules directly.
- Extensible: adding a new instrument = new adapter + registry entry.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from scorpio_pipe.metadata import FrameMeta


@dataclass(frozen=True)
class ModeKey:
    """Stable key identifying a unique observing setup/mode.

    The key is intentionally *data-oriented* and can be serialized.
    """

    instrument: str
    mode: str
    disperser: str
    slit: str
    binning: str
    shape: str
    readout: str


ModeProfile = dict[str, Any]


class InstrumentAdapter(Protocol):
    """Instrument adapter interface."""

    key: str  # e.g. 'scorpio1', 'scorpio2'
    display_name: str

    def parse_header(
        self,
        header: Mapping[str, Any],
        *,
        strict: bool = True,
        policy: Any | None = None,
        fallback_sources: Any | None = None,
    ) -> FrameMeta:
        """Parse a FITS header-like mapping into :class:`~scorpio_pipe.metadata.FrameMeta`."""

    def mode_from_meta(self, meta: FrameMeta) -> ModeKey:
        """Return :class:`ModeKey` for grouping/selection."""

    def defaults_for_mode(self, mode: ModeKey) -> ModeProfile:
        """Return default config overrides for this mode (may be empty)."""
