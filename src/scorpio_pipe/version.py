"""Project version + minimal pipeline provenance helpers.

Historically, several modules imported a :func:`get_provenance` helper from this
module. During refactors, it was accidentally removed, which broke imports in
both the CLI and the CI smoke/regression tests.

This module intentionally keeps provenance *stable* (no timestamps) so stage
hashing and caching remain deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

__version__ = "5.40.36"

# Backward-compatible alias used by the UI/CI and older QC JSON writers.
PIPELINE_VERSION = __version__


@dataclass(frozen=True)
class PipelineProvenance:
    """Small, stable pipeline provenance blob.

    Keep this deliberately minimal: it is embedded into stage hashes and state
    files, so volatile fields would create unnecessary cache misses.
    """

    name: str = "scorpio_pipe"
    version: str = __version__
    schema: str = "scorpio_pipe.provenance"
    schema_version: int = 1


def get_provenance() -> PipelineProvenance:
    """Return stable provenance for hashing/state files."""

    # Construct at call time to ensure it always reflects __version__.
    return PipelineProvenance(version=__version__)


# Legacy name some older code paths may still reference.
provenance = get_provenance()


def as_header_cards(prefix: str = "SCORP") -> dict[str, str]:
    """Minimal FITS provenance header cards.

    We keep the key namespace short to avoid header bloat.
    """

    p = str(prefix).strip().upper()
    return {
        f"{p}VER": __version__,
    }
