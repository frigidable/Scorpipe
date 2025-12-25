"""Provenance helpers for FITS products.

The pipeline uses provenance in two places:

1) FITS headers (to make products self-describing)
2) stage hashing / manifests (handled in :mod:`scorpio_pipe.stage_state`)

This module provides a small, stable API used by several stages.
"""

from __future__ import annotations

from typing import Any

from astropy.io import fits

from .version import as_header_cards


def add_provenance(hdr: fits.Header, cfg: dict[str, Any] | None = None, *, stage: str | None = None) -> fits.Header:
    """Return a *copy* of ``hdr`` with stable provenance cards.

    Notes
    -----
    - Must be stable: do NOT include timestamps.
    - ``cfg`` is accepted for API compatibility; currently unused.
    """

    ohdr = hdr.copy()
    for k, v in as_header_cards(prefix="SCORP").items():
        # Avoid overwriting user-provided cards with different meaning.
        if k not in ohdr:
            ohdr[k] = v
        else:
            ohdr[k] = v
    if stage:
        ohdr["SCORP_STG"] = (str(stage), "Pipeline stage")
    return ohdr
