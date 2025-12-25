"""Bitmask conventions for pipeline products (uint16 MASK plane).

Scorpio Pipe writes most science products as MEF FITS with extensions:
  - SCI  (float32)
  - VAR  (float32)
  - MASK (uint16)

This module defines the *strict, pipeline-wide* meaning of each MASK bit.

Notes
-----
- A pixel may have multiple flags; masks are combined with bitwise OR.
- The schema is versioned; outputs record the schema version in FITS headers
  and JSON manifests.

Keep this stable: changing bit meaning is a breaking change.
"""

from __future__ import annotations

import numpy as np


MASK_SCHEMA_VERSION = "v1"

# --- Canonical bit assignments (uint16) ---

# 0: no reliable data / outside coverage of the rectification mapping
NO_COVERAGE = np.uint16(1 << 0)

# 1: known bad pixel (static badpix map, detector defects, etc.)
BADPIX = np.uint16(1 << 1)

# 2: cosmic ray / transient (from a cosmic cleaning stage)
COSMIC = np.uint16(1 << 2)

# 3: saturation / non-linearity (pixel at/near saturation level)
SATURATED = np.uint16(1 << 3)

# 4: edge/interpolation artifact / partial-bin coverage (informational)
EDGE = np.uint16(1 << 4)

# 5: user-defined / manual exclusion (interactive masking)
USER = np.uint16(1 << 5)

# 6: rejected by robust combine (sigma-clip, outlier rejection)
REJECTED = np.uint16(1 << 6)


def header_cards(prefix: str = "SCORP") -> dict[str, str]:
    """FITS header cards describing the mask schema."""
    return {
        f"{prefix}_MKV": MASK_SCHEMA_VERSION,
        f"{prefix}_MB0": "NO_COVERAGE",
        f"{prefix}_MB1": "BADPIX",
        f"{prefix}_MB2": "COSMIC",
        f"{prefix}_MB3": "SATURATED",
        f"{prefix}_MB4": "EDGE",
        f"{prefix}_MB5": "USER",
        f"{prefix}_MB6": "REJECTED",
    }


def bitname(bit: int) -> str:
    """Return canonical name for a bit index."""
    names = {
        0: "NO_COVERAGE",
        1: "BADPIX",
        2: "COSMIC",
        3: "SATURATED",
        4: "EDGE",
        5: "USER",
        6: "REJECTED",
    }
    return names.get(int(bit), f"BIT{int(bit)}")


def summarize(mask: np.ndarray) -> dict[str, float]:
    """Return simple fraction-of-pixels stats for known bits."""
    m = np.asarray(mask, dtype=np.uint16)
    tot = float(m.size) if m.size else 1.0
    return {
        "no_coverage_frac": float(np.count_nonzero(m & NO_COVERAGE) / tot),
        "badpix_frac": float(np.count_nonzero(m & BADPIX) / tot),
        "cosmic_frac": float(np.count_nonzero(m & COSMIC) / tot),
        "saturated_frac": float(np.count_nonzero(m & SATURATED) / tot),
        "edge_frac": float(np.count_nonzero(m & EDGE) / tot),
        "user_frac": float(np.count_nonzero(m & USER) / tot),
        "rejected_frac": float(np.count_nonzero(m & REJECTED) / tot),
    }
