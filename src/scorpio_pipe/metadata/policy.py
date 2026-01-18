"""Missing-key policy for header normalization (P0-B2).

The pipeline must not silently guess critical metadata. This module defines a
small, serializable policy used by :func:`scorpio_pipe.metadata.parse_frame_meta`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class FieldRequirement(str, Enum):
    """Requirement level for a metadata field."""

    REQUIRED = "REQUIRED"
    OPTIONAL = "OPTIONAL"
    FALLBACK_ALLOWED = "FALLBACK_ALLOWED"


@dataclass(frozen=True)
class MetadataPolicy:
    """Policy describing how missing fields are handled."""

    requirements: Mapping[str, FieldRequirement]

    def requirement_for(self, field: str, *, strict: bool = True) -> FieldRequirement:
        req = self.requirements.get(field, FieldRequirement.OPTIONAL)
        # In non-strict mode, required fields become optional (but still recorded).
        if not strict and req == FieldRequirement.REQUIRED:
            return FieldRequirement.OPTIONAL
        return req


def default_metadata_policy() -> MetadataPolicy:
    """Return the default policy.

    We keep the default conservative but compatible with existing header samples:
    only keys that affect calibration association are required by default.
    """

    req = {
        # Core keys used across the pipeline.
        "instrument": FieldRequirement.REQUIRED,
        "mode": FieldRequirement.REQUIRED,
        "imagetyp": FieldRequirement.REQUIRED,
        # Spectral configuration keys (required in spectra modes; parser may apply
        # extra mode-dependent checks).
        "disperser": FieldRequirement.REQUIRED,
        "slit_width_arcsec": FieldRequirement.REQUIRED,
        # Geometry & readout keys used for hard matching.
        "binning": FieldRequirement.REQUIRED,
        "naxis1": FieldRequirement.REQUIRED,
        "naxis2": FieldRequirement.REQUIRED,
        "node": FieldRequirement.REQUIRED,
        "rate": FieldRequirement.REQUIRED,
        "gain": FieldRequirement.REQUIRED,
        # Time is required for association.
        "date_time_utc": FieldRequirement.REQUIRED,
        # Optional / frequently missing in legacy headers.
        "exptime": FieldRequirement.OPTIONAL,
        "rdnoise": FieldRequirement.OPTIONAL,
        "object_name": FieldRequirement.OPTIONAL,
        "slit_pos": FieldRequirement.OPTIONAL,
        "detector": FieldRequirement.OPTIONAL,
        "slitmask": FieldRequirement.OPTIONAL,
        "sperange": FieldRequirement.OPTIONAL,
    }
    return MetadataPolicy(requirements=req)
