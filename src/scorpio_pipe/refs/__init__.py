"""Reference resource subsystem (CRDS-lite).

This package provides stable helpers for:

* deterministic reference resolution
* reference file hashing
* building a frozen reference context (context_id)

The reference context is part of provenance and affects stage hashing.
"""

from __future__ import annotations

from .store import ReferenceResolution, file_hash, resolve_reference
from .context import (
    build_reference_context,
    ensure_reference_context,
    load_reference_context,
    reference_context_id,
    reference_context_path,
)

__all__ = [
    "ReferenceResolution",
    "file_hash",
    "resolve_reference",
    "build_reference_context",
    "ensure_reference_context",
    "load_reference_context",
    "reference_context_id",
    "reference_context_path",
]
