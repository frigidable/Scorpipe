from __future__ import annotations

"""Workspace port.

P0-ARCH-03.1
------------
This port defines a minimal interface for interacting with a pipeline
workspace ("run_root").

Why
---
- Provide a single place for layout/path conventions (stage dirs, per-exp dirs).
- Centralize atomic writes and product discovery.
- Offer a "single writer" lock to prevent concurrent processes from writing
  into the same run_root.
"""

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class WorkspaceLockedError(RuntimeError):
    """Raised when a workspace is locked by another process."""

    def __init__(self, message: str, *, lock_path: Path | None = None) -> None:
        super().__init__(message)
        self.lock_path = lock_path


@runtime_checkable
class Workspace(Protocol):
    """A pipeline workspace interface.

    Implementations should be lightweight and safe to import (GUI/CLI startup).
    """

    @property
    def run_root(self) -> Path:  # pragma: no cover (protocol)
        ...

    # ---- layout helpers ----
    def stage_dir(self, key: str) -> Path:  # pragma: no cover (protocol)
        """Return the stage directory for a stage/task key."""

    def per_exp_dir(self, stem: str) -> Path:  # pragma: no cover (protocol)
        """Return per-exposure directory for a raw stem (implementation-defined)."""

    # ---- atomic write helpers ----
    def atomic_write_text(self, path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:  # pragma: no cover
        ...

    def atomic_write_json(self, path: str | Path, obj: Any, *, indent: int = 2, ensure_ascii: bool = False) -> Path:  # pragma: no cover
        ...

    def atomic_write_bytes(self, path: str | Path, data: bytes) -> Path:  # pragma: no cover
        ...

    # ---- products ----
    def open_product(self, key: str) -> Path:  # pragma: no cover (protocol)
        """Resolve the best path for a named product key."""

    def list_products(self) -> list[Any]:  # pragma: no cover (protocol)
        """Return a stable listing of known products (implementation-defined records)."""

    # ---- retention/cleanup ----
    def cleanup(self, policy: str, *, dry_run: bool = False) -> dict[str, Any]:  # pragma: no cover (protocol)
        """Apply a retention policy and return a cleanup report.

        Policies
        --------
        - all: keep everything
        - canonical: keep only canonical publish products (ProductCatalog non-collections) + qc + provenance
        - none: aggressively remove heavy intermediates (whitelist-based)
        """

    
    # ---- concurrency ----
    def lock(self, *, override: bool = False) -> AbstractContextManager[None]:  # pragma: no cover (protocol)
        """Acquire a "single writer" lock for this run_root."""


__all__ = [
    "Workspace",
    "WorkspaceLockedError",
]
