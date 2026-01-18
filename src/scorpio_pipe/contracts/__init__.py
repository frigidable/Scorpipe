"""Pipeline contract layer ("law").

Downstream code should prefer importing from this package rather than internal
implementation modules.

This package intentionally re-exports very little to keep namespaces explicit.
"""

from __future__ import annotations

__all__ = [
    "data_model",
    "units",
    "variance",
    "maskbits",
    "validators",
]
