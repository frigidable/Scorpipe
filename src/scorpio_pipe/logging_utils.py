"""Small logging helpers.

Some stages (e.g. :mod:`scorpio_pipe.stages.linearize`) historically imported
``get_logger`` from a module that didn't ship in some builds.

This file is intentionally tiny and dependency-free.
"""

from __future__ import annotations

import logging


_CONFIGURED = False


def _ensure_configured() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    # Keep logging quiet by default; UI/CLI can reconfigure.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger with minimal default configuration."""
    _ensure_configured()
    return logging.getLogger(name)
