from __future__ import annotations

import logging
import os
import time

from rich.logging import RichHandler


def setup_logging(level: str | None = None) -> None:
    """Configure concise, readable console logging.

    Uses standard `logging` + RichHandler. Safe to call multiple times.

    Level resolution (first match wins):
      1) argument `level`
      2) env var `SCORPIO_LOG_LEVEL`
      3) default = "INFO"
    """

    if level is None:
        level = os.environ.get("SCORPIO_LOG_LEVEL", "INFO")

    level = str(level).upper().strip()
    if level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        level = "INFO"

    # Avoid duplicated handlers on re-init.
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                tracebacks_show_locals=False,
                show_path=False,
                show_time=True,
                omit_repeated_times=False,
            )
        ],
    )


class timer:
    """Lightweight context timer for logs.

    Example:
        with timer("build_superneon"):
            ...
    """

    def __init__(self, name: str, logger: logging.Logger | None = None):
        self.name = name
        self.logger = logger or logging.getLogger("scorpio")
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.logger.info("▶ %s…", self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        if exc is None:
            self.logger.info("✓ %s (%.2f s)", self.name, dt)
            return False
        self.logger.error("✗ %s FAILED (%.2f s): %s", self.name, dt, exc)
        return False
