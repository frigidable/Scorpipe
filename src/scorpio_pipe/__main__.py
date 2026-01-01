"""Module entry-point.

This enables running the project as a module:

    python -m scorpio_pipe

The canonical CLI entry-point is the console script ``scorpio-pipe``.
Nevertheless, some environments (and CI smoke-tests) prefer the ``-m`` form.

When invoked without arguments, we default to printing the version and exiting
successfully.
"""

from __future__ import annotations

import sys

from scorpio_pipe.cli import main


def _run() -> None:
    # If no args provided, print version (stable, succeeds) instead of argparse
    # error exit code.
    if len(sys.argv) == 1:
        sys.argv.append("version")
    main()


if __name__ == "__main__":
    _run()
