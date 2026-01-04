"""Convenient local test runner.

Why this exists
--------------
Pytest's `-m` option *selects* tests by marker.

So:
  - `pytest -m smoke` runs only tests marked `@pytest.mark.smoke` (others show as
    "deselected" — that's normal).

In CI we run two suites:
  - fast unit/regression tests: `pytest -m "not smoke"`
  - end-to-end smoke test:      `pytest -m smoke`

This helper mirrors that locally with one command.

Usage
-----
  python scripts/run_tests.py ci
  python scripts/run_tests.py fast
  python scripts/run_tests.py smoke
  python scripts/run_tests.py all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_on_path() -> None:
    """Allow running from a source checkout without `pip install -e .`."""

    root = _repo_root()
    src = root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> int:
    print("\n$", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(_repo_root()), env=env)


def main(argv: list[str] | None = None) -> int:
    _ensure_src_on_path()

    ap = argparse.ArgumentParser(description="Run Scorpio Pipe test suites")
    ap.add_argument(
        "suite",
        nargs="?",
        default="ci",
        choices={"ci", "fast", "smoke", "all"},
        help="Which suite to run (default: ci)",
    )
    ap.add_argument("--pytest", default="pytest", help="Pytest executable (default: pytest)")
    ap.add_argument("--quiet", action="store_true", help="Pass -q to pytest")
    args = ap.parse_args(argv)

    q = ["-q"] if args.quiet else []

    if args.suite == "fast":
        return _run([args.pytest, *q, "-m", "not smoke"])

    if args.suite == "smoke":
        # Optional: collect smoke artifacts just like CI.
        env = dict(os.environ)
        env.setdefault("SCORPIPE_CI_ARTIFACTS_DIR", str(_repo_root() / "ci_artifacts"))
        return _run([args.pytest, *q, "-m", "smoke"], env=env)

    if args.suite == "all":
        return _run([args.pytest, *q])

    # ci: fast + smoke
    code = _run([args.pytest, *q, "-m", "not smoke"])
    if code != 0:
        return code
    env = dict(os.environ)
    env.setdefault("SCORPIPE_CI_ARTIFACTS_DIR", str(_repo_root() / "ci_artifacts"))
    return _run([args.pytest, *q, "-m", "smoke"], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
