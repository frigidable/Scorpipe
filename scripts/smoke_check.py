"""Minimal smoke-check for Scorpio Pipe.

This script is intentionally lightweight; it does **not** require real FITS
inputs. It validates that:
  - package imports work
  - GUI runner compatibility wrappers exist
  - provenance functions are present (fixing the PyInstaller ImportError)

Run:
  python scripts/smoke_check.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Allow running directly from a source checkout without `pip install -e .`.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import scorpio_pipe.version as v
    from scorpio_pipe.ui import pipeline_runner as pr

    prov = v.get_provenance()
    print(f"Pipeline version: {v.PIPELINE_VERSION}")
    print(f"Package version:  {v.__version__}")
    print(f"Provenance:       {prov}")

    # GUI critical APIs
    assert hasattr(pr, "load_context"), "ui.pipeline_runner.load_context missing"
    assert hasattr(pr, "run_sequence"), "ui.pipeline_runner.run_sequence missing"
    assert hasattr(pr, "run_lineid_prepare"), "ui.pipeline_runner.run_lineid_prepare missing"
    assert hasattr(pr, "run_wavesolution"), "ui.pipeline_runner.run_wavesolution missing"

    if args.verbose:
        print("OK: gui runner API present")


if __name__ == "__main__":
    main()
