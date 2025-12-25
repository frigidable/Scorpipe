"""Version helpers.

This project keeps *two* version identifiers:

- ``__version__``: Python package version (PEP 440). This is what pip/packaging sees.
- ``PIPELINE_VERSION``: user-facing pipeline release (e.g. v5.2.0).

They are maintained together in this single module to avoid "Schrödinger versions".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import platform
import subprocess
import sys


__version__ = "5.5.0"
PIPELINE_VERSION = "5.3.0"


@dataclass(frozen=True)
class VersionInfo:
    package_version: str
    pipeline_version: str
    git_commit: str | None
    python: str
    platform: str


def _try_git_commit() -> str | None:
    """Return short git commit hash if available."""
    try:
        # If packaged as an exe, git may not exist.
        root = Path(__file__).resolve().parents[2]
        if not (root / ".git").exists():
            return None
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
        return out or None
    except Exception:
        return None


def get_version_info() -> VersionInfo:
    return VersionInfo(
        package_version=__version__,
        pipeline_version=PIPELINE_VERSION,
        git_commit=_try_git_commit(),
        python=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
    )


def as_header_cards(prefix: str = "SCORP") -> dict[str, str]:
    """Key-value cards to store in FITS/JSON provenance."""
    v = get_version_info()
    cards = {
        f"{prefix}_VER": v.pipeline_version,
        f"{prefix}_PKG": v.package_version,
        f"{prefix}_PY": v.python,
        f"{prefix}_PLAT": v.platform,
    }
    if v.git_commit:
        cards[f"{prefix}_GIT"] = v.git_commit
    return cards
