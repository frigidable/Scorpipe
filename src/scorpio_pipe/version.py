"""Version helpers.

This project keeps *two* version identifiers:

- ``__version__``: Python package version (PEP 440). This is what pip/packaging sees.
- ``PIPELINE_VERSION``: user-facing pipeline release (e.g. v5.2.0).

They are maintained together in this single module to avoid "Schrödinger versions".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import subprocess
import sys


__version__ = "5.38.2"
PIPELINE_VERSION = "v5.38.2"


@dataclass(frozen=True)
class Provenance:
    """Stable provenance snapshot for stage hashing and manifests.

    Important: keep this stable across runs on the same installation. Do NOT
    include timestamps, absolute working directories, random IDs, etc.
    """

    package_version: str
    pipeline_version: str
    git_commit: str | None
    python: str
    platform: str
    frozen: bool


def get_provenance() -> Provenance:
    """Compatibility API used by :mod:`scorpio_pipe.stage_state`.

    Older GUI builds import ``get_provenance``. In v5.5 this function was
    referenced but missing, which caused an ImportError in PyInstaller builds.
    """
    v = get_version_info()
    return Provenance(
        package_version=v.package_version,
        pipeline_version=v.pipeline_version,
        git_commit=v.git_commit,
        python=v.python,
        platform=v.platform,
        frozen=bool(getattr(sys, "frozen", False)),
    )


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
