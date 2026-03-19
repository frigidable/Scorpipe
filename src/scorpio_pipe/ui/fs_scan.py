from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


DEFAULT_SUFFIXES: Tuple[str, ...] = (".fits", ".fit", ".fts", ".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class ScanResult:
    paths: List[Path]
    truncated: bool
    scanned_files: int


def scan_tree(
    root: Path,
    *,
    suffixes: Sequence[str] = DEFAULT_SUFFIXES,
    max_files: int = 5000,
) -> ScanResult:
    """Fast recursive scan with early truncation.

    Uses os.walk (faster than Path.rglob for big trees) and returns sorted paths.
    """
    root = Path(root)
    out: List[Path] = []
    scanned = 0
    suf = tuple(s.lower() for s in suffixes)

    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            scanned += 1
            low = fn.lower()
            if not low.endswith(suf):
                continue
            out.append(Path(dirpath) / fn)
            if len(out) >= max_files:
                out.sort(key=lambda p: p.name)
                return ScanResult(paths=out, truncated=True, scanned_files=scanned)

    out.sort(key=lambda p: p.name)
    return ScanResult(paths=out, truncated=False, scanned_files=scanned)


def find_first(
    root: Path,
    *,
    suffixes: Sequence[str] = (".fits", ".fit", ".fts"),
) -> Path | None:
    """Find the first file with one of suffixes under root (depth-first)."""
    root = Path(root)
    suf = tuple(s.lower() for s in suffixes)
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(suf):
                return Path(dirpath) / fn
    return None
