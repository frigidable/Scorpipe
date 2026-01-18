from __future__ import annotations

"""Dataset manifest I/O helpers.

Stages should *not* implement their own ad-hoc path probing / JSON parsing.
This module provides a single, reusable implementation.

P0-B4: after ``dataset_manifest.json`` exists, it becomes the single source of
truth for calibration associations.
"""

from pathlib import Path
from typing import Any, Iterable

from scorpio_pipe.calib.manifest_schema import (
    DatasetManifestSchemaError,
    validate_dataset_manifest_file,
)
from scorpio_pipe.dataset.manifest import DatasetManifest


def candidate_manifest_paths(cfg: dict[str, Any], work_dir: Path) -> list[Path]:
    """Return candidate locations in priority order."""

    cand: list[Path] = []
    mp = cfg.get("dataset_manifest_path") or cfg.get("manifest_path")
    if mp:
        cand.append(Path(str(mp)).expanduser())

    cand.append((Path(work_dir) / "dataset_manifest.json").resolve())

    try:
        dd = cfg.get("data_dir")
        if dd:
            cand.append((Path(str(dd)).expanduser().resolve() / "dataset_manifest.json").resolve())
    except Exception:
        pass

    # De-duplicate while preserving order
    seen: set[str] = set()
    out: list[Path] = []
    for p in cand:
        sp = str(p)
        if sp in seen:
            continue
        seen.add(sp)
        out.append(p)
    return out


def resolve_dataset_manifest_path(cfg: dict[str, Any], work_dir: Path) -> Path | None:
    """Return the first existing candidate path, else None."""

    for p in candidate_manifest_paths(cfg, work_dir):
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            continue
    return None


def load_dataset_manifest(
    cfg: dict[str, Any],
    work_dir: Path,
    *,
    require: bool = False,
    require_v3: bool = True,
) -> tuple[DatasetManifest | None, Path | None]:
    """Load and validate ``dataset_manifest.json``.

    Returns
    -------
    (manifest, path)
        If not found and ``require=False``, returns (None, None).

    Raises
    ------
    RuntimeError
        If ``require=True`` and manifest is missing, or validation fails.
    """

    path = resolve_dataset_manifest_path(cfg, work_dir)
    if path is None:
        if require:
            cands = [str(p) for p in candidate_manifest_paths(cfg, work_dir)]
            raise RuntimeError(
                "dataset_manifest.json not found. Build it first (CLI: `scorpio-pipe dataset-manifest ...`). "
                f"Tried: {cands}"
            )
        return None, None

    try:
        man = validate_dataset_manifest_file(path, require_v3=require_v3)
    except DatasetManifestSchemaError as e:
        raise RuntimeError(str(e)) from e

    return man, path


def uniq_keep_order(xs: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out
