from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from scorpio_pipe.app_paths import user_cache_root, ensure_dir
from typing import Iterable


@dataclass(frozen=True)
class ResourceResolution:
    """Resolved resource location + provenance."""

    path: Path
    source: str  # e.g. absolute|work_dir|config_dir|project_root|package


def _default_cache_dir() -> Path:
    return ensure_dir(user_cache_root("Scorpipe") / "resources")


def _materialize_package_file(
    name: str, *, cache_dir: Path | None = None
) -> Path | None:
    """Return a stable on-disk path for a packaged resource.

    If the package is installed as a zip, importlib may expose resources via a
    temporary extracted file. We copy it into a user cache directory so other
    tools (Qt PDF viewers, etc.) can open the path reliably.
    """

    cache_dir = cache_dir or _default_cache_dir()

    try:
        ref = resources.files("scorpio_pipe.resources") / name
        if not ref.is_file():
            return None
        # Use a content hash to avoid stale cache.
        with resources.as_file(ref) as p:
            src = Path(p)
            data = src.read_bytes()
        h = hashlib.md5(data).hexdigest()[:10]
        dst = cache_dir / f"{name}.{h}"
        if not dst.exists():
            dst.write_bytes(data)
        return dst
    except Exception:
        return None


def resolve_resource(
    name_or_path: str | Path,
    *,
    work_dir: str | Path | None = None,
    config_dir: str | Path | None = None,
    project_root: str | Path | None = None,
    allow_package: bool = True,
) -> ResourceResolution:
    """Resolve a resource path with sensible fallbacks.

    Resolution order for relative inputs:
      1) work_dir / name
      2) config_dir / name
      3) project_root / name
      4) packaged resource scorpio_pipe/resources/<name>

    Returns ResourceResolution(path, source) or raises FileNotFoundError.
    """

    p = Path(str(name_or_path))
    if p.is_absolute():
        if p.exists():
            return ResourceResolution(path=p, source="absolute")
        raise FileNotFoundError(f"Resource not found: {p}")

    # Try work_dir
    if work_dir:
        cand = (Path(work_dir) / p).resolve()
        if cand.exists():
            return ResourceResolution(path=cand, source="work_dir")

    # Try config_dir
    if config_dir:
        cand = (Path(config_dir) / p).resolve()
        if cand.exists():
            return ResourceResolution(path=cand, source="config_dir")

    # Try project_root
    if project_root:
        cand = (Path(project_root) / p).resolve()
        if cand.exists():
            return ResourceResolution(path=cand, source="project_root")

    # Try packaged
    if allow_package:
        mat = _materialize_package_file(p.name)
        if mat and mat.exists():
            return ResourceResolution(path=mat, source="package")

    raise FileNotFoundError(
        f"Resource '{p}' not found. Tried work_dir/config_dir/project_root/package."
    )


def resolve_resource_maybe(
    name_or_path: str | Path,
    *,
    work_dir: str | Path | None = None,
    config_dir: str | Path | None = None,
    project_root: str | Path | None = None,
    allow_package: bool = True,
) -> ResourceResolution | None:
    try:
        return resolve_resource(
            name_or_path,
            work_dir=work_dir,
            config_dir=config_dir,
            project_root=project_root,
            allow_package=allow_package,
        )
    except Exception:
        return None
