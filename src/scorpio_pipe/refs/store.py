"""Reference store helpers (CRDS-lite).

This module provides *deterministic* reference resolution and hashing.

Design goals
------------
* No "guessing": if a reference is requested and cannot be found, we raise.
* Deterministic resolution order.
* Package resources are materialized into a stable cache path (content-hash).

This is the lowest layer: it should stay small and have minimal dependencies.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any

from scorpio_pipe.app_paths import ensure_dir, user_cache_root
from scorpio_pipe.prov_capture import sha256_file


@dataclass(frozen=True)
class ReferenceResolution:
    """Resolved reference + provenance."""

    requested: str
    resolved_path: Path
    source: str  # absolute|resources_dir|work_dir|config_dir|project_root|package

    @property
    def exists(self) -> bool:
        try:
            return self.resolved_path.exists()
        except Exception:
            return False


def file_hash(path: str | Path) -> str:
    """Return SHA256 hex digest for a reference file.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    OSError
        If the file cannot be read.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Reference not found: {p}")
    return sha256_file(p)


def _utc_mtime(path: Path) -> str | None:
    try:
        st = path.stat()
        return datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _default_cache_dir() -> Path:
    return ensure_dir(user_cache_root("Scorpipe") / "resources")


def _materialize_package_resource(
    rel_path: str | Path, *, cache_dir: Path | None = None
) -> Path | None:
    """Copy a packaged resource into a stable cache path and return it.

    We cannot rely on :func:`importlib.resources.as_file` paths because in
    zipped/one-file deployments they may point to temporary locations.
    """

    cache_dir = cache_dir or _default_cache_dir()
    rel = str(rel_path).replace("\\", "/").lstrip("/")

    try:
        ref = resources.files("scorpio_pipe.resources") / rel
        if not ref.is_file():
            return None
        with resources.as_file(ref) as p:
            src = Path(p)
            data = src.read_bytes()
    except Exception:
        return None

    h = hashlib.md5(data).hexdigest()[:10]
    safe = rel.replace("/", "__")
    dst = cache_dir / f"{safe}.{h}"
    try:
        if not dst.exists():
            dst.write_bytes(data)
        return dst
    except Exception:
        return None


def resolve_reference(
    name_or_path: str | Path,
    *,
    resources_dir: str | Path | None = None,
    work_dir: str | Path | None = None,
    config_dir: str | Path | None = None,
    project_root: str | Path | None = None,
    allow_package: bool = True,
) -> ReferenceResolution:
    """Resolve a reference in a deterministic order.

    Resolution order for relative inputs:
      1) resources_dir / name
      2) work_dir / name
      3) config_dir / name
      4) project_root / name
      5) packaged resource scorpio_pipe/resources/<name>

    Raises
    ------
    FileNotFoundError
        If the reference cannot be found.
    """

    requested = str(name_or_path)
    p = Path(requested)

    if p.is_absolute():
        if p.exists():
            return ReferenceResolution(requested=requested, resolved_path=p, source="absolute")
        raise FileNotFoundError(f"Reference not found: {p}")

    rel = Path(str(p)).as_posix().lstrip("/")

    def _try(root: str | Path | None, src: str) -> Path | None:
        if not root:
            return None
        try:
            cand = (Path(root) / rel).expanduser().resolve()
            return cand if cand.exists() else None
        except Exception:
            return None

    for root, src in (
        (resources_dir, "resources_dir"),
        (work_dir, "work_dir"),
        (config_dir, "config_dir"),
        (project_root, "project_root"),
    ):
        cand = _try(root, src)
        if cand is not None:
            return ReferenceResolution(requested=requested, resolved_path=cand, source=src)

    if allow_package:
        mat = _materialize_package_resource(rel)
        if mat and mat.exists():
            return ReferenceResolution(requested=requested, resolved_path=mat, source="package")

    raise FileNotFoundError(
        "Reference '{name}' not found. Tried resources_dir/work_dir/config_dir/project_root/package.".format(
            name=rel
        )
    )


def reference_stat(path: Path) -> dict[str, Any]:
    """Return small, JSON-safe file stats for manifests."""

    try:
        st = path.stat()
        return {
            "size": int(st.st_size),
            "mtime_utc": _utc_mtime(path),
        }
    except Exception:
        return {"size": None, "mtime_utc": None}
