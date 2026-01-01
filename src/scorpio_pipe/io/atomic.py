from __future__ import annotations

"""Small atomic write helpers.

Rationale
---------
GUI state (``ui/session.json``) and other small metadata files are written
frequently. A partial write (power loss, crash) can leave JSON corrupted.

We use an atomic temp-file + ``os.replace`` pattern within the same directory.

Note
----
We also provide a small, safe JSON encoder hook for pipeline metadata.
Stages occasionally place small dataclasses (ROI selections, version info,
paths, numpy scalars) into manifests. Standard ``json.dumps`` cannot encode
those objects.
"""

import dataclasses
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _json_default(o: Any) -> Any:
    """Best-effort serializer for small pipeline metadata JSON.

    This is intentionally conservative: metadata JSON should remain small.
    """

    # Dataclasses (e.g., ROISelection)
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)

    # Paths
    if isinstance(o, Path):
        return str(o)

    # Enums
    try:
        import enum

        if isinstance(o, enum.Enum):
            return o.value
    except Exception:
        pass

    # Numpy scalars / arrays (optional dependency)
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            # Avoid huge JSON by default; arrays shouldn't appear here.
            if o.size <= 1024:
                return o.tolist()
            return {
                "__ndarray__": True,
                "shape": list(o.shape),
                "dtype": str(o.dtype),
                "size": int(o.size),
            }
    except Exception:
        pass

    # Fallback: try to stringify (better than hard crash for manifests)
    return str(o)


def _atomic_replace(tmp_path: Path, dst_path: Path) -> None:
    # os.replace is atomic on POSIX and Windows when source and destination are
    # on the same filesystem.
    os.replace(str(tmp_path), str(dst_path))


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        _atomic_replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
    return path


def atomic_write_json(
    path: str | Path,
    obj: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    encoding: str = "utf-8",
) -> Path:
    s = json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, default=_json_default)
    return atomic_write_text(path, s, encoding=encoding)
