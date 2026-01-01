from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    # pydantic v2
    if hasattr(cfg, "model_dump"):
        try:
            return cfg.model_dump().get(key, default)
        except Exception:
            pass
    # pydantic v1 / other
    if hasattr(cfg, "dict"):
        try:
            return cfg.dict().get(key, default)
        except Exception:
            pass
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except Exception:
            pass
    return getattr(cfg, key, default)


def resolve_work_dir(cfg: Any) -> Path:
    """Resolve work_dir to an absolute path.

    If cfg['work_dir'] is relative, it is resolved relative to cfg['config_dir'].
    """

    # Many modern callers (especially GUI + P2 tests) pass the workspace root
    # under ``cfg['workspace']['root']`` while keeping ``work_dir`` unset.
    # Treat that as the base directory for resolving relative paths.
    ws = _cfg_get(cfg, "workspace", None)
    ws_root: str | None = None
    if isinstance(ws, Mapping):
        for k in ("root", "work_dir", "run_root", "path"):
            v = ws.get(k)
            if isinstance(v, (str, Path)) and str(v).strip():
                ws_root = str(v)
                break

    work_dir_raw = _cfg_get(cfg, "work_dir", None)
    config_dir_raw = _cfg_get(cfg, "config_dir", None)

    # If config_dir is not provided (or is a trivial "." placeholder), but the
    # workspace root is known, resolve relative work_dir against that.
    if config_dir_raw is None or str(config_dir_raw).strip() in {"", ".", "./"}:
        config_dir_raw = ws_root or "."
    if work_dir_raw is None or str(work_dir_raw).strip() == "":
        work_dir_raw = "."
    work_dir = Path(str(work_dir_raw)).expanduser()
    config_dir = Path(str(config_dir_raw)).expanduser()

    if not config_dir.is_absolute():
        config_dir = config_dir.resolve()

    if not work_dir.is_absolute():
        work_dir = (config_dir / work_dir).resolve()
    else:
        work_dir = work_dir.resolve()

    return work_dir
