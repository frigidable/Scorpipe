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
    work_dir_raw = _cfg_get(cfg, "work_dir", ".")
    config_dir_raw = _cfg_get(cfg, "config_dir", ".")
    work_dir = Path(str(work_dir_raw)).expanduser()
    config_dir = Path(str(config_dir_raw)).expanduser()

    if not config_dir.is_absolute():
        config_dir = config_dir.resolve()

    if not work_dir.is_absolute():
        work_dir = (config_dir / work_dir).resolve()
    else:
        work_dir = work_dir.resolve()

    return work_dir
