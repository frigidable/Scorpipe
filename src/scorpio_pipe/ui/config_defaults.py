"""UI helpers to obtain schema defaults for config keys."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from scorpio_pipe.schema import ConfigSchema


@lru_cache(maxsize=1)
def _default_cfg() -> ConfigSchema:
    # ``work_dir`` and ``data_dir`` are required by the schema.
    return ConfigSchema(work_dir=".", data_dir=".")


def schema_default(path: str) -> Any:
    """Return the default value for a dot-path in the resolved config.

    Examples:
      - "cosmics.k"
      - "sky.flexure.max_shift_pix"
    """
    obj: Any = _default_cfg()
    for token in (path or "").split("."):
        if not token:
            continue
        if isinstance(obj, dict):
            obj = obj.get(token)
        else:
            obj = getattr(obj, token)
    return obj
