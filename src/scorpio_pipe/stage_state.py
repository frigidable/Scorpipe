from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

from .version import get_provenance


def _is_numeric_scalar(v: Any) -> bool:
    # bool is intentionally included: toggles are meaningful for reruns.
    return isinstance(v, (int, float, bool))


def _numeric_only(obj: Any) -> Any:
    """Keep only numeric-ish values (int/float/bool) from nested configs.

    Variant A: stage is "dirty" only when numeric parameters change.
    We also keep bools (practically 0/1) because they affect behavior.
    """
    if obj is None:
        return None
    if _is_numeric_scalar(obj):
        # Normalize floats to a stable representation
        if isinstance(obj, float):
            return float(f"{obj:.12g}")
        return obj
    if isinstance(obj, (list, tuple)):
        out = [_numeric_only(v) for v in obj]
        out = [v for v in out if v is not None]
        return out
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k in sorted(obj.keys()):
            v = _numeric_only(obj[k])
            if v is not None:
                out[str(k)] = v
        return out
    return None


def _file_sig(p: Path) -> dict[str, Any]:
    try:
        st = p.stat()
        return {"path": str(p), "size": st.st_size, "mtime": int(st.st_mtime)}
    except Exception:
        return {"path": str(p), "missing": True}


def compute_stage_hash(
    *,
    stage: str,
    stage_cfg: dict[str, Any] | None = None,
    input_paths: Iterable[Path] | None = None,
) -> str:
    """Compute a stable hash for stage "up-to-date" checks.

    Variant A: stage becomes "dirty" only when numeric parameters and/or inputs change.
    We therefore keep only numeric-ish values from the stage config.
    """
    prov = get_provenance().__dict__
    stage_cfg_num = _numeric_only(stage_cfg or {})
    payload = {
        "pipeline": prov,
        "stage": stage,
        "stage_cfg": stage_cfg_num,
        "inputs": [_file_sig(Path(x)) for x in (input_paths or [])],
    }
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def is_stage_up_to_date(work_dir: Path, stage: str, expected_hash: str) -> bool:
    state = load_stage_state(work_dir)
    entry = get_stage_entry(state, stage)
    return bool(
        entry and entry.get("status") == "ok" and entry.get("hash") == expected_hash
    )


def record_stage_result(
    work_dir: Path,
    stage: str,
    *,
    status: str,
    stage_hash: str | None,
    message: str | None,
    trace: str | None,
) -> Path:
    state = load_stage_state(work_dir)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    entry = {
        "status": status,
        "hash": stage_hash,
        "message": message,
        "trace": trace,
        "updated_at": now,
    }
    set_stage_entry(state, stage, entry)
    return save_stage_state(work_dir, state)


def stage_state_path(work_dir: Path) -> Path:
    d = Path(work_dir) / "manifest"
    d.mkdir(parents=True, exist_ok=True)
    return d / "stage_state.json"


def load_stage_state(work_dir: Path) -> dict[str, Any]:
    p = stage_state_path(work_dir)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_stage_state(work_dir: Path, state: dict[str, Any]) -> Path:
    p = stage_state_path(work_dir)
    state.setdefault("pipeline", get_provenance().__dict__)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    p.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def get_stage_entry(state: dict[str, Any], stage: str) -> dict[str, Any] | None:
    return (state.get("stages") or {}).get(stage)


def set_stage_entry(state: dict[str, Any], stage: str, entry: dict[str, Any]) -> None:
    state.setdefault("stages", {})[stage] = entry
