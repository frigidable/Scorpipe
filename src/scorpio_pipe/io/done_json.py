from __future__ import annotations

"""Unified stage/task completion marker (done.json).

This module enforces a small, stable schema used across *all* stages and
UI/pipeline tasks.

Contract (P0-PROV-002)
----------------------
- done.json is written on both success and failure.
- must contain: status, error_code, error_message, input_hashes,
  effective_config, outputs_list.

Notes
-----
Stages typically know only inputs/params/outputs and QC flags.
The GUI runner may later *upsert* additional provenance fields
(input hashes, expanded config, timing, hashes).
"""

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from scorpio_pipe.io.atomic import atomic_write_json


_STATUS_ALIASES = {
    "ok": "ok",
    "success": "ok",
    "pass": "ok",
    "warn": "warn",
    "warning": "warn",
    "failed": "fail",
    "fail": "fail",
    "error": "fail",
    "skipped": "skipped",
    "skip": "skipped",
    "blocked": "blocked",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "cancel": "cancelled",
}

_ALLOWED_STATUSES = {"ok", "warn", "fail", "skipped", "blocked", "cancelled"}


def _utc_now() -> str:
    # RFC3339-ish UTC string.
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _norm_status(status: str | None) -> str:
    s = str(status or "ok").strip().lower()
    s = _STATUS_ALIASES.get(s, s)
    if s not in _ALLOWED_STATUSES:
        raise ValueError(f"Invalid status {status!r}. Allowed: {sorted(_ALLOWED_STATUSES)}")
    return s


def _norm_scalar(v: Any) -> Any:
    # Keep JSON small and robust.
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    try:
        # pathlib.Path
        return str(v)
    except Exception:
        return repr(v)


def _norm_mapping(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            out[str(k)] = _norm_mapping(v)
        return out
    if isinstance(obj, (list, tuple, set)):
        return [_norm_mapping(x) for x in obj]
    return _norm_scalar(obj)


def _outputs_list_from_outputs(outputs: Any) -> list[str]:
    out: list[str] = []

    def _add(v: Any) -> None:
        if v is None:
            return
        if isinstance(v, (str, Path)):
            out.append(str(v))
            return
        if isinstance(v, Mapping):
            for vv in v.values():
                _add(vv)
            return
        if isinstance(v, (list, tuple, set)):
            for vv in v:
                _add(vv)

    _add(outputs)

    # de-dup while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def _norm_flags(flags: Sequence[Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f in flags or []:
        if isinstance(f, dict):
            out.append(dict(f))
        else:
            # tolerate Flag dataclass or any object with attributes
            d: dict[str, Any] = {}
            for k in ("code", "severity", "message", "hint"):
                if hasattr(f, k):
                    d[k] = getattr(f, k)
            if not d:
                d = {"message": str(f)}
            out.append(d)
    return out


def write_done_json(
    *,
    stage: str,
    stage_dir: str | Path,
    status: str = "ok",
    error_code: str | None = None,
    error_message: str | None = None,
    inputs: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
    outputs: Mapping[str, Any] | None = None,
    input_hashes: list[dict[str, Any]] | None = None,
    effective_config: Mapping[str, Any] | None = None,
    outputs_list: list[str] | None = None,
    metrics: Mapping[str, Any] | None = None,
    flags: Sequence[Any] | None = None,
    qc: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    error: Mapping[str, Any] | None = None,
    legacy_paths: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Write ``done.json`` into ``stage_dir`` and return the payload."""

    st = _norm_status(status)
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    fl = _norm_flags(flags)

    # Decide error code/message.
    ecode = error_code
    emsg = error_message
    if st == "fail":
        if not ecode:
            ecode = "STAGE_FAILED"
        if not emsg:
            if error and isinstance(error, Mapping) and error.get("message"):
                emsg = str(error.get("message"))
            elif fl:
                emsg = str(fl[0].get("message") or fl[0].get("code") or "stage failed")
            else:
                emsg = "stage failed"
    else:
        # For non-failure statuses, keep codes/messages optional.
        ecode = ecode if ecode else None
        emsg = emsg if emsg else None

    payload: dict[str, Any] = {
        "schema": "scorpio_pipe.done_json",
        "schema_version": 1,
        "stage": str(stage),
        "status": st,
        "error_code": ecode,
        "error_message": emsg,
        "created_utc": _utc_now(),
        "inputs": _norm_mapping(inputs or {}),
        "params": _norm_mapping(params or {}),
        "outputs": _norm_mapping(outputs or {}),
        # Filled by runner when available; keep keys always present.
        "input_hashes": list(input_hashes or []),
        "effective_config": _norm_mapping(effective_config or {}),
        "outputs_list": list(outputs_list or _outputs_list_from_outputs(outputs or {})),
        "flags": fl,
    }

    if metrics is not None:
        payload["metrics"] = _norm_mapping(metrics)
    if qc is not None:
        payload["qc"] = _norm_mapping(qc)
    if extra is not None:
        payload["extra"] = _norm_mapping(extra)
    if error is not None:
        payload["error"] = _norm_mapping(error)

    # Write canonical marker.
    atomic_write_json(stage_dir / "done.json", payload, indent=2, ensure_ascii=False)

    # Optional legacy mirrors (best-effort).
    for lp in legacy_paths or []:
        try:
            atomic_write_json(Path(lp), payload, indent=2, ensure_ascii=False)
        except Exception:
            pass

    return payload
