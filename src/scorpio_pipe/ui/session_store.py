"""UI session persistence.

Contract (P1-G / UI-020)
-----------------------
We store GUI session state in:

    <run_root>/ui/session.json

and keep lightweight history snapshots in:

    <run_root>/ui/history/session_<timestamp>__<reason>.json

This is intentionally independent from the pipeline hash/state machine.
It is meant for *user experience* (restore UI state after restart and
provide a 'Previous state' rollback).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from scorpio_pipe.io.atomic import atomic_write_json, atomic_write_text


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def session_path(run_root: Path) -> Path:
    return Path(run_root) / "ui" / "session.json"


_STAGE_KEY_MAP: dict[str, str] = {
    # Canonical keys used by the UI session contract (P1-G / UI-020)
    "wavesolution": "wavesol",
    "wavesol": "wavesol",
    "sky": "sky_subtraction",
    "sky_sub": "sky_subtraction",
    "sky_subtraction": "sky_subtraction",
    "linearize": "linearization",
    "linearization": "linearization",
    "stack": "frame_stacking",
    "stack2d": "frame_stacking",
    "frame_stacking": "frame_stacking",
    "extract": "object_extraction",
    "extract1d": "object_extraction",
    "object_extraction": "object_extraction",
}


def _canon_stage_key(k: str) -> str:
    kk = str(k or "").strip().lower()
    return _STAGE_KEY_MAP.get(kk, kk or "unknown")


def _ensure_contract(payload: dict[str, Any], run_root: Path) -> dict[str, Any]:
    """Ensure both the historical UI schema and the P1-G contract keys exist.

    We keep backward compatibility for already-shipped keys:
      - schema_version / stages / history

    and also maintain the P1-G contract keys:
      - schema / workspace_root / active_run_dir / stage_state
    """

    payload = dict(payload or {})

    # --- legacy schema keys ---
    if payload.get("schema_version") != 1:
        payload["schema_version"] = 1
    payload.setdefault("created_at", _now_utc())
    payload.setdefault("updated_at", payload.get("created_at"))
    payload.setdefault("stages", {})
    payload.setdefault("history", [])

    # --- contract schema keys ---
    if payload.get("schema") != 1:
        payload["schema"] = 1
    # A conservative default: assume workspace_root is the parent of the night folder.
    payload.setdefault("workspace_root", "")
    # Store run_root as a relative hint when possible.
    payload.setdefault("active_run_dir", str(Path(run_root).name))
    payload.setdefault("stage_state", {})

    # Migrate from legacy "stages" into contract "stage_state" when missing.
    if isinstance(payload.get("stages"), dict) and isinstance(payload.get("stage_state"), dict):
        stage_state = payload["stage_state"]
        for k, v in payload["stages"].items():
            if not isinstance(v, dict):
                continue
            ck = _canon_stage_key(k)
            if ck in stage_state:
                continue
            stage_state[ck] = {
                "method": None,
                "cleanup_mode": None,
                "roi": None,
                "params": dict(v.get("params") or {}) if isinstance(v.get("params"), dict) else {},
                "last_done_path": v.get("last_done"),
                "last_status": v.get("status"),
            }

    return payload


def _read_json(p: Path) -> dict[str, Any] | None:
    try:
        if not p.exists():
            return None
        obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def load_session(run_root: Path) -> dict[str, Any]:
    run_root = Path(run_root)
    p = session_path(run_root)
    payload = _read_json(p) or {}
    return _ensure_contract(payload, run_root)


def save_session(run_root: Path, payload: dict[str, Any]) -> None:
    run_root = Path(run_root)
    p = session_path(run_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = _ensure_contract(dict(payload or {}), run_root)
    payload["schema_version"] = 1
    payload["schema"] = 1
    payload.setdefault("created_at", _now_utc())
    payload["updated_at"] = _now_utc()
    # Atomic write: protects against partially-written JSON on crash/power loss.
    atomic_write_json(p, payload, indent=2, ensure_ascii=False)


def update_stage(
    run_root: Path,
    stage: str,
    *,
    cfg_section: dict[str, Any] | None = None,
    done_json_rel: str | None = None,
    status: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """Update a single stage entry in the session."""

    payload = load_session(run_root)
    stages = payload.get("stages") if isinstance(payload.get("stages"), dict) else {}
    payload["stages"] = stages

    st = stages.get(stage) if isinstance(stages.get(stage), dict) else {}
    st = dict(st)
    if cfg_section is not None:
        st["params"] = cfg_section
    if done_json_rel is not None:
        st["last_done"] = str(done_json_rel)
    if status is not None:
        st["status"] = str(status)
    if message is not None:
        st["message"] = str(message)
    st["updated_at"] = _now_utc()
    stages[stage] = st

    # Contract stage_state mirror
    stage_state = payload.get("stage_state") if isinstance(payload.get("stage_state"), dict) else {}
    payload["stage_state"] = stage_state
    ck = _canon_stage_key(stage)
    ss = stage_state.get(ck) if isinstance(stage_state.get(ck), dict) else {}
    ss = dict(ss)
    ss.setdefault("method", None)
    ss.setdefault("cleanup_mode", None)
    ss.setdefault("roi", None)
    if cfg_section is not None:
        ss["params"] = dict(cfg_section)
        # Best-effort: store common method fields.
        if ck == "sky_subtraction":
            try:
                ss["method"] = str((cfg_section or {}).get("method") or ss.get("method") or "") or None
            except Exception:
                pass
        if ck == "linearization":
            try:
                ss["cleanup_mode"] = str(((cfg_section or {}).get("cleanup") or {}).get("mode") or ss.get("cleanup_mode") or "") or None
            except Exception:
                pass
    if done_json_rel is not None:
        ss["last_done_path"] = str(done_json_rel)
    if status is not None:
        ss["last_status"] = str(status)
    stage_state[ck] = ss

    save_session(run_root, payload)
    return payload


def _slug_reason(reason: str) -> str:
    r = (reason or "snapshot").strip().lower()
    r = re.sub(r"\s+", "_", r)
    r = re.sub(r"[^0-9a-zA-Z_\-]+", "", r)
    return (r or "snapshot")[:48]


def snapshot(
    run_root: Path,
    *,
    reason: str,
    cfg_path: Path | None = None,
    keep_last: int = 30,
) -> Path:
    """Write a snapshot file and register it in session history."""

    run_root = Path(run_root)
    ts = _now_utc().replace(":", "").replace("-", "")
    rslug = _slug_reason(reason)
    hist_dir = run_root / "ui" / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    snap_p = hist_dir / f"session_{ts}__{rslug}.json"

    sess = load_session(run_root)
    cfg_text: str | None = None
    if cfg_path is None:
        cfg_path = run_root / "config.yaml"
    try:
        if cfg_path.exists():
            cfg_text = cfg_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        cfg_text = None

    snap_payload = {
        "schema_version": 1,
        "schema": 1,
        "timestamp": _now_utc(),
        "reason": reason,
        "config_yaml": cfg_text,
        "session": sess,
    }
    # Atomic write: history snapshots should also never be partially written.
    atomic_write_json(snap_p, snap_payload, indent=2, ensure_ascii=False)

    # Register into history (latest-first)
    hist = sess.get("history") if isinstance(sess.get("history"), list) else []
    hist = [x for x in hist if isinstance(x, dict)]
    # P1-G contract requires history entries to contain stage_state snapshots.
    stage_state_snapshot = sess.get("stage_state") if isinstance(sess.get("stage_state"), dict) else {}
    hist.insert(
        0,
        {
            "timestamp": snap_payload["timestamp"],
            "reason": reason,
            "path": str(snap_p.relative_to(run_root)),
            "stage_state_snapshot": stage_state_snapshot,
        },
    )
    # Deduplicate by path
    seen: set[str] = set()
    dedup: list[dict[str, Any]] = []
    for x in hist:
        p = str(x.get("path", ""))
        if not p or p in seen:
            continue
        seen.add(p)
        dedup.append(x)
    sess["history"] = dedup[: max(0, int(keep_last))]
    save_session(run_root, sess)

    # Prune files on disk
    try:
        keep_paths = {str((run_root / str(h.get("path"))).resolve()) for h in sess["history"]}
        for p in hist_dir.glob("session_*.json"):
            if str(p.resolve()) not in keep_paths:
                try:
                    p.unlink()
                except Exception:
                    pass
    except Exception:
        pass

    return snap_p


def list_snapshots(run_root: Path, *, limit: int = 10) -> list[dict[str, Any]]:
    sess = load_session(run_root)
    hist = sess.get("history") if isinstance(sess.get("history"), list) else []
    out = [x for x in hist if isinstance(x, dict)]
    return out[: max(0, int(limit))]


def restore_snapshot(run_root: Path, snapshot_rel: str) -> bool:
    """Restore config.yaml + session.json from a history snapshot."""

    run_root = Path(run_root)
    snap_p = run_root / str(snapshot_rel)
    data = _read_json(snap_p)
    if not data:
        return False
    cfg_text = data.get("config_yaml") if isinstance(data.get("config_yaml"), str) else None
    session_obj = data.get("session") if isinstance(data.get("session"), dict) else None
    ok = False
    try:
        if cfg_text is not None:
            atomic_write_text(run_root / "config.yaml", cfg_text, encoding="utf-8")
            ok = True
    except Exception:
        pass
    try:
        if session_obj is not None:
            save_session(run_root, session_obj)
            ok = True
    except Exception:
        pass
    return ok
