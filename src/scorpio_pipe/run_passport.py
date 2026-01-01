"""Run passport (run.json).

Contract (P1-G / v5.40.6)
------------------------
Each run folder contains a lightweight "passport" file ``run.json``.
It is used by the GUI to:

- sort "Recent runs" by creation time;
- show stable run metadata;
- provide a single, canonical place to store the run identifier.

The passport is intentionally small and portable. It should *not* duplicate
stage outputs or large manifests.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scorpio_pipe.io.atomic import atomic_write_json
from scorpio_pipe.version import PIPELINE_VERSION


def passport_path(run_root: Path) -> Path:
    return Path(run_root) / "run.json"


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(p: Path) -> dict[str, Any] | None:
    try:
        if not p.exists():
            return None
        obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def read_run_passport(run_root: Path) -> dict[str, Any] | None:
    return _read_json(passport_path(run_root))


def _night_folder_from_path(run_root: Path) -> str:
    try:
        return Path(run_root).resolve().parent.name
    except Exception:
        return ""


def _night_date_from_night_dir(night_dir: str) -> str | None:
    """Convert night folder name to ISO date.

    Expected format: ``DD_MM_YYYY``.
    """

    s = str(night_dir or "").strip()
    try:
        import re

        m = re.match(r"^(\d{2})_(\d{2})_(\d{4})$", s)
        if not m:
            return None
        dd, mm, yyyy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if dd < 1 or dd > 31 or mm < 1 or mm > 12:
            return None
        return f"{yyyy:04d}-{mm:02d}-{dd:02d}"
    except Exception:
        return None


def _parse_object_disperser_from_name(run_root: Path) -> tuple[str | None, str | None]:
    """Best-effort parse of ``<object>_<disperser>_<NN>``."""

    name = Path(run_root).name
    if not name:
        return (None, None)
    parts = name.split("_")
    if len(parts) < 3:
        return (None, None)
    if not parts[-1].isdigit():
        return (None, None)
    # Assume last token is NN, token before is disperser, the rest is object.
    disp = parts[-2]
    obj = "_".join(parts[:-2])
    return (obj or None, disp or None)


def _parse_run_id_from_name(run_root: Path) -> int | None:
    name = Path(run_root).name
    # expected suffix: _NN
    try:
        if len(name) >= 3 and name[-3] == "_" and name[-2:].isdigit():
            return int(name[-2:])
    except Exception:
        pass
    return None


def _signature_from_config(cfg_path: Path) -> dict[str, Any] | None:
    try:
        from scorpio_pipe.workdir import signature_from_yaml

        sig = signature_from_yaml(cfg_path)
        if not sig:
            return None
        return {
            "object": sig.object_name,
            "disperser": sig.disperser,
            "slit": sig.slit,
            "binning": sig.binning,
        }
    except Exception:
        return None


def build_run_passport(
    run_root: Path,
    *,
    signature: dict[str, Any] | None = None,
    run_id: int | None = None,
    created_at: str | None = None,
    pipeline_version: str = PIPELINE_VERSION,
) -> dict[str, Any]:
    """Create a new passport payload.

    ``signature`` may contain instrument/setup information; it is treated as
    free-form JSON data.
    """

    run_root = Path(run_root)
    created = created_at or _now_utc()
    rid = int(run_id) if run_id is not None else _parse_run_id_from_name(run_root)

    night_dir = _night_folder_from_path(run_root)
    night_date = _night_date_from_night_dir(night_dir)

    obj = None
    disp = None
    if isinstance(signature, dict):
        obj = str(signature.get("object") or "").strip() or None
        disp = str(signature.get("disperser") or "").strip() or None
    if not obj or not disp:
        obj2, disp2 = _parse_object_disperser_from_name(run_root)
        obj = obj or obj2
        disp = disp or disp2

    rid_int: int | None = None
    if rid is not None:
        try:
            rid_int = int(rid)
        except Exception:
            rid_int = None
    rid_str = f"{rid_int:02d}" if isinstance(rid_int, int) else None

    # Contract (P1-G): minimal stable schema.
    payload: dict[str, Any] = {
        "schema": 1,
        "night_date": night_date,
        "night_dir": night_dir,
        "object": obj,
        "disperser": disp,
        "run_id": rid_str,
        "created_at": created,
        "pipeline_version": str(pipeline_version),
    }

    # Compatibility fields (read-only consumers may still expect them).
    payload["schema_version"] = 1
    if rid_int is not None:
        payload["run_id_int"] = int(rid_int)
    if signature:
        payload["signature"] = signature
    return payload


def ensure_run_passport(
    run_root: Path,
    *,
    signature: dict[str, Any] | None = None,
    run_id: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Ensure ``run.json`` exists.

    If the passport already exists, it is returned as-is unless ``overwrite=True``.
    If missing, we try to infer some fields from ``config.yaml``.
    """

    run_root = Path(run_root)
    p = passport_path(run_root)

    existing = _read_json(p)
    if existing and not overwrite:
        # Backfill minimal fields without changing created_at.
        changed = False
        if existing.get("schema") != 1:
            existing["schema"] = 1
            changed = True
        if "schema_version" not in existing:
            existing["schema_version"] = 1
            changed = True
        if "night_dir" not in existing:
            existing["night_dir"] = _night_folder_from_path(run_root)
            changed = True
        if "night_date" not in existing:
            existing["night_date"] = _night_date_from_night_dir(str(existing.get("night_dir") or ""))
            changed = True
        # Object/disperser (best effort): signature -> name parse.
        if not existing.get("object") or not existing.get("disperser"):
            sig = existing.get("signature") if isinstance(existing.get("signature"), dict) else None
            obj = str((sig or {}).get("object") or "").strip() if isinstance(sig, dict) else ""
            disp = str((sig or {}).get("disperser") or "").strip() if isinstance(sig, dict) else ""
            if not obj or not disp:
                obj2, disp2 = _parse_object_disperser_from_name(run_root)
                obj = obj or (obj2 or "")
                disp = disp or (disp2 or "")
            if obj:
                existing["object"] = obj
                changed = True
            if disp:
                existing["disperser"] = disp
                changed = True
        if existing.get("run_id") is None:
            rid = int(run_id) if run_id is not None else _parse_run_id_from_name(run_root)
            if rid is not None:
                existing["run_id"] = f"{int(rid):02d}"
                existing["run_id_int"] = int(rid)
                changed = True
        # If run_id is stored as int in legacy runs, add the string form.
        if isinstance(existing.get("run_id"), int) and "run_id_int" not in existing:
            try:
                existing["run_id_int"] = int(existing.get("run_id"))
                ri = int(existing.get("run_id_int"))
                existing["run_id"] = f"{ri:02d}"
                changed = True
            except Exception:
                pass
        if "pipeline_version" not in existing:
            existing["pipeline_version"] = str(PIPELINE_VERSION)
            changed = True
        if "signature" not in existing:
            sig = signature
            if sig is None:
                cfg_p = run_root / "config.yaml"
                if cfg_p.exists():
                    sig = _signature_from_config(cfg_p)
            if sig:
                existing["signature"] = sig
                changed = True
        if changed:
            try:
                atomic_write_json(p, existing, indent=2, ensure_ascii=False)
            except Exception:
                pass
        return existing

    # Build new passport
    if signature is None:
        cfg_p = run_root / "config.yaml"
        if cfg_p.exists():
            signature = _signature_from_config(cfg_p)

    payload = build_run_passport(run_root, signature=signature, run_id=run_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(p, payload, indent=2, ensure_ascii=False)
    return payload


def rewrite_run_passport_from_dir(
    run_root: Path, *, keep_created_at: bool = True
) -> dict[str, Any]:
    """Rewrite ``run.json`` so that it matches the folder name.

    Used by the GUI when the user opens a run directory whose ``run.json``
    diverges from ``workspace/<night>/<obj>_<disperser>_<run_id>/``.
    """

    run_root = Path(run_root)
    existing = read_run_passport(run_root) or {}
    created_at = str(existing.get("created_at")) if keep_created_at else None

    # Rebuild from folder name + (optional) signature extracted from config.yaml.
    sig: dict[str, Any] | None = None
    try:
        cfg_p = run_root / "config.yaml"
        if cfg_p.exists():
            sig = _signature_from_config(cfg_p)
    except Exception:
        sig = None

    payload = build_run_passport(
        run_root,
        signature=sig,
        run_id=None,
        created_at=created_at,
        pipeline_version=PIPELINE_VERSION,
    )
    p = passport_path(run_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(p, payload, indent=2, ensure_ascii=False)
    return payload


@dataclass(frozen=True)
class RunStamp:
    """Sortable timestamp helper."""

    sort_ts: float
    created_at: str | None


def get_run_stamp(run_root: Path) -> RunStamp:
    """Return (sort_ts, created_at) for Recent-runs ordering."""

    run_root = Path(run_root)
    p = passport_path(run_root)
    created: str | None = None
    ts: float | None = None
    data = _read_json(p)
    if data and isinstance(data.get("created_at"), str):
        created = str(data.get("created_at"))
        # parse ISO8601-ish '...Z'
        try:
            s = created.replace("Z", "+00:00")
            from datetime import datetime

            ts = datetime.fromisoformat(s).timestamp()
        except Exception:
            ts = None
    if ts is None:
        try:
            ts = run_root.stat().st_mtime
        except Exception:
            ts = 0.0
    return RunStamp(sort_ts=float(ts), created_at=created)
