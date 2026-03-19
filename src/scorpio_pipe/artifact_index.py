"""Incremental artifact index written by the engine after each stage.

PERF-20.1 / PERF-72.1
---------------------
The UI should avoid expensive recursive scans (os.walk / rglob) of stage
directories. The engine already knows which products a stage owns and can
resolve them via :class:`~scorpio_pipe.catalog.products.ProductCatalog`.

Historically the index lived as a full JSON snapshot::

    manifest/artifact_index.json

Starting with schema v2 the canonical format is append-only JSONL::

    manifest/artifact_index.jsonl

Each stage update is appended as a single record, so stage completion does not
rewrite the whole index on large runs.

JSONL records
-------------
Header (written once):
    {"kind":"meta","schema":2,"format":"jsonl","created_at":"...Z"}

Stage upsert (append-only):
    {
      "kind":"stage",
      "schema":2,
      "updated_at":"...Z",
      "stage":"wavesolution",
      "entry":{"files":[...],"keyed":{...}}
    }

Notes
-----
- Paths are stored **relative to work_dir** when possible.
- "files" is a de-duplicated list of existing files produced by the stage,
  derived from the stage's declared product_keys.
- "keyed" maps product_key -> representative path (best effort).
- Reading merges records with "last stage record wins" semantics.
- Legacy ``artifact_index.json`` is still readable and is used only as a
  one-time migration source when a v2 JSONL index is first created.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG
from scorpio_pipe.catalog.stages import CATALOG as STAGE_CATALOG
from scorpio_pipe.work_layout import ensure_work_layout


ARTIFACT_INDEX_SCHEMA = 2


def artifact_index_path(work_dir: Path) -> Path:
    """Return the canonical append-only artifact index path under work_dir."""

    layout = ensure_work_layout(Path(work_dir))
    return Path(layout.manifest) / "artifact_index.jsonl"


def artifact_index_legacy_path(work_dir: Path) -> Path:
    """Return the legacy snapshot JSON path under work_dir."""

    layout = ensure_work_layout(Path(work_dir))
    return Path(layout.manifest) / "artifact_index.json"


def artifact_index_cache_stamp(work_dir: Path) -> tuple[Path, int]:
    """Return the best path + mtime pair for cache invalidation.

    Prefer the canonical JSONL store. Fall back to the legacy JSON snapshot.
    """

    p = artifact_index_path(Path(work_dir))
    if p.exists():
        try:
            return p, int(p.stat().st_mtime_ns)
        except Exception:
            return p, -1
    legacy = artifact_index_legacy_path(Path(work_dir))
    if legacy.exists():
        try:
            return legacy, int(legacy.stat().st_mtime_ns)
        except Exception:
            return legacy, -1
    return p, -1


def _now_utc_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(work_dir: Path, p: str | Path) -> str:
    pp = Path(str(p))
    try:
        return str(pp.resolve().relative_to(Path(work_dir).resolve())).replace("\\", "/")
    except Exception:
        return str(pp).replace("\\", "/")


def _abs(work_dir: Path, rel_or_abs: str | Path) -> Path:
    p = Path(str(rel_or_abs))
    if p.is_absolute():
        return p
    return (Path(work_dir) / p).resolve()


def _load_legacy_artifact_index(work_dir: Path) -> dict[str, Any] | None:
    p = artifact_index_legacy_path(Path(work_dir))
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def load_artifact_index(work_dir: Path) -> dict[str, Any] | None:
    wd = Path(work_dir)
    p = artifact_index_path(wd)
    if p.exists():
        stages: dict[str, Any] = {}
        updated_at = ""
        try:
            with p.open("r", encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    kind = str(rec.get("kind") or "").strip().lower()
                    if kind == "meta":
                        ts = str(rec.get("created_at") or rec.get("updated_at") or "").strip()
                        if ts:
                            updated_at = ts
                        continue
                    if kind != "stage":
                        continue
                    stage = str(rec.get("stage") or "").strip().lower()
                    entry = rec.get("entry")
                    if not stage or not isinstance(entry, dict):
                        continue
                    stages[stage] = entry
                    ts = str(rec.get("updated_at") or "").strip()
                    if ts:
                        updated_at = ts
        except Exception:
            return None
        return {
            "schema": ARTIFACT_INDEX_SCHEMA,
            "format": "jsonl",
            "updated_at": updated_at or _now_utc_z(),
            "stages": stages,
        }
    return _load_legacy_artifact_index(wd)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl_record(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except Exception:
            pass


def _stable_unique_str(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        s = str(x).strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def build_stage_entry(
    cfg: Mapping[str, Any],
    *,
    work_dir: Path,
    stage: str,
    stage_products: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a single stage entry (files + keyed)."""

    stage_key = str(stage or "").strip().lower()
    if not stage_key:
        return {"files": [], "keyed": {}}

    product_keys: tuple[str, ...] = ()
    try:
        ti = STAGE_CATALOG.get_task(stage_key)
        product_keys = tuple(getattr(ti, "product_keys", ()) or ())
    except Exception:
        product_keys = ()

    keyed: dict[str, str] = {}
    files: list[str] = []
    hint = dict(stage_products or {})

    for pk in product_keys:
        k = str(pk).strip()
        if not k:
            continue

        hp = hint.get(k)
        if hp:
            try:
                ap = _abs(work_dir, hp)
                if ap.exists():
                    rr = _rel(work_dir, ap)
                    keyed[k] = rr
                    files.append(rr)
            except Exception:
                pass

        try:
            found = PRODUCT_CATALOG.find_existing_paths(dict(cfg), k, run_root=work_dir)
        except Exception:
            found = []

        for p in found:
            try:
                if Path(p).exists():
                    files.append(_rel(work_dir, p))
            except Exception:
                pass

        if k not in keyed:
            try:
                bp = PRODUCT_CATALOG.best_path(dict(cfg), k, run_root=work_dir)
                if bp is not None and Path(bp).exists():
                    keyed[k] = _rel(work_dir, bp)
            except Exception:
                pass

    files = sorted(_stable_unique_str(files))
    return {"files": files, "keyed": keyed}


def _ensure_jsonl_index_bootstrapped(work_dir: Path) -> Path:
    """Ensure canonical JSONL store exists, migrating legacy snapshot once."""

    wd = Path(work_dir)
    p = artifact_index_path(wd)
    if p.exists() and p.stat().st_size > 0:
        return p

    _append_jsonl_record(
        p,
        {
            "kind": "meta",
            "schema": ARTIFACT_INDEX_SCHEMA,
            "format": "jsonl",
            "created_at": _now_utc_z(),
        },
    )

    legacy = _load_legacy_artifact_index(wd)
    stages = legacy.get("stages") if isinstance(legacy, dict) else None
    if isinstance(stages, dict):
        for stage, entry in stages.items():
            if not isinstance(entry, dict):
                continue
            _append_jsonl_record(
                p,
                {
                    "kind": "stage",
                    "schema": ARTIFACT_INDEX_SCHEMA,
                    "updated_at": str(legacy.get("updated_at") or _now_utc_z()),
                    "stage": str(stage).strip().lower(),
                    "entry": {
                        "files": list(entry.get("files") or []),
                        "keyed": dict(entry.get("keyed") or {}),
                    },
                    "source": "legacy_bootstrap",
                },
            )
    return p


def update_after_stage(
    cfg: Mapping[str, Any],
    *,
    work_dir: Path,
    stage: str,
    stage_products: Mapping[str, str] | None = None,
) -> Path:
    """Append a stage upsert into the canonical append-only artifact index."""

    wd = Path(work_dir)
    p = _ensure_jsonl_index_bootstrapped(wd)
    entry = build_stage_entry(cfg, work_dir=wd, stage=stage, stage_products=stage_products)
    _append_jsonl_record(
        p,
        {
            "kind": "stage",
            "schema": ARTIFACT_INDEX_SCHEMA,
            "updated_at": _now_utc_z(),
            "stage": str(stage).strip().lower(),
            "entry": entry,
        },
    )
    return p


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def list_stage_files(
    work_dir: Path,
    stage: str,
    *,
    only_existing: bool = True,
) -> list[Path]:
    """Return absolute file paths for a stage from artifact index."""

    idx = load_artifact_index(Path(work_dir))
    if not idx:
        return []
    stages = idx.get("stages") or {}
    if not isinstance(stages, dict):
        return []
    ent = stages.get(str(stage).strip().lower())
    if not isinstance(ent, dict):
        return []
    files = ent.get("files") or []
    if not isinstance(files, list):
        return []

    out: list[Path] = []
    for s in files:
        try:
            p = _abs(Path(work_dir), str(s))
            if only_existing and not p.exists():
                continue
            out.append(p)
        except Exception:
            continue
    out = sorted(out, key=lambda x: x.name)
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        sp = str(p)
        if sp in seen:
            continue
        seen.add(sp)
        uniq.append(p)
    return uniq


def list_files_for_tasks(
    work_dir: Path,
    tasks: Iterable[str],
    *,
    under: Path | None = None,
    only_existing: bool = True,
) -> list[Path]:
    """Return union of indexed files for multiple tasks."""

    wd = Path(work_dir)
    base = Path(under).resolve() if under is not None else None
    out: list[Path] = []
    for t in tasks:
        for p in list_stage_files(wd, str(t), only_existing=only_existing):
            if base is not None:
                try:
                    rp = p.resolve()
                    if base not in rp.parents and rp != base:
                        continue
                except Exception:
                    pass
            out.append(p)

    out = sorted(out, key=lambda x: x.name)
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        sp = str(p)
        if sp in seen:
            continue
        seen.add(sp)
        uniq.append(p)
    return uniq


def tasks_for_gui_stage(gui_stage_key: str) -> list[str]:
    """Best-effort mapping from GUI stage key to engine task key(s)."""

    try:
        gs = STAGE_CATALOG.get_gui_stage(gui_stage_key)
        sk = str(gs.key).strip().lower()
    except Exception:
        sk = str(gui_stage_key or "").strip().lower()

    out: list[str] = []
    for t in STAGE_CATALOG.task_infos():
        try:
            if str(getattr(t, "stage_key", "")).strip().lower() == sk:
                out.append(str(t.key).strip().lower())
        except Exception:
            continue
    return out
