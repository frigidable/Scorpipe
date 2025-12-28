"""Persistent QC/metrics store.

The GUI already produces human-readable QC (HTML) and a stage state manifest.
For regression testing and future automation we also keep a single, stable
machine-readable summary under::

    work_dir/products/metrics.json

The intent is **not** to duplicate all intermediate files. Instead we store:
* per-stage status (ok/failed/skip),
* per-stage hash (so changes are traceable),
* a lightweight artifact table (exists/size/relative path),
* an optional metrics dict per stage.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scorpio_pipe.products import list_products, products_by_stage
from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.version import PIPELINE_VERSION
from scorpio_pipe.work_layout import ensure_work_layout
from scorpio_pipe.workspace_paths import stage_dir


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def metrics_path(work_dir: str | Path) -> Path:
    wd = Path(work_dir)
    layout = ensure_work_layout(wd)
    return layout.products / "metrics.json"


def load_metrics(work_dir: str | Path) -> dict[str, Any]:
    p = metrics_path(work_dir)
    if not p.exists():
        return {
            "schema": 1,
            "pipeline_version": PIPELINE_VERSION,
            "updated_at": _utc_now(),
            "stages": {},
        }
    try:
        js = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(js, dict):
            raise ValueError("metrics.json is not a dict")
        js.setdefault("schema", 1)
        js.setdefault("pipeline_version", PIPELINE_VERSION)
        js.setdefault("stages", {})
        return js
    except Exception:
        # Never break science because metrics got corrupted.
        return {
            "schema": 1,
            "pipeline_version": PIPELINE_VERSION,
            "updated_at": _utc_now(),
            "stages": {},
            "warning": "metrics.json was unreadable and has been reset",
        }


def save_metrics(work_dir: str | Path, metrics: Mapping[str, Any]) -> Path:
    wd = Path(work_dir)
    layout = ensure_work_layout(wd)
    p = layout.products / "metrics.json"
    out = dict(metrics)
    out["updated_at"] = _utc_now()
    out.setdefault("pipeline_version", PIPELINE_VERSION)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def _rel(work_dir: Path, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(work_dir.resolve())).replace("\\", "/")
    except Exception:
        return str(p)


def update_after_stage(
    cfg: Mapping[str, Any],
    *,
    stage: str,
    status: str,
    stage_hash: str | None,
    stage_metrics: Mapping[str, Any] | None = None,
) -> Path:
    """Update metrics.json after a stage execution.

    This is designed to be called from the GUI runner.
    """

    work_dir = resolve_work_dir(cfg)
    layout = ensure_work_layout(work_dir)
    metrics = load_metrics(layout.work_dir)

    # Artifact summary for this stage from the product registry.
    prods = products_by_stage(list_products(dict(cfg))).get(stage, [])
    artifacts: dict[str, Any] = {}
    for p in prods:
        artifacts[p.key] = {
            "path": _rel(layout.work_dir, p.path),
            "exists": bool(p.path.exists()),
            "size": p.size(),
            "kind": p.kind,
        }

    st = metrics.setdefault("stages", {}).setdefault(stage, {})
    st.update(
        {
            "status": str(status),
            "stage_hash": stage_hash,
            "updated_at": _utc_now(),
            "artifacts": artifacts,
        }
    )
    if stage_metrics is not None:
        # merge/update
        m = st.setdefault("metrics", {})
        if isinstance(m, dict):
            m.update(dict(stage_metrics))
        else:
            st["metrics"] = dict(stage_metrics)

    return save_metrics(layout.work_dir, metrics)


def mirror_qc_to_products(work_dir: str | Path) -> None:
    """Best-effort mirror from legacy work/qc into canonical products/NN_qc.

    Windows-friendly (no symlinks). Best-effort; never raises.
    """

    try:
        wd = Path(work_dir)
        layout = ensure_work_layout(wd)
        src = layout.qc
        dst = stage_dir(layout.work_dir, "qc_report")
        dst.mkdir(parents=True, exist_ok=True)
        if not src.is_dir():
            return
        for p in src.iterdir():
            if p.is_file():
                try:
                    (dst / p.name).write_bytes(p.read_bytes())
                except Exception:
                    pass
    except Exception:
        return
