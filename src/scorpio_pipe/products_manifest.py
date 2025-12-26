"""Products manifest (machine-readable index of produced artifacts).

This is *not* the reproducibility/provenance manifest (see :mod:`scorpio_pipe.manifest`).

Goal
----
Provide a stable JSON index of produced files including per-exposure trees.
This is consumed by QC/UI and is useful for debugging and downstream automation.

The manifest is intentionally conservative: it doesn't try to list every
intermediate file ever created, but it should cover all "scientific" products
and the most important quicklooks.
"""


from __future__ import annotations


import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scorpio_pipe.work_layout import ensure_work_layout
from scorpio_pipe.paths import resolve_work_dir


def _rel(work_dir: Path, path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(work_dir))
    except Exception:
        return str(p)


def build_products_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    work_dir = resolve_work_dir(cfg)
    layout = ensure_work_layout(work_dir)

    prod = layout.products
    qc = layout.qc

    def _stage_tree(stage_dir: Path) -> dict[str, Any]:
        out: dict[str, Any] = {
            "dir": _rel(work_dir, stage_dir),
            "files": [],
            "per_exposure": [],
        }
        if not stage_dir.exists():
            return out

        # top-level stable files (fits/png/json)
        for ext in ("*.fits", "*.png", "*.json", "*.csv", "*.txt"):
            for p in sorted(stage_dir.glob(ext)):
                out["files"].append(_rel(work_dir, p))

        per = stage_dir / "per_exp"
        if per.exists():
            # Group per-exposure artifacts by tag (prefix before first underscore)
            tags: set[str] = set()
            for p in per.glob("*_*.fits"):
                tags.add(p.name.split("_", 1)[0])
            for p in per.glob("*_*.png"):
                tags.add(p.name.split("_", 1)[0])
            for p in per.glob("*_*.json"):
                tags.add(p.name.split("_", 1)[0])

            for tag in sorted(tags):
                files: list[str] = []
                for ext in ("*.fits", "*.png", "*.json", "*.csv", "*.txt"):
                    for p in sorted(per.glob(f"{tag}_" + ext[1:])):
                        files.append(_rel(work_dir, p))
                if files:
                    out["per_exposure"].append({"tag": tag, "files": files})
        return out

    payload: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "work_dir": str(work_dir),
        "products_root": _rel(work_dir, prod),
        "qc_root": _rel(work_dir, qc),
        "stages": {
            "linearize": _stage_tree(prod / "lin"),
            "sky": _stage_tree(prod / "sky"),
            "stack": _stage_tree(prod / "stack"),
            "spec": _stage_tree(prod / "spec"),
        },
    }

    # helpful pointers if present
    grid = (prod / "lin" / "wave_grid.json")
    if grid.exists():
        payload["stages"]["linearize"]["wave_grid_json"] = _rel(work_dir, grid)

    lin_qc = qc / "linearize_qc.json"
    if lin_qc.exists():
        payload["stages"]["linearize"]["qc_json"] = _rel(work_dir, lin_qc)

    return payload


def write_products_manifest(*, cfg: dict[str, Any], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_products_manifest(cfg)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
