from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from scorpio_pipe.benchmark.long_compare import build_scorpipe_observer_bundle, write_observer_bundle
from scorpio_pipe.benchmark.certification import load_registry
from scorpio_pipe.calib.common import save_json
from scorpio_pipe.config import load_config_any


DEFAULT_CASE_METRICS: dict[str, dict[str, Any]] = {
    "wavelength_solution.rms": {"label": "Wave RMS [A]", "prefer": "lower"},
    "sky_subtraction.sky_rms": {"label": "Sky residual RMS", "prefer": "lower"},
    "extraction.snr": {"label": "Extracted S/N", "prefer": "higher"},
}


def _relativize(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except Exception:
        try:
            return str(path.resolve().relative_to(base_dir.parent.resolve()))
        except Exception:
            return str(path)


def export_bundle_from_config(config_path: str | Path, output_path: str | Path) -> Path:
    cfg = load_config_any(config_path)
    run_root = Path(cfg.get("work_dir") or ".").expanduser().resolve()
    bundle = build_scorpipe_observer_bundle(cfg, run_root=run_root)
    return write_observer_bundle(output_path, bundle)


def build_case_manifest(
    *,
    case_id: str,
    title: str,
    category: str,
    manifest_path: str | Path,
    scorpipe_bundle: str | Path,
    long_bundle: str | Path,
    source_type: str = "real_private",
    description: str = "",
    observer_notes: str = "",
    metrics: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    manifest_p = Path(manifest_path).expanduser().resolve()
    base_dir = manifest_p.parent
    sp_path = Path(scorpipe_bundle).expanduser().resolve()
    lg_path = Path(long_bundle).expanduser().resolve()
    return {
        "case_id": str(case_id),
        "title": str(title),
        "category": str(category),
        "source_type": str(source_type),
        "description": str(description or ""),
        "observer_notes": str(observer_notes or ""),
        "scorpipe_bundle": _relativize(sp_path, base_dir),
        "long_bundle": _relativize(lg_path, base_dir),
        "metrics": json.loads(json.dumps(dict(metrics or DEFAULT_CASE_METRICS))),
    }


def write_case_manifest(path: str | Path, payload: Mapping[str, Any]) -> Path:
    return save_json(Path(path).expanduser().resolve(), dict(payload))


def upsert_registry_entry(
    *,
    registry_path: str | Path,
    case_id: str,
    category: str,
    manifest_path: str | Path,
    status: str = "ready",
) -> Path:
    registry_p = Path(registry_path).expanduser().resolve()
    if registry_p.exists():
        registry = load_registry(registry_p)
        registry_p = Path(str(registry["_path"]))
    else:
        registry = {
            "schema": "scorpio_pipe.long_parity_registry",
            "schema_version": 1,
            "minimum_real_cases_required": 8,
            "entries": [],
            "_path": str(registry_p),
        }
    entries = list(registry.get("entries") or [])
    rel_manifest = _relativize(Path(manifest_path).expanduser().resolve(), registry_p.parent)

    updated = False
    for idx, raw in enumerate(entries):
        entry = dict(raw or {})
        if str(entry.get("case_id") or "") != str(case_id):
            continue
        entry["category"] = str(category)
        entry["status"] = str(status)
        entry["manifest"] = rel_manifest
        entries[idx] = entry
        updated = True
        break
    if not updated:
        entries.append(
            {
                "case_id": str(case_id),
                "status": str(status),
                "manifest": rel_manifest,
                "category": str(category),
            }
        )

    clean = {k: v for k, v in registry.items() if not str(k).startswith("_")}
    clean["entries"] = entries
    return save_json(registry_p, clean)
