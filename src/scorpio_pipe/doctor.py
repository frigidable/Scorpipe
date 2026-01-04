from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from scorpio_pipe.config import load_config
from scorpio_pipe.resource_utils import resolve_resource, resolve_resource_maybe
from scorpio_pipe.schema import find_unknown_keys, schema_validate
from scorpio_pipe.validation import validate_config
from scorpio_pipe.workspace_paths import stage_dir


def _check_optional_imports(modules: List[str]) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    for m in modules:
        try:
            __import__(m)
        except Exception:
            missing.append(m)
    return (len(missing) == 0), missing


def _ensure_dirs(cfg: Dict[str, Any]) -> List[str]:
    """Create key directories (safe). Returns list of created dirs."""
    created: List[str] = []
    work_dir = Path(str(cfg.get("work_dir", "work"))).expanduser().resolve()
    for d in [
        work_dir,
        work_dir / "report",
        work_dir / "calib",
        stage_dir(work_dir, "cosmics"),
        work_dir / "cosmics",  # legacy
    ]:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(str(d))
    return created


def _find_unique_by_basename(root: Path, basename: str) -> Optional[Path]:
    """Search root (shallow + recursive) for a unique file with given basename."""
    if not root.exists():
        return None

    # Prefer shallow matches
    shallow = [p for p in root.glob(basename) if p.is_file()]
    if len(shallow) == 1:
        return shallow[0]

    # Recursive search (can be slower, but safe enough for doctor)
    matches = []
    try:
        for p in root.rglob(basename):
            if p.is_file():
                matches.append(p)
            if len(matches) > 5:
                break
    except Exception:
        return None

    if len(matches) == 1:
        return matches[0]
    return None


def _maybe_patch_frame_paths(
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Attempt to patch missing frame paths using data_dir basename matching.

    Returns (new_cfg, edits).
    """
    data_dir = Path(str(cfg.get("data_dir", ""))).expanduser()
    frames = cfg.get("frames") if isinstance(cfg.get("frames"), dict) else {}

    if not isinstance(frames, dict) or not data_dir:
        return cfg, []

    edits: List[Dict[str, str]] = []
    out = dict(cfg)
    out_frames = dict(frames)

    for kind in ("bias", "flat", "neon", "obj", "sky"):
        arr = out_frames.get(kind)
        if arr is None:
            continue
        if isinstance(arr, str):
            arr = [arr]
        if not isinstance(arr, list):
            continue

        new_list: List[str] = []
        changed = False
        for item in arr:
            s = str(item)
            p = Path(s)
            # resolve relative to data_dir if not absolute
            if not p.is_absolute():
                p = (data_dir / p).resolve()
            if p.exists():
                new_list.append(str(p))
                continue

            cand = _find_unique_by_basename(data_dir, Path(s).name)
            if cand is not None and cand.exists():
                new_list.append(str(cand.resolve()))
                changed = True
                edits.append({"kind": kind, "from": s, "to": str(cand.resolve())})
            else:
                new_list.append(str(p))

        if changed:
            out_frames[kind] = new_list

    out["frames"] = out_frames
    return out, edits


def run_doctor(
    *, config_path: str | Path | None = None, fix: bool = False
) -> Dict[str, Any]:
    """Run environment + config diagnostics.

    Parameters
    ----------
    config_path : path to config.yaml (optional)
    fix : if True, apply safe autofixes (create dirs, materialize resources, write patched config)

    Returns a JSON-serializable report.
    """

    report: Dict[str, Any] = {
        "cwd": os.getcwd(),
        "python": {
            "executable": os.sys.executable,
            "version": os.sys.version,
        },
        "gui": {},
        "resources": [],
        "config": None,
        "fixes": [],
    }

    # Optional GUI deps
    ok, missing = _check_optional_imports(["PySide6", "pyqtgraph", "fitz"])
    report["gui"] = {
        "ok": ok,
        "missing": missing,
        "hint": "pip install scorpio-pipe[gui]" if not ok else "",
    }

    # Resources
    for name in ("neon_lines.csv", "HeNeAr_atlas.pdf"):
        r = resolve_resource_maybe(name, allow_package=True)
        if r is None and fix:
            try:
                resolve_resource(name, allow_package=True, prefer_cache=True)
            except Exception:
                pass
            r = resolve_resource_maybe(name, allow_package=True)

        report["resources"].append(
            {
                "name": name,
                "found": r is not None,
                "path": None if r is None else str(r.path),
                "source": None if r is None else str(r.source),
            }
        )

    if not config_path:
        return report

    cfg_path = Path(config_path).expanduser().resolve()
    try:
        cfg = load_config(cfg_path)
    except Exception as e:
        report["config"] = {"path": str(cfg_path), "load_error": str(e)}
        return report

    # Schema/type validation
    schema = schema_validate(cfg)
    report_config: Dict[str, Any] = {
        "path": str(cfg_path),
        "schema": {
            "ok": schema.ok,
            "errors": [vars(x) for x in schema.errors],
            "warnings": [vars(x) for x in schema.warnings],
            "unknown_keys": find_unknown_keys(cfg),
        },
    }

    # Full pipeline validation (path checks are optionally strict)
    vrep = validate_config(cfg, strict_paths=False)
    report_config["validate"] = {
        "ok": vrep.ok,
        "errors": [vars(x) for x in vrep.errors],
        "warnings": [vars(x) for x in vrep.warnings],
    }

    report["config"] = report_config

    if fix:
        created = _ensure_dirs(cfg)
        if created:
            report["fixes"].append({"action": "mkdir", "paths": created})

        patched_cfg, edits = _maybe_patch_frame_paths(cfg)
        if edits:
            patched_path = cfg_path.with_suffix(".patched.yaml")
            try:
                patched_path.write_text(
                    yaml.safe_dump(patched_cfg, sort_keys=False, allow_unicode=True),
                    encoding="utf-8",
                )
                report["fixes"].append(
                    {
                        "action": "write_patched_config",
                        "path": str(patched_path),
                        "edits": edits,
                    }
                )
            except Exception as e:
                report["fixes"].append(
                    {"action": "write_patched_config", "error": str(e), "edits": edits}
                )

        # Write a doctor.json report into work_dir/report when possible
        try:
            wd = Path(str(cfg.get("work_dir", "work"))).expanduser().resolve()
            out = wd / "report" / "doctor.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            report["fixes"].append({"action": "write_doctor_report", "path": str(out)})
        except Exception:
            pass

    return report
