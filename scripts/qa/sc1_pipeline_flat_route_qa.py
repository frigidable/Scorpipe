#!/usr/bin/env python3
"""Read-only routing inspection of a SCORPIO-1 config (not executed-stage proof)."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def inspect(config_path: Path) -> dict:
    import yaml
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict):
        raise ValueError("Expected YAML mapping")
    frame = cfg.get("frames") or {}
    flat = cfg.get("flatfield") or {}
    neon = cfg.get("superneon") or {}
    cal = cfg.get("calib") or {}
    if not all(isinstance(v, dict) for v in (frame, flat, neon, cal)):
        raise ValueError("Unexpected YAML sections")
    base = config_path.parent
    wd_raw = cfg.get("work_dir") or cfg.get("run_dir") or ""
    dd_raw = cfg.get("data_dir") or ""
    wd = Path(str(wd_raw)).expanduser() if wd_raw else base
    dd = Path(str(dd_raw)).expanduser() if dd_raw else base
    manifest_raw = cfg.get("dataset_manifest_path") or cfg.get("manifest_path")
    candidates = []
    if manifest_raw:
        candidates.append(Path(str(manifest_raw)).expanduser())
    candidates.extend((wd/"dataset_manifest.json", dd/"dataset_manifest.json"))
    manifest_paths = [str(p) for p in candidates]
    located = [str(p) for p in candidates if p.is_file()]
    manifest_active = bool(flat.get("use_manifest", True)) and bool(located)
    apply_to = flat.get("apply_to") or ["obj", "sky", "sunsky"]
    if not isinstance(apply_to, list):
        raise ValueError("flatfield.apply_to must be a list")
    counts = {}
    for role in ("obj", "bias", "flat", "neon", "sky", "sunsky", "standard"):
        items = frame.get(role) or []
        counts[role] = len(items) if isinstance(items, list) else None
    branch = "SKIPPED" if not bool(flat.get("enabled", False)) else (
        "MANIFEST_MASTERFLAT_FOR_OBJ" if manifest_active else "GLOBAL_SUPERFLAT_FOR_OBJ")
    need_global = (not manifest_active) or any(
        role != "obj" and counts.get(role, 0) for role in apply_to
    )
    return {
        "status": "CONFIG_INSPECTION_ONLY_NOT_RUNTIME_PROOF",
        "config_file": str(config_path),
        "flatfield_enabled": bool(flat.get("enabled", False)),
        "flatfield_use_manifest_requested": bool(flat.get("use_manifest", True)),
        "manifest_candidates": manifest_paths,
        "manifest_present": located,
        "object_flat_selection_if_flatfield_runs": branch,
        "global_superflat_required_by_flatfield_if_enabled": bool(need_global),
        "default_full_reduction_recipe_always_has_superflat_stage": True,
        "calib_bias_policy": str(cal.get("bias_policy", "degraded")),
        "superneon_bias_sub": bool(neon.get("bias_sub", True)),
        "selected_frame_counts_from_config": counts,
        "apply_to": apply_to,
        "caveat": "Executed recipe, stage skips/reuse and FITS provenance need checking run logs and outputs.",
    }


def selftest():
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as d:
        root = Path(d)
        cfg = root/"config.yaml"
        cfg.write_text(
            "frames:\n  obj: [s1.fits]\n  flat: [f1.fits]\n"
            "flatfield:\n  enabled: true\n  use_manifest: true\n",
            encoding="utf-8")
        result = inspect(cfg)
        assert result["object_flat_selection_if_flatfield_runs"] == "GLOBAL_SUPERFLAT_FOR_OBJ"
        (root/"dataset_manifest.json").write_text("{}", encoding="utf-8")
        result = inspect(cfg)
        assert result["object_flat_selection_if_flatfield_runs"] == "MANIFEST_MASTERFLAT_FOR_OBJ"
        cfg.write_text("frames: {}\nflatfield:\n  enabled: false\n", encoding="utf-8")
        result = inspect(cfg)
        assert result["object_flat_selection_if_flatfield_runs"] == "SKIPPED"
    print("SELFTEST PASSED")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: script.py --selftest | config.yaml")
    if sys.argv[1] == "--selftest":
        selftest()
    else:
        config = Path(sys.argv[1])
        if not config.is_file():
            raise FileNotFoundError(config)
        result = inspect(config)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        path = config.parent/"sc1_flat_route_config_inspection.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print("OUTPUT:", path)
