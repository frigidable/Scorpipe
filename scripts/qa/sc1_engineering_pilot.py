#!/usr/bin/env python3
"""One-off N2146sat engineering pilot: isolated config and canonical engine.

This is a DEVELOPMENT SMOKE RUN, not scientific validation. It does not
authorize BIASSEC/DATASEC, sky subtraction, wavelength accuracy, or publication.

Preparation uses a NEW empty run directory under the same observing-night
parent; source FITS remain referenced by ABSOLUTE paths and are never copied
or edited. The original config + pre-existing four MasterBias artifacts are
SHA256-guarded before/after the run.

Usage:
  python scripts/qa/sc1_engineering_pilot.py prepare source_config.yaml NEW_run_dir
  python scripts/qa/sc1_engineering_pilot.py run NEW_run_dir/config.yaml

Never use --force, QC override, or preflight override. Stops on first non-OK.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


EXPECTED = {"bias": 20, "obj": 7, "flat": 3, "neon": 2}
STAGES = (
    "manifest", "superbias", "superneon",
    "lineid_prepare", "arclineid", "wavesolution", "lsf_model",
    "sky", "sky_refine", "linearize", "stack2d", "extract1d",
    "qc_report",
)
ARTIFACTS = (
    "03_bias/master_bias.fits",
    "03_bias/master_bias_var.fits",
    "03_bias/master_bias_dq.fits",
    "03_bias/master_bias_index.json",
)
MANIFEST = "sc1_engineering_pilot_manifest.json"
REPORT = "sc1_engineering_pilot_result.json"


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_sources(source_cfg: Path) -> dict[str, str | None]:
    work = source_cfg.parent
    files = [source_cfg] + [work / item for item in ARTIFACTS]
    return {
        str(p): checksum(p) if p.is_file() else None
        for p in files
    }


def check_expected_frames(resolved: dict) -> None:
    entries = resolved.get("frames") or {}
    for kind, count in EXPECTED.items():
        paths = entries.get(kind) or []
        if not isinstance(paths, list) or len(paths) != count:
            raise RuntimeError(f"Expected {count} selected {kind} inputs")
        if len({str(x).lower() for x in paths}) != count:
            raise RuntimeError(f"Duplicate selected {kind} inputs")
        for raw in paths:
            if not Path(str(raw)).is_file():
                raise FileNotFoundError(f"Missing selected {kind}: {raw}")
    names = [Path(str(v)).name.lower() for v in entries["bias"]]
    expected_bias = {f"s228001{i:02d}.fts" for i in range(1, 21)}
    if set(names) != expected_bias:
        raise RuntimeError("Source config does not use audited early20 BIAS")


def check_pilot_destination(source_cfg: Path, new_dir: Path) -> None:
    if new_dir.exists():
        raise FileExistsError(f"Refusing existing engineering run: {new_dir}")
    source_work = source_cfg.parent
    if new_dir.parent != source_work.parent:
        raise ValueError("Pilot must be a new sibling under the SAME night")
    if not re.fullmatch(r".+_[0-9]{2}", new_dir.name):
        raise ValueError("Pilot run directory must use canonical _NN suffix")
    if new_dir == source_work:
        raise ValueError("Refusing original run as engineering pilot")


def prepare(config_path: str, new_directory: str) -> None:
    import yaml
    from scorpio_pipe.config import load_config

    source = Path(config_path).expanduser().resolve(strict=True)
    if source.name != "config.yaml":
        raise ValueError("Expected original run config.yaml")
    new_dir = Path(new_directory).expanduser().resolve()
    check_pilot_destination(source, new_dir)

    # Resolve frame paths BEFORE writing the isolated config.
    cfg = load_config(source)
    if Path(str(cfg.get("work_dir") or "")).resolve() != source.parent:
        raise RuntimeError("Original config work_dir mismatches original folder")
    check_expected_frames(cfg)
    if (cfg.get("calib") or {}).get("bias_policy") != "strict":
        raise RuntimeError("Original config must have bias_policy=strict")
    if bool((cfg.get("flatfield") or {}).get("enabled", False)):
        raise RuntimeError("Original config must have flatfield.enabled=false")
    if bool((cfg.get("superneon") or {}).get("bias_sub", True)):
        raise RuntimeError("Original config must have superneon.bias_sub=false")

    raw = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise TypeError("Source config is not a mapping")
    raw["work_dir"] = str(new_dir)
    raw["data_dir"] = str(cfg.get("data_dir") or "")
    for key, vals in (cfg.get("frames") or {}).items():
        if isinstance(vals, list):
            raw.setdefault("frames", {})[key] = [str(v) for v in vals]
    for key in ("resources_dir",):
        val = raw.get(key)
        if val and not Path(str(val)).is_absolute():
            raw[key] = str((source.parent / str(val)).resolve())
    calib = raw.setdefault("calib", {})
    calib["bias_policy"] = "strict"
    calib["bias_combine"] = "median"
    calib["bias_sigma_clip"] = 0.0
    calib["bias_gross_reject_adu"] = 50.0
    # Avoid accidentally loading an existing calibration in the source run.
    calib["superbias_path"] = "03_bias/master_bias.fits"
    if calib.get("superflat_path"):
        calib["superflat_path"] = "04_flat/superflat.fits"
    raw.setdefault("flatfield", {})["enabled"] = False
    raw.setdefault("superneon", {})["bias_sub"] = False

    # Do not carry path strings to original science PRODUCTS into the new run.
    # Reading raw FITS at their unchanged absolute paths is allowed.
    source_work_string = str(source.parent).lower()
    raw_data = str(Path(str(raw["data_dir"])).resolve()).lower()
    if source_work_string == raw_data:
        raise RuntimeError("Raw data_dir unexpectedly equals source work_dir")
    def reject_old_run_product_refs(value: object, location: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                # Existing RAW exposures may legitimately live under the
                # source run; they are only ever opened for reading.
                if not location and key == "frames":
                    continue
                reject_old_run_product_refs(child, f"{location}.{key}" if location else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                reject_old_run_product_refs(child, f"{location}[{index}]")
        elif isinstance(value, str) and source_work_string in value.lower():
            raise RuntimeError(f"Unsafe original-run reference: {location}")

    reject_old_run_product_refs(raw)

    before = fingerprint_sources(source)
    new_dir.mkdir(parents=False)
    new_config = new_dir / "config.yaml"
    with new_config.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(raw, stream, sort_keys=False, allow_unicode=True)
    record = {
        "status": "PREPARED_ENGINEERING_ONLY_SCIENCE_HOLD",
        "source_config": str(source),
        "original_protected_sha256": before,
        "pilot_config": str(new_config),
        "work_dir": str(new_dir),
        "optional_gross_reject_adu": 50.0,
        "stage_plan": list(STAGES),
        "selected_raw_paths": {
            kind: [str(p) for p in cfg["frames"][kind]]
            for kind in EXPECTED
        },
        "flatfield_enabled": False,
        "superneon_bias_sub": False,
        "science_e2e_approved": False,
    }
    with (new_dir / MANIFEST).open("x", encoding="utf-8") as output:
        json.dump(record, output, indent=2, ensure_ascii=False)
    print("ENGINEERING PILOT PREPARED:", new_config, flush=True)
    print("NO PIPELINE STAGES RUN; SCIENCE E2E HOLD", flush=True)


def run(pilot_config: str) -> None:
    from scorpio_pipe.pipeline.public_api import run_sequence_locked, canonical_task_name

    config = Path(pilot_config).expanduser().resolve(strict=True)
    if config.name != "config.yaml":
        raise ValueError("Expected isolated config.yaml")
    work = config.parent
    manifest_file = work / MANIFEST
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if Path(manifest["pilot_config"]).resolve() != config:
        raise RuntimeError("Pilot config does not match recorded manifest")
    source = Path(manifest["source_config"]).resolve(strict=True)
    if work == source.parent or work.parent != source.parent.parent:
        raise RuntimeError("Pilot would run in original workspace")
    from scorpio_pipe.config import load_config
    cfg = load_config(config)
    if Path(str(cfg["work_dir"])).resolve() != work:
        raise RuntimeError("Pilot work_dir does not match config location")
    check_expected_frames(cfg)
    if tuple(manifest.get("stage_plan") or ()) != STAGES:
        raise RuntimeError("Pilot stage plan differs from the prepared manifest")
    for task in STAGES:
        if canonical_task_name(task) != task:
            raise RuntimeError(f"Noncanonical/aliased pilot task: {task}")
    for kind in EXPECTED:
        prepared = manifest.get("selected_raw_paths", {}).get(kind)
        current = [str(p) for p in cfg["frames"][kind]]
        if current != prepared:
            raise RuntimeError(f"Selected {kind} RAW files changed after preparation")
    if (cfg.get("calib") or {}).get("bias_policy") != "strict":
        raise RuntimeError("Pilot bias_policy must remain strict")
    if bool((cfg.get("flatfield") or {}).get("enabled", False)):
        raise RuntimeError("Pilot flatfield must remain disabled")
    if bool((cfg.get("superneon") or {}).get("bias_sub", True)):
        raise RuntimeError("Pilot superneon bias_sub must remain disabled")
    if (cfg.get("calib") or {}).get("bias_gross_reject_adu") != 50.0:
        raise RuntimeError("Pilot gross rejection no longer matches manifest")
    if not all(
        checksum(Path(p)) == digest
        if digest is not None else not Path(p).exists()
        for p, digest in manifest["original_protected_sha256"].items()
    ):
        raise RuntimeError("Original files changed since pilot preparation")
    if (work / REPORT).exists():
        raise FileExistsError("Pilot report exists; refusing second execution")

    output = {
        "status": "ENGINEERING_RUN_INCOMPLETE_SCIENCE_HOLD",
        "config": str(config),
        "stage_plan": list(STAGES),
        "completed_stages": [],
        "stopped_at": None,
        "error": None,
        "qc_override": False,
        "preflight_override": False,
        "force": False,
        "original_files_unchanged": None,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        for name in STAGES:
            print("ENGINEERING STAGE:", name, flush=True)
            outcome = run_sequence_locked(
                config, [name], resume=True, force=False,
                qc_override=False, preflight_override=False,
                config_path=config,
            )
            task_result = outcome.get(name, {}) if isinstance(outcome, dict) else {}
            status = str(task_result.get("status", "")).lower()
            if status != "ok":
                raise RuntimeError(
                    f"Stage {name} returned status={status!r}: "
                    + str(task_result.get("error_message") or task_result)[:600]
                )
            output["completed_stages"].append(name)
        output["status"] = "ENGINEERING_STAGES_COMPLETED_NOT_SCIENCE_APPROVED"
    except Exception as exc:
        output["stopped_at"] = (
            STAGES[len(output["completed_stages"])]
            if len(output["completed_stages"]) < len(STAGES)
            else "postrun"
        )
        output["error"] = f"{type(exc).__name__}: {exc}"
        output["traceback_tail"] = traceback.format_exc(limit=6)
        raise
    finally:
        now = fingerprint_sources(source)
        output["original_files_unchanged"] = (
            now == manifest["original_protected_sha256"]
        )
        if not output["original_files_unchanged"]:
            output["status"] = "SOURCE_INTEGRITY_FAILURE_SCIENCE_HOLD"
            output["error"] = (
                str(output.get("error") or "") +
                " | ORIGINAL PROTECTED FILES CHANGED"
            ).strip(" |")
        output["finished_utc"] = datetime.now(timezone.utc).isoformat()
        with (work / REPORT).open("x", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2, ensure_ascii=False)
        print("PILOT REPORT:", work / REPORT)
        print("ORIGINAL FILES UNCHANGED:",
              output["original_files_unchanged"])
        print("SCIENCE E2E: HOLD")
    if output["original_files_unchanged"] is False:
        raise RuntimeError("Original protected files changed during engineering run")


def selftest() -> None:
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        source = root / "run_01"
        source.mkdir()
        cfg = source / "config.yaml"
        cfg.write_text("work_dir: .", encoding="utf-8")
        check_pilot_destination(cfg, root / "run_91")
        try:
            check_pilot_destination(cfg, source)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Accepted existing run directory")
        assert fingerprint_sources(cfg)[str(cfg)] == checksum(cfg)
        assert len(STAGES) == 13
        assert "biascorr" not in STAGES
    print("SELFTEST PASSED")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) == 4 and sys.argv[1] == "prepare":
        prepare(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        raise SystemExit(
            "Usage: script.py --selftest | prepare original/config.yaml NEW_dir | "
            "run NEW_dir/config.yaml"
        )
