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
    "manifest", "superbias", "cosmics", "superneon",
    "lineid_prepare", "wavesolution", "lsf_model",
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


# These keys are generated afresh by scorpio_pipe.config.load_config().
# Persisting their OLD values into a copied YAML config risks routing outputs
# or provenance back to the original workspace.
LOADER_DERIVED_ROOT_KEYS = (
    "project_root", "config_path", "config_dir", "work_dir_abs",
    "_project_manifest",
)


def strip_loader_derived_root_keys(raw: dict) -> None:
    """Drop cached original-run metadata; the loader reconstructs it for the pilot."""
    for key in LOADER_DERIVED_ROOT_KEYS:
        raw.pop(key, None)


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
    strip_loader_derived_root_keys(raw)
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
    # Sky requires BIASSUB=True for instrument-identified SCORPIO frames.
    # Cosmics provides bias-corrected OBJ products without touching fast-readout
    # FLAT/NEON, which have no compatible MasterBias in this data set.
    cosmics = raw.setdefault("cosmics", {})
    cosmics["enabled"] = True
    cosmics["apply_to"] = ["obj"]
    cosmics["bias_subtract"] = True
    cosmics["method"] = "auto"  # Seven OBJ => stack_mad; engineering, NOT certified
    # Disallow the Sky pass-through fallback from masquerading as a completed
    # engineering reduction when sky fitting actually failed.
    raw.setdefault("sky", {})["failure_policy"] = "strict"

    # Do not carry path strings to original science PRODUCTS into the new run.
    # Reading raw FITS at their unchanged absolute paths is allowed.
    source_work_string = str(source.parent).lower()
    raw_data = str(Path(str(raw["data_dir"])).resolve()).lower()
    if source_work_string == raw_data:
        raise RuntimeError("Raw data_dir unexpectedly equals source work_dir")
    unsafe_locations: list[str] = []

    def collect_old_run_product_refs(value: object, location: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                # Existing RAW exposures may legitimately live under the
                # source run; they are only ever opened for reading.
                if not location and key == "frames":
                    continue
                collect_old_run_product_refs(child, f"{location}.{key}" if location else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                collect_old_run_product_refs(child, f"{location}[{index}]")
        elif isinstance(value, str) and source_work_string in value.lower():
            unsafe_locations.append(location)

    collect_old_run_product_refs(raw)
    if unsafe_locations:
        raise RuntimeError(
            "Unsafe original-run reference(s): " + ", ".join(unsafe_locations)
        )

    before = fingerprint_sources(source)
    new_dir.mkdir(parents=False)
    new_config = new_dir / "config.yaml"
    with new_config.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(raw, stream, sort_keys=False, allow_unicode=True)

    # Verify the actual config loader against the prepared RAW selection before
    # declaring this pilot usable. In particular, manifest role overrides must
    # not silently change the 20-BIAS cohort or the seven science frames.
    loaded_pilot = load_config(new_config)
    if Path(str(loaded_pilot.get("work_dir") or "")).resolve() != new_dir:
        raise RuntimeError("Prepared pilot loader resolved work_dir outside pilot")
    check_expected_frames(loaded_pilot)
    for kind in EXPECTED:
        if loaded_pilot["frames"][kind] != cfg["frames"][kind]:
            raise RuntimeError(f"Prepared pilot changed selected {kind} RAW frames")

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
        "cosmics_apply_to": ["obj"],
        "cosmics_bias_subtract": True,
        "cosmics_method": "auto",
        "sky_failure_policy": "strict",
        "science_e2e_approved": False,
    }
    with (new_dir / MANIFEST).open("x", encoding="utf-8") as output:
        json.dump(record, output, indent=2, ensure_ascii=False)
    print("ENGINEERING PILOT PREPARED:", new_config, flush=True)
    print("NO PIPELINE STAGES RUN; SCIENCE E2E HOLD", flush=True)


LINEID_PHASE_REPORT = "sc1_engineering_pilot_lineid_phase.json"
LINEID_BOUNDARY = STAGES.index("wavesolution")


def phase_tasks(phase: str) -> tuple[str, ...]:
    if phase == "lineid":
        return STAGES[:LINEID_BOUNDARY]
    if phase == "resume":
        return STAGES[LINEID_BOUNDARY:]
    raise ValueError(f"Unknown engineering pilot phase: {phase}")


def review_lineid_pairs(cfg: dict, work: Path) -> dict[str, object]:
    """Check a manually reviewed hand_pairs.txt, never auto-accept CSV proposals."""
    import numpy as np
    from scorpio_pipe.wavesol_paths import wavesol_dir
    from scorpio_pipe.stages.wavesolution import _read_hand_pairs

    wcfg = cfg.get("wavesol") or {}
    outdir = wavesol_dir(cfg, run_root=work)
    hp = outdir / "hand_pairs.txt"
    hp_raw = str(wcfg.get("hand_pairs_path") or "").strip()
    if hp_raw:
        explicit = Path(hp_raw)
        explicit = (explicit if explicit.is_absolute() else work / explicit).resolve()
        if explicit != hp.resolve():
            raise RuntimeError("Manual LineID file must be in the isolated pilot wavesol directory")
    if not hp.is_file():
        raise FileNotFoundError(
            f"Manually reviewed LineID required before resuming: {hp}. "
            "lineid_prepare outputs are suggestions, not hand_pairs.txt."
        )
    x, lam, _, disabled = _read_hand_pairs(hp)
    keep = ~disabled
    xs, ls = x[keep], lam[keep]
    required = max(5, int(wcfg.get("poly_deg_1d", 4)) + 1)
    if xs.size < required:
        raise RuntimeError(f"LineID needs >={required} active hand pairs, found {xs.size}")
    if not np.isfinite(xs).all() or not np.isfinite(ls).all():
        raise RuntimeError("LineID contains nonfinite active pairs")
    if len({round(float(v), 3) for v in xs}) != xs.size:
        raise RuntimeError("LineID has duplicate active pixel positions")
    if len({round(float(v), 3) for v in ls}) != ls.size:
        raise RuntimeError("LineID has duplicate active laboratory wavelengths")
    coverage = wcfg.get("expected_coverage_A")
    if isinstance(coverage, (tuple, list)) and len(coverage) == 2:
        lower, upper = map(float, coverage)
        if (np.any(ls < lower - 20.0) or np.any(ls > upper + 20.0)):
            raise RuntimeError("LineID wavelengths outside setup prior coverage")
    if float(np.ptp(xs)) < 300.0:
        raise RuntimeError("LineID pixel coverage <300 px; review lamp identifications")
    return {
        "manual_pairs_path": str(hp),
        "manual_pairs_sha256": checksum(hp),
        "active_pairs": int(xs.size),
        "pixel_span": float(np.ptp(xs)),
        "wavelength_span_A": float(np.ptp(ls)),
        "user_review_confirmed": True,
    }


def run(pilot_config: str, *, phase: str = "lineid", lineid_reviewed: bool = False) -> None:
    """Run to the manual LineID gate, then explicitly resume after review."""
    from scorpio_pipe.pipeline.public_api import run_sequence_locked, canonical_task_name

    tasks = phase_tasks(phase)
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
    cosmic = cfg.get("cosmics") or {}
    if (not bool(cosmic.get("enabled", False))
            or cosmic.get("apply_to") != ["obj"]
            or not bool(cosmic.get("bias_subtract", False))
            or str(cosmic.get("method") or "").lower() != "auto"):
        raise RuntimeError("Pilot Cosmics must bias-correct only OBJ frames in auto mode")
    if str((cfg.get("sky") or {}).get("failure_policy") or "").lower() != "strict":
        raise RuntimeError("Pilot Sky failure_policy must remain strict")
    if (cfg.get("calib") or {}).get("bias_gross_reject_adu") != 50.0:
        raise RuntimeError("Pilot gross rejection no longer matches manifest")
    if not all(
        checksum(Path(p)) == digest
        if digest is not None else not Path(p).exists()
        for p, digest in manifest["original_protected_sha256"].items()
    ):
        raise RuntimeError("Original files changed since pilot preparation")

    phase1_file = work / LINEID_PHASE_REPORT
    final_file = work / REPORT
    if final_file.exists():
        raise FileExistsError("Final engineering result exists; refusing another execution")
    review_info = None
    if phase == "lineid":
        if phase1_file.exists():
            raise FileExistsError("LineID phase already attempted; refusing another execution")
        if lineid_reviewed:
            raise RuntimeError("LineID review cannot be asserted before phase one")
        report_file = phase1_file
    else:
        if not lineid_reviewed:
            raise RuntimeError("Resume requires explicit --lineid-reviewed confirmation")
        if not phase1_file.is_file():
            raise FileNotFoundError("No completed LineID phase report; run first phase")
        phase1 = json.loads(phase1_file.read_text(encoding="utf-8"))
        if (phase1.get("status") != "ENGINEERING_PAUSED_FOR_LINEID_SCIENCE_HOLD"
                or phase1.get("completed_stages") != list(phase_tasks("lineid"))
                or phase1.get("original_files_unchanged") is not True):
            raise RuntimeError("LineID phase did not complete cleanly; resume forbidden")
        if phase1.get("pilot_config_sha256") != checksum(config):
            raise RuntimeError("Pilot config changed after LineID phase; resume forbidden")
        review_info = review_lineid_pairs(cfg, work)
        report_file = final_file

    output = {
        "status": "ENGINEERING_RUN_INCOMPLETE_SCIENCE_HOLD",
        "phase": phase,
        "config": str(config),
        "pilot_config_sha256": checksum(config),
        "stage_plan": list(STAGES),
        "phase_stages": list(tasks),
        "completed_stages": [],
        "stopped_at": None,
        "error": None,
        "qc_override": False,
        "preflight_override": False,
        "force": False,
        "original_files_unchanged": None,
        "lineid_review": review_info,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        for name in tasks:
            print("ENGINEERING STAGE:", name, flush=True)
            outcome = run_sequence_locked(
                config, [name], resume=True, force=False,
                qc_override=False, preflight_override=False,
                config_path=config,
            )
            task_result = outcome.get(name, {}) if isinstance(outcome, dict) else {}
            status = str(task_result.get("status", "")).lower() if isinstance(task_result, dict) else ""
            if status != "ok":
                raise RuntimeError(
                    f"Stage {name} returned status={status!r}: "
                    + str(task_result.get("error_message") or task_result)[:600]
                    if isinstance(task_result, dict) else f"Stage {name} has no valid result"
                )
            output["completed_stages"].append(name)
        if phase == "lineid":
            output["status"] = "ENGINEERING_PAUSED_FOR_LINEID_SCIENCE_HOLD"
            output["stopped_at"] = "MANUAL_LINEID_REVIEW_REQUIRED"
        else:
            output["status"] = "ENGINEERING_STAGES_COMPLETED_NOT_SCIENCE_APPROVED"
    except Exception as exc:
        output["stopped_at"] = (
            tasks[len(output["completed_stages"])]
            if len(output["completed_stages"]) < len(tasks)
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
        with report_file.open("x", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2, ensure_ascii=False)
        print("PILOT REPORT:", report_file)
        print("ORIGINAL FILES UNCHANGED:", output["original_files_unchanged"])
        print("SCIENCE E2E: HOLD")
    if output["original_files_unchanged"] is False:
        raise RuntimeError("Original protected files changed during engineering run")
    if phase == "lineid":
        print("PAUSED: inspect manual_pairs_auto.csv, curate hand_pairs.txt in the pilot wave directory.")
        print("Do NOT auto-approve the suggested lamp matches.")



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
        assert "cosmics" in STAGES
        assert "biascorr" not in STAGES
        assert "arclineid" not in STAGES  # Alias for lineid_prepare, not a separate task
        assert phase_tasks("lineid")[-1] == "lineid_prepare"
        assert phase_tasks("resume")[0] == "wavesolution"
        assert phase_tasks("lineid") + phase_tasks("resume") == STAGES
    print("SELFTEST PASSED")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) == 4 and sys.argv[1] == "prepare":
        prepare(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] == "run":
        run(sys.argv[2], phase="lineid")
    elif len(sys.argv) == 4 and sys.argv[1] == "resume" and sys.argv[3] == "--lineid-reviewed":
        run(sys.argv[2], phase="resume", lineid_reviewed=True)
    else:
        raise SystemExit(
            "Usage: script.py --selftest | prepare original/config.yaml NEW_dir | "
            "run NEW_dir/config.yaml | resume NEW_dir/config.yaml --lineid-reviewed"
        )
