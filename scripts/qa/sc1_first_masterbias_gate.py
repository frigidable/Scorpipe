#!/usr/bin/env python3
"""Guarded first SCORPIO-1 N2146sat calibration stage. Exact-night only.

Two explicit modes:
    --check config.yaml qa_report.json         # FITS headers, no stage run
    --run-superbias config.yaml qa_report.json  # invokes ONLY engine task superbias

Does not edit YAML or raw FITS. Stage run writes engine provenance and MasterBias
products under the verified existing work_dir. This is an engineering validation
and does NOT authorize science reduction, cross-readout FLAT, or E2E release.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_N = {"bias": 20, "obj": 7, "flat": 3, "neon": 2}
EXPECTED_SHAPE = (1032, 2080)
EXPECTED_BIASES = {f"s228001{i:02d}.fts" for i in range(1, 21)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(config_path: Path) -> tuple[dict, list[Path]]:
    import yaml
    from astropy.io import fits
    from scorpio_pipe.frame_signature import FrameSignature

    config_path = config_path.expanduser().resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict):
        raise RuntimeError("Config must be a YAML mapping")
    workdir = Path(str(cfg.get("work_dir", ""))).expanduser()
    if not workdir.is_absolute() or workdir.resolve() != config_path.parent:
        raise RuntimeError("Configured work_dir must exactly equal this config directory")
    project = cfg.get("project") or {}
    if not isinstance(project, dict):
        raise RuntimeError("project must be a YAML mapping")
    wave_ref = str(project.get("wave_reference") or "").strip().lower()
    if wave_ref != "air":
        raise RuntimeError(
            "SCORPIO-1/2 He-Ne-Ar reference requires project.wave_reference=air "
            f"for this night (got {wave_ref!r}); correct saved config before stage run. "
            "Do not bypass typed validation or alter detector FITS."
        )
    if (cfg.get("calib") or {}).get("bias_policy") != "strict":
        raise RuntimeError("Cross-readout fallback prohibited: require strict bias policy")
    if bool((cfg.get("flatfield") or {}).get("enabled", False)):
        raise RuntimeError("First MasterBias gate requires flatfield.enabled=false")
    if bool((cfg.get("superneon") or {}).get("bias_sub", True)):
        raise RuntimeError("First MasterBias gate requires superneon.bias_sub=false")

    data_dir = Path(str(cfg.get("data_dir") or config_path.parent)).expanduser()
    if not data_dir.is_absolute():
        data_dir = (config_path.parent / data_dir).resolve()
    frame_lists = cfg.get("frames") or {}
    if not isinstance(frame_lists, dict):
        raise RuntimeError("frames must be a dictionary")

    signature = {}
    all_paths: list[Path] = []
    info = {}
    for role, expected_n in EXPECTED_N.items():
        names = frame_lists.get(role, [])
        if not isinstance(names, list) or len(names) != expected_n:
            raise RuntimeError(
                f"Unexpected frame selection for {role}: need {expected_n}, "
                f"got {len(names) if isinstance(names, list) else type(names).__name__}"
            )
        files: list[Path] = []
        for raw in names:
            p = Path(str(raw)).expanduser()
            if not p.is_absolute():
                p = data_dir / p
            p = p.resolve()
            if not p.is_file():
                raise FileNotFoundError(f"Missing {role} frame: {p}")
            files.append(p)
            if p in all_paths:
                raise RuntimeError(f"Duplicate FITS selection (within/across roles): {p}")
            all_paths.append(p)
        if role == "bias" and {p.name.lower() for p in files} != EXPECTED_BIASES:
            raise RuntimeError("Selected 20 BIAS are not the audited early20 cohort")

        sigs = []
        for p in files:
            try:
                h = fits.getheader(p, 0, ignore_missing_end=True)
                shape = (int(h.get("NAXIS2", 0)), int(h.get("NAXIS1", 0)))
                if shape != EXPECTED_SHAPE:
                    raise RuntimeError(f"{role}: unexpected shape {shape} in {p.name}")
                rate = float(h["RATE"])
                gain = float(h["GAIN"])
                node = str(h.get("NODE", "")).strip().upper()
            except Exception as exc:
                raise RuntimeError(f"Cannot validate FITS {p.name}: {exc}") from exc
            required_rate, required_gain = ((145.0, 0.9)
                                            if role in ("bias", "obj")
                                            else (225.0, 3.5))
            if (abs(rate-required_rate) > 0.01
                    or abs(gain-required_gain) > 0.001 or node != "A"):
                raise RuntimeError(
                    f"Unexpected readout for {role} {p.name}: "
                    f"RATE={rate}, GAIN={gain}, NODE={node}"
                )
            sigs.append(FrameSignature.from_header(h, fallback_shape=shape))
        reference = sigs[0]
        for p, sig in zip(files, sigs):
            diffs = sig.diff(reference)
            if diffs:
                raise RuntimeError(f"Incompatible {role} frame {p.name}: {diffs}")
        signature[role] = reference
        info[role] = [p.name for p in files]

    for role in ("sky", "sunsky", "standard", "telluric_standard"):
        if frame_lists.get(role):
            raise RuntimeError(f"Unexpected optional frames in {role}")
    for role in ("bias", "obj"):
        diffs = signature[role].diff(signature["bias"])
        if diffs:
            raise RuntimeError(f"Science and BIAS mismatch ({role}): {diffs}")
    for role in ("flat", "neon"):
        diffs = signature[role].diff(signature["flat"])
        if diffs:
            raise RuntimeError(f"Fast FLAT and NEON mismatch ({role}): {diffs}")

    from scorpio_pipe.workspace_paths import stage_dir
    bias_output = stage_dir(workdir, "superbias")
    forbidden = [bias_output]
    forbidden.extend(workdir.glob("??_superbias"))
    forbidden.extend(workdir.glob("??_superflat"))
    forbidden.extend(workdir.glob("??_flat*"))
    forbidden.extend(workdir.glob("products/??_*"))
    found = [str(p) for p in set(forbidden) if p.exists()]
    if found:
        raise RuntimeError(
            "Prior stage products found; abort first-build to avoid overwriting: "
            + ", ".join(sorted(found))
        )
    report = {
        "status": "PRECHECK_PASSED_NOT_SCIENCE_VALIDATION",
        "config": str(config_path), "work_dir": str(workdir),
        "input_counts": dict(EXPECTED_N),
        "audited_early20_bias_confirmed": True,
        "readout_modes": {
            role: signature[role].to_dict() for role in EXPECTED_N
        },
        "selected_filenames": info,
        "missing_raw_files": 0,
        "first_masterbias_target": str(bias_output),
        "expected_stage_tasks": ["superbias"],
        "fast_flat_neon_bias_status": "HOLD_NO_FAST_MODE_MASTERBIAS",
    }
    return report, all_paths


def run(config: Path, out: Path, *, execute: bool = False) -> None:
    report, raw_paths = collect(config)
    report["created_utc"] = datetime.now(timezone.utc).isoformat()
    if execute:
        pre_hash = {str(p): sha256(p) for p in [config] + raw_paths}
        try:
            from scorpio_pipe.pipeline.public_api import run_sequence_locked
            result = run_sequence_locked(
                config, ["superbias"], resume=False, force=False,
                qc_override=False, preflight_override=False,
            )
            report["engine_result"] = str(result)
            stage = Path(report["first_masterbias_target"])
            needed = (
                "master_bias.fits", "master_bias_var.fits",
                "master_bias_dq.fits", "master_bias_index.json"
            )
            missing = [f for f in needed if not (stage / f).is_file()]
            if missing:
                raise RuntimeError(f"Missing MasterBias products: {missing}")
            idx = json.loads((stage/"master_bias_index.json").read_text(encoding="utf-8"))
            groups = idx.get("groups") or []
            if len(groups) != 1:
                raise RuntimeError(f"Unexpected MasterBias group count: {len(groups)}")
            report["n_master_bias_groups"] = len(groups)
            report["master_bias_files"] = list(needed)
            report["status"] = "ENGINEERING_MASTERBIAS_BUILT_SCIENCE_HOLD"
        except Exception as exc:
            report["status"] = "ENGINEERING_STAGE_FAILED_SCIENCE_HOLD"
            report["failure"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            modified = [str(p) for p in [config]+raw_paths
                        if sha256(p) != pre_hash[str(p)]]
            report["modified_config_or_raw_fits"] = modified
            if modified:
                report["status"] = "CRITICAL_INPUT_MUTATION_DETECTED"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
            if modified:
                raise RuntimeError("CRITICAL: source FITS/config hash changed")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    print("STATUS:", report["status"])
    print("WORK:", report["work_dir"])
    print("INPUT COUNTS:", report["input_counts"])
    print("OUTPUT:", out)
    print("SCIENCE E2E: HOLD")


def selftest():
    # Predicate sanity check; synthetic FITS creation is in pytest.
    assert len(EXPECTED_BIASES) == 20
    assert "s22800101.fts" in EXPECTED_BIASES
    assert "s22800120.fts" in EXPECTED_BIASES
    assert set(EXPECTED_N) == {"bias", "obj", "flat", "neon"}
    print("SELFTEST PASSED")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) == 4 and sys.argv[1] in ("--check", "--run-superbias"):
        run(Path(sys.argv[2]), Path(sys.argv[3]),
            execute=(sys.argv[1] == "--run-superbias"))
    else:
        raise SystemExit(
            "Usage: script.py --selftest | --check config.yaml report.json | "
            "--run-superbias config.yaml report.json"
        )
