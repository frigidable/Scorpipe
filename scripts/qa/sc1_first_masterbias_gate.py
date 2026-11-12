#!/usr/bin/env python3
"""Guarded first SCORPIO-1 N2146sat calibration stage. Exact-night only.

Two read-only-science-input modes:
    --check config.yaml qa_report.json  # first-build gate: refuses prior stages
    --audit-existing config.yaml qa_report.json  # inspect saved workspace, no stage run

No production engine invocation is present. The new diagnostic report must
be outside the selected RAW and work directories, and is created exclusively;
existing files, YAML, RAW FITS and reduction products cannot be overwritten.
This is engineering QA, never scientific qualification.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_N = {"bias": 20, "obj": 7, "flat": 3, "neon": 2}
EXPECTED_SHAPE = (1032, 2080)
EXPECTED_BIASES = {f"s228001{i:02d}.fts" for i in range(1, 21)}


def collect(
    config_path: Path, *, require_clean_workdir: bool = True,
) -> tuple[dict, list[Path]]:
    """Validate exact-night inputs, optionally forbidding existing stage output.

    Both modes are strictly read-only with respect to RAW and workspace data.
    Only require_clean_workdir=True is suitable for *first-build* readiness.
    """
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
        data_dir = config_path.parent / data_dir
    data_dir = data_dir.resolve(strict=True)
    if not data_dir.is_dir():
        raise RuntimeError("Configured RAW data_dir must be an existing directory")
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
            # Comparisons with NaN are always False: without this explicit
            # check an entire coherent cohort of NaN headers can pass the
            # tolerance test below and appear readout-compatible.
            if not math.isfinite(rate) or not math.isfinite(gain):
                raise RuntimeError(
                    f"Non-finite CCD readout for {role} {p.name}: "
                    f"RATE={rate}, GAIN={gain}"
                )
            required_rate, required_gain = ((145.0, 0.9)
                                            if role in ("bias", "obj")
                                            else (225.0, 3.5))
            if (abs(rate-required_rate) > 0.01
                    or abs(gain-required_gain) > 0.001 or node != "A"):
                raise RuntimeError(
                    f"Unexpected readout for {role} {p.name}: "
                    f"RATE={rate}, GAIN={gain}, NODE={node}"
                )
            sig = FrameSignature.from_header(h, fallback_shape=shape)
            # Equal *missing* metadata across all frames must not be treated
            # as evidence of compatible CCD binning for this exact-night QA.
            if sig.bx is None or sig.by is None:
                raise RuntimeError(
                    f"Unknown CCD binning for {role} {p.name}; "
                    "required header metadata is absent or unparseable"
                )
            sigs.append(sig)
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
    if found and require_clean_workdir:
        raise RuntimeError(
            "Prior stage products found; abort first-build to avoid overwriting: "
            + ", ".join(sorted(found))
        )
    report = {
        "status": (
            "PRECHECK_PASSED_NOT_SCIENCE_VALIDATION"
            if require_clean_workdir else
            "EXISTING_WORKSPACE_HEADER_AUDIT_NOT_SCIENCE_VALIDATION"
        ),
        "mode": "first_build_preflight" if require_clean_workdir else "audit_existing_readonly",
        "existing_stage_products": sorted(found),
        "first_build_clear": require_clean_workdir and not found,
        "stage_execution_authorized": False,
        "config": str(config_path), "work_dir": str(workdir),
        "raw_data_dir": str(data_dir),
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
        # A FITS-header and config preflight neither reads detector pixels
        # nor verifies detector regions, overscan physics, or data corruption.
        "header_only_preflight": True,
        "raw_pixel_integrity_verified": False,
        "physical_overscan_region_certified": False,
        "masterbias_science_validated": False,
    }
    return report, all_paths


def run(config: Path, out: Path, *, audit_existing: bool = False) -> None:
    """Read-only header check; audit-existing cannot authorize a first build."""
    report, raw_paths = collect(
        config, require_clean_workdir=not audit_existing,
    )
    config_path = Path(config).expanduser().resolve(strict=True)
    out = Path(out).expanduser().resolve()
    work_dir = Path(report["work_dir"]).resolve(strict=True)
    raw_dir = Path(report["raw_data_dir"]).resolve(strict=True)
    input_paths = {config_path, *(p.resolve(strict=True) for p in raw_paths)}
    raw_roots = {raw_dir, *(p.parent.resolve() for p in raw_paths)}
    if (out in input_paths or out == work_dir or out.is_relative_to(work_dir)
            or any(out == root or out.is_relative_to(root) for root in raw_roots)):
        raise ValueError("Preflight report must be outside the work and RAW directories")
    if out.suffix.lower() != ".json":
        raise ValueError("Preflight report must be a JSON file")
    if out.exists():
        raise FileExistsError(f"Refuse to overwrite existing diagnostic report: {out}")
    report["created_utc"] = datetime.now(timezone.utc).isoformat()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
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
    elif len(sys.argv) == 4 and sys.argv[1] in {"--check", "--audit-existing"}:
        run(
            Path(sys.argv[2]), Path(sys.argv[3]),
            audit_existing=sys.argv[1] == "--audit-existing",
        )
    else:
        raise SystemExit(
            "Usage: script.py --selftest | "
            "--check config.yaml report.json | "
            "--audit-existing config.yaml report.json"
        )
