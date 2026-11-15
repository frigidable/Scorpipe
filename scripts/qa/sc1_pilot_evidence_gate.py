#!/usr/bin/env python3
"""Read-only SCORPIO-1 N2146sat pilot evidence reconciliation.

Reconcile existing raw-readability and 6+1 role-plan reports; do not rerun
CCD analyses or authorize physical calibration. No RAW FITS pixel reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

SCHEMA = "SC1_N2146SAT_PILOT_EVIDENCE_GATE_V1"
RAW_SCHEMA = "SC1_RAW_PIXEL_READABILITY_V1"
ROLE_SCHEMA = "SC1_N2146SAT_OFFSET_SKY_ROLE_CONTRACT_V1"
RAW_STATUS = "READ_ONLY_PIXEL_READABILITY_NOT_SCIENCE_QUALIFICATION"
ROLE_STATUS = "ROLE_CONTRACT_RESOLVED_NO_SCIENCE_EXECUTION"
ORIGINAL_COUNTS = {"bias": 20, "obj": 7, "flat": 3, "neon": 2}
OFF = "s22800604.fts"
ON = tuple(f"s228006{i:02d}.fts" for i in range(7, 13))
MAX_EVIDENCE_BYTES = 20_000_000


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _path(text: Any, *, require_file: bool = True) -> Path:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Evidence path is missing or invalid")
    p = Path(text).expanduser().resolve(strict=require_file)
    if require_file and not p.is_file():
        raise ValueError(f"Evidence path must be a file: {p}")
    return p


def _read(path: Path, *, limit: int = MAX_EVIDENCE_BYTES) -> bytes:
    if path.stat().st_size > limit:
        raise ValueError(f"Evidence too large: {path}")
    with path.open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError(f"Evidence grew beyond limit: {path}")
    return content


def _json(path: Path) -> tuple[dict[str, Any], str]:
    content = _read(path)
    obj = json.loads(content.decode("utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError(f"Evidence root must be JSON object: {path}")
    return obj, _digest(content)


def _yaml(path: Path) -> tuple[dict[str, Any], str]:
    content = _read(path)
    obj = yaml.safe_load(content.decode("utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return obj, _digest(content)


def _expect(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _files(rows: Any, label: str) -> list[Path]:
    _expect(isinstance(rows, list), f"Missing list of {label}")
    return [_path(name) for name in rows]


def _validate_raw(raw: dict[str, Any], saved: Path) -> tuple[dict[str, list[Path]], str]:
    _expect(raw.get("schema") == RAW_SCHEMA, "Wrong RAW source evidence schema")
    _expect(raw.get("status") == RAW_STATUS, "RAW source evidence not successful")
    _expect(raw.get("header_audit_mode") == "audit_existing_readonly", "Unexpected RAW audit mode")
    _expect(raw.get("all_selected_primary_arrays_readable_and_finite") is True,
            "RAW pixel readability not documented")
    _expect(raw.get("stage_execution_authorized") is False, "Unsafe RAW report flag")
    _expect(_path(raw.get("source_config")) == saved, "RAW report belongs to another config")
    _expect(raw.get("input_counts") == ORIGINAL_COUNTS, "Unexpected original frame counts")
    rows = raw.get("input_files")
    _expect(isinstance(rows, list) and len(rows) == 32, "Expected exactly 32 RAW source records")

    groups: dict[str, list[Path]] = {role: [] for role in ORIGINAL_COUNTS}
    canonical: list[str] = []
    seen: set[Path] = set()
    expected_roles = [role for role, n in ORIGINAL_COUNTS.items() for _ in range(n)]
    for i, (row, role) in enumerate(zip(rows, expected_roles)):
        _expect(isinstance(row, dict) and row.get("role") == role,
                f"RAW manifest role/order mismatch at index {i}")
        p = _path(row.get("source_path"))
        _expect(p not in seen, "RAW manifest duplicates source path")
        seen.add(p)
        _expect(row.get("filename", "").lower() == p.name.lower(),
                f"RAW manifest filename mismatch: {p}")
        _expect(row.get("size_bytes") == p.stat().st_size,
                f"RAW file size changed or report stale: {p}")
        digest = row.get("source_sha256")
        _expect(isinstance(digest, str) and len(digest) == 64
                and all(c in '0123456789abcdef' for c in digest.lower()),
                f"Missing SHA256 in RAW report for: {p}")
        groups[role].append(p)
        canonical.append(f"{role}\t{p.name}\t{digest}")

    _expect({p.name.lower() for p in groups["obj"]} == set(ON) | {OFF},
            "Original object cohort is not OFF04 plus ON07..12")
    manifest_sha = _digest("\n".join(canonical).encode("utf-8"))
    _expect(raw.get("input_manifest_sha256") == manifest_sha,
            "RAW report content does not match its pinned manifest digest")
    return groups, manifest_sha


def _validate_role(role: dict[str, Any], raw_file: Path, saved: Path,
                   saved_sha: str, original: dict[str, list[Path]],
                   manifest_sha: str) -> tuple[dict[str, str], dict[str, Any]]:
    _expect(role.get("schema") == ROLE_SCHEMA, "Wrong role-plan schema")
    _expect(role.get("status") == ROLE_STATUS, "Role plan not successfully resolved")
    _expect(role.get("stage_execution_authorized") is False, "Unsafe stage authorization flag")
    for k in ("sky_spatial_cleanliness_verified", "sky_temporal_transfer_verified",
              "sky_subtraction_approved", "stack_pointing_approved", "wavelength_lsf_approved",
              "science_e2e_approved"):
        _expect(role.get(k) is False, f"Unexpected authorization in role plan: {k}")
    _expect(_path(role.get("source_config")) == saved, "Role contract from another config")
    _expect(role.get("source_config_sha256") == saved_sha, "Saved config bytes changed")
    _expect(_path(role.get("source_raw_inventory")) == raw_file,
            "Role plan references another RAW inventory")
    _expect(role.get("source_raw_manifest_sha256") == manifest_sha,
            "Role plan pinned another RAW manifest")

    selected = role.get("resolved_roles")
    _expect(isinstance(selected, dict), "Role plan has no resolved roles")
    approved: dict[str, list[Path]] = {}
    for role_name in (*ORIGINAL_COUNTS, "sky"):
        approved[role_name] = _files(selected.get(role_name), role_name)
    _expect([p.name.lower() for p in approved["obj"]] == list(ON),
            "Incorrect ON exposure order or identity")
    _expect([p.name.lower() for p in approved["sky"]] == [OFF],
            "Incorrect dedicated OFF exposure")
    for other in ("bias", "flat", "neon"):
        _expect(approved[other] == original[other], f"Modified {other} calibration role")
    _expect({*approved["obj"], *approved["sky"]} == set(original["obj"]),
            "ON/OFF transfer no longer matches original source cohort")
    all_roles = [p for part in approved.values() for p in part]
    _expect(len(all_roles) == 32 and len(set(all_roles)) == 32,
            "Duplicated or omitted original FITS sources")
    observer_roles = role.get("observational_roles")
    _expect(isinstance(observer_roles, dict)
            and observer_roles.get(OFF) == "OFFSET_SKY_OBSERVER_CONFIRMED"
            and all(observer_roles.get(p) == "ON_TARGET_SCIENCE" for p in ON),
            "Missing observer-confirmed role distinction")

    isolated_config = _path(role.get("isolated_config"))
    isolated_manifest = _path(role.get("isolated_manifest"))
    _expect(_path(role.get("resolved_manifest_path")) == isolated_manifest,
            "Isolated manifest path not resolved as intended")
    _expect(isolated_manifest.parent == isolated_config.parent,
            "Isolated manifest and config in different directories")
    _expect(role.get("resolved_frames_source") == "manifest",
            "Role loader did not select explicit manifest")
    isolated, isolated_sha = _yaml(isolated_config)
    manifest, project_sha = _yaml(isolated_manifest)
    _expect(_path(isolated.get("work_dir"), require_file=False) == isolated_config.parent,
            "Isolated config work root changed")
    _expect(_path(isolated.get("project_manifest_path")) == isolated_manifest,
            "Isolated config lost pinned manifest")
    frames = isolated.get("frames")
    _expect(isinstance(frames, dict), "Isolated config frames missing")
    for label, expected in approved.items():
        _expect(_files(frames.get(label), f"isolated config {label}") == expected,
                f"Isolated config {label} paths changed")
    _expect(manifest.get("schema") == "scorpio-pipe.project-manifest.v1.2"
            and manifest.get("role_policy") == "explicit", "Manifest contract changed")
    roles = manifest.get("roles")
    _expect(isinstance(roles, dict), "Explicit manifest roles missing")
    for manifest_name, label in (("OBJECT_FRAMES", "obj"), ("SKY_FRAMES", "sky")):
        entry = roles.get(manifest_name)
        _expect(isinstance(entry, dict), f"Missing manifest {manifest_name}")
        _expect(_files(entry.get("files"), manifest_name) == approved[label],
                f"Manifest {manifest_name} source selection changed")
        _expect(not entry.get("globs") and not entry.get("exclude"),
                f"Manifest {manifest_name} unexpected discovery/exclusion")
    exclusion = manifest.get("exclude")
    _expect(isinstance(exclusion, dict)
            and not exclusion.get("files") and not exclusion.get("globs"),
            "Manifest unexpectedly excludes source files")

    return ({"isolated_config_sha256": isolated_sha,
             "isolated_manifest_sha256": project_sha},
            {"on_target": [p.name for p in approved["obj"]],
             "offset_sky": [p.name for p in approved["sky"]],
             "n_original_source_frames": len(all_roles)})


def prepare(saved_config: Path, raw_inventory: Path, role_contract: Path,
            output: Path) -> dict[str, Any]:
    """Only metadata/report concordance; science-qualification is ALWAYS HOLD."""
    saved = _path(str(saved_config))
    raw_file = _path(str(raw_inventory))
    role_file = _path(str(role_contract))
    out = Path(output).expanduser().resolve()
    _expect(out.suffix.lower() == ".json", "Output must be a new .json")
    if out.exists():
        raise FileExistsError(f"Refuse to overwrite existing evidence report: {out}")
    original, saved_sha = _yaml(saved)
    raw, raw_sha = _json(raw_file)
    role, role_sha = _json(role_file)
    _expect(original.get("work_dir") is not None,
            "Saved config has no workspace reference")
    old_work = _path(original["work_dir"], require_file=False)
    raw_root = _path(raw.get("raw_data_dir"), require_file=False)
    isolated_root = _path(role.get("isolated_config")).parent
    for protected in (old_work, raw_root, isolated_root):
        _expect(out != protected and protected not in out.parents,
                "Output path overlaps protected source, workspace or isolated plan")
    groups, raw_manifest_sha = _validate_raw(raw, saved)
    hashes, selection = _validate_role(role, raw_file, saved,
                                       saved_sha, groups, raw_manifest_sha)
    _expect(raw.get("work_dir") is not None
            and _path(raw["work_dir"], require_file=False) == old_work,
            "Saved workspace and RAW report disagree")
    evidence = {"saved_config_sha256": saved_sha,
                "raw_inventory_json_sha256": raw_sha,
                "role_contract_json_sha256": role_sha,
                "raw_input_manifest_sha256": raw_manifest_sha,
                **hashes}
    # Evidence may have changed during reading; prevent a stale success report.
    _expect(_digest(_read(saved)) == saved_sha, "Saved config changed during audit")
    _expect(_digest(_read(raw_file)) == raw_sha, "RAW report changed during audit")
    _expect(_digest(_read(role_file)) == role_sha, "Role report changed during audit")
    for path, digest in ((role.get("isolated_config"), hashes["isolated_config_sha256"]),
                         (role.get("isolated_manifest"), hashes["isolated_manifest_sha256"])):
        _expect(_digest(_read(_path(path))) == digest, "Isolated plan changed during audit")

    report = {
        "schema": SCHEMA,
        "status": "EVIDENCE_CONSISTENT_SCIENCE_HOLD",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "source_paths": {"saved_config": str(saved), "raw_inventory": str(raw_file),
                         "role_contract": str(role_file)},
        "evidence_sha256": evidence,
        "observational_selection": selection,
        "completed_engineering_checks": [
            "SAVED_CONFIG_SHA256_MATCHES_OBSERVER_ROLE_PLAN",
            "EXACT_32_RAW_INVENTORY_MANIFEST_INTERNALLY_CONSISTENT",
            "ISOLATED_6_ON_1_OFF_ROLE_PLAN_UNCHANGED",
            "ISOLATED_CONFIG_AND_EXPLICIT_MANIFEST_CONSISTENT",
        ],
        "unresolved_science_gates": {
            "B_CCD_OVERSCAN_PHYSICS": "HOLD",
            "C_MASTERBIAS_FLAT_AND_CALIBRATION_COVARIANCE": "HOLD",
            "D_WAVE2D_FLEXURE_AND_EFFECTIVE_LSF": "HOLD",
            "E_SOURCE_FREE_SKY_AND_OFF_ON_TRANSFER": "HOLD",
            "F_SCI_VAR_MASK_FULL_CHAIN_COVARIANCE": "HOLD",
            "G_INDEPENDENT_REAL_NIGHT_E2E_BENCHMARKS": "HOLD",
        },
        "raw_fits_pixels_read": False,
        "raw_fits_source_hashes_recomputed": False,
        "calibrated_science_fits_created": False,
        "stages_executed": False,
        "stage_execution_authorized": False,
        "sky_only_rows_certified": 0,
        "sky_subtraction_approved": False,
        "science_qualified": False,
        "observatory_release_approved": False,
        "science_hold": True,
        "limitations": [
            "Reconciles archived engineering observations, not current RAW pixel contents.",
            "Raw source file existence and reported byte sizes were checked; byte hashes were not recalculated.",
            "The saved original config, isolated role plan and input JSON reports were byte-checked.",
            "An observer-confirmed OFF role does not mean a source-free SKY field or constant sky spectrum.",
            "Successful engineering evidence is not physical CCD, wavelength, LSF or covariance validation.",
            "This report is never an executable reduction authorization or a SCIENCE_QUALIFIED certificate.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-config", required=True, type=Path)
    parser.add_argument("--raw-inventory", required=True, type=Path)
    parser.add_argument("--role-contract", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.saved_config, args.raw_inventory,
                     args.role_contract, args.out)
    print("STATUS:", result["status"])
    print("ON:", result["observational_selection"]["on_target"])
    print("OFF:", result["observational_selection"]["offset_sky"])
    print("OPEN PHYSICAL GATES:", ", ".join(result["unresolved_science_gates"]))
    print("RAW PIXELS READ: NO; SCIENCE HOLD: ACTIVE; STAGES EXECUTED: NONE")
    print("REPORT:", args.out)


if __name__ == "__main__":
    main()
