#!/usr/bin/env python3
"""Prepare a NEW isolated N2146sat OFF-SKY role plan; never execute science.

This exact-night tool is for observer-confirmed 2024-12-06:
  s22800604.fts = intentional dedicated OFF-SKY, 600 s;
  s22800607..12.fts = six ON-TARGET science exposures, 1200 s.

The source saved config, source RAW and previous stage products are read only.
Uses the independently recorded raw SHA-256 inventory as a content lock.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROLE_PLAN_SCHEMA = "SC1_N2146SAT_OFFSET_SKY_ROLE_CONTRACT_V1"
OFF_SKY = "s22800604.fts"
ON_TARGET = tuple(f"s228006{i:02d}.fts" for i in range(7, 13))
COUNTS = {"bias": 20, "obj": 7, "flat": 3, "neon": 2}
CHUNK = 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected(cfg: dict[str, Any], raw_root: Path) -> dict[str, list[Path]]:
    lists = cfg.get("frames")
    if not isinstance(lists, dict):
        raise ValueError("Saved config frames must be a mapping")
    selected: dict[str, list[Path]] = {}
    all_paths: set[Path] = set()
    for role, count in COUNTS.items():
        names = lists.get(role)
        if not isinstance(names, list) or len(names) != count:
            raise ValueError(f"Expected {count} saved {role} paths")
        files: list[Path] = []
        for name in names:
            candidate = Path(str(name)).expanduser()
            if not candidate.is_absolute():
                candidate = raw_root / candidate
            p = candidate.resolve(strict=True)
            if not p.is_file() or p in all_paths:
                raise ValueError(f"Missing or duplicate source {role}: {p}")
            all_paths.add(p)
            files.append(p)
        selected[role] = files
    obj_names = {p.name.lower() for p in selected["obj"]}
    if obj_names != {OFF_SKY, *ON_TARGET}:
        raise ValueError(
            "Saved OBJ cohort is not exactly OFF04 + ON07..12; "
            f"found {sorted(obj_names)}"
        )
    existing_sky = lists.get("sky") or []
    if not isinstance(existing_sky, list) or existing_sky:
        raise ValueError("Saved SKY must be empty; no implicit role replacement")
    for name in lists.get("exclude", []) or []:
        if Path(str(name)).name.lower() in obj_names:
            raise ValueError("Source config excludes a required science/sky frame")
    return selected


def _verify_inventory(
    inventory_path: Path, config_path: Path, selected: dict[str, list[Path]],
) -> str:
    record = json.loads(inventory_path.read_text(encoding="utf-8"))
    if record.get("schema") != "SC1_RAW_PIXEL_READABILITY_V1":
        raise ValueError("Need exact-night RAW pixel inventory V1")
    if record.get("status") != "READ_ONLY_PIXEL_READABILITY_NOT_SCIENCE_QUALIFICATION":
        raise ValueError("RAW inventory status is not a successful readability pass")
    if not record.get("all_selected_primary_arrays_readable_and_finite"):
        raise ValueError("RAW inventory does not establish selected pixel readability")
    if Path(str(record.get("source_config"))).resolve() != config_path:
        raise ValueError("RAW inventory belongs to another saved config")
    entries = record.get("input_files")
    if not isinstance(entries, list) or len(entries) != 32:
        raise ValueError("RAW inventory must contain exactly 32 source hashes")
    source_paths = [
        (role, path) for role, files in selected.items() for path in files
    ]
    canonical: list[str] = []
    for (role, source), entry in zip(source_paths, entries):
        if (
            not isinstance(entry, dict)
            or entry.get("role") != role
            or Path(str(entry.get("source_path"))).resolve() != source
            or entry.get("filename", "").lower() != source.name.lower()
            or entry.get("size_bytes") != source.stat().st_size
        ):
            raise ValueError(f"RAW inventory selection/order differs: {source.name}")
        digest = entry.get("source_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or _sha256(source) != digest
        ):
            raise ValueError(f"RAW bytes changed since inventory: {source.name}")
        canonical.append(f"{role}\t{source.name}\t{digest}")
    manifest_hash = hashlib.sha256("\n".join(canonical).encode("utf-8")).hexdigest()
    if record.get("input_manifest_sha256") != manifest_hash:
        raise ValueError("RAW inventory manifest digest inconsistent with inputs")
    return manifest_hash


def prepare(config_file: Path, inventory_file: Path, output_dir: Path) -> dict[str, Any]:
    """Create only a NEW external isolated plan; verify canonical loader contract."""
    from scorpio_pipe.config import load_config

    source = Path(config_file).expanduser().resolve(strict=True)
    inventory = Path(inventory_file).expanduser().resolve(strict=True)
    output = Path(output_dir).expanduser().resolve()
    original_bytes = source.read_bytes()
    cfg = yaml.safe_load(original_bytes.decode("utf-8-sig"))
    if not isinstance(cfg, dict):
        raise ValueError("Saved config must be a mapping")

    raw_dir = Path(str(cfg.get("data_dir") or source.parent)).expanduser()
    if not raw_dir.is_absolute():
        raw_dir = source.parent / raw_dir
    raw_dir = raw_dir.resolve(strict=True)
    work_dir = Path(str(cfg.get("work_dir") or "")).expanduser()
    if not work_dir.is_absolute() or work_dir.resolve() != source.parent:
        raise ValueError("Saved work_dir must equal the saved config's directory")
    for protected in (raw_dir, work_dir):
        if output == protected or output.is_relative_to(protected):
            raise ValueError("Isolated output must remain outside RAW and existing work")
    if output.exists():
        raise FileExistsError(f"Never overwrite an isolated role plan: {output}")

    selected = _selected(cfg, raw_dir)
    digest = _verify_inventory(inventory, source, selected)
    by_name = {p.name.lower(): p for p in selected["obj"]}
    on_paths = [by_name[name] for name in ON_TARGET]
    off_path = by_name[OFF_SKY]
    roles = {
        "OBJECT_FRAMES": {
            "files": [str(p) for p in on_paths], "globs": [], "exclude": [],
        },
        "SKY_FRAMES": {
            "files": [str(off_path)], "globs": [], "exclude": [],
        },
    }
    manifest = {
        "schema": "scorpio-pipe.project-manifest.v1.2",
        "role_policy": "explicit",
        "exclude": {"files": [], "globs": []},
        "roles": roles,
    }
    isolated = copy.deepcopy(cfg)
    isolated["work_dir"] = str(output)
    isolated["project_manifest_path"] = str(output / "project_manifest.yaml")
    isolated_frames = isolated["frames"]
    isolated_frames["obj"] = [str(p) for p in on_paths]
    isolated_frames["sky"] = [str(off_path)]
    # Other calibration groups retain exactly their saved raw assignments.
    # No stage is called. load_config creates its usual bookkeeping *only*
    # inside the NEW isolated output, never in the original work directory.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    isolated_config = output / "config.yaml"
    isolated_manifest = output / "project_manifest.yaml"
    with isolated_config.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(isolated, stream, allow_unicode=True, sort_keys=False)
    with isolated_manifest.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, allow_unicode=True, sort_keys=False)

    resolved = load_config(isolated_config)
    meta = resolved.get("_project_manifest") or {}
    resolved_frames = resolved.get("frames") or {}
    if (
        meta.get("frames_source") != "manifest"
        or meta.get("role_selection_status") != "explicit"
        or set(meta.get("roles_applied") or []) != {"OBJECT_FRAMES", "SKY_FRAMES"}
        or Path(str(meta.get("path"))).resolve() != isolated_manifest
        or not meta.get("has_sky_frames")
        or meta.get("n_sky_frames") != 1
        or Path(str(resolved.get("work_dir"))).resolve() != output
    ):
        raise RuntimeError("Resolved loader DID NOT apply isolated role manifest")
    expected = {"obj": on_paths, "sky": [off_path]}
    expected.update({key: value for key, value in selected.items() if key != "obj"})
    for role, paths in expected.items():
        actual = [Path(str(p)).resolve() for p in resolved_frames.get(role, [])]
        if actual != paths:
            raise RuntimeError(f"Resolved loader role mismatch: {role}")
    if set(resolved_frames["obj"]) & set(resolved_frames["sky"]):
        raise RuntimeError("OBJECT and SKY overlap")
    if _sha256(source) != hashlib.sha256(original_bytes).hexdigest():
        raise RuntimeError("Source config changed while preparing isolated plan")

    report: dict[str, Any] = {
        "schema": ROLE_PLAN_SCHEMA,
        "status": "ROLE_CONTRACT_RESOLVED_NO_SCIENCE_EXECUTION",
        "source_config": str(source),
        "source_config_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "source_raw_inventory": str(inventory),
        "source_raw_manifest_sha256": digest,
        "isolated_config": str(isolated_config),
        "isolated_manifest": str(isolated_manifest),
        "resolved_manifest_path": str(meta["path"]),
        "resolved_frames_source": meta["frames_source"],
        "resolved_roles": {
            role: [str(p) for p in paths] for role, paths in expected.items()
        },
        "observational_roles": {
            "s22800604.fts": "OFFSET_SKY_OBSERVER_CONFIRMED",
            **{name: "ON_TARGET_SCIENCE" for name in ON_TARGET},
        },
        "sky_spatial_cleanliness_verified": False,
        "sky_temporal_transfer_verified": False,
        "sky_subtraction_approved": False,
        "stack_pointing_approved": False,
        "wavelength_lsf_approved": False,
        "science_e2e_approved": False,
        "stage_execution_authorized": False,
    }
    with (output / "role_contract.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-config", required=True, type=Path)
    parser.add_argument("--raw-inventory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.saved_config, args.raw_inventory, args.output_dir)
    print("STATUS:", result["status"])
    print("ON TARGET SCIENCE:", len(result["resolved_roles"]["obj"]))
    print("DEDICATED OFFSET SKY:", len(result["resolved_roles"]["sky"]))
    print("RAW SHA256:", result["source_raw_manifest_sha256"])
    print("PLAN:", str(Path(result["isolated_config"]).parent))
    print("STAGES EXECUTED: NONE. SCIENCE HOLD: ACTIVE")


if __name__ == "__main__":
    main()
