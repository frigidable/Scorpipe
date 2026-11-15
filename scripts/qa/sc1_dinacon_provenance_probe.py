#!/usr/bin/env python3
"""Search for DINACON acquisition provenance without approving CCD geometry.

Evidence: successful original P0 and B0 reports. Only selected FITS primary
headers are read; no pixels, controller instructions, stages or corrections.
CFG-like files are hashed and listed, NEVER executed, parsed as authoritative,
or associated automatically with the December 2024 observing night.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "SC1_DINACON_PROVENANCE_PROBE_V1"
B0_SCHEMA = "SC1_N2146SAT_CCD_HEADER_SECTIONS_V1"
P0_SCHEMA = "SC1_N2146SAT_PILOT_EVIDENCE_GATE_V1"
ANCHORS = ("s22800101.fts", "s22800604.fts", "s22800607.fts",
           "s22800614.fts", "s22800606.fts")
HEADER_KEYS = (
    "ORIGIN", "FILE", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
    "CTYPE1", "CTYPE2", "CDELT1", "CDELT2",
    "DETECTOR", "NODE", "BINNING", "RATE", "GAIN", "NAXIS1", "NAXIS2",
    "MODE", "IMAGETYP", "DATE", "DATE-OBS", "START", "UT",
    "CAMERA", "CCDNAME", "CCDSUM", "INSTRUME",
)
EXACT_NAMES = {"chip.cfg", "fits.hdr", "fits_.hdr", "f_0.txt"}
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache"}
MAX_NAMES = 200_000
MAX_CANDIDATES = 250
MAX_SCAN_FILE = 5_000_000
MAX_JSON = 20_000_000


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as inp:
        for b in iter(lambda: inp.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    if path.stat().st_size > MAX_JSON:
        raise ValueError(f"Evidence JSON too large: {path}")
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError("Evidence JSON must contain an object")
    return obj


def real_file(name: object) -> Path:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Missing evidence source path")
    original = Path(name).expanduser()
    if original.is_symlink():
        raise ValueError(f"Symlinked evidence path prohibited: {original}")
    p = original.resolve(strict=True)
    if not p.is_file():
        raise ValueError(f"Evidence must be a regular file: {p}")
    return p


def source_kind(name: str) -> str | None:
    s = name.lower()
    if s in EXACT_NAMES:
        return "DINASYSTEM_NAMED_CONFIGURATION"
    if s.startswith("init") and s.endswith(".cfg"):
        return "DINACON_INITIALIZATION_CANDIDATE"
    if s.startswith("tele") and s.endswith(".cfg"):
        return "DINACON_TELEMETRY_CANDIDATE"
    if s.endswith((".ngr", ".ldr")):
        return "DINACON_CLOCK_OR_FIRMWARE_CANDIDATE"
    if s.endswith((".bat", ".cmd")) and any(x in s for x in ("dina", "ccd", "scorpio")):
        return "DINACMD_COMMAND_CANDIDATE"
    if s.endswith((".cfg", ".ini")) and any(x in s for x in ("dina", "ccd", "camera", "controller")):
        return "CAMERA_CONFIGURATION_CANDIDATE"
    return None


def scan(roots: list[Path]) -> tuple[list[dict], int]:
    visited = 0
    rows: list[dict] = []
    discovered: set[Path] = set()
    for root in roots:
        for base, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [
                d for d in dirs
                if d.lower() not in SKIP_DIRS
                and not (Path(base) / d).is_symlink()
            ]
            for name in files:
                visited += 1
                if visited > MAX_NAMES:
                    raise ValueError("File-name scan safety limit reached")
                kind = source_kind(name)
                if kind is None:
                    continue
                p = Path(base) / name
                if p.is_symlink():
                    continue
                p = p.resolve()
                if p in discovered:
                    continue
                discovered.add(p)
                if len(rows) >= MAX_CANDIDATES:
                    raise ValueError("Too many technical candidates for bounded intake")
                st = p.stat()
                record = {
                    "path": str(p), "name": name, "kind": kind,
                    "size_bytes": st.st_size,
                    "mtime_utc": datetime.fromtimestamp(
                        st.st_mtime, timezone.utc).isoformat(),
                    "2024_night_origin_authenticated": False,
                    "2024_readout_settings_authenticated": False,
                }
                if st.st_size <= MAX_SCAN_FILE:
                    record["sha256"] = digest(p)
                else:
                    record["skip_reason"] = "SIZE_LIMIT_NO_HASH_NO_CONTENT_READ"
                rows.append(record)
    return sorted(rows, key=lambda r: r["path"].lower()), visited


def probe_header(path: Path) -> dict:
    from astropy.io import fits
    h = fits.getheader(path, 0, ignore_missing_end=False)
    cards = {k: str(h[k])[:300] for k in HEADER_KEYS if k in h}
    notes = []
    for card in h.cards:
        if card.keyword in {"COMMENT", "HISTORY"}:
            if len(notes) < 40:
                notes.append({
                    "kind": card.keyword,
                    "text": str(card.value)[:250],
                })
    return {"cards": cards, "comment_history_sample": notes}


def prepare(p0_file: Path, b0_file: Path, roots: list[Path], output: Path) -> dict:
    p0_path = real_file(str(p0_file))
    b0_path = real_file(str(b0_file))
    b0_fingerprint = digest(b0_path)
    p0 = load_json(p0_path)
    b0 = load_json(b0_path)
    if (p0.get("schema") != P0_SCHEMA or
        p0.get("status") != "EVIDENCE_CONSISTENT_SCIENCE_HOLD" or
        p0.get("science_hold") is not True or
        p0.get("science_qualified") is not False or
        p0.get("stage_execution_authorized") is not False):
        raise ValueError("P0 source evidence is not the science-held pilot")
    if (b0.get("schema") != B0_SCHEMA or
        b0.get("status") != "HEADER_SECTION_INVENTORY_DIAGNOSTIC_SCIENCE_HOLD" or
        b0.get("science_hold") is not True or
        b0.get("physical_overscan_region_certified") is not False or
        b0.get("stage_execution_authorized") is not False or
        b0.get("raw_frames_sha256_reverified") != 32 or
        b0.get("pilot_report_sha256") != digest(p0_path)):
        raise ValueError("B0 source evidence not matched to science-held P0")
    if (p0.get("evidence_sha256", {}).get("raw_input_manifest_sha256")
            != b0.get("raw_input_manifest_sha256")):
        raise ValueError("P0/B0 RAW manifest fingerprint differs")
    raw_path = real_file(p0["source_paths"]["raw_inventory"])
    if digest(raw_path) != p0["evidence_sha256"]["raw_inventory_json_sha256"]:
        raise ValueError("Original RAW inventory JSON changed")
    original = load_json(raw_path)
    entries = original.get("input_files")
    frames = b0.get("frames")
    if not isinstance(entries, list) or len(entries) != 32:
        raise ValueError("Original evidence must contain 32 source entries")
    if not isinstance(frames, list) or len(frames) != 32:
        raise ValueError("B0 must contain 32 source entries")
    b0_names = {row.get("name"): row for row in frames}
    if len(b0_names) != 32:
        raise ValueError("Duplicated B0 frame identities")
    raw_names = {row.get("filename"): row for row in entries}
    if len(raw_names) != 32:
        raise ValueError("Duplicated original manifest names")
    for name in ANCHORS:
        if (name not in b0_names or name not in raw_names or
            b0_names[name].get("sha256") != raw_names[name].get("source_sha256")):
            raise ValueError(f"Missing/mismatched anchor: {name}")

    out = Path(output).expanduser().resolve()
    if out.suffix.lower() != ".json" or out.exists():
        raise ValueError("Output must be a new, nonexisting JSON file")
    protected = [
        Path(str(original["raw_data_dir"])).resolve(strict=True),
        real_file(p0["source_paths"]["saved_config"]).parent,
        real_file(p0["source_paths"]["role_contract"]).parent,
    ]
    if any(out == p or p in out.parents for p in protected):
        raise ValueError("Output would modify a protected original directory")
    targets = [Path(r).expanduser().resolve(strict=True) for r in roots]
    if not targets or any(not x.is_dir() for x in targets):
        raise ValueError("At least one existing scan directory is required")

    header_rows = []
    for name in ANCHORS:
        row = raw_names[name]
        src = real_file(row["source_path"])
        current = digest(src)
        if current != row["source_sha256"]:
            raise ValueError(f"Anchor RAW source changed: {name}")
        header_rows.append({
            "name": name, "sha256_reverified": current, **probe_header(src)
        })
    candidate_rows, count = scan(targets)
    if digest(p0_path) != b0["pilot_report_sha256"]:
        raise ValueError("P0 changed during examination")
    if digest(b0_path) != b0_fingerprint:
        raise ValueError("B0 changed during examination")
    if digest(raw_path) != p0["evidence_sha256"]["raw_inventory_json_sha256"]:
        raise ValueError("Source inventory changed during examination")

    result = {
        "schema": SCHEMA,
        "status": "TECHNICAL_PROVENANCE_DISCOVERY_ONLY_SCIENCE_HOLD",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "p0_sha256": digest(p0_path),
        "b0_sha256": b0_fingerprint,
        "raw_manifest_sha256": b0["raw_input_manifest_sha256"],
        "raw_anchor_headers_read": len(header_rows),
        "raw_anchor_source_sha256_reverified": len(header_rows),
        "raw_fits_pixel_arrays_read": False,
        "raw_anchor_headers": header_rows,
        "scan_roots": [str(p) for p in targets],
        "file_names_screened": count,
        "technical_file_candidates": candidate_rows,
        "candidate_counts": dict(Counter(r["kind"] for r in candidate_rows)),
        "control_2024_readout_configuration_found_and_authenticated": False,
        "source_controller_firmware_authenticated": False,
        "raw_active_image_section_certified": False,
        "physical_overscan_region_certified": False,
        "overscan_subtraction_approved": False,
        "stage_execution_authorized": False,
        "science_qualified": False,
        "science_hold": True,
        "limitations": [
            "No raw FITS pixel samples were opened; no scientific stage executed.",
            "Technical candidates identified only by filenames and hashed; no candidate contents executed or interpreted.",
            "Filename, timestamp, and SHA alone cannot prove a file was used on 2024-12-06.",
            "Historic Dina System and DINACMD manuals describe configurable overscan counts, not the actual 2024 controller settings.",
            "No section/correction approval is possible from this discovery report.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0", type=Path, required=True)
    parser.add_argument("--b0", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    r = prepare(a.p0, a.b0, a.scan_root, a.out)
    print("STATUS:", r["status"])
    print("HEADER ANCHORS VERIFIED:", r["raw_anchor_source_sha256_reverified"])
    print("FILES SCREENED:", r["file_names_screened"])
    print("TECHNICAL CANDIDATES:", len(r["technical_file_candidates"]))
    print("CANDIDATE TYPES:", r["candidate_counts"])
    print("REPORT:", a.out)
    print("ORIGINAL 2024 CONTROLLER CONFIG AUTHENTICATED: NO")
    print("OVERSCAN AND SCIENCE: HOLD; STAGES EXECUTED: NONE")


if __name__ == "__main__":
    main()
