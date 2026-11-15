#!/usr/bin/env python3
"""Read-only SC1 NGC2146sat CCD-header geometry intake; NOT overscan approval.

Rehash all 32 existing original RAW FITS against a successful real pilot P0
report and record detector/amplifier section header cards. Never read CCD
pixel arrays, apply trim/overscan or create a science product.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "SC1_N2146SAT_CCD_HEADER_SECTIONS_V1"
PILOT_SCHEMA = "SC1_N2146SAT_PILOT_EVIDENCE_GATE_V1"
RAW_SCHEMA = "SC1_RAW_PIXEL_READABILITY_V1"
EXPECTED_ROLES = {"bias": 20, "obj": 7, "flat": 3, "neon": 2}
EXPECTED_ON = {f"s228006{i:02d}.fts" for i in range(7, 13)}
EXPECTED_OFF = "s22800604.fts"
EXPECTED_SHAPE_YX = (1032, 2080)
SECTION_KEYS = (
    "BIASSEC", "DATASEC", "TRIMSEC", "CCDSEC", "DETSEC", "IMAGESEC",
    "BIASSECA", "BIASSECB", "DATASECA", "DATASECB",
    "BSEC11", "BSEC12", "DSEC11", "DSEC12",
    "CSEC11", "CSEC12", "ASEC11", "ASEC12",
)
HEADER_KEYS = (
    "INSTRUME", "OBJECT", "IMAGETYP", "FILTERS", "NODE", "RATE", "GAIN",
    "BINNING", "CCDSUM", "CCDNAME", "DETNAME", "DETECTOR", "AMPLIFIER",
    "NAXIS1", "NAXIS2", "BITPIX", "BSCALE", "BZERO",
)
SECTION_RE = re.compile(
    r"^\s*\[\s*(\d+)\s*:\s*(\d+)\s*,\s*(\d+)\s*:\s*(\d+)\s*\]\s*$"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    if path.stat().st_size > 20_000_000:
        raise ValueError("Oversized evidence JSON")
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError("Evidence must be JSON object")
    return obj


def source_file(name: str) -> Path:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Missing evidence source path")
    p = Path(name).expanduser().resolve(strict=True)
    if not p.is_file():
        raise ValueError("Evidence source not a regular file")
    return p


def parse_section(text: str, *, nx: int = 2080, ny: int = 1032) -> dict:
    """FITS one-based inclusive sections; do NOT infer physical meaning."""
    m = SECTION_RE.fullmatch(str(text))
    if not m:
        return {"literal": str(text), "parse_ok": False, "reason": "NOT_SIMPLE_2D_SECTION"}
    x1, x2, y1, y2 = map(int, m.groups())
    if not (1 <= x1 <= nx and 1 <= x2 <= nx and 1 <= y1 <= ny and 1 <= y2 <= ny):
        return {"literal": str(text), "parse_ok": False, "reason": "OUTSIDE_RAW_HEADER_BOUNDS"}
    return {
        "literal": str(text), "parse_ok": True,
        "python_x_zero_based_half_open": [min(x1, x2) - 1, max(x1, x2)],
        "python_y_zero_based_half_open": [min(y1, y2) - 1, max(y1, y2)],
        "reversed_x": x2 < x1, "reversed_y": y2 < y1,
    }


def overlap(a: dict, b: dict) -> bool:
    if not (a.get("parse_ok") and b.get("parse_ok")):
        return False
    ax, bx = a["python_x_zero_based_half_open"], b["python_x_zero_based_half_open"]
    ay, by = a["python_y_zero_based_half_open"], b["python_y_zero_based_half_open"]
    return (max(ax[0], bx[0]) < min(ax[1], bx[1])
            and max(ay[0], by[0]) < min(ay[1], by[1]))


def inspect_header(path: Path) -> dict:
    # fits.getheader accesses primary header only. Never touch HDU.data.
    from astropy.io import fits
    h = fits.getheader(path, 0, ignore_missing_end=False)
    if (int(h.get("NAXIS", -1)) != 2
            or (int(h.get("NAXIS2", -1)), int(h.get("NAXIS1", -1))) != EXPECTED_SHAPE_YX):
        raise ValueError(f"Unexpected RAW detector shape: {path.name}")
    cards = {k: str(h[k]) for k in HEADER_KEYS if k in h}
    raw_sec = {k: str(h[k]) for k in SECTION_KEYS if k in h}
    for k in h.keys():
        name = str(k).upper()
        if re.fullmatch(r"(?:BSEC|DSEC|CSEC|ASEC)\d{1,3}", name):
            raw_sec[name] = str(h[k])
    sections = {k: parse_section(v) for k, v in sorted(raw_sec.items())}
    conflict = []
    for d, b in (("DATASEC", "BIASSEC"), ("DATASECA", "BIASSECA"),
                 ("DATASECB", "BIASSECB")):
        if d in sections and b in sections and overlap(sections[d], sections[b]):
            conflict.append(f"DECLARED_{d}_{b}_OVERLAP")
    return {
        "cards": cards, "declared_sections": sections,
        "section_conflicts": conflict,
        "unparseable_sections": [k for k, v in sections.items() if not v["parse_ok"]],
    }


def prepare(pilot_report: Path, destination: Path) -> dict:
    pilot_path = source_file(str(pilot_report))
    pilot_sha = sha256(pilot_path)
    out = Path(destination).expanduser().resolve()
    if out.suffix.lower() != ".json":
        raise ValueError("Output must be a new JSON file")
    if out.exists():
        raise FileExistsError(f"Refuse to overwrite: {out}")
    pilot = read_json(pilot_path)
    if (pilot.get("schema") != PILOT_SCHEMA
            or pilot.get("status") != "EVIDENCE_CONSISTENT_SCIENCE_HOLD"
            or pilot.get("science_hold") is not True
            or pilot.get("science_qualified") is not False
            or pilot.get("stage_execution_authorized") is not False
            or pilot.get("sky_only_rows_certified") != 0):
        raise ValueError("Pilot P0 evidence not approved for metadata study")
    evidence = pilot.get("evidence_sha256") or {}
    paths = pilot.get("source_paths") or {}
    saved = source_file(paths.get("saved_config"))
    raw_file = source_file(paths.get("raw_inventory"))
    role_file = source_file(paths.get("role_contract"))
    for key, p in (("saved_config_sha256", saved),
                   ("raw_inventory_json_sha256", raw_file),
                   ("role_contract_json_sha256", role_file)):
        if sha256(p) != evidence.get(key):
            raise ValueError(f"Pilot evidence changed: {key}")
    role_plan = read_json(role_file)
    isolated = []
    for key, sha_field in (("isolated_config", "isolated_config_sha256"),
                           ("isolated_manifest", "isolated_manifest_sha256")):
        p = source_file(role_plan.get(key))
        if sha256(p) != evidence.get(sha_field):
            raise ValueError(f"Isolated role file changed: {key}")
        isolated.append((p, sha_field))

    raw = read_json(raw_file)
    if (raw.get("schema") != RAW_SCHEMA
            or raw.get("input_manifest_sha256") != evidence.get("raw_input_manifest_sha256")):
        raise ValueError("RAW inventory identity inconsistent")
    input_files = raw.get("input_files")
    if not isinstance(input_files, list) or len(input_files) != 32:
        raise ValueError("Exactly 32 original RAW entries required")
    roles = [r for role, n in EXPECTED_ROLES.items() for r in [role] * n]
    raw_root = Path(str(raw["raw_data_dir"])).expanduser().resolve(strict=True)
    protected = [raw_root, saved.parent, isolated[0][0].parent]
    protected += [source_file(x["source_path"]).parent for x in input_files]
    if any(out == p or p in out.parents for p in protected):
        raise ValueError("Report must be outside source RAW, old and isolated roots")

    observed = []
    names = {k: [] for k in EXPECTED_ROLES}
    seen = set()
    readout_modes = Counter()
    for idx, (entry, role) in enumerate(zip(input_files, roles)):
        if not isinstance(entry, dict) or entry.get("role") != role:
            raise ValueError(f"RAW role order changed at {idx}")
        p = source_file(entry.get("source_path"))
        if p in seen or p.name.lower() != str(entry.get("filename", "")).lower():
            raise ValueError(f"Duplicate/inconsistent original file: {p.name}")
        seen.add(p)
        before = p.stat()
        if before.st_size != entry.get("size_bytes"):
            raise ValueError(f"Source size changed: {p.name}")
        h = inspect_header(p)
        current_sha = sha256(p)
        after = p.stat()
        if current_sha != entry.get("source_sha256") or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Original FITS changed: {p.name}")
        names[role].append(p.name.lower())
        fields = h["cards"]
        readout_modes[(role, *(fields.get(k, "MISSING") for k in ("NODE", "RATE", "GAIN", "BINNING")))] += 1
        observed.append({"name": p.name, "original_role": role, "sha256": current_sha, **h})
    if set(names["obj"]) != EXPECTED_ON | {EXPECTED_OFF} or len(names["obj"]) != 7:
        raise ValueError("Original OBJ cohort changed")

    keyword_counts = Counter(k for row in observed for k in row["declared_sections"])
    result = {
        "schema": SCHEMA,
        "status": "HEADER_SECTION_INVENTORY_DIAGNOSTIC_SCIENCE_HOLD",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_report_sha256": pilot_sha,
        "raw_input_manifest_sha256": evidence["raw_input_manifest_sha256"],
        "raw_frames_sha256_reverified": len(observed),
        "original_raw_fits_headers_examined": len(observed),
        "readout_modes": [{"role": k[0], "node": k[1], "rate": k[2],
                           "gain": k[3], "binning": k[4], "n": n}
                          for k, n in sorted(readout_modes.items())],
        "declared_section_keyword_counts": dict(sorted(keyword_counts.items())),
        "any_unparseable_sections": any(x["unparseable_sections"] for x in observed),
        "any_section_overlaps": any(x["section_conflicts"] for x in observed),
        "previous_candidate_regions_only_not_certified": {
            "left_x_python_half_open": [2, 14],
            "right_x_python_half_open": [2066, 2078],
            "interior_x_python_half_open": [32, 2048],
        },
        "frames": observed,
        "raw_fits_pixels_read": False,
        "physical_overscan_region_certified": False,
        "overscan_correction_approved": False,
        "science_qualified": False,
        "stage_execution_authorized": False,
        "science_hold": True,
        "interpretation": (
            "Header geometry declarations, if present, must be confirmed "
            "with the exact detector/readout documentation and independent "
            "light-response diagnostics before any physical overscan subtraction. "
            "No BIASSEC is not evidence that no overscan exists."
        ),
    }
    for key, p in (("saved_config_sha256", saved),
                   ("raw_inventory_json_sha256", raw_file),
                   ("role_contract_json_sha256", role_file)):
        if sha256(p) != evidence[key]:
            raise ValueError(f"Pilot evidence changed during inspection: {key}")
    for p, key in isolated:
        if sha256(p) != evidence[key]:
            raise ValueError(f"Isolated evidence changed during inspection: {key}")
    if sha256(pilot_path) != pilot_sha:
        raise ValueError("Pilot report changed during inspection")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = prepare(args.pilot_report, args.out)
    print("STATUS:", report["status"])
    print("VERIFIED ORIGINAL FITS:", report["raw_frames_sha256_reverified"])
    print("SECTION KEYWORDS:", report["declared_section_keyword_counts"])
    print("READOUT MODES:", report["readout_modes"])
    print("RESULT:", args.out)
    print("OVERSCAN: UNQUALIFIED; SCIENCE HOLD: ACTIVE; STAGES: NONE")


if __name__ == "__main__":
    main()
