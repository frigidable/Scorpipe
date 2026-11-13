#!/usr/bin/env python3
"""Discover saved LineID26c/d/e/f evidence, without rerunning observations.

Only original archival metadata and SHA-256 are inventoried; no CCD reduction,
sky-line analysis or source-free sky certification is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PHASE_RE = re.compile(r"lineid[\s_-]*26[\s_-]*([cdef])", re.IGNORECASE)
SUFFIXES = {".json", ".csv", ".npz", ".txt", ".png", ".zip"}
MAX_SINGLE_BYTES = 150 * 1024 * 1024
MAX_DISCOVERED = 2000


def _inside(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def _phase(path: Path) -> str | None:
    # Prefer a phase marker in the file basename over its containing directory.
    for piece in (path.name, *(part for part in reversed(path.parts[:-1]))):
        m = PHASE_RE.search(piece)
        if m:
            return "26" + m.group(1).lower()
    return None


def _hash_and_check(path: Path) -> tuple[str, int]:
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed during hashing: {path}")
    return h.hexdigest(), after.st_size


def _inspect_structure(path: Path, suffix: str, size: int) -> dict:
    """Read limited archive structure, not pixels or arrays."""
    try:
        if suffix == ".json" and size <= 10 * 1024 * 1024:
            with path.open("r", encoding="utf-8-sig") as stream:
                payload = json.load(stream)
            return {
                "json_root_type": type(payload).__name__,
                "json_root_keys": sorted(str(k) for k in payload)[:80]
                if isinstance(payload, dict) else [],
            }
        if suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                head = stream.readline(8192)
            return {"csv_first_record": next(csv.reader([head]), [])[:80]}
        if suffix in {".npz", ".zip"}:
            with zipfile.ZipFile(path, "r") as archive:
                members = archive.infolist()
            return {
                "archive_n_members": len(members),
                "archive_member_names": [m.filename for m in members[:70]],
                "archive_members_truncated": len(members) > 70,
            }
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError,
            zipfile.BadZipFile, StopIteration) as exc:
        return {"structure_error": f"{type(exc).__name__}: {exc}"}
    return {}


def inventory(roots: list[Path], output: Path) -> dict:
    """No source modifications; exclusive external JSON, no scientific approval."""
    roots = [Path(p).expanduser().resolve(strict=True) for p in roots]
    if not roots or any(not root.is_dir() for root in roots):
        raise ValueError("Specify existing source directories")
    if len({str(root).casefold() for root in roots}) != len(roots):
        raise ValueError("Duplicate source roots")
    output = Path(output).expanduser().resolve()
    if output.suffix.lower() != ".json":
        raise ValueError("Output must be a .json file")
    if output.exists():
        raise FileExistsError(f"Will not overwrite existing report: {output}")
    if any(_inside(output, root) for root in roots):
        raise ValueError("Output must be outside every inspected source root")

    candidates: dict[str, tuple[Path, str]] = {}
    for root in roots:
        for path in root.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            if path.suffix.lower() not in SUFFIXES:
                continue
            phase = _phase(path)
            if phase is None:
                continue
            resolved = path.resolve(strict=True)
            if not _inside(resolved, root):
                raise ValueError(f"File escapes declared root: {path}")
            candidates[str(resolved).casefold()] = (resolved, phase)
            if len(candidates) > MAX_DISCOVERED:
                raise RuntimeError("Too many archival candidates; narrow --root")

    files: list[dict] = []
    skipped_large: list[dict] = []
    for path, phase in sorted(candidates.values(), key=lambda pair: str(pair[0]).casefold()):
        size = path.stat().st_size
        if size > MAX_SINGLE_BYTES:
            skipped_large.append({
                "path": str(path), "phase": phase, "size_bytes": size,
                "reason": "EXCEEDS_ARCHIVAL_INVENTORY_LIMIT_NO_SHA",
            })
            continue
        digest, size = _hash_and_check(path)
        suffix = path.suffix.lower()
        files.append({
            "path": str(path), "phase": phase, "size_bytes": size,
            "sha256": digest, "extension": suffix,
            "structure": _inspect_structure(path, suffix, size),
        })

    present = sorted(set(record["phase"] for record in files))
    report = {
        "schema": "SC1_N2146SAT_LINEID26_CDEF_ARCHIVE_INVENTORY_V1",
        "status": "ARCHIVE_DISCOVERY_DIAGNOSTIC_ONLY",
        "source_roots": [str(p) for p in roots],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files_found": len(files),
        "phases_found": present,
        "phases_missing": sorted({"26c", "26d", "26e", "26f"} - set(present)),
        "files_skipped_large": skipped_large,
        "files": files,
        "science_hold": True,
        "stages_executed": False,
        "original_fits_read": False,
        "source_free_sky_mask_approved": False,
        "certified_sky_rows": 0,
        "sky_subtraction_approved": False,
        "interpretation": (
            "Source-discovery and file-integrity metadata only; archived "
            "measurements are not revalidated or interpreted. A low signal, "
            "valid ROI or absent compact source cannot certify SKY_ONLY."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.root, args.out)
    print("STATUS:", result["status"])
    print("FOUND:", result["files_found"], "PHASES:", result["phases_found"])
    print("MISSING:", result["phases_missing"])
    print("LARGE SKIPPED:", len(result["files_skipped_large"]))
    for item in result["files"]:
        print("  ", item["phase"], item["extension"], item["size_bytes"],
              item["sha256"][:12], item["path"])
    print("REPORT:", args.out)
    print("SKY_ONLY: ZERO APPROVED; SCIENCE HOLD: ACTIVE; STAGES: NONE")


if __name__ == "__main__":
    main()
