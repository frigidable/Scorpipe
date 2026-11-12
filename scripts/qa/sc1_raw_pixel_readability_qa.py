#!/usr/bin/env python3
"""Read-only SCORPIO-1 N2146sat raw FITS pixel readability and SHA-256 inventory.

This is NOT a detector noise, physical overscan, MasterBias, or science E2E
qualification. It does not run stages, modify FITS, or write inside RAW/work.

Usage:
  python scripts/qa/sc1_raw_pixel_readability_qa.py --inspect config.yaml external-new.json
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = "SC1_RAW_PIXEL_READABILITY_V1"
CHUNK_BYTES = 1024 * 1024


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_file(path: Path, role: str, expected_shape: tuple[int, int]) -> dict:
    from astropy.io import fits

    before = path.stat()
    source_hash = _file_sha256(path)
    with fits.open(
        path, mode="readonly", memmap=False, do_not_scale_image_data=True,
        ignore_missing_end=False,
    ) as hdus:
        info = hdus.fileinfo(0)
        if (not info or info.get("datLoc") is None
                or info.get("datSpan") is None):
            raise RuntimeError(f"Cannot establish native FITS image extent: {path}")
        expected_end = int(info["datLoc"]) + int(info["datSpan"])
        if before.st_size < expected_end:
            raise RuntimeError(
                f"Truncated primary FITS payload ({role} {path.name}): "
                f"file has {before.st_size} bytes; need >= {expected_end}"
            )
        primary = hdus[0]
        image = primary.data
        if image is None:
            raise RuntimeError(f"Missing primary raw image: {path}")
        pixels = np.asarray(image)
        if pixels.shape != expected_shape or pixels.ndim != 2:
            raise RuntimeError(
                f"Raw array shape mismatch ({role} {path.name}): "
                f"got {pixels.shape}, expected {expected_shape}"
            )
        if not (
            np.issubdtype(pixels.dtype, np.integer)
            or np.issubdtype(pixels.dtype, np.floating)
        ):
            raise RuntimeError(f"Non-real raw CCD image dtype: {path}")
        bad = int(pixels.size - np.count_nonzero(np.isfinite(pixels)))
        if bad:
            raise RuntimeError(
                f"Non-finite raw pixels ({role} {path.name}): {bad}"
            )
        # No scaling is applied: values are storage samples, not certified ADU.
        min_sample = float(np.min(pixels))
        max_sample = float(np.max(pixels))
        mean_sample = float(np.mean(pixels, dtype=np.float64))
        for value in (min_sample, max_sample, mean_sample):
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite raw pixel summary: {path}")
        result = {
            "role": role,
            "filename": path.name,
            "source_path": str(path),
            "size_bytes": int(before.st_size),
            "source_sha256": source_hash,
            "shape_numpy_y_x": list(pixels.shape),
            "stored_dtype": str(pixels.dtype),
            "stored_min_sample": min_sample,
            "stored_max_sample": max_sample,
            "stored_mean_sample": mean_sample,
            "hdu_count": len(hdus),
            "fits_checksum_keywords_present": sorted(
                key for key in ("CHECKSUM", "DATASUM") if key in primary.header
            ),
            "fits_checksum_validated": False,
        }
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size, after.st_mtime_ns
    ):
        raise RuntimeError(f"RAW file changed during inventory: {path}")
    return result


def inspect(config: Path) -> tuple[dict, list[Path]]:
    """Validate 32 exact-night headers, then load each primary image read-only."""
    from sc1_first_masterbias_gate import (
        EXPECTED_N, EXPECTED_SHAPE, collect,
    )

    header_report, sources = collect(Path(config))
    manifest = []
    index = 0
    for role, count in EXPECTED_N.items():
        group = sources[index:index + count]
        if len(group) != count:
            raise RuntimeError(f"Incomplete {role} RAW group")
        manifest.extend(
            _inspect_file(path, role, EXPECTED_SHAPE) for path in group
        )
        index += count
    if index != len(sources) or len(manifest) != sum(EXPECTED_N.values()):
        raise RuntimeError("Unexpected input manifest length")

    # Root-independent digest of exact role/name/content sequence.
    manifest_canonical = "\n".join(
        f"{entry['role']}\t{entry['filename']}\t{entry['source_sha256']}"
        for entry in manifest
    )
    manifest_sha256 = hashlib.sha256(
        manifest_canonical.encode("utf-8")
    ).hexdigest()
    return {
        "schema": SCHEMA,
        "status": "READ_ONLY_PIXEL_READABILITY_NOT_SCIENCE_QUALIFICATION",
        "header_preflight_status": header_report["status"],
        "source_config": header_report["config"],
        "work_dir": header_report["work_dir"],
        "raw_data_dir": header_report["raw_data_dir"],
        "input_counts": header_report["input_counts"],
        "input_manifest_sha256": manifest_sha256,
        "input_files": manifest,
        "all_selected_primary_arrays_readable_and_finite": True,
        "raw_file_origin_authenticated": False,
        "raw_pixel_units_and_gain_physically_verified": False,
        "physical_overscan_region_certified": False,
        "masterbias_and_covariance_validated": False,
        "wavelength_lsf_science_validated": False,
        "warnings": [
            "SHA-256 fingerprints document present bytes; they do not authenticate observational origin.",
            "Only selected primary FITS arrays are inspected, not unselected frames.",
            "FITS CHECKSUM/DATASUM keyword validity is not asserted.",
            "Stored pixel samples are unscaled; these summaries are not a calibrated ADU/RN measurement.",
            "No science calibration, physical overscan or downstream uncertainty is qualified.",
        ],
    }, sources


def run(config: Path, report_file: Path) -> None:
    report, selected = inspect(config)
    target = Path(report_file).expanduser().resolve()
    work = Path(report["work_dir"]).resolve(strict=True)
    raw_root = Path(report["raw_data_dir"]).resolve(strict=True)
    source = Path(report["source_config"]).resolve(strict=True)
    forbidden_roots = {work, raw_root, *(p.parent.resolve() for p in selected)}
    if (target == source or target in selected
            or any(target == root or target.is_relative_to(root)
                   for root in forbidden_roots)):
        raise ValueError("Report must be outside the work and configured RAW roots")
    if target.suffix.lower() != ".json":
        raise ValueError("Report filename must end in .json")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {target}")
    report["created_utc"] = datetime.now(timezone.utc).isoformat()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
    print("RAW READABILITY:", report["status"])
    print("FRAMES:", len(report["input_files"]))
    print("MANIFEST SHA256:", report["input_manifest_sha256"])
    print("OUTPUT:", target)
    print("SCIENCE E2E: HOLD")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--inspect":
        run(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit(
            "Usage: sc1_raw_pixel_readability_qa.py --inspect "
            "config.yaml external-new-report.json"
        )
