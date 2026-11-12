#!/usr/bin/env python3
"""Read-only replay of the archived LineID26c native 5199-A window.

Reports conditional raw-CCD sky-feature proxies, NOT true sky flux,
source-free slit windows, flexure, time-transfer coefficients or permission
to run SkySub. LineID26c/d/e/f observations are archival science evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "SC1_N2146SAT_LINEID26G_TEMPORAL_REVIEW_V1"
FILENAMES = ["s22800604.fts", *[f"s228006{i:02d}.fts" for i in range(7, 13)]]
SECONDS = [600.0] + [1200.0] * 6
# Candidate regions from LineID26f; neither is a certified SKY_ONLY mask.
REGIONS = {"LOWER_REVIEW": (96, 320), "UPPER_REVIEW": (640, 864)}
SPATIAL_BLOCK = 16
# 37 native pixels in the original LineID26c extraction around 5199.
WING_LEFT = slice(0, 7)
WING_RIGHT = slice(30, 37)
CORE = slice(13, 24)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_contract(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "SC1_N2146SAT_OFFSET_SKY_ROLE_CONTRACT_V1":
        raise ValueError("Require a verified N2146sat role_contract.json")
    if report.get("status") != "ROLE_CONTRACT_RESOLVED_NO_SCIENCE_EXECUTION":
        raise ValueError("Role resolution has not succeeded")
    roles = report.get("resolved_roles") or {}
    observed = [str(p).replace("\\", "/").split("/")[-1].lower()
                for p in roles.get("sky", [])]
    observed += [str(p).replace("\\", "/").split("/")[-1].lower()
                 for p in roles.get("obj", [])]
    if observed != FILENAMES:
        raise ValueError(f"Wrong OFF/ON role order in role contract: {observed}")
    if (report.get("sky_subtraction_approved") is not False
            or report.get("stage_execution_authorized") is not False):
        raise ValueError("Source report must forbid science execution")
    return report


def npz_inventory(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: {
                "shape": list(archive[key].shape),
                "dtype": str(archive[key].dtype),
            }
            for key in sorted(archive.files)
        }


def read_cube(path: Path, key: str | None) -> tuple[np.ndarray, str, dict]:
    inventory = npz_inventory(path)
    candidates = [
        name for name, record in inventory.items()
        if record["shape"] == [7, 1032, 37]
        and np.dtype(record["dtype"]).kind in "iuf"
    ]
    if key is None:
        if len(candidates) != 1:
            raise ValueError(
                "No unique (7,1032,37) numeric cube. Use --inspect and "
                f"explicit --cube-key. Candidates={candidates}"
            )
        key = candidates[0]
    if key not in inventory:
        raise ValueError(f"Cube key {key!r} missing; use --inspect")
    with np.load(path, allow_pickle=False) as archive:
        arr = archive[key]
        if arr.shape != (7, 1032, 37):
            raise ValueError(f"Cube {key!r} must be 7x1032x37, got {arr.shape}")
        if arr.dtype.kind not in "iuf":
            raise ValueError("Cube must be real numerical raw CCD samples")
        cube = np.asarray(arr, dtype=np.float64)
    if not np.isfinite(cube).all():
        raise ValueError("Nonfinite raw counts in selected sample cube")
    return cube, key, inventory


def row_feature_proxy(cube: np.ndarray) -> np.ndarray:
    """Subtract wing-interpolated continuum and sum 11 core pixels.

    This is a CONDITIONAL raw CCD contrast proxy, with no positivity
    clipping, measured sky line identity, physical variance or calibration.
    """
    if cube.shape != (7, 1032, 37) or not np.isfinite(cube).all():
        raise ValueError("Expected seven finite 1032x37 original CCD cutouts")
    low = np.median(cube[..., WING_LEFT], axis=-1)
    high = np.median(cube[..., WING_RIGHT], axis=-1)
    xcore = np.arange(13, 24, dtype=np.float64)
    baseline = (
        low[..., None]
        + ((high - low) / 30.0)[..., None] * (xcore - 3.0)
    )
    return np.sum(cube[..., CORE] - baseline, axis=-1)


def _finite_number(x: float) -> float | None:
    return float(x) if math.isfinite(float(x)) else None


def region_statistics(features: np.ndarray) -> list[dict]:
    if features.shape != (7, 1032):
        raise ValueError("Features must have seven 1032-row frame profiles")
    rows: list[dict] = []
    for label, (start, stop) in REGIONS.items():
        # Alternating 16-row blocks are diagnostic crosschecks, not
        # independent stochastic draws or proven source-free sky masks.
        nblocks = (stop - start) // SPATIAL_BLOCK
        medians = np.stack([
            np.median(
                features[:, start + b * SPATIAL_BLOCK:
                            start + (b + 1) * SPATIAL_BLOCK],
                axis=1,
            )
            for b in range(nblocks)
        ], axis=1)
        off_rate = float(np.median(features[0, start:stop]) / SECONDS[0])
        for i, name in enumerate(FILENAMES):
            block = medians[i]
            center = float(np.median(features[i, start:stop]))
            rate = center / SECONDS[i]
            ratio = rate / off_rate if off_rate > 1e-12 else None
            even_med = float(np.median(block[::2]))
            odd_med = float(np.median(block[1::2]))
            rows.append({
                "frame": name,
                "observational_role": "OFF_SKY" if i == 0 else "ON_TARGET",
                "region": label,
                "y_start_inclusive": start,
                "y_stop_exclusive": stop,
                "n_rows": stop - start,
                "block_size_rows": SPATIAL_BLOCK,
                "block_medians_raw_counts": [_finite_number(v) for v in block],
                "median_line_core_proxy_raw_counts": _finite_number(center),
                "median_proxy_raw_counts_per_sec": _finite_number(rate),
                "rate_ratio_relative_to_OFF04": (
                    _finite_number(ratio) if ratio is not None else None
                ),
                "even_block_median_counts": _finite_number(even_med),
                "odd_block_median_counts": _finite_number(odd_med),
                "even_odd_absolute_difference_counts": abs(even_med - odd_med),
            })
    return rows


def prepare(
    archive_npz: Path,
    role_contract: Path,
    output: Path,
    *,
    key: str | None = None,
    asserted_order: bool = False,
) -> dict:
    archive_npz = archive_npz.expanduser().resolve(strict=True)
    role_contract = role_contract.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    contract = source_contract(role_contract)
    old_work = Path(contract["source_config"]).parent.resolve()
    raw_roots = {
        Path(path).parent.resolve()
        for role in ("obj", "sky", "bias", "flat", "neon")
        for path in contract["resolved_roles"].get(role, [])
    }
    protected = {old_work, *raw_roots}
    if (output in {archive_npz, role_contract}
            or any(output == root or output.is_relative_to(root)
                   for root in protected)):
        raise ValueError("Output must not be inside prior source, RAW or work roots")
    if output.exists():
        raise FileExistsError("Refusing to overwrite sky transfer review")
    if not asserted_order:
        raise ValueError(
            "Array frame axis is not self-verifying. Supply explicit "
            "--assert-26c-frame-order after reviewing --inspect"
        )
    cube, cube_key, inventory = read_cube(archive_npz, key)
    comparisons = region_statistics(row_feature_proxy(cube))
    report = {
        "schema": SCHEMA,
        "status": "RAW_FEATURE_TIME_TRANSFER_DIAGNOSTIC_ONLY",
        "role_contract_sha256": sha256_file(role_contract),
        "raw_inventory_manifest_sha256": contract["source_raw_manifest_sha256"],
        "lineid26c_npz": str(archive_npz),
        "lineid26c_npz_sha256": sha256_file(archive_npz),
        "source_cube_key": cube_key,
        "source_array_schema": inventory,
        "assumed_frame_axis": FILENAMES,
        "frame_axis_basis":
            "operator_asserted_lineid26c_original_exports_NOT_source_cryptographic_proof",
        "nominal_exposure_seconds_from_historical_observing_metadata": SECONDS,
        "reference_line_label": "5199_A_UNIDENTIFIED_CONDITIONAL_FEATURE",
        "method":
            "native 37pix core11 minus linear interpolation between median 7pix wings",
        "spatial_review_regions": REGIONS,
        "comparisons": comparisons,
        "source_spatial_mask_approved": False,
        "independent_atomic_line_id_approved": False,
        "time_transfer_coefficient_approved": False,
        "sky_subtraction_approved": False,
        "masterbias_physical_model_validated": False,
        "wavelength_lsf_validated": False,
        "covariance_qualified": False,
        "stages_executed": False,
        "limitations": [
            "Raw ADC line-minus-wing proxies, NOT calibrated sky fluxes or S/N.",
            "Approximate old-map 5199 center; fixed native windows can drift.",
            "Wings may contain emission lines, continuum and readout structure.",
            "OFF differs in pointing and time from every ON exposure.",
            "Neither provisional y region is certified source-free sky.",
            "Alternating blocks share detector/sky systematics.",
            "Rate ratios are diagnostics, not sky-subtraction coefficients.",
            "Frame axis was operator-asserted, not authenticated from the NPZ.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--role-contract", required=True, type=Path)
    parser.add_argument("--inspect", action="store_true",
                        help="Only display NPZ keys, dimensions, dtypes and SHA-256")
    parser.add_argument("--cube-key")
    parser.add_argument("--assert-26c-frame-order", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.inspect:
        print(json.dumps({
            "file": str(args.npz),
            "sha256": sha256_file(args.npz),
            "npz_keys": npz_inventory(args.npz),
        }, indent=2))
        return
    if args.out is None:
        parser.error("--out is required for analysis")
    result = prepare(
        args.npz, args.role_contract, args.out, key=args.cube_key,
        asserted_order=args.assert_26c_frame_order,
    )
    print("STATUS:", result["status"])
    print("COMPARISONS:", len(result["comparisons"]))
    print("OUTPUT:", args.out)
    print("SCIENCE HOLD: ACTIVE; STAGES EXECUTED: NONE")


if __name__ == "__main__":
    main()
