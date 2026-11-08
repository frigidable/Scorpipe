#!/usr/bin/env python3
"""Read-only SCORPIO-1 N2146sat edge-electronics characterization.

Two 16-column strips are CANDIDATES only, not certified overscan. Tests
whether these strips have real read noise, stable column boundaries, stable
left-right offsets, and common temporal behavior in 41 BIAS and 7 OBJ.
Also checks optical nonresponse in 3 FLAT + 2 NEON of *different readout*.
Never modifies existing raw/stage files and NEVER applies an overscan
correction, updates a config, or enables scientific E2E.

Usage: python scripts/qa/sc1_serial_edge_candidate_qa.py --selftest
       python scripts/qa/sc1_serial_edge_candidate_qa.py config.yaml external.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EXPECTED_SHAPE = (1032, 2080)
# Deliberately provisional: test only, NOT detector-section authority.
EDGE = {"left": (0, 16), "right": (2064, 2080)}
INNER = {"left": (16, 32), "right": (2048, 2064)}
Y_START, Y_STOP = 32, 992
Y_SEGMENTS = ((32, 288), (288, 736), (736, 992))
REQUIRED_COUNTS = {
    "bias_early20": 20,
    "bias_late19": 19,
    "bias_flagged2": 2,
    "obj_selected7": 7,
    "flat_selected3": 3,
    "neon_selected2": 2,
}
SCHEMA = "SC1_SERIAL_EDGE_CANDIDATE_READ_ONLY_V1"


def robust_sample_metrics(sample):
    """Trim 30-ADU transients, return robust mean AND untrimmed distribution."""
    x = np.asarray(sample, dtype=np.float64).ravel()
    if not x.size or not np.all(np.isfinite(x)):
        raise ValueError("Invalid candidate stripe pixel sample")
    median = float(np.median(x))
    inlier = np.abs(x - median) <= 30.0
    if np.mean(inlier) < .99:
        raise ValueError("Excessive candidate edge outlier fraction")
    v = x[inlier]
    _, counts = np.unique(x, return_counts=True)
    return {
        "sample_pixel_count": int(x.size),
        "median_adu": median,
        "trimmed_mean_adu": float(np.mean(v)),
        "trimmed_std_adu": float(np.std(v)),
        "trimmed_pixel_count": int(v.size),
        "fraction_rejected_at_30adu": float(1 - v.size / x.size),
        "unique_integer_level_count": int(np.unique(x).size),
        "modal_value_fraction": float(counts.max() / x.size),
        "min_adu": float(np.min(x)),
        "max_adu": float(np.max(x)),
    }


def characterize(data):
    """No correction, no detector-geometry certification."""
    a = np.asarray(data)
    if a.shape != EXPECTED_SHAPE:
        raise ValueError("Unexpected CCD shape")
    if not np.issubdtype(a.dtype, np.number):
        raise TypeError("CCD array must be numerical")
    out = {
        "raw_shape_numpy_y_x": list(EXPECTED_SHAPE),
        "candidate_overscan_certified": False,
        "x_illuminated_section_certified": False,
    }
    for label, bands in (("candidate", EDGE), ("neighbor", INNER)):
        out[label] = {}
        for side, (lo, hi) in bands.items():
            block = a[Y_START:Y_STOP, lo:hi]
            met = robust_sample_metrics(block)
            met["x0_x1"] = [lo, hi]
            met["y0_y1"] = [Y_START, Y_STOP]
            met["column_medians_adu"] = [
                float(np.median(block[:, j]))
                for j in range(hi-lo)
            ]
            met["temporal_row_mean_std_adu"] = float(
                np.std(block.astype(np.float64).mean(axis=1))
            )
            met["y_segment_trimmed_mean_adu"] = [
                robust_sample_metrics(a[ya:yb, lo:hi])["trimmed_mean_adu"]
                for ya, yb in Y_SEGMENTS
            ]
            out[label][side] = met
    out["right_minus_left_candidate_trimmed_mean_adu"] = float(
        out["candidate"]["right"]["trimmed_mean_adu"]
        - out["candidate"]["left"]["trimmed_mean_adu"]
    )
    out["neighbor_minus_candidate_adu"] = {
        side: float(
            out["neighbor"][side]["median_adu"]
            - out["candidate"][side]["median_adu"]
        ) for side in ("left", "right")
    }
    central = np.asarray(a[128:896, 256:1792], dtype=np.float64)
    if not np.isfinite(central).all():
        raise ValueError("Nonfinite central sample")
    central_reference = float(np.median(central))
    out["central_reference_y128_896_x256_1792_median_adu"] = central_reference
    out["vertical_extreme_row_profile_unverified"] = {}
    for label, lo, hi in (("top_0_23", 0, 24),
                          ("bottom_1008_1031", 1008, 1032)):
        slab = np.asarray(a[lo:hi, 256:1792], dtype=np.float64)
        if not np.isfinite(slab).all():
            raise ValueError("Nonfinite vertical edge sample")
        medians = np.median(slab, axis=1)
        out["vertical_extreme_row_profile_unverified"][label] = {
            "y0_y1": [lo, hi],
            "row_median_adu": medians.tolist(),
            "row_minus_middle_reference_adu":
                (medians - central_reference).tolist(),
            "physical_parallel_overscan_certified": False,
        }
    return out


def summarize_modes(results):
    """Avoid cross-mode electronic-zero subtraction."""
    bias = (results["bias_early20"] + results["bias_late19"]
            + results["bias_flagged2"])
    obj = results["obj_selected7"]
    for frame in bias+obj:
        header = frame["header"]
        if (header["NODE"], float(header["RATE"]),
                float(header["GAIN"]), header["BINNING"]) != (
                    "A", 145.0, .9, "1x2"):
            raise RuntimeError("Incompatible BIAS / OBJ CCD readout")
    for group in ("flat_selected3", "neon_selected2"):
        for frame in results[group]:
            header = frame["header"]
            if (header["NODE"], float(header["RATE"]),
                    float(header["GAIN"]), header["BINNING"]) != (
                        "A", 225.0, 3.5, "1x2"):
                raise RuntimeError("Unexpected lamp CCD readout")
    ref = {
        side: float(np.median([
            f["statistics"]["candidate"][side]["trimmed_mean_adu"]
            for f in results["bias_early20"]
        ]))
        for side in ("left", "right")
    }
    diagnostic = []
    for kind in ("bias_early20", "bias_late19", "bias_flagged2",
                 "obj_selected7"):
        for f in results[kind]:
            edges = f["statistics"]["candidate"]
            row = {
                "group": kind, "filename": f["filename"],
                "DATE": f["header"]["DATE"],
                "delta_left_against_early20_median_adu": float(
                    edges["left"]["trimmed_mean_adu"] - ref["left"]),
                "delta_right_against_early20_median_adu": float(
                    edges["right"]["trimmed_mean_adu"] - ref["right"]),
                "left_minus_right_delta_disagreement_adu": float(
                    edges["left"]["trimmed_mean_adu"] - ref["left"]
                    - edges["right"]["trimmed_mean_adu"] + ref["right"]),
            }
            diagnostic.append(row)
    return {
        "reference_early20_edge_trimmed_means_adu": ref,
        "same_readout_temporal_candidate_edge_diagnostics": diagnostic,
        "science_epoch_offset_established": False,
        "physical_overscan_certified": False,
        "cross_mode_zero_subtraction_performed": False,
    }


def selftest():
    rng = np.random.default_rng(331179)
    arr = np.full(EXPECTED_SHAPE, 5000, dtype=np.int16)
    for x0, x1 in EDGE.values():
        arr[:, x0:x1] = rng.integers(994, 1005, size=(1032, x1-x0))
    q = characterize(arr)
    assert not q["candidate_overscan_certified"]
    assert q["neighbor_minus_candidate_adu"]["right"] > 3000
    assert q["neighbor_minus_candidate_adu"]["left"] > 3000
    assert q["candidate"]["left"]["unique_integer_level_count"] == 11
    assert q["candidate"]["right"]["modal_value_fraction"] < .2
    try:
        characterize(arr[:1024])
    except ValueError:
        pass
    else:
        raise AssertionError("Wrong shape accepted")
    print("SELFTEST PASSED")


def run(config_file, report_file):
    from astropy.io import fits
    import yaml

    from sc1_masterbias_raw_sample_qa import read_config_and_products
    from sc1_bias_night_timeline_qa import (
        LATE_NAMES, FLAGGED_NAMES, locate_frames,
    )

    cfg_path = Path(config_file).expanduser().resolve(strict=True)
    out_path = Path(report_file).expanduser().resolve()
    early20, _, _, _, work = read_config_and_products(cfg_path)
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))
    raw_root = Path(str(config.get("data_dir") or work)).expanduser()
    if not raw_root.is_absolute():
        raw_root = work / raw_root
    raw_root = raw_root.resolve(strict=True)

    if out_path.is_relative_to(work) or out_path.is_relative_to(raw_root):
        raise RuntimeError("Refusing QA output within source or work directory")
    if out_path.exists():
        raise FileExistsError("Refusing to overwrite existing QA report")

    frames = config.get("frames") or {}
    other = {}
    for name, n in (("obj", 7), ("flat", 3), ("neon", 2)):
        values = frames.get(name) or []
        if len(values) != n:
            raise ValueError(f"Expected exactly {n} {name.upper()} inputs")
        arr = []
        for value in values:
            p = Path(str(value)).expanduser()
            if not p.is_absolute():
                p = raw_root / p
            arr.append(p.resolve(strict=True))
        other[name] = arr

    discovered = locate_frames(raw_root, early20)
    groups = {
        "bias_early20": early20,
        "bias_late19": [discovered[k] for k in sorted(LATE_NAMES)],
        "bias_flagged2": [discovered[k] for k in sorted(FLAGGED_NAMES)],
        "obj_selected7": other["obj"],
        "flat_selected3": other["flat"],
        "neon_selected2": other["neon"],
    }
    paths = [p for v in groups.values() for p in v]
    if (len(paths) != 53 or len(set(paths)) != 53
            or {k: len(v) for k, v in groups.items()} != REQUIRED_COUNTS):
        raise RuntimeError("Invalid exact 53-input manifest")

    results = {}
    for group, group_paths in groups.items():
        results[group] = []
        for p in group_paths:
            with fits.open(p, memmap=False) as hdu:
                hdr = hdu[0].header
                if (hdr.get("NAXIS2"), hdr.get("NAXIS1")) != EXPECTED_SHAPE:
                    raise RuntimeError(f"Unexpected detector shape: {p.name}")
                image = hdu[0].data
                if image is None:
                    raise RuntimeError(f"Missing CCD data: {p.name}")
                stats = characterize(image)
                header = {
                    k: str(hdr.get(k, "")) for k in
                    ("DATE", "UT", "EXPTIME", "IMAGETYP",
                     "NODE", "RATE", "GAIN", "BINNING")
                }
                results[group].append({
                    "filename": p.name, "header": header,
                    "statistics": stats,
                })
        print(f"PROFILED {group}: {len(results[group])} frames", flush=True)

    summary = summarize_modes(results)
    report = {
        "schema": SCHEMA,
        "status": "CANDIDATE_SERIAL_EDGE_TESTED_OVERSCAN_UNVERIFIED_SCIENCE_HOLD",
        "source_config": str(cfg_path),
        "groups": results,
        "same_mode_comparison": summary,
        "warnings": [
            "Geometry is a hypothesis; no DATASEC/BIASSEC/overscan was certified.",
            "Candidate strips are 0:16 and 2064:2080; Y32:992 is arbitrary.",
            "The medians and robust means are descriptive, NOT approved overscan.",
            "Entire source images are only READ, never changed.",
            "Do NOT use flat/neon (rate225/gain3.5) zero against normal rate145/gain0.9.",
            "The delta_left/right values do NOT authorize science epoch bias correction.",
            "SCIENCE E2E HOLD until optical+electronic interpretation and VAR fixed.",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("REPORT:", out_path)
    print("OVERSCAN NOT CERTIFIED; SCIENCE E2E HOLD")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) == 3:
        run(sys.argv[1], sys.argv[2])
    else:
        raise SystemExit("Usage: script.py --selftest | config.yaml external_report.json")
