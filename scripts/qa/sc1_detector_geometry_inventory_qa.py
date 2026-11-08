#!/usr/bin/env python3
"""Read-only actual SCORPIO-1 CCD geometry/header/edge-profile inventory.

The frame array shape (1032, 2080) differs from a commonly cited SCORPIO
1x2-binning example (1046, 2068). No detector section is treated as a physical
overscan until independently identified. This script NEVER subtracts or trims
overscan, estimates a science-epoch electronic zero, or alters any source
product. It writes one new JSON report outside raw and stage directories.

Usage:
  python scripts/qa/sc1_detector_geometry_inventory_qa.py --selftest
  python scripts/qa/sc1_detector_geometry_inventory_qa.py config.yaml external.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EXPECTED_SHAPE = (1032, 2080)  # numpy y,x
X_BANDS = (
    (0, 16), (16, 32), (32, 64), (64, 128), (128, 256),
    (256, 512), (512, 1024), (1024, 1536), (1536, 1792),
    (1792, 1920), (1920, 2016), (2016, 2048),
    (2048, 2064), (2064, 2080),
)
Y_BANDS = (
    (0, 8), (8, 16), (16, 32), (32, 64), (64, 128),
    (128, 256), (256, 768), (768, 896), (896, 960),
    (960, 992), (992, 1016), (1016, 1032),
)
HEADER_KEYS = (
    "DATE", "DATE-OBS", "UT", "EXPTIME", "OBJECT", "IMAGETYP",
    "NAXIS1", "NAXIS2", "BINNING", "BINX", "BINY",
    "NODE", "RATE", "GAIN", "DETECTOR", "CCD",
    "BIASSEC", "DATASEC", "CCDSEC", "DETSEC", "TRIMSEC",
    "BIASSEC1", "DATASEC1", "PRESEC", "ORSEC",
    "OVERSCAN", "OVSCX", "OVSCY",
)
SCHEMA = "SC1_READ_ONLY_GEOMETRY_INVENTORY_V1"


def profile(data):
    """Raw region medians only; never a physical overscan classification."""
    arr = np.asarray(data)
    if arr.shape != EXPECTED_SHAPE:
        raise ValueError("Unexpected raw detector geometry")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError("Non-numeric FITS array")
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite raw CCD data")
    # These are arbitrary diagnostic windows, NOT verified active regions.
    middle = float(np.median(arr[128:896, 256:1792]))
    xx = []
    for a, b in X_BANDS:
        m = float(np.median(arr[128:896, a:b]))
        xx.append({"x0": a, "x1": b, "median_adu": m,
                   "minus_middle_adu": m - middle})
    yy = []
    for a, b in Y_BANDS:
        m = float(np.median(arr[a:b, 256:1792]))
        yy.append({"y0": a, "y1": b, "median_adu": m,
                   "minus_middle_adu": m - middle})
    return {
        "middle_reference_adu": middle,
        "middle_reference_roi_y0_y1_x0_x1": [128, 896, 256, 1792],
        "x_edge_band_medians": xx,
        "y_edge_band_medians": yy,
        "physical_overscan_region_certified": False,
    }


def header_fields(header):
    """Preserve raw keyword values verbatim, including unknown or absent."""
    return {
        key: (None if key not in header else str(header[key]))
        for key in HEADER_KEYS
    }


def selftest():
    arr = np.full(EXPECTED_SHAPE, 100, dtype=np.int16)
    arr[128:896, 256:1792] = 200
    found = profile(arr)
    assert found["middle_reference_adu"] == 200
    assert found["x_edge_band_medians"][0]["median_adu"] == 100
    assert found["x_edge_band_medians"][-1]["minus_middle_adu"] == -100
    assert found["physical_overscan_region_certified"] is False
    try:
        profile(arr[:900])
    except ValueError:
        pass
    else:
        raise AssertionError("Wrong image shape accepted")
    h = {"BIASSEC": "[1:16,1:100]", "RATE": 145}
    fields = header_fields(h)
    assert fields["BIASSEC"] == "[1:16,1:100]"
    assert fields["DATASEC"] is None
    assert fields["RATE"] == "145"
    print("SELFTEST PASSED")


def resolve_config_frame(raw, raw_root):
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = raw_root / p
    return p.resolve(strict=True)


def run(config_file, report_file):
    from astropy.io import fits
    import yaml

    from sc1_masterbias_raw_sample_qa import read_config_and_products
    from sc1_bias_night_timeline_qa import (
        FLAGGED_NAMES, LATE_NAMES, locate_frames,
    )

    config_file = Path(config_file).expanduser().resolve(strict=True)
    report_file = Path(report_file).expanduser().resolve()
    first20, _, _, _, work = read_config_and_products(config_file)
    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8-sig"))
    raw_root = Path(str(cfg.get("data_dir") or work)).expanduser()
    if not raw_root.is_absolute():
        raw_root = work / raw_root
    raw_root = raw_root.resolve(strict=True)

    if report_file.is_relative_to(work) or report_file.is_relative_to(raw_root):
        raise RuntimeError("QA output must be outside raw and work directories")
    if report_file.exists():
        raise FileExistsError("Refusing to overwrite an existing QA report")

    frames = cfg.get("frames") or {}
    expected = {"obj": 7, "flat": 3, "neon": 2}
    selected = {}
    for kind, n in expected.items():
        values = frames.get(kind) or []
        if len(values) != n:
            raise RuntimeError(f"Expected {n} {kind.upper()} input paths")
        selected[kind] = [resolve_config_frame(p, raw_root) for p in values]

    discovered = locate_frames(raw_root, first20)
    groups = {
        "bias_early20": first20,
        "bias_late19_diagnostic": [discovered[k] for k in sorted(LATE_NAMES)],
        "bias_flagged2_excluded": [
            discovered[k] for k in sorted(FLAGGED_NAMES)],
        "obj_selected7": selected["obj"],
        "flat_selected3_other_readout": selected["flat"],
        "neon_selected2_other_readout": selected["neon"],
    }
    all_paths = [p for v in groups.values() for p in v]
    if len(all_paths) != 53 or len(set(all_paths)) != 53:
        raise RuntimeError("Missing or duplicate raw path in 53-file manifest")

    manifest = []
    for kind, paths in groups.items():
        for p in paths:
            hdr = fits.getheader(p)
            shape = (int(hdr.get("NAXIS2", -1)),
                     int(hdr.get("NAXIS1", -1)))
            if shape != EXPECTED_SHAPE:
                raise RuntimeError(f"Unexpected header geometry {p.name}: {shape}")
            manifest.append({
                "group": kind,
                "filename": p.name,
                "shape_npy_y_x": list(shape),
                "header": header_fields(hdr),
            })
    print(f"VERIFIED {len(manifest)} RAW HEADERS", flush=True)

    samples = [
        ("early_first", first20[0]),
        ("early_last", first20[-1]),
        ("late_first", groups["bias_late19_diagnostic"][0]),
        ("late_last", groups["bias_late19_diagnostic"][-1]),
        ("excluded_bias_first", groups["bias_flagged2_excluded"][0]),
        ("obj_first", selected["obj"][0]),
        ("obj_middle", selected["obj"][3]),
        ("obj_last", selected["obj"][-1]),
        ("flat_first", selected["flat"][0]),
        ("neon_first", selected["neon"][0]),
    ]
    profiles = {}
    for label, p in samples:
        with fits.open(p, memmap=False) as hdus:
            raw = hdus[0].data
            if raw is None:
                raise RuntimeError(f"No raw primary image in {p.name}")
            profiles[label] = {"filename": p.name, **profile(raw)}
        print("PROFILE", label, p.name, flush=True)

    output = {
        "schema": SCHEMA,
        "status": "HEADER_AND_PIXEL_PROFILE_ONLY_OVERSCAN_UNVERIFIED_SCIENCE_HOLD",
        "source_config": str(config_file),
        "expected_numpy_shape_y_x": list(EXPECTED_SHAPE),
        "group_counts": {k: len(v) for k, v in groups.items()},
        "all_raw_headers": manifest,
        "raw_edge_profiles_selected_files": profiles,
        "warnings": [
            "Raw sections in these arrays are NOT established from shape alone.",
            "BIASSEC and DATASEC headers, even if present, require optical validation.",
            "Raw X/Y bands are arbitrary and are NOT approved electronic overscan.",
            "FLAT and NEON use a different readout/gain from OBJ and normal BIAS; compare geometry within a frame, not absolute ADU between modes.",
            "An illuminated/science frame may contaminate suspected edge regions.",
            "The FITS source timestamp convention was NOT certified.",
            "The rejected two bias frames are documented only, never used to calibrate.",
            "Read-only: no raw FITS, masterbias, variance, DQ or configuration changed.",
            "SCIENCE E2E HOLD. This report never authorizes overscan correction.",
        ],
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with report_file.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print("RAW GEOMETRY INVENTORY:", report_file)
    print("OVERSCAN: NOT VERIFIED; SCIENCE E2E HOLD")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) == 3:
        run(sys.argv[1], sys.argv[2])
    else:
        raise SystemExit("Usage: script.py --selftest | config.yaml QA_report.json")
