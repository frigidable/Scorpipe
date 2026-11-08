#!/usr/bin/env python3
"""Read-only within-epoch placebo for early/late SCORPIO-1 MasterBias maps.

The previous 20-vs-19 BIAS median difference yielded tile mean RMS 0.025 ADU
in a provisional 960x1984 ROI. This script compares its noise-normalized
spatial tile RMS to early-only and late-only 10/10, 9/10 placebo divisions.

It is a *diagnostic*, NOT a p-value, confidence bound, approved active region,
overscan correction, temporal interpolation, or production calibration.
Frame-level common offsets are removed from each difference map before the
spatial tile RMS is assessed. All input raw FITS and original stage products
are strictly read only. Writes exactly one new external JSON report.
"""
from __future__ import annotations

import json
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np

EXPECTED_SHAPE = (1032, 2080)
ROI = (32, 992, 32, 2016)
ROWS_PER_BLOCK = 32
TILE = 64
N_TILE_Y, N_TILE_X = (15, 31)


def partitions():
    """Zero-based indices into early20 + late19 (39) canonical sorted frames."""
    early = np.arange(20, dtype=int)
    late = np.arange(20, 39, dtype=int)
    rng = np.random.default_rng(20261108)
    scrambled_early = rng.permutation(early)
    scrambled_late = rng.permutation(late)
    return {
        "between_early20_late19": (late, early),
        "early_chronological_10v10": (early[:10], early[10:]),
        "early_odd_even_10v10": (early[::2], early[1::2]),
        "early_seeded_random_10v10": (scrambled_early[:10], scrambled_early[10:]),
        "late_chronological_9v10": (late[:9], late[9:]),
        "late_odd_even_10v9": (late[::2], late[1::2]),
        "late_seeded_random_9v10": (scrambled_late[:9], scrambled_late[9:]),
    }


def empty_stats():
    return {
        "tile_sum": np.zeros((N_TILE_Y, N_TILE_X), dtype=np.float64),
        "tile_n": np.zeros((N_TILE_Y, N_TILE_X), dtype=np.int64),
        "pixel_sum": 0.0,
        "pixel_sumsq": 0.0,
        "pixels": 0,
    }


def add_block(acc, delta, first_row):
    """Accumulate a full-width delta block using uncorrelated row indexing."""
    d = np.asarray(delta, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] != EXPECTED_SHAPE[1] or d.shape[0] <= 0:
        raise ValueError("Expected (block_rows, 2080) difference block")
    if not np.isfinite(d).all():
        raise ValueError("Nonfinite median-epoch difference")
    y0, y1, x0, x1 = ROI
    top = max(y0, first_row)
    stop = min(y1, first_row + d.shape[0])
    if top >= stop:
        return
    cropped = d[top - first_row:stop - first_row, x0:x1]
    acc["pixels"] += int(cropped.size)
    acc["pixel_sum"] += float(cropped.sum())
    acc["pixel_sumsq"] += float(np.square(cropped).sum())
    for j, y in enumerate(range(top, stop)):
        ty = (y - y0) // TILE
        grouped = cropped[j].reshape(N_TILE_X, TILE)
        acc["tile_sum"][ty] += grouped.sum(axis=1)
        acc["tile_n"][ty] += TILE


def finish(acc):
    y0, y1, x0, x1 = ROI
    if acc["pixels"] != (y1-y0) * (x1-x0):
        raise RuntimeError("Not all active-candidate ROI pixels processed")
    if not np.all(acc["tile_n"] == TILE*TILE):
        raise RuntimeError("Incomplete 64x64 tile aggregation")
    n = acc["pixels"]
    m = acc["pixel_sum"] / n
    pixel_var = acc["pixel_sumsq"]/n - m*m
    if pixel_var <= 0:
        raise RuntimeError("Zero or negative empirical difference-map variance")
    tile_means = acc["tile_sum"] / acc["tile_n"]
    centered = tile_means - np.mean(tile_means)
    rms = float(np.std(centered))
    shot_expectation = float(np.sqrt(pixel_var) / TILE)
    return {
        "roi_n_pixels": n,
        "difference_pixel_mean_adu": float(m),
        "difference_pixel_std_adu": float(np.sqrt(pixel_var)),
        "spatial_tile_rms_after_scalar_removal_adu": rms,
        "independent_pixel_reference_tile_std_adu": shot_expectation,
        "normalized_tile_variance_factor": float(
            rms*rms / (pixel_var/(TILE*TILE))
        ),
        "tile_mean_p05_adu": float(np.percentile(tile_means, 5)),
        "tile_mean_p95_adu": float(np.percentile(tile_means, 95)),
        "tile_min_mean_adu": float(np.min(tile_means)),
        "tile_max_mean_adu": float(np.max(tile_means)),
        "all_tile_means_adu_15_by_31": tile_means.tolist(),
    }


def selftest():
    p = partitions()
    assert len(p) == 7
    for a,b in p.values():
        assert not (set(a) & set(b))
        assert len(set(a)) == len(a) and len(set(b)) == len(b)
        assert np.all(np.asarray(a)<39) and np.all(np.asarray(b)<39)
    a = empty_stats()
    rng = np.random.default_rng(20261108)
    # Two 32-row portions combine into one entire 64-row tile stripe.
    for first in range(0, EXPECTED_SHAPE[0], ROWS_PER_BLOCK):
        nrows = min(ROWS_PER_BLOCK, EXPECTED_SHAPE[0]-first)
        delta = rng.normal(-0.11,1.4,size=(nrows,2080))
        add_block(a,delta,first)
    out = finish(a)
    assert out["roi_n_pixels"] == 960*1984
    assert abs(out["difference_pixel_mean_adu"]+0.11) < 0.01
    assert 0.7 < out["normalized_tile_variance_factor"] < 1.3
    assert len(out["all_tile_means_adu_15_by_31"])==15
    print("SELFTEST PASSED")


def run(config_file, report_file):
    from astropy.io import fits
    import yaml

    # Imports explicitly deferred so the standalone math tests remain runnable.
    from sc1_masterbias_raw_sample_qa import read_config_and_products
    from sc1_bias_night_timeline_qa import (
        EARLY_NAMES, LATE_NAMES, locate_frames,
    )

    config_file = Path(config_file).expanduser().resolve(strict=True)
    report_file = Path(report_file).expanduser().resolve()
    early_selected, (master_sci,_,_), group, _, work = read_config_and_products(config_file)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8-sig"))
    raw_root = Path(str(config.get("data_dir") or work)).expanduser()
    if not raw_root.is_absolute():
        raw_root = work / raw_root
    raw_root = raw_root.resolve(strict=True)
    if report_file.is_relative_to(work) or report_file.is_relative_to(raw_root):
        raise RuntimeError("Refusing output within raw or Scorpipe work directory")
    if report_file.exists():
        raise FileExistsError("Refuse to overwrite a prior spatial placebo report")

    discovered = locate_frames(raw_root, early_selected)
    paths = [discovered[name] for name in sorted(EARLY_NAMES)] + [
        discovered[name] for name in sorted(LATE_NAMES)
    ]
    if len(paths)!=39 or len(set(paths))!=39 or set(paths[:20]) != set(early_selected):
        raise RuntimeError("Wrong BIAS cohorts or duplicate paths")
    maps = partitions()
    accum = {name:empty_stats() for name in maps}
    early_reconstruction_max_abs_error = 0.0
    with ExitStack() as scope:
        hdus = [scope.enter_context(fits.open(f, memmap=False)) for f in paths]
        arrays = []
        for p,hdu in zip(paths,hdus):
            h = hdu[0].header
            image = hdu[0].data
            if image is None or image.shape != EXPECTED_SHAPE:
                raise RuntimeError(f"Wrong BIAS dimensions: {p.name}")
            if (float(h["RATE"]),float(h["GAIN"]),
                str(h["NODE"]).strip().upper()) != (145.0,0.9,"A"):
                raise RuntimeError(f"Wrong CCD readout for {p.name}")
            arrays.append(image)
        for y in range(0,EXPECTED_SHAPE[0],ROWS_PER_BLOCK):
            end = min(y+ROWS_PER_BLOCK, EXPECTED_SHAPE[0])
            block = np.stack([x[y:end] for x in arrays],axis=0).astype(np.float64)
            early_med = np.median(block[:20],axis=0)
            err = float(np.max(np.abs(early_med-master_sci[y:end])))
            early_reconstruction_max_abs_error = max(
                early_reconstruction_max_abs_error,err
            )
            if err > 1e-4:
                raise RuntimeError(f"Saved MasterBias does not match early20 at Y={y}")
            for name,(aa,bb) in maps.items():
                if name=="between_early20_late19":
                    med_a = np.median(block[20:],axis=0)
                    med_b = early_med
                else:
                    med_a = np.median(block[aa],axis=0)
                    med_b = np.median(block[bb],axis=0)
                add_block(accum[name],med_a-med_b,y)
            print(f"VERIFIED {end}/{EXPECTED_SHAPE[0]} CCD rows",flush=True)

    results={name:finish(a) for name,a in accum.items()}
    between=results["between_early20_late19"]
    placebo=[x["normalized_tile_variance_factor"]
             for k,x in results.items() if k!="between_early20_late19"]
    normalized_median=float(np.median(placebo))
    report={
        "status":"READ_ONLY_WITHIN_EPOCH_SPATIAL_PLACEBO_SCIENCE_HOLD",
        "source_masterbias_gid":group["gid"],
        "config":str(config_file),
        "early_bias_count":20, "late_bias_count":19,
        "full_detector_rows_verified":EXPECTED_SHAPE[0],
        "saved_masterbias_max_abs_sci_reconstruction_error_adu":
            early_reconstruction_max_abs_error,
        "candidate_roi_halfopen_y0_y1_x0_x1":list(ROI),
        "comparisons":{
            name:{"group_A":[paths[int(i)].name for i in aa],
                  "group_B":[paths[int(i)].name for i in bb],
                  **results[name]}
            for name,(aa,bb) in maps.items()
        },
        "heuristic_cross_over_median_placebo_normalized_variance_factor":
            float(between["normalized_tile_variance_factor"]/normalized_median),
        "warnings":[
            "Normalization by pixel variance uses an independent-pixel reference; CCD noise correlations can change its interpretation.",
            "The six within-epoch partitions overlap and are NOT independent placebo replicates.",
            "Different epoch group sizes 20/19 versus 10/10, 9/10 change quantized-median statistics.",
            "Subtracting one global scalar from tile means does not correct spatially varying bias.",
            "A low cross/placebo ratio does NOT prove identical spatial patterns or validate absolute bias at OBJ times.",
            "No physical overscan/trim region is certified here.",
            "Read-only: neither source FITS nor existing calibration products have been modified.",
        ],
    }
    report_file.parent.mkdir(parents=True,exist_ok=True)
    report_file.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print("BETWEEN-EPOCH TILE STD =",round(between["spatial_tile_rms_after_scalar_removal_adu"],6),"ADU")
    print("BETWEEN-EPOCH NORMALIZED TILE FACTOR =",round(between["normalized_tile_variance_factor"],6))
    print("MEDIAN WITHIN-EPOCH PLACEBO NORMALIZED FACTOR =",round(normalized_median,6))
    print("HEURISTIC BETWEEN / PLACEBO =",round(report[
        "heuristic_cross_over_median_placebo_normalized_variance_factor"],6))
    print("REPORT:",report_file)
    print("SCIENCE E2E HOLD")


if __name__=="__main__":
    if len(sys.argv)==2 and sys.argv[1]=="--selftest":
        selftest()
    elif len(sys.argv)==3:
        run(Path(sys.argv[1]),Path(sys.argv[2]))
    else:
        raise SystemExit("Usage: script.py --selftest | config.yaml external_QA_report.json")
