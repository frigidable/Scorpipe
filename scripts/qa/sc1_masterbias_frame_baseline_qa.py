#!/usr/bin/env python3
"""SCORPIO-1 early20 BIAS: read-only per-frame common-mode drift diagnosis.

Measures each source BIAS relative to the saved 20-frame MasterBias in the
provisional ROI [32:992, 32:2016], using a *gross* residual cutoff of 50 ADU
to suppress bright transients. NO physical overscan correction is inferred.
Reports 64px-tile means, chronological versus interleaved offsets, and a
frame-label permutation diagnostic. Does NOT write any FITS/config or edit
calibration outputs; writes JSON outside work/raw directories only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from sc1_masterbias_raw_sample_qa import (
    ACTIVE, EXPECTED_SHAPE, read_config_and_products,
)

CLIP_ADU = 50.0
TILE = 64
PERMUTATIONS = 50000
SEED = 20261108


def tile_stats(raw_roi: np.ndarray, reference_roi: np.ndarray,
               threshold_adu: float = CLIP_ADU) -> tuple[np.ndarray, dict]:
    """Compute residual tile means with gross outlier exclusion only."""
    raw = np.asarray(raw_roi, dtype=np.float64)
    ref = np.asarray(reference_roi, dtype=np.float64)
    if raw.shape != (960, 1984) or ref.shape != raw.shape:
        raise ValueError("Expected 960x1984 provisional ROI")
    if not np.isfinite(raw).all() or not np.isfinite(ref).all():
        raise ValueError("Nonfinite ROI")
    if not (np.isfinite(threshold_adu) and threshold_adu > 0):
        raise ValueError("Invalid diagnostic residual threshold")
    diff = raw - ref
    good = np.abs(diff) <= threshold_adu
    values = np.where(good, diff, 0.0)
    sums = values.reshape(15, 64, 31, 64).sum(axis=(1, 3))
    counts = good.reshape(15, 64, 31, 64).sum(axis=(1, 3))
    if np.any(counts < 0.98 * TILE**2):
        raise RuntimeError("Excessively contaminated 64px tile; do not infer drift")
    means = sums / counts
    return means, {
        "n_pixels": int(diff.size),
        "n_discarded": int(np.count_nonzero(~good)),
        "mean_residual_adu": float(values.sum()/good.sum()),
        "median_tile_mean_residual_adu": float(np.median(means)),
        "std_tile_means_adu": float(np.std(means)),
        "tile_mean_min_adu": float(np.min(means)),
        "tile_mean_max_adu": float(np.max(means)),
    }


def partition_stats(tile_cube: np.ndarray, a: np.ndarray,
                    b: np.ndarray) -> dict:
    """Compare *frame-mean* detector baselines, not half-master medians."""
    tiles = np.asarray(tile_cube, dtype=float)
    if tiles.shape != (20, 15, 31):
        raise ValueError("Expected 20 × 15 × 31 tile means")
    x = set(map(int, a))
    y = set(map(int, b))
    if len(x) != 10 or len(y) != 10 or x & y or x | y != set(range(20)):
        raise ValueError("Invalid 10+10 partition")
    difference = np.mean(tiles[np.asarray(a,dtype=int)],axis=0) - np.mean(
        tiles[np.asarray(b,dtype=int)],axis=0)
    return {
        "mean_difference_adu": float(np.mean(difference)),
        "median_tile_difference_adu": float(np.median(difference)),
        "std_tile_difference_adu": float(np.std(difference)),
        "tile_p05_adu": float(np.percentile(difference,5)),
        "tile_p95_adu": float(np.percentile(difference,95)),
        "fraction_tiles_difference_negative": float(np.mean(difference<0)),
        "fraction_tiles_difference_positive": float(np.mean(difference>0)),
        "min_tile_difference_adu":float(np.min(difference)),
        "max_tile_difference_adu":float(np.max(difference)),
    }


def permutation_test(frame_means, *, seed=SEED,
                     n_permutations=PERMUTATIONS) -> dict:
    """Two-sided random label test. Exchangeability is an *assumption*."""
    x = np.asarray(frame_means,dtype=np.float64)
    if x.shape!=(20,) or not np.isfinite(x).all() or n_permutations<100:
        raise ValueError("Expected 20 finite frame means and >=100 permutations")
    rng=np.random.default_rng(seed)
    splits=np.argsort(rng.random((n_permutations,20)),axis=1)
    perm=x[splits[:,:10]].mean(axis=1)-x[splits[:,10:]].mean(axis=1)
    observed=float(x[:10].mean()-x[10:].mean())
    p=float((1+np.count_nonzero(np.abs(perm)>=abs(observed)))/
            (n_permutations+1))
    return {
        "observed_first10_minus_last10_adu":observed,
        "two_sided_permutation_p_under_exchangeability":p,
        "permutations":int(n_permutations),
        "seed":int(seed),
        "null_requires_exchangeable_independent_frame_baselines":True,
        "not_a_test_of_causal_electronic_drift":True,
    }


def run(config_path:Path, report_path:Path)->None:
    from astropy.io import fits

    config_path=config_path.expanduser().resolve(strict=True)
    report_path=report_path.expanduser().resolve()
    paths,(sci,_,_),group,_,work=read_config_and_products(config_path)
    if report_path.is_relative_to(work) or any(
        report_path.is_relative_to(p.parent) for p in paths
    ):
        raise RuntimeError("QA report must be outside work and raw directories")
    if report_path.exists():
        raise FileExistsError(f"QA report already exists: {report_path}")
    if group.get("n_inputs")!=20 or group.get("n_used")!=20:
        raise RuntimeError("Unexpected MasterBias source count")

    y0,y1,x0,x1=ACTIVE
    reference=np.asarray(sci[y0:y1,x0:x1],dtype=np.float64)
    tiles=[]
    frames=[]
    for n,path in enumerate(paths,1):
        with fits.open(path,memmap=False) as h:
            hdr=h[0].header
            if (float(hdr["RATE"]),float(hdr["GAIN"]),
                str(hdr.get("NODE","")).upper().strip())!=(145.0,0.9,"A"):
                raise RuntimeError(f"Unexpected CCD mode: {path}")
            dat=h[0].data
            if dat is None or dat.shape!=EXPECTED_SHAPE:
                raise RuntimeError(f"Unexpected raw geometry: {path}")
            tile,meta=tile_stats(dat[y0:y1,x0:x1],reference)
            meta.update({
                "index":n,"frame":path.name,
                "date_obs_header":str(hdr.get("DATE-OBS","")),
                "date_header":str(hdr.get("DATE","")),
                "ut_header":str(hdr.get("UT","")),
            })
        tiles.append(tile)
        frames.append(meta)
        print(f"{n:02d}/20 {path.name}: "
              f"{meta['mean_residual_adu']:+.5f} ADU, "
              f"reject={meta['n_discarded']}",flush=True)

    cube=np.stack(tiles)
    from sc1_masterbias_split_half_qa import splits
    partition_results={
        name:partition_stats(cube,ia,ib)
        for name,(ia,ib) in splits().items()
    }
    means=np.array([f["mean_residual_adu"] for f in frames],dtype=float)
    slope=float(np.polyfit(np.arange(1,21),means,1)[0])
    result={
        "status":"READ_ONLY_FRAME_BASELINE_DIAGNOSTIC_SCIENCE_HOLD",
        "config":str(config_path),"group_gid":group["gid"],
        "method":"frame minus saved median20 MasterBias, exclude |residual|>50 ADU, mean remaining provisional ROI pixels and 64x64 tile means; NO overscan correction",
        "candidate_active_region_halfopen":[y0,y1,x0,x1],
        "frame_offsets":frames,
        "chronological_linear_slope_adu_per_frame_index":slope,
        "split_frame_mean_results":partition_results,
        "frame_level_permutation_test":permutation_test(means),
        "cautions":[
            "Reference MasterBias uses all 20 frames; this in-sample reference can induce correlations.",
            "An observed ordered baseline change does not establish a physical electronics drift without timestamp and independent telemetry.",
            "Frame-label permutation requires exchangeability and may not account for temporal autocorrelation.",
            "Across-tile means are spatially correlated, NOT independent evidence for 465 replicates.",
            "This provisional ROI is not an instrument-certified trim or overscan geometry.",
            "No changes to source FITS, YAML, stage data, or existing MasterBias products.",
        ]
    }
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    for name,rec in partition_results.items():
        print(f"{name}: mean offset A-B = {rec['mean_difference_adu']:+.5f} ADU; "
              f"negative tile fraction = {rec['fraction_tiles_difference_negative']:.3f}",
              flush=True)
    print(f"ORDERED-SPLIT PERMUTATION p = "
          f"{result['frame_level_permutation_test']['two_sided_permutation_p_under_exchangeability']:.5g}")
    print("REPORT:",report_path)
    print("SCIENCE E2E: HOLD")


def selftest():
    rng=np.random.default_rng(241206)
    ref=np.full((960,1984),999.0)
    test=999.0 + rng.normal(0.15,3.0,size=ref.shape)
    test[42,128]+=17000
    tiles,stats=tile_stats(test,ref)
    assert stats["n_discarded"]==1
    assert abs(stats["mean_residual_adu"]-0.15)<0.02
    assert np.max(np.abs(tiles-0.15))<0.2
    data=np.zeros((20,15,31),dtype=float)
    data[:10]-=0.2
    stat=partition_stats(data,np.arange(10),np.arange(10,20))
    assert abs(stat["mean_difference_adu"]+0.2)<1e-10
    assert stat["fraction_tiles_difference_negative"]==1.0
    p=permutation_test(np.r_[np.full(10,-0.2),np.zeros(10)],n_permutations=1000)
    assert p["observed_first10_minus_last10_adu"]==-0.2
    assert 0<p["two_sided_permutation_p_under_exchangeability"]<=1
    print("SELFTEST PASSED")


if __name__=="__main__":
    if len(sys.argv)==2 and sys.argv[1]=="--selftest":
        selftest()
    elif len(sys.argv)==3:
        run(Path(sys.argv[1]),Path(sys.argv[2]))
    else:
        raise SystemExit("Usage: script.py --selftest | config.yaml external_report.json")
