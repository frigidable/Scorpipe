#!/usr/bin/env python3
"""SCORPIO-1 MasterBias: independent 10+10 real-frame cross-validation.

Read-only diagnostics for the 20 *selected* BIAS exposures. Tests the predicted
variance of the DIFFERENCE of independent 10-frame half-masters, NOT the final
20-frame MasterBias itself. Three partitions probe temporal drift and dependence
on split construction; they reuse the same exposures and are NOT independent.

Covers the provisional candidate CCD ROI Y[32:992], X[32:2016], which is NOT
instrument-certified as active silicon. Never edits inputs or stage products.
"""
from __future__ import annotations

import hashlib
import json
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scorpio_pipe.bias_uncertainty import normal_median_variance_factor
from sc1_masterbias_raw_sample_qa import read_config_and_products, ACTIVE, EXPECTED_SHAPE

N_HALF = 10
THRESHOLD_ADU = 50.0
CHUNK_ROWS = 32
PARTITION_SEED = 1072412


def splits():
    chronological = (np.arange(10,dtype=int),np.arange(10,20,dtype=int))
    interleaved = (np.arange(0,20,2,dtype=int),np.arange(1,20,2,dtype=int))
    order=np.random.default_rng(PARTITION_SEED).permutation(20)
    randomized = (order[:10],order[10:])
    return {
        "chronological_early_vs_late":chronological,
        "odd_vs_even":interleaved,
        "seeded_random_10_vs_10":randomized,
    }


def estimate_half(stack, *, threshold=None):
    """Return (estimator, plugin variance, accepted count, discarded count).

    If threshold is set, the plug-in variance is an UNCALIBRATED heuristic
    after sample-dependent rejection; it must not be interpreted as a
    science-approved uncertainty model.
    """
    a=np.asarray(stack,dtype=np.float64)
    if a.ndim != 3 or a.shape[0] != N_HALF or not np.isfinite(a).all():
        raise ValueError("Expected ten finite bias exposures per half")
    center=np.median(a,axis=0)
    if threshold is None:
        n=np.full(center.shape,N_HALF,dtype=np.int16)
        variance=np.var(a,axis=0,ddof=1)*normal_median_variance_factor(N_HALF)
        return center,variance,n,0
    if not (np.isfinite(threshold) and threshold>0):
        raise ValueError("Invalid rejection threshold")
    keep=np.abs(a-center)<=float(threshold)
    n=np.sum(keep,axis=0,dtype=np.int16)
    if np.any(n<2):
        raise RuntimeError("Candidate half-master has fewer than two samples")
    values=np.where(keep,a,np.nan)
    center=np.nanmedian(values,axis=0)
    mean=np.sum(np.where(keep,a,0.0),axis=0)/n
    s2=np.sum(np.where(keep,(a-mean)**2,0.0),axis=0)/(n-1)
    factors=np.zeros_like(center)
    for nn in np.unique(n):
        factors[n==nn]=normal_median_variance_factor(int(nn))
    return center, s2*factors, n, int(np.sum(~keep))


def _stat_accum():
    return dict(n=0,valid_n=0,within_1sigma=0,within_1p96sigma=0,
                sum_delta=0.0,sum_delta2=0.0,sum_predicted_var=0.0,
                excluded_samples=0,min_n_half=10,
                tile_sums=np.zeros((15,31),dtype=np.float64),
                tile_counts=np.zeros((15,31),dtype=np.int64))


def _add(acc, delta, pred_var, ystart, accepted_a, accepted_b, discarded):
    y0,y1,x0,x1=ACTIVE
    ystop=ystart+delta.shape[0]
    left=max(ystart,y0); right=min(ystop,y1)
    if right<=left:
        return
    sl=slice(left-ystart,right-ystart)
    d=delta[sl,x0:x1]
    v=pred_var[sl,x0:x1]
    na=accepted_a[sl,x0:x1]; nb=accepted_b[sl,x0:x1]
    if np.any(~np.isfinite(d)) or np.any(~np.isfinite(v)) or np.any(v<0):
        raise RuntimeError("Nonfinite or negative half-difference variance")
    valid=v>0
    with np.errstate(divide="ignore",invalid="ignore"):
        z=np.where(valid,np.abs(d)/np.sqrt(v),np.inf)
    acc["n"]+=int(d.size)
    acc["valid_n"]+=int(np.count_nonzero(valid))
    acc["within_1sigma"]+=int(np.count_nonzero(z<=1.0))
    acc["within_1p96sigma"]+=int(np.count_nonzero(z<=1.96))
    acc["sum_delta"]+=float(np.sum(d))
    acc["sum_delta2"]+=float(np.sum(d*d))
    acc["sum_predicted_var"]+=float(np.sum(v))
    acc["excluded_samples"]+=int(np.sum(20-na-nb))
    acc["min_n_half"]=min(acc["min_n_half"],int(np.min(na)),int(np.min(nb)))
    # 64×64 tile means of differences reveal coherent temporal structures.
    # Handle row chunks crossing tiles, without allocating full mosaics.
    for y in range(left,right):
        tile_y=(y-y0)//64
        vector=d[y-left,:].reshape(31,64)
        acc["tile_sums"][tile_y,:]+=vector.sum(axis=1)
        acc["tile_counts"][tile_y,:]+=64


def _finish(acc):
    n=acc["n"]
    if n<100 or acc["valid_n"] < 0.99*n:
        raise RuntimeError("Insufficient valid predictions in candidate active ROI")
    mean_delta=acc["sum_delta"]/n
    empirical_var=acc["sum_delta2"]/n-mean_delta**2
    prediction=acc["sum_predicted_var"]/n
    if empirical_var<=0:
        raise RuntimeError("No measurable independent half-MasterBias scatter")
    tiles=acc["tile_sums"]/np.maximum(acc["tile_counts"],1)
    tile_values=tiles[acc["tile_counts"]==64*64]
    if not len(tile_values):
        raise RuntimeError("No fully observed 64x64 tiles")
    return {
        "n_pixels":n, "valid_variance_pixels":acc["valid_n"],
        "frac_abs_difference_le_1sigma":acc["within_1sigma"]/n,
        "frac_abs_difference_le_1p96sigma":acc["within_1p96sigma"]/n,
        "mean_half_difference_adu":mean_delta,
        "empirical_spatial_var_of_half_difference_adu2":empirical_var,
        "mean_predicted_half_difference_variance_adu2":prediction,
        "predicted_to_empirical_spatial_var_ratio":prediction/empirical_var,
        "excluded_samples_from_active_roi":acc["excluded_samples"],
        "minimum_accepted_in_either_half":acc["min_n_half"],
        "n_complete_tiles_64x64":int(len(tile_values)),
        "tile_mean_difference_median_adu":float(np.median(tile_values)),
        "tile_mean_difference_p05_adu":float(np.percentile(tile_values,5)),
        "tile_mean_difference_p95_adu":float(np.percentile(tile_values,95)),
        "tile_mean_difference_min_adu":float(np.min(tile_values)),
        "tile_mean_difference_max_adu":float(np.max(tile_values)),
    }


def run(config, output):
    from astropy.io import fits
    config=Path(config).expanduser().resolve(strict=True)
    output=Path(output).expanduser().resolve()
    paths, (_,_,_), group, products, work=read_config_and_products(config)
    raw_roots={p.parent for p in paths}
    if output.is_relative_to(work) or any(output.is_relative_to(root) for root in raw_roots):
        raise RuntimeError("QA output must be outside work and raw directories")
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite prior report: {output}")
    if float(group["key"]["readout"]["rate"])!=145.0 or group["n_used"]!=20:
        raise RuntimeError("Unexpected MasterBias readout or input count")

    parts=splits()
    aggregates={
        name:{"unclipped":_stat_accum(),"gross_50adu_diagnostic":_stat_accum()}
        for name in parts
    }
    with ExitStack() as exit_stack:
        hdu=[exit_stack.enter_context(fits.open(p,memmap=False)) for p in paths]
        raw=[]
        for p,h in zip(paths,hdu):
            hdr=h[0].header
            d=h[0].data
            if d is None or d.shape!=EXPECTED_SHAPE:
                raise RuntimeError(f"Invalid BIAS shape: {p.name}")
            if (float(hdr["RATE"]),float(hdr["GAIN"]),
                str(hdr["NODE"]).strip().upper())!=(145.0,0.9,"A"):
                raise RuntimeError(f"Readout mismatch: {p.name}")
            raw.append(d)
        for y in range(0,EXPECTED_SHAPE[0],CHUNK_ROWS):
            end=min(y+CHUNK_ROWS,EXPECTED_SHAPE[0])
            block=np.stack([p[y:end] for p in raw],axis=0).astype(np.float64)
            for label,(aa,bb) in parts.items():
                a=block[aa]; b=block[bb]
                for mode,thr in (("unclipped",None),
                                 ("gross_50adu_diagnostic",THRESHOLD_ADU)):
                    ma,va,na,discard_a=estimate_half(a,threshold=thr)
                    mb,vb,nb,discard_b=estimate_half(b,threshold=thr)
                    _add(aggregates[label][mode],ma-mb,va+vb,y,na,nb,
                         discarded=discard_a+discard_b)
            print(f"VERIFIED ROWS {end}/{EXPECTED_SHAPE[0]}",flush=True)
    reports={p:{model:_finish(a) for model,a in modes.items()}
             for p,modes in aggregates.items()}
    payload={
        "status":"READ_ONLY_EMPIRICAL_SPLIT_HALF_DIAGNOSTIC_SCIENCE_HOLD",
        "input_config":str(config), "source_group_gid":group["gid"],
        "source_count":20,
        "frame_sets":{
            name:{"A":[paths[i].name for i in ia],
                  "B":[paths[i].name for i in ib]}
            for name,(ia,ib) in parts.items()
        },
        "candidate_active_region_halfopen":[32,992,32,2016],
        "tile_size_pixels":64, "rejection_threshold_adu":THRESHOLD_ADU,
        "models":reports,
        "input_masterbias_sha256":{
            p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in products
        },
        "limitations":[
            "A 10+10 split is not an independent reference for the final 20-exposure MasterBias.",
            "Different partitions reuse frames and their metrics are statistically dependent.",
            "Spatially correlated detector readout and artifacts make CCD pixels non-independent replicates.",
            "The candidate active ROI is provisional and not an instrument-certified trim/overscan definition.",
            "The 50 ADU rejection threshold and post-selection s_inlier²*c_N uncertainty are diagnostic, not validated.",
            "No raw source, existing MasterBias, YAML, or pipeline stages have been changed.",
        ],
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
    for name,m in reports.items():
        print("SPLIT",name)
        for mode,r in m.items():
            print(f"  {mode}: 1sigma={r['frac_abs_difference_le_1sigma']:.5f}, "
                  f"1.96sigma={r['frac_abs_difference_le_1p96sigma']:.5f}, "
                  f"ratio={r['predicted_to_empirical_spatial_var_ratio']:.5f}, "
                  f"mean_diff={r['mean_half_difference_adu']:.4f} ADU",flush=True)
    print("REPORT:",output)
    print("SCIENCE E2E: HOLD")


def selftest():
    rng=np.random.default_rng(24460)
    raw=999.0+rng.normal(0,3.1,(20,4,7))
    raw[0,0,:]+=559.
    old=estimate_half(raw[:10],threshold=None)
    clean=estimate_half(raw[:10],threshold=50.)
    assert old[2].min()==10
    assert clean[2][0,0]==9
    assert np.all(clean[2][1:,:]==10)
    assert clean[1][0,0]<old[1][0,0]
    assert len(splits())==3
    for a,b in splits().values():
        assert len(a)==10 and len(b)==10 and not set(a)&set(b)
        assert set(a)|set(b)==set(range(20))
    print("SELFTEST PASSED")


if __name__=="__main__":
    if len(sys.argv)==2 and sys.argv[1]=="--selftest":
        selftest()
    elif len(sys.argv)==3:
        run(sys.argv[1],sys.argv[2])
    else:
        raise SystemExit("Usage: script.py --selftest | config.yaml QA_output.json")
