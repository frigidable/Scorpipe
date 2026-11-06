#!/usr/bin/env python3
"""Read-only independent sample audit of SCORPIO-1 MasterBias SCI/VAR/DQ.

Visits all 20 selected early BIAS frames without building or changing any stage.
Samples entire detector rows Y=0 and Y=1, every high-VAR pixel, plus a
reproducible sample of the science ROI. Recomputes the exact sample median and
sample variance times the saved SB_VFAC from raw pixels, and identifies which
exposure(s) caused large temporal deviations. Writes only into external QA dir.

Does NOT certify a physical overscan region or a corrected variance/DQ model.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

EXPECTED_EARLY20 = {f"s228001{i:02d}.fts" for i in range(1, 21)}
EXPECTED_SHAPE = (1032, 2080)
ACTIVE = (32, 992, 32, 2016)
SEED = 241206


def quantiles(a):
    a = np.asarray(a, dtype=float)
    return dict(
        n=int(a.size),
        min=float(np.min(a)) if a.size else None,
        median=float(np.median(a)) if a.size else None,
        p95=float(np.percentile(a, 95)) if a.size else None,
        p99=float(np.percentile(a, 99)) if a.size else None,
        max=float(np.max(a)) if a.size else None,
    )


def sample_metrics(stack, expected_sci, expected_var, factor):
    stack = np.asarray(stack, dtype=np.float64)
    expected_sci = np.asarray(expected_sci, dtype=np.float64)
    expected_var = np.asarray(expected_var, dtype=np.float64)
    if stack.ndim != 2 or stack.shape[0] != 20:
        raise ValueError("Expected 20 time samples at N selected pixels")
    if stack.shape[1] != expected_sci.size or expected_sci.shape != expected_var.shape:
        raise ValueError("Sample sizes disagree")
    if not (0 < float(factor) < 1):
        raise ValueError("Invalid median variance factor")
    med = np.median(stack, axis=0)
    unbiased_var = np.var(stack, axis=0, ddof=1)
    est_var = unbiased_var * factor
    residual = stack - med
    spike_count_100 = np.sum(np.abs(residual) > 100, axis=0)
    robust_noise = 1.4826*np.median(np.abs(residual), axis=0)
    sci_abs = np.abs(expected_sci-med)
    var_abs = np.abs(expected_var-est_var)
    var_rel = var_abs/np.maximum(est_var,1e-6)
    return dict(
        n_pixels=int(stack.shape[1]),
        max_abs_sci_difference_adu=float(np.max(sci_abs)),
        p99_abs_sci_difference_adu=float(np.percentile(sci_abs, 99)),
        max_rel_var_difference=float(np.max(var_rel)),
        p99_rel_var_difference=float(np.percentile(var_rel,99)),
        max_abs_var_difference_adu2=float(np.max(var_abs)),
        n_pixels_raw_deviation_over_100adu=int(np.count_nonzero(spike_count_100>0)),
        n_pixels_with_multiple_deviations_over_100adu=int(np.count_nonzero(spike_count_100>1)),
        n_raw_values_deviation_over_100adu=int(spike_count_100.sum()),
        median_robust_sigma_adu=float(np.median(robust_noise)),
        max_var_adu2=float(np.max(expected_var)),
    ), med, residual, spike_count_100, robust_noise


def read_config_and_products(config_path):
    import yaml
    from astropy.io import fits

    config_path = Path(config_path).expanduser().resolve(strict=True)
    c = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(c, dict):
        raise ValueError("Config must be mapping")
    work = Path(str(c.get("work_dir") or config_path.parent)).expanduser().resolve()
    if work != config_path.parent:
        raise RuntimeError("Unexpected work_dir (require saved config's own run)")
    frames = (c.get("frames") or {}).get("bias") or []
    if len(frames) != 20:
        raise RuntimeError("Expected 20 BIAS paths")
    data_root = Path(str(c.get("data_dir") or work)).expanduser()
    if not data_root.is_absolute():
        data_root = (work/data_root).resolve()
    paths = []
    for raw in frames:
        pp = Path(str(raw)).expanduser()
        if not pp.is_absolute():
            pp = data_root/pp
        pp = pp.resolve(strict=True)
        paths.append(pp)
    if len(set(paths)) != 20 or {p.name.lower() for p in paths} != EXPECTED_EARLY20:
        raise RuntimeError("Invalid early20 BIAS selection")

    stage = work/"03_bias"
    ip = stage/"master_bias_index.json"
    index = json.loads(ip.read_text(encoding="utf-8-sig"))
    groups = index.get("groups") or []
    if index.get("schema_version") != 3 or len(groups) != 1:
        raise RuntimeError("Unexpected bias index schema or group count")
    group = groups[0]
    if group.get("n_used") != 20 or group.get("n_inputs") != 20:
        raise RuntimeError("Incorrect BIAS count in index")
    if group.get("variance",{}).get("model") != "GAUSSMED":
        raise RuntimeError("Expected GAUSSMED, no sigma clipping")
    key = group.get("key") or {}
    ro = key.get("readout") or {}
    if (key.get("nx"),key.get("ny"),ro.get("node"),float(ro.get("rate",0)),
        float(ro.get("gain",0))) != (2080,1032,"A",145.0,0.9):
        raise RuntimeError("Index group readout mismatch")
    sci_p = stage/"master_bias.fits"
    var_p = stage/"master_bias_var.fits"
    dq_p = stage/"master_bias_dq.fits"
    sci = np.asarray(fits.getdata(sci_p),dtype=np.float64)
    var = np.asarray(fits.getdata(var_p),dtype=np.float64)
    dq = np.asarray(fits.getdata(dq_p),dtype=np.uint16)
    if any(a.shape != EXPECTED_SHAPE for a in (sci,var,dq)):
        raise RuntimeError("Output products have wrong geometry")
    if not np.isfinite(sci).all() or not np.isfinite(var).all() or np.any(var<0):
        raise RuntimeError("Nonfinite or negative SCI/VAR")
    return paths, (sci,var,dq), group, (sci_p,var_p,dq_p,ip), work


def run(config_path, output_path):
    from astropy.io import fits

    raw_paths, (sci,var,dq), group, products, work = read_config_and_products(config_path)
    out = Path(output_path).expanduser().resolve()
    if out.is_relative_to(work) or any(out.is_relative_to(p.parent) for p in raw_paths):
        raise RuntimeError("QA report must be outside run and raw input directories")
    if out.exists():
        raise FileExistsError(f"Refuse to overwrite existing QA report: {out}")

    n_y,n_x = sci.shape
    r0x = np.arange(n_x,dtype=int)
    coords = [np.column_stack((np.zeros(n_x,dtype=int),r0x)),
              np.column_stack((np.ones(n_x,dtype=int),r0x))]
    high_y,high_x = np.where(var>10.0)
    coords.append(np.column_stack((high_y,high_x)))
    rng = np.random.default_rng(SEED)
    y0,y1,x0,x1 = ACTIVE
    yy = rng.integers(y0,y1,size=8192)
    xx = rng.integers(x0,x1,size=8192)
    coords.append(np.column_stack((yy,xx)))
    selected = np.unique(np.concatenate(coords,axis=0),axis=0)
    ys,xs = selected[:,0],selected[:,1]
    raw_sample = np.empty((20,len(selected)),dtype=np.float64)
    rows = []
    for j, path in enumerate(raw_paths):
        with fits.open(path,memmap=False) as hdul:
            h = hdul[0].header
            data = hdul[0].data
            if data is None or data.shape != EXPECTED_SHAPE:
                raise RuntimeError(f"Invalid raw geometry: {path.name}")
            if (float(h["RATE"]),float(h["GAIN"]),str(h["NODE"]).strip().upper()) != (145.0,0.9,"A"):
                raise RuntimeError(f"Invalid raw readout: {path.name}")
            raw_sample[j] = np.asarray(data[ys,xs],dtype=np.float64)
            row0 = np.asarray(data[0,:],dtype=np.float64)
            row1 = np.asarray(data[1,:],dtype=np.float64)
            rows.append({
                "file":path.name,
                "row0_median_adu":float(np.median(row0)),
                "row1_median_adu":float(np.median(row1)),
                "row0_minus_row1_median_adu":float(np.median(row0-row1)),
                "row0_range_adu":[float(np.min(row0)),float(np.max(row0))],
                "row1_range_adu":[float(np.min(row1)),float(np.max(row1))],
            })
        print(f"READ {j+1}/20 {path.name}: "
              f"row0 - row1 = {rows[-1]['row0_minus_row1_median_adu']:.2f} ADU",flush=True)
    summary,med,residual,spikes,sigma_rob=sample_metrics(raw_sample,sci[ys,xs],var[ys,xs],
                                                         group["variance"]["factor"])
    full_high=(var>100)
    in_roi=np.zeros(sci.shape,dtype=bool)
    in_roi[y0:y1,x0:x1] = True
    high_selected=var[ys,xs]>100
    scmask=(ys>=y0)&(ys<y1)&(xs>=x0)&(xs<x1)
    top_indices=np.where(high_selected & scmask)[0]
    top_indices=top_indices[np.argsort(var[ys[top_indices],xs[top_indices]])[::-1][:16]]
    examples=[]
    for idx in top_indices:
        values=raw_sample[:,idx]
        examples.append({
            "y":int(ys[idx]),"x":int(xs[idx]),
            "master_sci_adu":float(sci[ys[idx],xs[idx]]),
            "master_var_adu2":float(var[ys[idx],xs[idx]]),
            "raw_frame_values_adu":[float(x) for x in values],
            "n_deviations_over_100adu":int(spikes[idx]),
            "robust_raw_sigma_adu":float(sigma_rob[idx]),
            "deviant_frames":[raw_paths[k].name for k in np.flatnonzero(np.abs(residual[:,idx])>100)],
        })
    mask_hi=high_selected & scmask
    frame_spikes=[
        {"file":raw_paths[j].name,
         "n_selected_pixels_deviation_gt_100adu":int(np.count_nonzero(np.abs(residual[j])>100)),
         "n_selected_active_highvar_pixels_deviation_gt_100adu":int(np.count_nonzero(
            (np.abs(residual[j])>100)&mask_hi))}
        for j in range(len(raw_paths))
    ]
    report={
        "status":"DIAGNOSTIC_ONLY_SCIENCE_HOLD",
        "inputs":{"config":str(Path(config_path).resolve()),"source_files":[p.name for p in raw_paths],
                  "product_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in products}},
        "index_group":group["gid"],
        "sample_recompute":summary,
        "full_frame":{"pixels":int(sci.size),
                      "dq_nonzero_count":int(np.count_nonzero(dq)),
                      "median_sci_adu":float(np.median(sci)),
                      "median_var_adu2":float(np.median(var)),
                      "row0_median_var_adu2":float(np.median(var[0])),
                      "row1_median_var_adu2":float(np.median(var[1])),
                      "high_var_gt100_full":int(np.count_nonzero(full_high)),
                      "high_var_gt100_row0":int(np.count_nonzero(full_high[0])),
                      "high_var_gt100_active_roi":int(np.count_nonzero(full_high&in_roi))},
        "raw_per_frame_row0_vs_row1":rows,
        "deviant_frame_summary":frame_spikes,
        "active_high_var_examples":examples,
        "limitations":[
            "The threshold 100 ADU is a gross-deviation diagnostic, not a calibrated cosmic-ray threshold.",
            "This verifies selected-pixel arithmetic, not the full 2.1M-pixel stack.",
            "High temporal sample variance does not automatically imply uncertainty of the sample median.",
            "No bias or FLAT reprocessing, DQ flagging, or science E2E authorized.",
        ],
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print("OUTPUT:",out)
    print("SCIENCE E2E HOLD")


def selftest():
    rng=np.random.default_rng(2024)
    base=np.full((20,4),1000.0)+rng.normal(0,3,(20,4))
    base[8,0]+=1000
    factor=.07343702770505486
    expected_sci=np.median(base,axis=0)
    expected_var=np.var(base,axis=0,ddof=1)*factor
    s,_,_,spikes,_=sample_metrics(base,expected_sci,expected_var,factor)
    assert s["max_abs_sci_difference_adu"]==0
    assert s["max_rel_var_difference"]==0
    assert spikes[0]==1 and spikes[1]==0
    assert expected_var[0]>100 and expected_var[1]<10
    print("SELFTEST PASSED")


if __name__=="__main__":
    if len(sys.argv)==2 and sys.argv[1]=="--selftest":
        selftest()
    elif len(sys.argv)==3:
        run(Path(sys.argv[1]),Path(sys.argv[2]))
    else:
        raise SystemExit("Usage: script.py --selftest | config.yaml qa_output.json")
