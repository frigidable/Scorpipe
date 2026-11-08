#!/usr/bin/env python3
"""Read-only epoch-median spatial comparison for 2024-12-06 SCORPIO-1 N2146sat.

Reads exactly early20 and late19 NORMAL BIAS frames; reconstructs the saved
early20 MasterBias SCI *pixel for pixel* as an arithmetic integrity gate.
Computes the late19 per-pixel median only IN MEMORY and measures the signed
late19-minus-early20 difference over the provisional ROI and two unverified
side strips, without creating or replacing any FITS stage products.

Not a calibration algorithm. Integer-ADU quantization, temporal structure,
post-median uncertainty, and the meaning of sideband columns remain unresolved.
Output JSON must be new and outside both raw and work directories.
"""
from __future__ import annotations

import hashlib
import json
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from sc1_masterbias_raw_sample_qa import (
    ACTIVE, EXPECTED_SHAPE, read_config_and_products,
)
from sc1_bias_night_timeline_qa import (
    EARLY_NAMES, LATE_NAMES, SIDES, locate_frames,
)

CHUNK_ROWS=32


def summarize_epoch_difference(delta_roi:np.ndarray,
                               left:np.ndarray,
                               right:np.ndarray) -> dict:
    """Pure helper; 64×64 tiles are spatial diagnostics, NOT independent trials."""
    d=np.asarray(delta_roi,dtype=np.float64)
    l=np.asarray(left,dtype=np.float64)
    r=np.asarray(right,dtype=np.float64)
    if d.shape!=(960,1984) or l.shape!=(960,12) or r.shape!=(960,12):
        raise ValueError("Expected exact ROI and two 12-column candidate-side regions")
    if not all(np.isfinite(a).all() for a in (d,l,r)):
        raise ValueError("Nonfinite epoch-difference pixels")

    tile=d.reshape(15,64,31,64).mean(axis=(1,3))
    row=d.mean(axis=1)
    column=d.mean(axis=0)
    centered=tile-tile.mean()
    def stats(x):
        return {
            "mean_adu":float(np.mean(x)),
            "median_adu":float(np.median(x)),
            "std_adu":float(np.std(x)),
            "p05_adu":float(np.percentile(x,5)),
            "p95_adu":float(np.percentile(x,95)),
            "min_adu":float(np.min(x)),
            "max_adu":float(np.max(x)),
        }
    x0=np.asarray(d[96:-96,96:-96])
    roi_summary=stats(d)
    tight_summary=stats(x0)
    result={
        "provisional_roi_yx_halfopen":[32,992,32,2016],
        "roi_pixel_count":int(d.size),
        "roi_difference":roi_summary,
        "tightened_roi_yx_halfopen":[128,896,128,1920],
        "tightened_roi_difference":tight_summary,
        "tile_count_64x64":int(tile.size),
        "tile_mean_difference":stats(tile),
        "tile_mean_residual_after_subtracting_scalar":stats(centered),
        "tile_mean_fraction_positive":float(np.mean(tile>0)),
        "tile_mean_fraction_negative":float(np.mean(tile<0)),
        "per_row_mean_difference":stats(row),
        "per_column_mean_difference":stats(column),
        "sidebands_uncertified":{
            "left_x_halfopen":[2,14],
            "right_x_halfopen":[2066,2078],
            "left_difference":stats(l),
            "right_difference":stats(r),
            "left_minus_roi_mean_adu":float(np.mean(l)-np.mean(d)),
            "right_minus_roi_mean_adu":float(np.mean(r)-np.mean(d)),
            "physical_overscan_certified":False,
        },
    }
    return result


def run(config:Path,output:Path):
    from astropy.io import fits
    import yaml

    config=Path(config).expanduser().resolve(strict=True)
    output=Path(output).expanduser().resolve()
    selected,(saved,_,_),group,products,work=read_config_and_products(config)
    cfg=yaml.safe_load(config.read_text(encoding="utf-8-sig")) or {}
    raw_root=Path(str(cfg.get("data_dir") or work)).expanduser()
    if not raw_root.is_absolute():
        raw_root=(work/raw_root).resolve()
    raw_root=raw_root.resolve(strict=True)
    if output.is_relative_to(work) or output.is_relative_to(raw_root):
        raise RuntimeError("QA output must be OUTSIDE workspace and raw inputs")
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite existing QA report: {output}")

    found=locate_frames(raw_root,selected)
    early=[found[n] for n in sorted(EARLY_NAMES)]
    late=[found[n] for n in sorted(LATE_NAMES)]
    if len(early)!=20 or len(late)!=19 or set(early)&set(late):
        raise RuntimeError("Unexpected early/late cohort membership")
    if set(early)!=set(selected):
        raise RuntimeError("Selected early20 differs from actual reference inputs")

    y0,y1,x0,x1=ACTIVE
    roi=np.empty((960,1984),dtype=np.float32)
    sides=[
        np.empty((960,12),dtype=np.float32),
        np.empty((960,12),dtype=np.float32),
    ]
    max_err=0.0
    verified_rows=0
    with ExitStack() as stack:
        all_paths=early+late
        hdus=[stack.enter_context(fits.open(p,memmap=False)) for p in all_paths]
        arrays=[]
        for p,hdu in zip(all_paths,hdus):
            hdr=hdu[0].header
            data=hdu[0].data
            if data is None or data.shape!=EXPECTED_SHAPE:
                raise RuntimeError(f"Invalid BIAS image dimensions in {p}")
            if (float(hdr["RATE"]),float(hdr["GAIN"]),
                str(hdr["NODE"]).strip().upper())!=(145.0,0.9,"A"):
                raise RuntimeError(f"Invalid CCD readout mode in {p}")
            arrays.append(data)

        for ystart in range(0,EXPECTED_SHAPE[0],CHUNK_ROWS):
            yend=min(ystart+CHUNK_ROWS,EXPECTED_SHAPE[0])
            block=np.stack([a[ystart:yend] for a in arrays],axis=0).astype(np.float64)
            early_med=np.median(block[:20],axis=0)
            late_med=np.median(block[20:],axis=0)
            if not np.isfinite(early_med).all() or not np.isfinite(late_med).all():
                raise RuntimeError("Non-finite median BIAS")
            err=float(np.max(np.abs(early_med-saved[ystart:yend])))
            max_err=max(max_err,err)
            if err>1e-4:
                raise RuntimeError(
                    f"Saved MasterBias SCI is not median of selected early20 (row={ystart}, maxdiff={err})"
                )

            lo=max(ystart,y0)
            hi=min(yend,y1)
            if lo<hi:
                start=lo-y0
                end=hi-y0
                src=slice(lo-ystart,hi-ystart)
                delta=late_med[src]-early_med[src]
                roi[start:end]=delta[:,x0:x1]
                for i,(a,b) in enumerate(SIDES):
                    sides[i][start:end]=delta[:,a:b]
            verified_rows+=yend-ystart
            print(f"RECONSTRUCTED {verified_rows}/{EXPECTED_SHAPE[0]} rows",flush=True)

    stats=summarize_epoch_difference(roi,*sides)
    report={
        "status":"READ_ONLY_EPOCH_SPATIAL_COMPARISON_NOT_MASTERBIAS_APPROVAL",
        "config":str(config),
        "source_masterbias_gid":group["gid"],
        "early20_n":len(early),"late19_n":len(late),
        "early20_first_last":[early[0].name,early[-1].name],
        "late19_first_last":[late[0].name,late[-1].name],
        "algorithm":"Per-pixel median of early20 and late19; difference late-minus-early; no clipping, no interpolation, no overscan, no science-frame processing",
        "saved_early20_masterbias_sci_max_abs_reconstruction_error_adu":max_err,
        "verified_full_detector_rows":verified_rows,
        "spatial_diagnostics":stats,
        "saved_masterbias_product_sha256":{
            p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in products
        },
        "limitations":[
            "The 19 late BIAS were never selected by the production configuration: these values are a separate diagnostic, not an authoritative MasterBias.",
            "Difference of two noisy per-pixel medians includes sampling noise (and ADC quantization), so its pixel scatter is not directly a measure of spatial-pattern change.",
            "Coherent 64x64 tile means are descriptive; pixel/tile correlations invalidate naive independent-sample significances.",
            "The chronological gap spans hours and OBJ electronics zero cannot be inferred by interpolating these two snapshots.",
            "Candidate CCD region and side strips are NOT verified physical overscan/trim areas.",
            "Two flagged neighboring late BIAS are deliberately excluded from comparison.",
            "No raw FITS, stage FITS, configuration, or other pipeline products have been edited.",
        ],
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("ROI late19 - early20 mean:",round(stats["roi_difference"]["mean_adu"],6),"ADU")
    print("64x64 tile spatial std:",round(stats["tile_mean_residual_after_subtracting_scalar"]["std_adu"],6),"ADU")
    print("Candidate side minus ROI means:",
          round(stats["sidebands_uncertified"]["left_minus_roi_mean_adu"],6),
          round(stats["sidebands_uncertified"]["right_minus_roi_mean_adu"],6),"ADU")
    print("SAVED SCI RECONSTRUCTION MAX ERROR:",max_err,"ADU")
    print("REPORT:",output)
    print("SCIENCE E2E: HOLD")


def selftest():
    roi=np.full((960,1984),-0.15,dtype=np.float32)
    left=np.full((960,12),-0.10,dtype=np.float32)
    right=np.full((960,12),-0.07,dtype=np.float32)
    summary=summarize_epoch_difference(roi,left,right)
    assert summary["tile_count_64x64"]==465
    assert abs(summary["roi_difference"]["mean_adu"]+0.15)<1e-6
    assert abs(summary["sidebands_uncertified"]["left_minus_roi_mean_adu"]-0.05)<1e-6
    assert abs(summary["sidebands_uncertified"]["right_minus_roi_mean_adu"]-0.08)<1e-6
    assert summary["tile_mean_residual_after_subtracting_scalar"]["std_adu"]<1e-6
    print("SELFTEST PASSED")


if __name__=="__main__":
    if len(sys.argv)==2 and sys.argv[1]=="--selftest":
        selftest()
    elif len(sys.argv)==3:
        run(Path(sys.argv[1]),Path(sys.argv[2]))
    else:
        raise SystemExit("Usage: script.py --selftest | config.yaml external_QA_report.json")
