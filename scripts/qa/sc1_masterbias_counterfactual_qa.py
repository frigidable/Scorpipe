#!/usr/bin/env python3
"""Read-only SCORPIO-1 MasterBias VAR counterfactual sensitivity diagnostic.

Compares saved 20-frame median/VAR to a *hypothetical* median and plugin VAR
after per-pixel exclusion of gross transient samples (absolute residual >30,
>50, >100 ADU). This is NOT a validated production estimator: data-dependent
clipping changes the median's sampling distribution; c_N times inlier sample
variance is diagnostic only, not a statistically calibrated uncertainty.

No product FITS/config/raw files are changed. Output JSON path must be outside
both the work_dir and raw inputs and must not already exist.
"""
from __future__ import annotations

import hashlib
import json
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from scorpio_pipe.bias_uncertainty import normal_median_variance_factor
from sc1_masterbias_raw_sample_qa import read_config_and_products, EXPECTED_SHAPE, ACTIVE

THRESHOLDS_ADU = (30.0, 50.0, 100.0)
ROWS_PER_CHUNK = 32


def candidate(raw, threshold: float):
    """Hypothetical clipped-median SCI and inlier-sample plug-in variance."""
    a = np.asarray(raw, dtype=np.float64)
    if a.ndim != 3 or a.shape[0] != 20 or not np.isfinite(a).all():
        raise ValueError("Expected 20 finite raw-bias images in an (N,y,x) block")
    if not (np.isfinite(threshold) and threshold > 0):
        raise ValueError("Threshold must be positive")
    old_med = np.median(a, axis=0)
    old_var = np.var(a, axis=0, ddof=1) * normal_median_variance_factor(20)
    keep = np.abs(a - old_med) <= threshold
    n = keep.sum(axis=0)
    valid = n >= 2
    # Even for diagnostic models, never impute missing data as zero uncertainty.
    clean = np.where(keep, a, np.nan)
    with np.errstate(invalid="ignore"):
        # Under very rare insufficient-support cases, signal NaN downstream.
        clean_med = np.nanmedian(clean, axis=0)
    clean_mean = np.sum(np.where(keep, a, 0.0), axis=0) / np.maximum(n, 1)
    s2 = np.sum(np.where(keep, (a - clean_mean)**2, 0.0), axis=0) / np.maximum(n-1, 1)
    factor = np.empty(n.shape, dtype=np.float64)
    for nval in np.unique(n):
        factor[n == nval] = (normal_median_variance_factor(int(nval))
                             if nval >= 2 else np.nan)
    plugin_var = s2 * factor
    clean_med[~valid] = np.nan
    plugin_var[~valid] = np.nan
    return dict(old_med=old_med, old_var=old_var, clean_med=clean_med,
                plugin_var=plugin_var, keep=keep, n=n, valid=valid)


def _new_accumulator():
    return dict(
        pixels=0, affected_pixels=0, discarded_samples=0,
        first_row_affected_pixels=0, first_row_discarded_samples=0,
        active_affected_pixels=0, active_discarded_samples=0,
        invalid_pixels=0, sci_change_gt_0p5=0, sci_change_gt_1=0,
        active_sci_change_gt_1=0, old_var_gt100=0, candidate_var_gt100=0,
        row0_oldvar_median_parts=[], row0_newvar_median_parts=[],
        active_oldvar_medians=[], active_newvar_medians=[],
        active_sci_change_medians=[],
        max_abs_sci_delta=0.0, max_abs_var_recalc_diff=0.0,
        max_rel_var_recalc_diff=0.0, min_support=20,
        per_frame_discards=np.zeros(20,dtype=np.int64),
    )


def _sample_list_for_sanity(a):
    return float(np.percentile(np.asarray(a,dtype=float),99)) if len(a) else None


def run(config: Path, out: Path):
    from astropy.io import fits
    config = Path(config).expanduser().resolve(strict=True)
    out = Path(out).expanduser().resolve()
    paths, (saved_sci, saved_var, saved_dq), group, products, work = read_config_and_products(config)
    if out.is_relative_to(work) or any(out.is_relative_to(p.parent) for p in paths):
        raise RuntimeError("QA output must not be inside work_dir or source data folder")
    if out.exists():
        raise FileExistsError(f"Refuse to overwrite existing QA output: {out}")
    if int(np.count_nonzero(saved_dq)) != 0:
        raise RuntimeError("Diagnostic expects original zero-DQ MasterBias")
    params = {thr:_new_accumulator() for thr in THRESHOLDS_ADU}
    verified_chunks=0
    verified_rows=0
    with ExitStack() as cleanup:
        hdus=[cleanup.enter_context(fits.open(p, memmap=False)) for p in paths]
        heads=[h[0].header for h in hdus]
        for p, h, open_hdu in zip(paths,heads,hdus):
            if open_hdu[0].data is None or open_hdu[0].data.shape != EXPECTED_SHAPE:
                raise RuntimeError(f"Bad source shape: {p.name}")
            if (float(h["RATE"]), float(h["GAIN"]), str(h["NODE"]).strip().upper()) != (145.0,0.9,"A"):
                raise RuntimeError(f"Readout mismatch in {p.name}")
        arrays=[np.asarray(h[0].data) for h in hdus]
        y0,y1,x0,x1=ACTIVE
        for ystart in range(0, EXPECTED_SHAPE[0], ROWS_PER_CHUNK):
            ystop=min(ystart+ROWS_PER_CHUNK,EXPECTED_SHAPE[0])
            raw=np.stack([a[ystart:ystop,:] for a in arrays],axis=0).astype(np.float64)
            ref_sci=saved_sci[ystart:ystop]
            ref_var=saved_var[ystart:ystop]
            yslice=np.arange(ystart,ystop)
            roi_rows=(yslice>=y0)&(yslice<y1)
            roi_mask=np.zeros(ref_sci.shape,dtype=bool)
            roi_mask[roi_rows,x0:x1]=True
            for thr in THRESHOLDS_ADU:
                res=candidate(raw,thr)
                acc=params[thr]
                sci_difference=np.abs(res["old_med"]-ref_sci)
                var_difference=np.abs(res["old_var"]-ref_var)
                if np.max(sci_difference) > 1e-4:
                    raise RuntimeError(f"Source SCI differs from recomputation at row {ystart}")
                relative=var_difference/np.maximum(res["old_var"],1e-9)
                max_rel=float(np.max(relative))
                if max_rel>1e-6:
                    raise RuntimeError(f"Source VAR differs from recomputation at row {ystart}")
                acc["max_abs_var_recalc_diff"]=max(acc["max_abs_var_recalc_diff"],
                                                   float(np.max(var_difference)))
                acc["max_rel_var_recalc_diff"]=max(acc["max_rel_var_recalc_diff"],max_rel)
                old_var=res["old_var"]; new_var=res["plugin_var"]
                if not np.isfinite(new_var[res["valid"]]).all():
                    raise RuntimeError("Nonfinite candidate variance")
                n_drop=20-res["n"]
                affected=n_drop>0
                delta=np.abs(res["clean_med"]-res["old_med"])
                acc["pixels"]+=int(affected.size)
                acc["affected_pixels"]+=int(np.count_nonzero(affected))
                acc["discarded_samples"]+=int(n_drop.sum())
                acc["active_affected_pixels"]+=int(np.count_nonzero(affected&roi_mask))
                acc["active_discarded_samples"]+=int(n_drop[roi_mask].sum())
                acc["invalid_pixels"]+=int(np.count_nonzero(~res["valid"]))
                acc["sci_change_gt_0p5"]+=int(np.count_nonzero(delta>0.5))
                acc["sci_change_gt_1"]+=int(np.count_nonzero(delta>1))
                acc["active_sci_change_gt_1"]+=int(np.count_nonzero((delta>1)&roi_mask))
                acc["old_var_gt100"]+=int(np.count_nonzero(old_var>100))
                acc["candidate_var_gt100"]+=int(np.count_nonzero(new_var>100))
                acc["max_abs_sci_delta"]=max(acc["max_abs_sci_delta"],float(np.nanmax(delta)))
                acc["min_support"]=min(acc["min_support"],int(np.min(res["n"])))
                acc["per_frame_discards"]+=np.count_nonzero(~res["keep"],axis=(1,2))
                if ystart==0:
                    acc["first_row_affected_pixels"]=int(np.count_nonzero(affected[0]))
                    acc["first_row_discarded_samples"]=int(n_drop[0].sum())
                    acc["row0_oldvar_median_parts"].append(float(np.median(old_var[0])))
                    acc["row0_newvar_median_parts"].append(float(np.median(new_var[0])))
                if np.any(roi_mask):
                    # Per-chunk medians are a diagnostic, not an exact global percentile.
                    acc["active_oldvar_medians"].append(float(np.median(old_var[roi_mask])))
                    acc["active_newvar_medians"].append(float(np.median(new_var[roi_mask])))
                    acc["active_sci_change_medians"].append(float(np.median(delta[roi_mask])))
            verified_chunks+=1
            verified_rows+=ystop-ystart
            print(f"RECOMPUTE {ystop}/{EXPECTED_SHAPE[0]} rows",flush=True)
    results={}
    for thr,acc in params.items():
        result={k:v for k,v in acc.items()
                if k not in ("per_frame_discards","active_oldvar_medians",
                             "active_newvar_medians","active_sci_change_medians",
                             "row0_oldvar_median_parts","row0_newvar_median_parts")}
        result.update({
            "per_frame_discarded_samples":{
                p.name:int(n) for p,n in zip(paths,acc["per_frame_discards"])
            },
            "row0_median_old_var_adu2":acc["row0_oldvar_median_parts"][0],
            "row0_median_plugin_var_adu2":acc["row0_newvar_median_parts"][0],
            "median_of_chunk_active_old_var_medians_adu2":float(np.median(acc["active_oldvar_medians"])),
            "median_of_chunk_active_plugin_var_medians_adu2":float(np.median(acc["active_newvar_medians"])),
            "active_sci_change_median_adu":float(np.median(acc["active_sci_change_medians"])),
        })
        results[str(int(thr))]=result
    result={
        "status":"READ_ONLY_COUNTERFACTUAL_NOT_PRODUCTION_CALIBRATION",
        "input_config":str(config), "index_gid":group["gid"],
        "model":"Reject |BIAS_ij - median_j| > fixed ADU threshold; median retained samples; candidate var = s_inlier^2 * Gaussian c_(n_retained). This is a plugin sensitivity proxy, not calibrated after data-dependent rejection.",
        "thresholds_adu":results,
        "original_full_frame_verified_rows":int(verified_rows),
        "original_full_frame_verified_chunks":int(verified_chunks),
        "original_fits_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in products},
        "warnings":[
            "The 30/50/100 ADU cuts are intentionally extreme diagnostic thresholds, not validated cosmic-ray detection settings.",
            "The inlier variance times c_n ignores selection conditioning and changing effective sample size; use physical injection/bootstrap checks before publishing corrected VAR.",
            "One highly contaminated row in a single exposure does not imply a persistent detector defect or science-object contamination.",
            "A zero MasterBias DQ indicates no invalid support by current contract, not that all temporal samples are clean.",
            "Do not replace current MasterBias/VAR or rerun a science stage based on this diagnostic.",
        ]
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("OUTPUT:",out)
    print("SCIENCE E2E HOLD")


def selftest():
    rng=np.random.default_rng(20261107)
    raw=1000+rng.normal(0,3,(20,2,3))
    raw[0,0,:]+=559
    raw[8,1,1]+=17000
    original=candidate(raw,100.0)
    assert original["n"][0,0]==19 and original["n"][1,1]==19
    assert int(np.sum(~original["keep"]))==4
    assert float(original["old_var"][0,0])>100
    assert float(original["plugin_var"][0,0])<10
    assert float(original["plugin_var"][1,1])<10
    assert np.max(np.abs(original["old_med"]-original["clean_med"]))<3
    assert np.isfinite(original["plugin_var"]).all()
    print("SELFTEST PASSED")


if __name__=="__main__":
    if len(sys.argv)==2 and sys.argv[1]=="--selftest":
        selftest()
    elif len(sys.argv)==3:
        run(Path(sys.argv[1]),Path(sys.argv[2]))
    else:
        raise SystemExit("Usage: script.py --selftest | config.yaml qa_output.json")
