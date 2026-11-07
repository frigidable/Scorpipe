#!/usr/bin/env python3
"""Read-only simulation of MasterBias median variance *coverage* under ADC quantization.

Real QA establishes: early20 BIAS, RN estimate ~3.145 ADU, one defective
row in one exposure (+559 ADU) and rare transient events of up to 17k ADU.

Simulation intentionally injects exactly one positive gross outlier per
20-sample pixel series, not the observed *rate* of bad pixels. It measures
conditional performance when a pixel is contaminated.

The Gaussian c_N * inlier sample-variance formula is a candidate *diagnostic*
and NOT calibrated science noise. The initial noise distribution is an IID
quantized-Gaussian toy model; correlated row noise/non-Gaussian tails are not
captured. No FITS or pipeline stage is modified.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

from scorpio_pipe.bias_uncertainty import normal_median_variance_factor

SEED = 20261107
REPLICATES = 100000
NBIAS = 20
THRESHOLD_ADU = 50.0
OFFSET_ADU = 999.0
SCENARIOS = {"clean":0.0, "row_559":559.0, "transient_17166":17166.0}


def simulate(rn_adu: float, *, n_replicates: int=REPLICATES,
             threshold_adu: float=THRESHOLD_ADU,
             seed: int=SEED, bias_offset_adu: float=OFFSET_ADU) -> dict:
    if not (np.isfinite(rn_adu) and rn_adu>0 and 100<=n_replicates<=2_000_000
            and np.isfinite(threshold_adu) and threshold_adu>0
            and np.isfinite(bias_offset_adu)):
        raise ValueError("Invalid simulation scale, realization count or threshold")
    rng=np.random.default_rng(seed)
    clean=np.rint(bias_offset_adu+rng.normal(0,rn_adu,size=(n_replicates,NBIAS)))
    factors=np.asarray([np.nan if n<2 else normal_median_variance_factor(n)
                        for n in range(NBIAS+1)])
    results={}
    for label,added in SCENARIOS.items():
        sample=clean.copy()
        if added:
            sample[:,0]+=added

        original_median=np.median(sample,axis=1)
        original_var=np.var(sample,axis=1,ddof=1)*factors[NBIAS]

        accepted=np.abs(sample-original_median[:,None])<=threshold_adu
        n=accepted.sum(axis=1)
        if np.min(n)<2:
            raise RuntimeError("Simulated clipping leaves <2 samples: no estimator")
        clipped=np.where(accepted,sample,np.nan)
        revised_median=np.nanmedian(clipped,axis=1)
        revised_mean=np.nanmean(clipped,axis=1)
        revised_s2=(np.nansum(
            np.where(accepted,(sample-revised_mean[:,None])**2,0.0),
            axis=1)/(n-1))
        candidate_var=revised_s2*factors[n]

        def measure(median,var):
            valid=np.isfinite(var)&(var>=0)
            if not np.all(valid):
                raise RuntimeError("Simulated variances are invalid")
            empirical_var=float(np.var(median,ddof=1))
            mean_reported=float(np.mean(var))
            sigma=np.sqrt(var)
            residual=np.abs(median-bias_offset_adu)
            return {
                "empirical_estimator_variance_adu2":empirical_var,
                "mean_reported_variance_adu2":mean_reported,
                "reported_to_empirical_variance_ratio":mean_reported/empirical_var,
                "coverage_abs_err_le_reported_1sigma":float(np.mean(residual<=sigma)),
                "coverage_abs_err_le_reported_1p96sigma":float(np.mean(residual<=1.96*sigma)),
                "mean_signed_estimator_bias_adu":float(np.mean(median-bias_offset_adu)),
            }

        results[label]={
            "injected_positive_outlier_adu":added,
            "per_replicate_discarded_fraction":float(np.mean(n<NBIAS)),
            "support_counts":{str(int(k)):int(count)
                              for k,count in zip(*np.unique(n,return_counts=True))},
            "old_estimator":measure(original_median,original_var),
            "candidate_clipped_estimator":measure(revised_median,candidate_var),
            "median_abs_change_adu":float(np.median(np.abs(revised_median-original_median))),
        }
    return {
        "status":"SYNTHETIC_COVERAGE_DIAGNOSTIC_ONLY_NOT_PRODUCTION",
        "parameters":{
            "seed":int(seed),"n_replicates_per_scenario":int(n_replicates),
            "n_bias":NBIAS,"readnoise_sigma_adu":float(rn_adu),
            "adc_rounding_to_integer_adu":True,
            "gross_rejection_threshold_adu":float(threshold_adu),
            "true_bias_offset_adu":float(bias_offset_adu),
            "finite_n_factor_20":float(factors[20]),
            "finite_n_factor_19":float(factors[19]),
        },
        "ideal_normal_coverage_reference":{"1sigma":0.682689492137,
                                           "1p96sigma":0.9500042097},
        "conditional_scenarios":results,
        "caveats":[
            "All contaminated scenarios force an outlier into each replicate; these are stress tests, not an estimated observational rate.",
            "Noise is IID rounded Gaussian with a uniform true bias pedestal, not an empirical SCORPIO detector noise distribution.",
            "Coverage depends on the unknown sub-ADU phase of each pixel's true electronic pedestal; the phase-sensitivity table is essential.",
            "Data-dependent clipping, finite n and integer quantization can yield undercoverage of Gaussian plug-in intervals.",
            "Real validated error bars must be benchmarked against held-out observations or physically grounded injection/recovery.",
            "No source FITS or MasterBias files are read or modified except the input index JSON.",
        ]
    }


def run(index_path:Path,report_path:Path):
    index_path=index_path.expanduser().resolve(strict=True)
    report_path=report_path.expanduser().resolve()
    root=index_path.parent.parent
    if report_path.is_relative_to(root):
        raise RuntimeError("Refuse to write QA report inside Scorpipe workspace")
    if report_path.exists():
        raise FileExistsError("Refuse to overwrite an existing QA report")
    index=json.loads(index_path.read_text(encoding="utf-8-sig"))
    groups=index.get("groups") or []
    if index.get("schema_version")!=3 or len(groups)!=1 or groups[0].get("n_used")!=20:
        raise RuntimeError("Unexpected MasterBias index; expected one 20-frame v3 group")
    rn=groups[0].get("noise",{}).get("rn_est_adu")
    if rn is None or not np.isfinite(float(rn)) or float(rn)<=0:
        raise RuntimeError("Missing positive independent-readnoise estimate in index")
    payload=simulate(float(rn))
    # ADC phase is unknown for real pixels; show sensitivity rather than
    # treating a rounded-integer true pedestal as representative of all phases.
    phase_coverage={}
    for frac in (0.0,0.25,0.5,0.75):
        phase=simulate(float(rn),n_replicates=25000,seed=SEED,
                       bias_offset_adu=OFFSET_ADU+frac)
        phase_coverage[str(frac)]={
            label:{
                "old_1sigma":case["old_estimator"]["coverage_abs_err_le_reported_1sigma"],
                "candidate_1sigma":case["candidate_clipped_estimator"]["coverage_abs_err_le_reported_1sigma"],
                "old_1p96sigma":case["old_estimator"]["coverage_abs_err_le_reported_1p96sigma"],
                "candidate_1p96sigma":case["candidate_clipped_estimator"]["coverage_abs_err_le_reported_1p96sigma"],
                "candidate_reported_to_empirical_variance_ratio":
                    case["candidate_clipped_estimator"]["reported_to_empirical_variance_ratio"],
            } for label,case in phase["conditional_scenarios"].items()
        }
    payload["quantization_phase_sensitivity"]=phase_coverage
    payload["source_index"]=str(index_path)
    payload["source_group_gid"]=groups[0]["gid"]
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    for name,sc in payload["conditional_scenarios"].items():
        old=sc["old_estimator"]
        new=sc["candidate_clipped_estimator"]
        print(name,"original_var/empirical",round(old["reported_to_empirical_variance_ratio"],5),
              "candidate_var/empirical",round(new["reported_to_empirical_variance_ratio"],5),
              "candidate_1sigma_coverage",round(new["coverage_abs_err_le_reported_1sigma"],5),
              flush=True)
    print("REPORT:",report_path,"| SCIENCE HOLD")


def selftest():
    x=simulate(3.14506954,n_replicates=2000,seed=726)
    assert x["conditional_scenarios"]["clean"]["support_counts"]=={"20":2000}
    assert x["conditional_scenarios"]["row_559"]["support_counts"]=={"19":2000}
    assert x["conditional_scenarios"]["transient_17166"]["support_counts"]=={"19":2000}
    assert x["conditional_scenarios"]["row_559"]["old_estimator"]["mean_reported_variance_adu2"]>1000
    assert x["conditional_scenarios"]["row_559"]["candidate_clipped_estimator"]["mean_reported_variance_adu2"]<2
    print("SELFTEST PASSED")


if __name__=="__main__":
    if len(sys.argv)==2 and sys.argv[1]=="--selftest":
        selftest()
    elif len(sys.argv)==3:
        run(Path(sys.argv[1]),Path(sys.argv[2]))
    else:
        raise SystemExit("Usage: script.py --selftest | master_bias_index.json QA_report.json")
