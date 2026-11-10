"""SHA-locked v21 original NEON06/NEON13 red-multiplet read-only diagnostic.

Refits original ARC pixels under historical and atlas line hypotheses.
Identical spectral fitting interval for every comparison; spatial
trace-registered extraction is sensitivity-only. No science FITS writes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from lsf19_profile_probe import read_frozen_primary
from scorpio_pipe.lsf_neon_multiplet_diagnostic import (
    HISTORIC_RED_PAIR_A, NIST_NEI_RED_AIR_A, ADDITIONAL_NEI_5326_A,
    fit_original_arc_multiplet, red_lineid_review, residual_agreement,
    trace_registered_median,
)

EXPECTED_SHA256 = {
    "arc06":"54a6d638d2284b9bfd1d947df02399eaf0264f998667a830e144b1c437a61dac",
    "arc13":"9e2c98f2d2ba1125e18f84722854099f73e435a1fd7810f0b4df84ea07ab0c05",
    "map":"37808e73c00279ced67ea520e2245767427bd71dc7edec9af22177fbc8df07ab",
    "hand_pairs":"2d88b6948d9225326bcbf16125dfa844544da6a44ec1e5e8af6438975433d623",
}
if any(len(v)!=64 or any(ch not in "0123456789abcdef" for ch in v)
       for v in EXPECTED_SHA256.values()):
    raise ValueError("Malformed canonical original-source SHA256")


def _sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as src:
        for data in iter(lambda:src.read(1024*1024),b""):
            digest.update(data)
    return digest.hexdigest()


def _median_finite(items: list[float]) -> float | None:
    return float(np.median(items)) if items else None


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("arc06","arc13","map","hand-pairs","output-dir"):
        p.add_argument("--"+arg,required=True,type=Path)
    args=p.parse_args()
    inputs={
        key:getattr(args,key).expanduser().resolve(strict=True)
        for key in ("arc06","arc13","map","hand_pairs")
    }
    expected_names={
        "arc06":"s22800606.fts","arc13":"s22800613.fts",
        "map":"lambda_map.fits","hand_pairs":"hand_pairs.txt",
    }
    for key,file in inputs.items():
        if not file.is_file() or file.name!=expected_names[key]:
            raise RuntimeError("Unexpected v21 original input identity: "+key)
        if _sha(file)!=EXPECTED_SHA256[key]:
            raise RuntimeError("Original source SHA256 mismatch: "+key)
    line_list=inputs["hand_pairs"].read_text(encoding="utf-8")
    if not any("5341.60" in line for line in line_list.splitlines()):
        raise RuntimeError("Historic hand_pairs does not contain 5341.60 Å")
    wave,wh=read_frozen_primary(inputs["map"])
    if str(wh.get("WAVEREF","")).strip().lower()!="air":
        raise RuntimeError("Frozen Wave2D map must be air wavelength")
    images={
        "06":read_frozen_primary(inputs["arc06"])[0],
        "13":read_frozen_primary(inputs["arc13"])[0],
    }
    if any(frame.shape!=wave.shape for frame in images.values()):
        raise ValueError("Raw ARC and wavelength map geometry differs")
    pairs={
        "historic_2":tuple(HISTORIC_RED_PAIR_A),
        "nist_3":tuple(NIST_NEI_RED_AIR_A),
        "nist_4":tuple(sorted((ADDITIONAL_NEI_5326_A,*NIST_NEI_RED_AIR_A))),
    }
    rows=[]
    fitted={}
    for y in range(74,939,27):
        for arc_id,data in images.items():
            raw=np.median(data[y-10:y+11],axis=0)
            for label,positions in pairs.items():
                fit=fit_original_arc_multiplet(wave[y],raw,positions)
                record={
                    "y":int(y),"arc":arc_id,"model":label,
                    "status":fit["status"],
                    "normalized_rmse":fit.get("normalized_rmse"),
                    "fwhm_gaussian_equivalent_A":fit.get("fwhm_gaussian_equivalent_A"),
                    "shared_shift_px":fit.get("shared_shift_px"),
                    "component_strengths":fit.get("amplitudes"),
                }
                rows.append(record)
                fitted[(y,arc_id,label)]=fit
            aligned=trace_registered_median(data,wave,int(y),half_strip=10)
            # Model must not silently accept any NaNs introduced at detector edges.
            # All fit windows lie well away from those edges on this setup.
            if not np.isfinite(aligned).all():
                valid=np.isfinite(aligned)
                aligned=np.interp(
                    np.arange(aligned.size),
                    np.flatnonzero(valid),aligned[valid],
                )
            pair_fit=fit_original_arc_multiplet(
                wave[y],aligned,pairs["historic_2"]
            )
            rows.append({
                "y":int(y),"arc":arc_id,"model":"historic_2_registered",
                "status":pair_fit["status"],
                "normalized_rmse":pair_fit.get("normalized_rmse"),
                "fwhm_gaussian_equivalent_A":pair_fit.get(
                    "fwhm_gaussian_equivalent_A"
                ),
                "shared_shift_px":pair_fit.get("shared_shift_px"),
                "component_strengths":pair_fit.get("amplitudes"),
            })
    comparisons={}
    for label in pairs:
        successful=[
            r for r in rows if r["model"]==label and r["status"]=="DIAGNOSTIC_ONLY"
        ]
        comparisons[label]={
            "n_successful":len(successful),
            "median_fwhm_A":_median_finite([
                r["fwhm_gaussian_equivalent_A"] for r in successful
            ]),
            "median_normalized_rmse":_median_finite([
                r["normalized_rmse"] for r in successful
            ]),
        }
    agreement=[]
    for y in range(74,939,27):
        q1=fitted[(y,"06","historic_2")]
        q2=fitted[(y,"13","historic_2")]
        if q1["status"]=="DIAGNOSTIC_ONLY" and q2["status"]=="DIAGNOSTIC_ONLY":
            qc=residual_agreement(q1,q2)
            if qc["status"]=="DIAGNOSTIC_ONLY":
                agreement.append(qc["pearson_correlation"])
    output=args.output_dir.expanduser().resolve()
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError("Output must be a NEW directory in an existing parent")
    if any(output.is_relative_to(path.parent) or path.is_relative_to(output)
           for path in inputs.values()):
        raise ValueError("Refusing to write any file in immutable source directories")
    report={
        "status":"REVIEW_REQUIRED",
        "source_sha256":dict(EXPECTED_SHA256),
        "source_kind":"2 separate original unstacked raw Ne frames",
        "lab_catalog_check":red_lineid_review(),
        "n_original_arc_spatial_bins_per_lamp":33,
        "n_profile_model_attempts":len(rows),
        "hypotheses":comparisons,
        "median_correlated_residual_pearson_two_lamps":_median_finite(agreement),
        "note":(
            "Model comparisons are on identical wavelength pixels; "
            "FWHM remains strongly conditional on chosen line membership. "
            "Shared wavelength map and detector systematic errors make "
            "repeated lamps non-independent for absolute calibration."
        ),
        "science_lsf_validated":False,
        "wave2d_calibration_approved":False,
        "auto_replace_production_lsf":False,
        "original_science_fits_written":False,
    }
    serialized=json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)
    output.mkdir(exist_ok=False)
    (output/"lsf21_atlas_lineid_profile.json").write_text(
        serialized,encoding="utf-8"
    )
    fields=("y","arc","model","status","normalized_rmse",
            "fwhm_gaussian_equivalent_A","shared_shift_px","component_strengths")
    with (output/"lsf21_fits_by_y.csv").open(
        "x",encoding="utf-8",newline=""
    ) as out:
        writer=csv.DictWriter(out,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("LSF21 original NEON line-list review:",report["status"])
    print("Model comparison (same detector window):",comparisons)
    print("SCIENCE HOLD. Original FITS unchanged. Output:",output)


if __name__=="__main__":
    main()
