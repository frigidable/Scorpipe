"""Read-only provenance-checked NEON spatial curvature validation (v20).

This reuses original-frame profile fits from v19 and a frozen v18 candidate.
No production LSF file or original science/exposure file is written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from scorpio_pipe.lsf_spatial_curvature_audit import audit_neon_spatial_curvature


ORIGINAL_SHA = {
    "arc06":"54a6d638d2284b9bfd1d947df02399eaf0264f998667a830e144b1c437a61dac",
    "arc13":"9e2c98f2d2ba1125e18f84722854099f73e435a1fd7810f0b4df84ea07ab0c05",
    "lambda_map":"37808e73c00279ced67ea520e2245767427bd71dc7edec9af22177fbc8df07ab",
}
if any(len(x)!=64 or any(c not in "0123456789abcdef" for c in x)
       for x in ORIGINAL_SHA.values()):
    raise ValueError("Invalid canonical original-ARC SHA256")


def sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda:src.read(4*1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in (
        "arc06","arc13","lambda-map","v18-json",
        "v19-json","v19-csv","output-json",
    ):
        parser.add_argument("--"+name,required=True,type=Path)
    args=parser.parse_args()
    inputs={
        name:getattr(args,name).expanduser().resolve(strict=True)
        for name in ("arc06","arc13","lambda_map","v18_json","v19_json","v19_csv")
    }
    if not all(path.is_file() for path in inputs.values()):
        raise ValueError("Input must be regular, immutable original files")
    for key,basename in (
        ("arc06","s22800606.fts"),("arc13","s22800613.fts"),
        ("lambda_map","lambda_map.fits")
    ):
        path=inputs[key]
        if path.name!=basename or sha(path)!=ORIGINAL_SHA[key]:
            raise RuntimeError("Original ARC or frozen Wave2D source SHA mismatch: "+key)
    previous=json.loads(inputs["v18_json"].read_text(encoding="utf-8"))
    v19=json.loads(inputs["v19_json"].read_text(encoding="utf-8"))
    if (previous.get("status")!="DIAGNOSTIC_CANDIDATE"
            or previous.get("science_lsf_validated") is not False
            or previous.get("auto_replace_production_lsf") is not False
            or previous.get("wavelength_map_sha256")!=ORIGINAL_SHA["lambda_map"]
            or previous.get("source_arc_sha256")!={"NEON06":ORIGINAL_SHA["arc06"],
                                                    "NEON13":ORIGINAL_SHA["arc13"]}):
        raise RuntimeError("Previous v18 diagnostic does not match original sources")
    if (v19.get("status")!="DIAGNOSTIC_ONLY"
            or v19.get("science_lsf_validated") is not False
            or v19.get("auto_apply_allowed") is not False
            or v19.get("original_source_sha256")!=ORIGINAL_SHA
            or v19.get("n_attempts")!=2640
            or v19.get("n_numerically_accepted")!=2425):
        raise RuntimeError("Previous original-ARC v19 profile diagnostic not verified")
    records=[]
    with inputs["v19_csv"].open(encoding="utf-8-sig",newline="") as stream:
        for row in csv.DictReader(stream):
            records.append(row)
    if len(records)!=v19["n_attempts"] or sum(
        row["status"]=="DIAGNOSTIC_FIT" for row in records
    )!=v19["n_numerically_accepted"]:
        raise ValueError("Original-ARC CSV inconsistent with v19 JSON counts")
    result=audit_neon_spatial_curvature(records,previous)
    result["source_sha256"]={
        key:sha(value) for key,value in inputs.items()
    }
    result["v18_v19_sources_are_precomputed"]=True
    result["raw_pixels_refitted_this_run"]=False
    result["science_fits_written"]=False
    result["uncertainty_status"]="SPATIAL_MODEL_SYSTEMATICS_NOT_FORMAL_CI"

    output=args.output_json.expanduser().resolve()
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError("Output JSON must not exist inside an existing new audit directory")
    if any(output==p or output.is_relative_to(p.parent)
           for p in inputs.values()):
        raise ValueError("Output path overlaps protected original input directory")
    # Serialize prior to writing, including finite-number validation.
    serialized=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
    with output.open("x",encoding="utf-8") as sink:
        sink.write(serialized)
    print("LSF20 ORIGINAL-ARC SPATIAL AUDIT:",result["status"])
    print("Accepted two-lamp pair rows:",
          result["n_pairs_with_sufficient_repeatable_y"],"/4")
    for pair in result["pair_diagnostics"]:
        if pair["status"]=="DIAGNOSTIC_ONLY":
            print(pair["pair_air_A"],
                  "arc curvature v2=",pair["fwhm_quadratic_coeff_v2_by_arc"],
                  "v18 shared=",pair["shared_v18_quadratic_coeff_v2"],
                  "sign agrees=",pair["curvature_sign_matches_v18_for_both_arcs"])
    print("No FITS written; no science LSF approval. Output:",output)


if __name__=="__main__":
    main()
