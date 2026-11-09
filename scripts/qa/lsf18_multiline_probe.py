"""Read-only N2146sat two-ARC LSF consistency audit from frozen CSV fits.

The measurement CSVs contain *precomputed* independent line-profile fits;
this script checks their source hashes but does not refit the original FITS.
No science FITS or production LSF map is written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from scorpio_pipe.lsf_multiline_validation import validate_lsf_multiline

HASHES = {
    "neon06": "54a6d638d2284b9bfd1d947df02399eaf0264f998667a830e144b1c437a61dac",
    "neon13": "9e2c98f2d2ba1125e18f84722854099f73e435a1fd7810f0b4df84ea07ab0c05",
    "map": "37808e73c00279ced67ea520e2245767427bd71dc7edec9af22177fbc8df07ab",
    "lineid": "2d88b6948d9225326bcbf16125dfa844544da6a44ec1e5e8af6438975433d623",
    "singles": "bd59115bfa256621acf2a0e7ba95ffeb50eb91d1d844671f34117a2c345796fb",
    "pairs": "85c967f8d697498c30484f987db543d882f9378fc4a2dc5560bcb3d836cad7ac",
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b""):
            result.update(chunk)
    return result.hexdigest()


def load_arc_measurements(
    single_path: Path, pair_path: Path, source_line_list: Path
) -> list[dict]:
    refs=set()
    for line in source_line_list.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cols=line.split()
        if len(cols)<2:
            raise ValueError("Invalid manually reviewed hand_pairs row")
        refs.add(round(float(cols[1]),2))
    if len(refs)!=19:
        raise ValueError("Source reviewed LineID must contain exactly 19 lines")
    entries=[]
    with single_path.open(encoding="utf-8-sig",newline="") as stream:
        for row in csv.DictReader(stream):
            lab=round(float(row["line_A"]),2)
            if lab not in refs:
                raise ValueError("Unknown single ARC laboratory line")
            basename=Path(row["file"]).name
            if basename not in {"s22800606.fts","s22800613.fts"}:
                raise ValueError("Unexpected original ARC label in single-line CSV")
            entries.append({
                "feature_id":f"single:{lab:.2f}",
                "arc_id":"NEON06" if basename=="s22800606.fts" else "NEON13",
                "fit_kind":"single",
                "wavelength_A":lab,
                "lab_lines_A":[lab],
                "y_pix":float(row["y"]),
                "fwhm_A":float(row["fwhm_A"]),
            })
    with pair_path.open(encoding="utf-8-sig",newline="") as stream:
        for row in csv.DictReader(stream):
            if row["status"]!="DIAGNOSTIC_ONLY":
                continue
            a=round(float(row["line1_air_A"]),2)
            b=round(float(row["line2_air_A"]),2)
            if a not in refs or b not in refs or a>=b:
                raise ValueError("Unreviewed joint Ne pair")
            if row["arc_id"] not in {"06","13"}:
                raise ValueError("Unexpected original ARC label in pair CSV")
            entries.append({
                "feature_id":f"pair:{a:.2f}+{b:.2f}",
                "arc_id":"NEON"+row["arc_id"],
                "fit_kind":"pair",
                "wavelength_A":round((a+b)/2.0,4),
                "lab_lines_A":[a,b],
                "y_pix":float(row["y_center_0based"]),
                "fwhm_A":float(row["fwhm_A"]),
            })
    return entries


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for key in HASHES:
        parser.add_argument("--"+key,required=True,type=Path)
    parser.add_argument("--out-json",required=True,type=Path)
    args=parser.parse_args()
    paths={
        k:getattr(args,k).expanduser().resolve(strict=True)
        for k in HASHES
    }
    for key,path in paths.items():
        if not path.is_file() or digest(path)!=HASHES[key]:
            raise RuntimeError(
                f"Immutable N2146sat source SHA mismatch: {key}: {path}"
            )
    if (paths["neon06"].name!="s22800606.fts"
            or paths["neon13"].name!="s22800613.fts"):
        raise RuntimeError("Wrong original ARC basenames")
    records=load_arc_measurements(
        paths["singles"],paths["pairs"],paths["lineid"]
    )
    report=validate_lsf_multiline(
        records,arc_ids=("NEON06","NEON13"),
        source_arc_sha256={
            "NEON06":HASHES["neon06"],"NEON13":HASHES["neon13"]
        },
        wavelength_map_sha256=HASHES["map"],
        manual_lineid_sha256=HASHES["lineid"],
    )
    report["provenance"]={
        "measurement_csv_sha256":{
            "singles":HASHES["singles"],"pairs":HASHES["pairs"]
        },
        "measurements_are_precomputed":True,
        "raw_arcs_not_refit_by_this_script":True,
        "lambda_map_not_modified":True,
        "upstream_science_stage_executed":False,
    }
    target=args.out_json.expanduser().resolve()
    if target.exists() or not target.parent.is_dir():
        raise FileExistsError(
            "Output JSON must not exist; parent must already exist"
        )
    if any(target.is_relative_to(p.parent) for p in paths.values()):
        raise ValueError(
            "Refusing to write inside a protected input directory"
        )
    with target.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print("LSF18 audit:",report["status"])
    print("Two-ARC candidate features:",report.get("n_cv_accepted_features"))
    print("Candidate span:",report.get("wavelength_support_A"))
    print("SCIENCE HOLD: no science LSF approval or FITS output.")
    print("Audit report:",target)


if __name__=="__main__":
    main()
