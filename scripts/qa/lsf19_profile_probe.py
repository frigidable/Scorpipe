"""Source-SHA-locked original N2146sat Ne line-profile model sensitivity.

Read only s22800606/13.fts, _95/_96 lambda_map and v18 JSON.
Write diagnostic CSV/JSON to a NEW destination. No production FITS.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scorpio_pipe.lsf_profile_systematics import fit_neon_pair_hypothesis

ORIGINAL_SHA = {
    "arc06": "54a6d638d2284b9bfd1d947df02399eaf0264f998667a830e144b1c437a61dac",
    "arc13": "9e2c98f2d2ba1125e18f84722854099f73e435a1fd7810f0b4df84ea07ab0c05",
    "wavelength_map": "37808e73c00279ced67ea520e2245767427bd71dc7edec9af22177fbc8df07ab",
}
if any(len(x)!=64 or any(ch not in "0123456789abcdef" for ch in x)
       for x in ORIGINAL_SHA.values()):
    raise ValueError("Malformed original-ARC SHA256 provenance")
NEON_PAIRS = ((4181.88,4190.71),(4333.56,4345.17),
              (5037.75,5047.74),(5330.78,5341.60))
WINDOWS = (11,13,16,20,24)
ORDERS = (1,2)
Y_BINS = tuple(range(74,939,27))


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read_frozen_primary(path: Path) -> tuple[np.ndarray,dict]:
    """Minimal strict reader for only the *SHA-locked* 2D original FITS.

    General FITS reduction must use Astropy. This function is restricted
    to the exact original files and their known 2D uncompressed primary HDU.
    """
    with path.open("rb") as fd:
        header={}
        while True:
            block=fd.read(2880)
            if len(block)!=2880:
                raise ValueError("Incomplete FITS header")
            done=False
            for start in range(0,2880,80):
                card=block[start:start+80].decode("ascii","strict")
                if card.startswith("END "):
                    done=True
                    break
                if card[8:10]=="= ":
                    raw=card[10:30].strip()
                    if raw.startswith("'"):
                        val=raw.strip("'").strip()
                    else:
                        val=raw.split("/")[0].strip()
                        try:
                            val=float(val) if any(c in val for c in ".Ee") else int(val)
                        except ValueError:
                            pass
                    header[card[:8].strip()]=val
            if done:
                break
        if (header.get("NAXIS")!=2 or header.get("NAXIS1")!=2080
            or header.get("NAXIS2")!=1032 or header.get("BITPIX") not in (16,-32)):
            raise ValueError("Unexpected frozen original FITS geometry/encoding")
        dtype={16:">i2",-32:">f4"}[header["BITPIX"]]
        data=np.fromfile(fd,dtype=dtype,count=2080*1032)
        if data.size!=2080*1032:
            raise ValueError("Truncated frozen ARC FITS")
        array=data.reshape(1032,2080).astype(float)
    array=array*float(header.get("BSCALE",1))+float(header.get("BZERO",0))
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite frozen primary image")
    return array,header


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arc06",type=Path,required=True)
    parser.add_argument("--arc13",type=Path,required=True)
    parser.add_argument("--wavelength-map",type=Path,required=True)
    parser.add_argument("--v18-report",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    sources={key:getattr(args,key).expanduser().resolve(strict=True)
             for key in ("arc06","arc13")}
    sources["wavelength_map"]=args.wavelength_map.expanduser().resolve(strict=True)
    for key,name in (("arc06","s22800606.fts"),("arc13","s22800613.fts"),
                     ("wavelength_map","lambda_map.fits")):
        if sources[key].name!=name or _sha(sources[key])!=ORIGINAL_SHA[key]:
            raise RuntimeError("Source SHA256/name mismatch: "+key)
    previous=args.v18_report.expanduser().resolve(strict=True)
    old=json.loads(previous.read_text(encoding="utf-8"))
    if (old.get("status")!="DIAGNOSTIC_CANDIDATE"
        or old.get("science_lsf_validated") is not False
        or old.get("auto_replace_production_lsf") is not False
        or old.get("wavelength_map_sha256")!=ORIGINAL_SHA["wavelength_map"]
        or old.get("source_arc_sha256")!={"NEON06":ORIGINAL_SHA["arc06"],
                                         "NEON13":ORIGINAL_SHA["arc13"]}):
        raise RuntimeError("Unverified immutable v18 source provenance")
    output=args.output_dir.expanduser().resolve()
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError("Must select a NEW audit output directory")
    if any(output.is_relative_to(p.parent) or p.is_relative_to(output)
           for p in sources.values()):
        raise ValueError("Output overlaps frozen original FITS inputs")
    if output==previous.parent or previous.is_relative_to(output):
        raise ValueError("Output overlaps frozen v18 report")
    waves,hdr=read_frozen_primary(sources["wavelength_map"])
    if str(hdr.get("WAVEREF","")).lower()!="air":
        raise ValueError("Expected original air wavelengths")
    originals={key:read_frozen_primary(sources[key])[0]
               for key in ("arc06","arc13")}
    results=[]
    for key,arc in originals.items():
        for j,pair in enumerate(NEON_PAIRS):
            for y in Y_BINS:
                profile=np.median(arc[y-10:y+11],axis=0)
                for hw in WINDOWS:
                    for degree in ORDERS:
                        trial=fit_neon_pair_hypothesis(
                            waves[y],profile,pair,
                            half_window_px=hw,continuum_degree=degree)
                        results.append({
                            "arc_id":key,"pair":j,"y_pix":y,
                            "half_strip_pix":10,
                            "half_window_px":hw,"continuum_degree":degree,
                            **trial})
    good=[r for r in results if r["status"]=="DIAGNOSTIC_FIT"]
    summaries=[]
    for j,pair in enumerate(NEON_PAIRS):
        segment=[r for r in good if r["pair"]==j]
        medians={}
        for degree in ORDERS:
            for hw in WINDOWS:
                vals=[r["fwhm_A"] for r in segment
                      if r["continuum_degree"]==degree
                      and r["half_window_px"]==hw]
                medians[f"degree{degree}_window{hw}"] = (
                    float(np.median(vals)) if vals else None
                )
        summaries.append({"pair_A":list(pair),"n_good":len(segment),
                          "median_fwhm_by_hypothesis_A":medians})
    center=[r for r in good if r["pair"]==3 and r["y_pix"]==506
            and r["half_window_px"]==16 and r["continuum_degree"]==1]
    coeff=np.asarray(old["coefficients"],float)
    w=5336.19
    y=506
    u=(w-4600)/1000
    v=(y-515.5)/515.5
    predicted=float(np.dot(coeff,[1,u,v,v*v,u*v]))
    report={
        "status":"DIAGNOSTIC_ONLY",
        "n_attempts":len(results),"n_numerically_accepted":len(good),
        "original_source_sha256":ORIGINAL_SHA,
        "v18_report_sha256":_sha(previous),
        "y_bins":list(Y_BINS),"windows_px":list(WINDOWS),
        "continuum_degrees":list(ORDERS),
        "feature_summaries":summaries,
        "red_pair_y506":{
            "original_lamp_fwhm_A":{r["arc_id"]:r["fwhm_A"] for r in center},
            "v18_smooth_model_fwhm_A":predicted,
        },
        "uncertainty_status":"MODEL_DEPENDENCE_NOT_FORMAL_CI",
        "science_lsf_validated":False,
        "auto_apply_allowed":False,
        "science_FITS_written":False,
    }
    # Validate all output content before making any new directory.
    json.dumps(report,allow_nan=False)
    output.mkdir(exist_ok=False)
    columns=("arc_id","pair","y_pix","half_strip_pix",
             "half_window_px","continuum_degree","status","reason",
             "fwhm_A","fwhm_px","shift_px","normalized_rms",
             "residual_odd_projection","amplitude_ratio","rough_noise_proxy_adu")
    with (output/"profile_model_sensitivity.csv").open(
        "x",encoding="utf-8",newline=""
    ) as sink:
        writer=csv.DictWriter(sink,fieldnames=columns,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    with (output/"lsf19_science_diagnostic.json").open(
        "x",encoding="utf-8"
    ) as sink:
        json.dump(report,sink,ensure_ascii=False,indent=2,allow_nan=False)
    print("LSF19 ORIGINAL-ARC DIAGNOSTIC:",len(good),"/",len(results))
    print("Red pair y=506:",report["red_pair_y506"])
    print("SCIENCE HOLD. No science FITS. Audit:",output)


if __name__=="__main__":
    main()
