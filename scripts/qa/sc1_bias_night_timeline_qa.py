#!/usr/bin/env python3
"""Read-only SCORPIO-1 N2146sat chronology and late-BIAS cross-epoch QA.

Uses EXACT early20, late19, flagged2 filenames established by the independent
41-frame inventory. Queries the 7 *selected* OBJ headers, never infers exposure
dates from FITS filenames. The output is diagnostic, never a calibration stage.

Every CCD mode and image shape is validated before a pixel-level comparison.
Candidate sidebands are NOT certified physical overscan. DATE-OBS and DATE
fields are reported verbatim: the instrument timestamp convention is unverified.

Usage: python scripts/qa/sc1_bias_night_timeline_qa.py --selftest
       python scripts/qa/sc1_bias_night_timeline_qa.py config.yaml QA_report.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from sc1_masterbias_raw_sample_qa import (
    read_config_and_products, ACTIVE, EXPECTED_SHAPE,
)

EARLY_NAMES = {f"s228001{i:02d}.fts" for i in range(1,21)}
LATE_NAMES = {f"s228017{i:02d}.fts" for i in range(2,21)}
FLAGGED_NAMES = {"s22801620.fts", "s22801701.fts"}
ALL_NAMES = EARLY_NAMES | LATE_NAMES | FLAGGED_NAMES
SIDES = ((2,14),(2066,2078))
THRESHOLD = 50.0
TILE_SHAPE=(15,31)


def locate_frames(raw_dir:Path, first20:list[Path])->dict[str,Path]:
    """Find all 41 known filenames without selecting extra raw frames.

    The raw directory is scanned once if the expected direct/<series> paths
    are not enough. Duplicate basenames under the source dataset are fatal.
    """
    raw_dir=raw_dir.expanduser().resolve(strict=True)
    if not raw_dir.is_dir():
        raise ValueError("Expected existing raw data directory")
    known={p.name.lower():p.resolve(strict=True) for p in first20}
    if set(known)!=EARLY_NAMES or len(first20)!=20:
        raise RuntimeError("Early BIAS selection is not exact early20")
    choices={}
    for name in ALL_NAMES:
        series=name[:7]  # s228001, s228017 or s228016
        candidates=[raw_dir/name, raw_dir/series/name]
        existing=[p.resolve() for p in candidates if p.is_file()]
        if len(set(existing))>1:
            raise RuntimeError(f"Ambiguous source FITS: {name}: {existing}")
        if existing:
            choices[name]=existing[0]
    if len(choices)<len(ALL_NAMES):
        for p in raw_dir.rglob("*"):
            if not p.is_file() or p.name.lower() not in ALL_NAMES:
                continue
            name=p.name.lower()
            rp=p.resolve()
            if name in choices and choices[name]!=rp:
                raise RuntimeError(f"Ambiguous source FITS: {name}: {choices[name]} / {rp}")
            choices[name]=rp
    if set(choices)!=ALL_NAMES:
        missing=sorted(ALL_NAMES-set(choices))
        raise FileNotFoundError(f"Expected 41 audited normal-readout BIAS files; missing {missing}")
    for p in first20:
        if choices[p.name.lower()]!=p:
            raise RuntimeError(f"Raw discovery disagrees with selected early20: {p.name}")
    return choices


def header_metadata(header) -> dict:
    """Report observed metadata, DO NOT reinterpret DATE-OBS format."""
    keys=("DATE","DATE-OBS","UT","UTC","ST","START","EXPTIME",
          "EXPOSURE","OBJECT","IMAGETYP","IMAGETYPE","RATE","GAIN",
          "NODE","BINNING","NAXIS1","NAXIS2")
    return {k:str(header.get(k,"")) for k in keys}


def roi_summary(data, master, *, cutoff=THRESHOLD):
    """BIAS-only residuals: full-frame and candidate side strips.

    Returns mean *per-exposure* residual, plus 64x64 tile means for
    investigating spatially uniform additive offsets.
    """
    y0,y1,x0,x1=ACTIVE
    if data.shape!=EXPECTED_SHAPE or master.shape!=EXPECTED_SHAPE:
        raise ValueError("Unexpected detector shape")
    r=np.asarray(data[y0:y1,x0:x1],dtype=np.float64) - master[y0:y1,x0:x1]
    valid=np.isfinite(r)&(np.abs(r)<=cutoff)
    if valid.mean()<0.99:
        raise RuntimeError("Excessive transient pixels in candidate ROI")
    vals=np.where(valid,r,0.0)
    tile_sums=vals.reshape(15,64,31,64).sum(axis=(1,3))
    tile_n=valid.reshape(15,64,31,64).sum(axis=(1,3))
    if np.any(tile_n<4000):
        raise RuntimeError("Too few valid pixels in 64x64 tile")
    tiles=tile_sums/tile_n
    bands={}
    for label,(lo,hi) in zip(("left","right"),SIDES):
        rside=np.asarray(data[y0:y1,lo:hi],dtype=np.float64)-master[y0:y1,lo:hi]
        good=np.isfinite(rside)&(np.abs(rside)<=cutoff)
        if good.mean()<0.95:
            raise RuntimeError(f"Too many outliers in unverified {label} side region")
        bands[label]={
            "mean_residual_adu":float(np.mean(rside[good])),
            "n_discarded":int((~good).sum()),
            "candidate_side_columns_halfopen":[lo,hi],
            "physical_overscan_certified":False,
        }
    return {
        "mean_roi_residual_adu":float(vals.sum()/valid.sum()),
        "median_tile_residual_adu":float(np.median(tiles)),
        "std_tile_residual_adu":float(np.std(tiles)),
        "roi_outlier_count":int((~valid).sum()),
        "candidate_side_strips":bands,
    }


def linear_summary(records):
    vals=np.array([r["mean_roi_residual_adu"] for r in records],dtype=float)
    t=np.arange(len(vals),dtype=float)
    if len(vals)<3 or not np.isfinite(vals).all():
        raise ValueError("Need three or more finite independent BIAS-frame baselines")
    slope=float(np.polyfit(t,vals,1)[0])
    return {
        "n_frames":len(vals),
        "first_offset_adu":float(vals[0]),
        "last_offset_adu":float(vals[-1]),
        "end_minus_start_adu":float(vals[-1]-vals[0]),
        "mean_offset_adu":float(np.mean(vals)),
        "median_offset_adu":float(np.median(vals)),
        "std_across_frame_offsets_adu":float(np.std(vals,ddof=1)),
        "slope_adu_per_frame_index":slope,
        "monotonically_increasing":bool(np.all(np.diff(vals)>0)),
        "max_64px_tile_std_per_frame_adu":max(
            float(r["std_tile_residual_adu"]) for r in records
        ),
    }


def date_value(raw):
    """Strict ISO DATE only, without assuming verified time-scale correctness."""
    if not raw:
        return None
    try:
        # Do not parse the ambiguous DATE-OBS 2024/06/12.
        return datetime.fromisoformat(str(raw))
    except ValueError:
        return None


def run(config_path:Path,out_path:Path):
    from astropy.io import fits

    config_path=config_path.expanduser().resolve(strict=True)
    out_path=out_path.expanduser().resolve()
    original_paths,(sci,_,_),group,_,work=read_config_and_products(config_path)
    cfg=yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
    data_root=Path(str(cfg.get("data_dir") or work)).expanduser()
    if not data_root.is_absolute():
        data_root=(work/data_root).resolve()
    data_root=data_root.resolve(strict=True)
    if out_path.is_relative_to(work) or out_path.is_relative_to(data_root):
        raise RuntimeError("QA output must be outside work_dir and raw directory")
    if out_path.exists():
        raise FileExistsError(f"Refuse to overwrite existing QA report: {out_path}")
    frames=cfg.get("frames") or {}
    objs=frames.get("obj") or []
    if not isinstance(objs,list) or len(objs)!=7:
        raise RuntimeError("Expected exactly 7 selected science OBJ frames")
    resolved_objects=[]
    for source in objs:
        p=Path(str(source)).expanduser()
        if not p.is_absolute():
            p=data_root/p
        p=p.resolve(strict=True)
        if p in resolved_objects or p in original_paths:
            raise RuntimeError("Duplicated or overlapping OBJ frame")
        resolved_objects.append(p)

    inventory=locate_frames(data_root,original_paths)
    reports={}
    raw_paths=[]
    for category,names in (
        ("early20",sorted(EARLY_NAMES)),
        ("late19",sorted(LATE_NAMES)),
        ("flagged2",sorted(FLAGGED_NAMES)),
    ):
        group_rows=[]
        for i,name in enumerate(names,1):
            path=inventory[name]
            with fits.open(path,memmap=False) as hdul:
                h=hdul[0].header
                frame=hdul[0].data
                if frame is None or frame.shape!=EXPECTED_SHAPE:
                    raise RuntimeError(f"Unexpected raw detector geometry {path}")
                if (float(h["RATE"]),float(h["GAIN"]),str(h["NODE"]).upper().strip()) != (145.0,0.9,"A"):
                    raise RuntimeError(f"Unexpected raw CCD mode {path}")
                row={"file":path.name,"group":category,"instrument_header":header_metadata(h)}
                row.update(roi_summary(frame,sci))
            group_rows.append(row)
            raw_paths.append(path)
            print(f"BIAS {category} {i}/{len(names)} {name} "
                  f"offset={row['mean_roi_residual_adu']:+.5f} ADU",flush=True)
        reports[category]=group_rows

    sci_headers=[]
    for i,p in enumerate(resolved_objects,1):
        with fits.open(p,memmap=False) as hdul:
            h=hdul[0].header
            if hdul[0].data is None or hdul[0].data.shape!=EXPECTED_SHAPE:
                raise RuntimeError(f"Unexpected OBJ geometry: {p.name}")
            if (float(h["RATE"]),float(h["GAIN"]),str(h["NODE"]).strip().upper())!=(145.0,0.9,"A"):
                raise RuntimeError(f"OBJ readout mismatches selected MasterBias: {p.name}")
            sci_headers.append({
                "file":p.name,"role":"OBJ",
                "instrument_header":header_metadata(h),
                "pixel_flux_not_used":True,
            })
        print(f"OBJ {i}/{len(resolved_objects)} {p.name}: DATE={sci_headers[-1]['instrument_header']['DATE']}",
              flush=True)

    early=linear_summary(reports["early20"])
    late=linear_summary(reports["late19"])
    # DATE can encode an instrument clock, not necessarily authoritative UTC.
    def stamp(row):
        return date_value(row["instrument_header"].get("DATE"))
    all_bias=reports["early20"]+reports["late19"]+reports["flagged2"]
    parsed=[stamp(row) for row in all_bias+sci_headers]
    ordered=[{
        "file":r["file"],
        "group":r.get("group",r.get("role")),
        "DATE_raw":r["instrument_header"]["DATE"],
        "UT_raw":r["instrument_header"]["UT"],
        "DATE_OBS_raw":r["instrument_header"]["DATE-OBS"],
    } for r in sorted(all_bias+sci_headers,key=lambda z:stamp(z) or datetime.max)]

    payload={
        "status":"READ_ONLY_LATE_BIAS_AND_SCIENCE_TIMELINE_SCIENCE_HOLD",
        "config":str(config_path),
        "masterbias_group_gid":group["gid"],
        "measurement":"Mean over provisional Y[32:992],X[32:2016] of (BIAS - saved early20 MasterBias), with |difference|>50 ADU excluded",
        "bias_early20":reports["early20"],
        "bias_late19":reports["late19"],
        "bias_flagged2_excluded_from_reference":reports["flagged2"],
        "obj_header_only":sci_headers,
        "early20_summary":early,
        "late19_summary":late,
        "late_mean_minus_early_mean_adu":float(late["mean_offset_adu"]-early["mean_offset_adu"]),
        "instrument_DATE_order_unverified":ordered,
        "parsed_iso_DATE_count":int(sum(x is not None for x in parsed)),
        "total_frames_read":len(parsed),
        "warnings":[
            "FITS DATE/UT/DATE-OBS temporal semantics and timestamps have not been instrument-certified.",
            "The provisional ROI and both candidate 12-column sidebands are NOT certified overscan/trim geometry.",
            "Science OBJ data were NOT used to estimate CCD zero level; science continuum+sky precludes direct bias estimation.",
            "Flagged BIAS frames are shown separately and not used as stable late calibration candidates.",
            "Late mean offsets relative to the early master do not prove that interpolation is suitable for science exposures.",
            "No original FITS, existing MasterBias, run configuration, or scientific pipeline products have been altered.",
        ],
    }
    out_path.parent.mkdir(parents=True,exist_ok=True)
    out_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print("EARLY:",early)
    print("LATE:",late)
    print(f"LATE minus EARLY mean={payload['late_mean_minus_early_mean_adu']:+.5f} ADU")
    print("REPORT:",out_path)
    print("SCIENCE E2E HOLD")


def selftest():
    r=np.full(EXPECTED_SHAPE,999.5,dtype=np.float64)
    s=np.full(EXPECTED_SHAPE,999.0,dtype=np.float64)
    r[100,700]+=15000
    stat=roi_summary(r,s)
    assert stat["roi_outlier_count"]==1
    assert stat["mean_roi_residual_adu"]==0.5
    for side in stat["candidate_side_strips"].values():
        assert side["mean_residual_adu"]==0.5
        assert side["physical_overscan_certified"] is False
    assert len(ALL_NAMES)==41
    assert not (EARLY_NAMES & LATE_NAMES)
    fake=[{"mean_roi_residual_adu":float(i),
           "std_tile_residual_adu":0.05} for i in range(3)]
    assert linear_summary(fake)["slope_adu_per_frame_index"]==1
    assert date_value("2024-12-06T12:24:12") is not None
    assert date_value("2024/06/12") is None
    print("SELFTEST PASSED")


if __name__=="__main__":
    if len(sys.argv)==2 and sys.argv[1]=="--selftest":
        selftest()
    elif len(sys.argv)==3:
        run(Path(sys.argv[1]),Path(sys.argv[2]))
    else:
        raise SystemExit("Usage: script.py --selftest | config.yaml external_report.json")
