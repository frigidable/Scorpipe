"""Source-locked, read-only original SCORPIO-1 19-line He-Ne-Ar audit.

Does not query NIST over the network; external ASD 5.12 full line export
remains a separate provenance requirement. Does not reduce raw ARC FITS.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from scorpio_pipe.lineid_atlas_review import (
    review_sc1_original_lineid,gaussian_component_identifiability,
    FROZEN_HAND_PAIRS_SHA256,FROZEN_LINE_LIST_SHA256,
)

_ARC_HASHES = {
    "06":"54a6d638d2284b9bfd1d947df02399eaf0264f998667a830e144b1c437a61dac",
    "13":"9e2c98f2d2ba1125e18f84722854099f73e435a1fd7810f0b4df84ea07ab0c05",
}
if any(len(x)!=64 for x in _ARC_HASHES.values()):
    raise ValueError("Original arc digest must be complete SHA256")


def _hash(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(2*1024*1024),b""):
            h.update(part)
    return h.hexdigest()


def _header(path: Path) -> dict[str,str]:
    """Read only a primary FITS header, never pixel buffers or mmap handles."""
    cards={}
    with path.open("rb") as f:
        for _ in range(100):  # fail closed on malformed 2880-byte headers
            block=f.read(2880)
            if len(block)!=2880:
                raise ValueError("Incomplete primary ARC FITS header")
            for i in range(0,2880,80):
                row=block[i:i+80].decode("ascii","strict")
                if row.startswith("END "):
                    return cards
                if row[8:10]=="= ":
                    raw=row[10:30].strip()
                    if raw.startswith("'"):
                        value=raw.split("'")[1].strip()
                    else:
                        value=raw.split("/")[0].strip()
                    cards[row[:8].strip()]=value
    raise ValueError("ARC FITS header exceeds strict audit limit")


def _flat_record(row: dict) -> dict:
    return {
        "x_manual_zero_based":row["x_manual_zero_based"],
        "historic_air_A":row["historic_air_A"],
        "species_candidate":row["species_candidate"],
        "published_transition_air_A":row["published_transition_air_A"],
        "historic_minus_published_A":row["historic_minus_published_A"],
        "status":row["status"],
        "neighbors":";".join(
            f"{x['air_A']:.5f} {x['species_or_evidence']}"
            for x in row["neighbor_candidates"]
        ),
        "reference_quality":row["reference_quality"],
        "isolation_verified_from_raw_arc":False,
        "lab_line_approved_for_new_wave2d":False,
    }


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("hand-pairs","moiseev-list","arc06","arc13","output-dir"):
        parser.add_argument("--"+name,required=True,type=Path)
    args=parser.parse_args()
    paths={
        key:getattr(args,key).expanduser().resolve(strict=True)
        for key in ("hand_pairs","moiseev_list","arc06","arc13")
    }
    names={"hand_pairs":"hand_pairs.txt","moiseev_list":"lines_long(1).tab",
           "arc06":"s22800606.fts","arc13":"s22800613.fts"}
    for key,path in paths.items():
        if not path.is_file() or path.name!=names[key]:
            raise RuntimeError("Unexpected input filename or missing immutable input: "+key)
    for suffix in ("06","13"):
        if _hash(paths["arc"+suffix])!=_ARC_HASHES[suffix]:
            raise RuntimeError("Raw original SCORPIO-1 ARC SHA mismatch: "+suffix)
    if _hash(paths["hand_pairs"])!=FROZEN_HAND_PAIRS_SHA256:
        raise RuntimeError("Frozen hand_pairs source SHA mismatch")
    if _hash(paths["moiseev_list"])!=FROZEN_LINE_LIST_SHA256:
        raise RuntimeError("Frozen Moiseev source SHA mismatch")
    hdr={}
    for suffix in ("06","13"):
        h=_header(paths["arc"+suffix])
        if (h.get("INSTRUME","").upper()!="SCORPIO-1"
            or h.get("IMAGETYP","").lower()!="neon"
            or h.get("DISPERSE","").upper()!="VPHG1200B"):
            raise RuntimeError("Raw ARC header contradicts original instrument identity")
        hdr["NEON"+suffix]={
            k:h.get(k) for k in (
                "INSTRUME","IMAGETYP","DISPERSE","OBJECT",
                "DATE","DATE-OBS","ROTANGLE","EXPTIME","FILTERS","SLITWID"
            )
        }
    report=review_sc1_original_lineid(
        paths["hand_pairs"],paths["moiseev_list"]
    )
    report["original_arc_source_sha256"]=dict(_ARC_HASHES)
    report["observed_2024_header_cards"]=hdr
    report["atlas_matching_method"]=(
        "Curated air-wavelength NIST abbreviated Strong Lines screen, "
        "supplemented with independent 5080.38 Ne I line studies and "
        "provisional 5221.27 Ar I candidate; full ASD export not yet obtained"
    )
    report["blend_diagnostics"]={
        "Ar_I_4190_713_plus_4191_029":gaussian_component_identifiability(
            4190.713,4191.029,fwhm_A=6.
        ),
        "He_I_4713_146_plus_4713_380":gaussian_component_identifiability(
            4713.146,4713.380,fwhm_A=6.
        ),
        "Ne_I_5341_0938_plus_5343_2834":gaussian_component_identifiability(
            5341.0938,5343.2834,fwhm_A=6.
        ),
    }
    report["raw_ARC_pixels_reduced"]=False
    report["science_fits_written"]=False
    output=args.output_dir.expanduser().resolve()
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError("Output must be an exclusive NEW directory")
    if any(output.is_relative_to(path.parent) or path.is_relative_to(output)
           for path in paths.values()):
        raise ValueError("Output would overlap frozen source files")
    serialized=json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)
    records=[_flat_record(q) for q in report["entries"]]
    output.mkdir(exist_ok=False)
    with (output/"lineid22_19_review.json").open("x",encoding="utf-8") as f:
        f.write(serialized)
    with (output/"lineid22_19_review.csv").open(
        "x",encoding="utf-8",newline=""
    ) as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("LINEID22:",report["status"],
          "19 of",report["n_historic_anchors"],"source-verified")
    print("UNRESOLVED historical IDs:",report["arity_and_line_id_risks"][
        "unresolved_atomic_identifications_air_A"])
    print("SCIENCE HOLD: no original FITS were written. Output:",output)


if __name__=="__main__":
    main()
