"""LSF20 v19 JSON/CSV provenance and numeric matching contract."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from statistics import median

import pytest
from scorpio_pipe.lsf_spatial_curvature_audit import FWHM_PAIRS_A


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qa" / (
    "lsf20_spatial_curvature_probe.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("lsf20_qa", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sample():
    rows=[]
    for pair in range(4):
        for arc in ("arc06","arc13"):
            for degree in (1,2):
                for width in (11,13,16,20,24):
                    fwhm=5.7+pair*0.07+(width-16)*0.005+degree*0.02
                    fwhm+=0.025 if arc=="arc13" else 0.0
                    rows.append({
                        "pair":str(pair), "arc_id":arc,
                        "y_pix":"506", "continuum_degree":str(degree),
                        "half_window_px":str(width), "status":"DIAGNOSTIC_FIT",
                        "fwhm_A":str(fwhm),
                    })
    summary=[]
    for pair in range(4):
        sub=[row for row in rows if int(row["pair"])==pair]
        medians={}
        for degree in (1,2):
            for width in (11,13,16,20,24):
                medians[f"degree{degree}_window{width}"]=median(
                    float(row["fwhm_A"]) for row in sub
                    if int(row["continuum_degree"])==degree
                    and int(row["half_window_px"])==width
                )
        summary.append({
            "pair_A":list(FWHM_PAIRS_A[pair]),
            "n_good":len(sub),
            "median_fwhm_by_hypothesis_A":medians,
        })
    red={
        arc:next(
            float(row["fwhm_A"]) for row in rows
            if int(row["pair"])==3 and row["arc_id"]==arc
            and int(row["half_window_px"])==16
            and int(row["continuum_degree"])==1
        ) for arc in ("arc06","arc13")
    }
    return rows, {
        "feature_summaries":summary,
        "red_pair_y506":{"original_lamp_fwhm_A":red},
    }


def test_qa_v20_original_v19_per_feature_summary_consistency():
    rows,summary=_sample()
    _load().validate_v19_csv_against_summary(rows,summary)


def test_qa_v20_must_reject_csv_with_altered_profile_width():
    rows,summary=_sample()
    rows[0]=dict(rows[0],fwhm_A="99.0")
    with pytest.raises(ValueError,match="median differs"):
        _load().validate_v19_csv_against_summary(rows,summary)


def test_qa_v20_must_reject_original_red_pair_mismatch():
    rows,summary=_sample()
    summary["red_pair_y506"]["original_lamp_fwhm_A"]["arc06"]+=.1
    with pytest.raises(ValueError,match="red pair FITS profile"):
        _load().validate_v19_csv_against_summary(rows,summary)


def test_qa_v20_canonical_map_sha_manifest_format():
    tree=ast.parse(SCRIPT.read_text(encoding="utf-8"))
    assignments=[
        ast.literal_eval(n.value)
        for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=="ORIGINAL_SHA"
                for t in n.targets)
    ]
    assert len(assignments)==1
    h=assignments[0]
    assert set(h)=={"arc06","arc13","lambda_map"}
    assert all(len(x)==64 for x in h.values())
    assert h["lambda_map"]==(
        "37808e73c00279ced67ea520e2245767427bd71dc7edec9af22177fbc8df07ab"
    )



def test_qa_v20_source_metadata_matches_original_v19_json_schema():
    v19={
        "status":"DIAGNOSTIC_ONLY",
        "science_lsf_validated":False,
        "auto_apply_allowed":False,
        "n_attempts":2640,
        "n_numerically_accepted":2425,
        "original_source_sha256":{
            "arc06":"54a6d638d2284b9bfd1d947df02399eaf0264f998667a830e144b1c437a61dac",
            "arc13":"9e2c98f2d2ba1125e18f84722854099f73e435a1fd7810f0b4df84ea07ab0c05",
            "wavelength_map":"37808e73c00279ced67ea520e2245767427bd71dc7edec9af22177fbc8df07ab",
        },
    }
    _load().validate_v19_source_metadata(v19)
    legacy=dict(v19)
    legacy["source_files_sha256"]=legacy.pop("original_source_sha256")
    with pytest.raises(RuntimeError,match="source provenance"):
        _load().validate_v19_source_metadata(legacy)
    forged=dict(v19,science_lsf_validated=True)
    with pytest.raises(RuntimeError,match="source provenance"):
        _load().validate_v19_source_metadata(forged)


def test_qa_v20_rejects_wrong_reviewed_lab_pair_even_with_same_row_count():
    rows,summary=_sample()
    summary["feature_summaries"][3]["pair_A"]=[5330.78,5341.61]
    with pytest.raises(ValueError,match="laboratory pair identity"):
        _load().validate_v19_csv_against_summary(rows,summary)


def test_qa_v20_rejects_changed_median_even_when_profile_counts_match():
    rows,summary=_sample()
    summary["feature_summaries"][1]["median_fwhm_by_hypothesis_A"]["degree1_window20"]+=0.015
    with pytest.raises(ValueError,match="median differs"):
        _load().validate_v19_csv_against_summary(rows,summary)
