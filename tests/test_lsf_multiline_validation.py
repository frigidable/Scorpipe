"""Two-ARC feature-blocked candidate validation is diagnostics only."""
import numpy as np
import pytest
from scorpio_pipe.lsf_multiline_validation import (
    validate_lsf_multiline, predict_lsf_candidate,
)

HASHES = {"ARC_A": "a"*64, "ARC_B": "b"*64}
KW = {
    "arc_ids": ("ARC_A","ARC_B"),
    "source_arc_sha256": HASHES,
    "wavelength_map_sha256": "c"*64,
    "manual_lineid_sha256": "d"*64,
}


def synthetic_records(*, anomalous_feature=None, mismatched_arc=False):
    data=[]
    for j,lam in enumerate(
        [3900,4050,4200,4370,4500,4800,5000,5080,5336.19]
    ):
        for arc in ("ARC_A","ARC_B"):
            for y in np.arange(50,1010,32):
                value = 5.8+0.00028*(lam-4600)+0.35*((y-516)/516)**2
                if arc=="ARC_B":
                    value+=0.025
                if j==anomalous_feature:
                    value+=2.5
                if j==6 and arc=="ARC_B" and mismatched_arc:
                    value+=0.8
                data.append({
                    "feature_id":f"feature-{j}",
                    "arc_id":arc,
                    "fit_kind":"pair" if j==8 else "single",
                    "lab_lines_A":(5330.78,5341.60) if j==8 else (float(lam),),
                    "wavelength_A":float(lam),
                    "y_pix":float(y),
                    "fwhm_A":float(value),
                })
    return data


def test_clean_two_arc_lolo_and_masked_2d_candidate():
    qc=validate_lsf_multiline(synthetic_records(),**KW)
    assert qc["status"]=="DIAGNOSTIC_CANDIDATE"
    assert qc["n_independent_features"]==9
    assert qc["n_cv_accepted_features"]==9
    assert qc["wavelength_support_A"]==[3900.0,5336.19]
    assert np.isfinite(qc["feature_blocked_lolo_rms_of_median_abs_A"])
    assert qc["science_lsf_validated"] is False
    assert qc["auto_replace_production_lsf"] is False
    wave=np.broadcast_to(np.linspace(3800,5400,200),(25,200))
    y=np.broadcast_to(np.linspace(20,1020,25)[:,None],wave.shape)
    model,mask=predict_lsf_candidate(wave,y,qc)
    assert np.isnan(model[0]).all()
    assert mask[10].any()
    assert not mask[10,0] and not mask[10,-1]
    assert np.isnan(model[10,0])


def test_anomalous_line_excluded_as_a_whole_lab_feature():
    qc=validate_lsf_multiline(
        synthetic_records(anomalous_feature=4),**KW
    )
    assert qc["status"]=="DIAGNOSTIC_CANDIDATE"
    assert "feature-4" not in qc["accepted_feature_ids"]
    f=next(v for v in qc["features"] if v["feature_id"]=="feature-4")
    assert f["two_arc_consistent"] is True
    assert f["heldout_model_consistent"] is False
    assert qc["n_cv_accepted_features"]>=7


def test_two_lamps_are_not_independent_when_arc_shifted():
    qc=validate_lsf_multiline(
        synthetic_records(mismatched_arc=True),**KW
    )
    assert qc["status"]=="DIAGNOSTIC_CANDIDATE"
    assert "feature-6" not in qc["accepted_feature_ids"]
    bad=next(v for v in qc["features"] if v["feature_id"]=="feature-6")
    assert bad["two_arc_consistent"] is False


def test_original_arc_source_sha_must_be_distinct_and_valid():
    with pytest.raises(ValueError,match="hashes"):
        validate_lsf_multiline(
            synthetic_records(),**dict(
                KW,source_arc_sha256={"ARC_A":"a"*64,"ARC_B":"a"*64}
            )
        )
    with pytest.raises(ValueError,match="digests"):
        validate_lsf_multiline(
            synthetic_records(),**dict(
                KW,wavelength_map_sha256="altered"
            )
        )


def test_no_physical_lab_line_can_be_double_counted_across_features():
    rows=synthetic_records()
    alias=dict(rows[0],feature_id="fake-second-reference",y_pix=901.)
    with pytest.raises(ValueError,match="double-counted"):
        validate_lsf_multiline(rows+[alias],**KW)


def test_pair_identity_and_wavelength_centroid_must_match():
    rows=synthetic_records()
    rows[0]=dict(
        rows[0],fit_kind="pair",lab_lines_A=(3900.,3910.)
    )
    with pytest.raises(ValueError,match="Invalid ARC line"):
        validate_lsf_multiline(rows,**KW)


def test_no_duplicate_spatial_measurements_or_fake_lab_ids():
    rows=synthetic_records()
    with pytest.raises(ValueError,match="Duplicate"):
        validate_lsf_multiline(rows+[dict(rows[0])],**KW)
    rows=synthetic_records()
    rows[0]=dict(rows[0],lab_lines_A=(np.nan,))
    with pytest.raises(ValueError,match="Invalid ARC line"):
        validate_lsf_multiline(rows,**KW)


def test_fewer_than_six_repeatable_features_rejected():
    records=synthetic_records()
    qc=validate_lsf_multiline(
        [r for r in records if r["feature_id"] in {
            "feature-0","feature-1","feature-2","feature-3"
        }],**KW
    )
    assert qc["status"]=="REJECT"
    assert qc["reason"]=="insufficient_repeatable_features"
    assert qc["auto_replace_production_lsf"] is False
    with pytest.raises(ValueError,match="Candidate requires"):
        predict_lsf_candidate(np.ones((2,3)),np.ones((2,3)),qc)


def test_all_failed_lines_do_not_become_a_candidate():
    records=synthetic_records()
    # Remove all paired measurements from ARC_B.
    qc=validate_lsf_multiline(
        [r for r in records if r["arc_id"]=="ARC_A"],**KW
    )
    assert qc["status"]=="REJECT"
    assert qc["n_repeatable_features"] if "n_repeatable_features" in qc else True
    assert qc["science_lsf_validated"] is False
