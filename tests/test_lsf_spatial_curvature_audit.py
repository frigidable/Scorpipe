"""Do not let original-ARC repeatability silently certify science LSF."""
import numpy as np
import pytest

from scorpio_pipe.lsf_spatial_curvature_audit import audit_neon_spatial_curvature


REPORT = {
    "status":"DIAGNOSTIC_CANDIDATE",
    "science_lsf_validated":False,
    "auto_replace_production_lsf":False,
    "coefficients":[5.6369,0.2083,0.0355,1.3753,-0.1997],
}


def _records(*,n=33,wrong_arc=False):
    records=[]
    for pair in range(4):
        for arc in ("arc06","arc13"):
            for y in np.linspace(74,938,n):
                v=(y-515.5)/515.5
                quadratic=[2.05,2.29,0.20,-0.40][pair]
                fwhm=6.0+[0.0,0.4,-0.2,0.1][pair]+quadratic*v*v
                fwhm+=0.015 if arc=="arc13" else 0.0
                records.append({
                    "pair":pair, "arc_id":"missing_arc" if wrong_arc else arc,
                    "y_pix":float(y), "status":"DIAGNOSTIC_FIT",
                    "half_window_px":16, "continuum_degree":1,
                    "fwhm_A":float(fwhm),
                })
    return records


def test_cross_arc_spatial_curvature_catches_red_sign_reversal():
    qc=audit_neon_spatial_curvature(_records(),REPORT)
    assert qc["status"]=="DIAGNOSTIC_ONLY"
    assert qc["n_pairs_with_sufficient_repeatable_y"]==4
    assert qc["science_lsf_validated"] is False
    assert qc["safe_for_stellar_kinematics"] is False
    blue=qc["pair_diagnostics"][1]
    red=qc["pair_diagnostics"][3]
    assert blue["curvature_sign_matches_v18_for_both_arcs"] is True
    assert red["curvature_sign_matches_v18_for_both_arcs"] is False
    assert red["fwhm_quadratic_coeff_v2_by_arc"]["arc06"]==pytest.approx(-0.4,abs=.01)
    assert red["fwhm_quadratic_coeff_v2_by_arc"]["arc13"]==pytest.approx(-0.4,abs=.01)
    assert red["arc_difference_median_abs_A"]==pytest.approx(.015,abs=.001)
    assert red["center_zone_v18_residual_median_A"] is not None
    assert red["outer_zone_v18_residual_median_A"] is not None


def test_two_arcs_required_no_fake_replication():
    with pytest.raises(ValueError,match="distinct original-ARC"):
        audit_neon_spatial_curvature(_records(),REPORT,arc_ids=("arc06","arc06"))
    with pytest.raises(ValueError,match="Unrecognized original ARC"):
        audit_neon_spatial_curvature(_records(wrong_arc=True),REPORT)


def test_duplicate_y_lamp_pair_sample_fails():
    rows=_records()
    with pytest.raises(ValueError,match="Duplicate ARC"):
        audit_neon_spatial_curvature(rows+[dict(rows[0])],REPORT)


def test_too_few_common_rows_rejected_for_each_line():
    qc=audit_neon_spatial_curvature(_records(n=7),REPORT)
    assert qc["n_pairs_with_sufficient_repeatable_y"]==0
    assert all(row["status"]=="REJECT" for row in qc["pair_diagnostics"])


def test_diagnostic_only_v18_report_required():
    with pytest.raises(ValueError,match="diagnostic-only"):
        audit_neon_spatial_curvature(
            _records(),dict(REPORT,science_lsf_validated=True)
        )
    with pytest.raises(ValueError,match="five finite"):
        audit_neon_spatial_curvature(
            _records(),dict(REPORT,coefficients=[0.3])
        )


def test_nonfinite_width_or_unexpected_pair_fails():
    rows=_records()
    rows[0]=dict(rows[0],fwhm_A=np.nan)
    with pytest.raises(ValueError,match="Nonfinite"):
        audit_neon_spatial_curvature(rows,REPORT)
    rows=_records()
    rows[0]=dict(rows[0],pair=10)
    with pytest.raises(ValueError,match="Unrecognized original ARC"):
        audit_neon_spatial_curvature(rows,REPORT)


def test_no_production_lsf_stage_auto_adoption():
    from pathlib import Path
    from scorpio_pipe.stages import lsf_model
    source=Path(lsf_model.__file__).read_text(encoding="utf-8")
    assert "lsf_spatial_curvature_audit" not in source
    assert "audit_neon_spatial_curvature(" not in source
    assert "science_lsf_validated" in source
