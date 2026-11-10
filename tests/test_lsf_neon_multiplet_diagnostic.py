"""No science LSF approval from correlated original-ARC Ne multiplet fits."""
import numpy as np
import pytest
from scorpio_pipe.lsf_neon_multiplet_diagnostic import (
    fit_original_arc_multiplet, trace_registered_median,
    red_lineid_review, residual_agreement
)


def _three_neon():
    wave=5280+0.75*np.arange(160,dtype=float)
    x=np.arange(160,dtype=float)
    centers=np.interp([5330.7775,5341.0938,5343.2834],wave,x)
    signal=800+0.2*(x-90)
    for amp,c in zip([800,600,300],centers):
        signal=signal+amp*np.exp(-.5*((x-c-0.15)/2.9)**2)
    return wave,signal


def test_lineid_catalog_discrepancy_not_silent_wavesolution_rewrite():
    q=red_lineid_review()
    assert q["status"]=="REVIEW_REQUIRED"
    assert q["wave2d_calibration_approved"] is False
    assert q["closest_nist_nei_to_historic_5341_60_A"]==5341.0938
    assert q["difference_historic_minus_nearest_lab_A"]==pytest.approx(.5062)
    assert q["another_nei_component_in_5341_blend_A"]==5343.2834


def test_true_three_component_synthetic_beats_legacy_two_at_same_pixels():
    wave,signal=_three_neon()
    original=fit_original_arc_multiplet(wave,signal,(5330.78,5341.60))
    triple=fit_original_arc_multiplet(
        wave,signal,(5330.7775,5341.0938,5343.2834)
    )
    assert original["status"]==triple["status"]=="DIAGNOSTIC_ONLY"
    assert original["n_fit_pixels"]==triple["n_fit_pixels"]
    assert triple["normalized_rmse"] < original["normalized_rmse"]*.1
    assert triple["fwhm_gaussian_equivalent_A"]==pytest.approx(
        2.354820045*2.9*.75,abs=.05
    )
    assert triple["auto_replace_production_lsf"] is False


def test_wrong_line_order_or_wave_map_fails_closed():
    wave,signal=_three_neon()
    with pytest.raises(ValueError,match="ordered"):
        fit_original_arc_multiplet(wave,signal,(5341.1,5330.8))
    with pytest.raises(ValueError,match="increasing"):
        fit_original_arc_multiplet(wave[::-1],signal,(5330.8,5341.6))
    with pytest.raises(ValueError,match="span every line"):
        fit_original_arc_multiplet(
            wave,signal,(5330.8,5341.6),fit_range_air_A=(5332,5352)
        )


def test_spatial_trace_resampling_does_not_retain_line_tilt():
    ny,nx=23,140
    yy=np.arange(ny)
    xx=np.arange(nx)
    wave=5300+0.8*xx[None,:]+0.2*(yy[:,None]-11)
    arc=50+1000*np.exp(-.5*((wave-5360.)/1.8)**2)
    aligned=trace_registered_median(arc,wave,11,half_strip=10)
    unaligned=np.median(arc[1:22],axis=0)
    assert aligned.shape==(nx,)
    assert np.nanmax(aligned)>np.max(unaligned)
    assert np.isfinite(aligned[15:110]).all()


def test_trace_registration_rejects_bad_wavelength_map():
    arc=np.ones((20,80))
    wave=5300+.8*np.arange(80)[None,:]+np.zeros((20,1))
    wave[10,40]=1
    with pytest.raises(ValueError,match="strictly increase"):
        trace_registered_median(arc,wave,10,half_strip=2)


def test_two_original_lamp_residual_correlations_do_not_license_science():
    wave,signal=_three_neon()
    first=fit_original_arc_multiplet(wave,signal,(5330.78,5341.60))
    second=fit_original_arc_multiplet(
        wave,signal*1.05+20,(5330.78,5341.60)
    )
    q=residual_agreement(first,second)
    assert q["status"]=="DIAGNOSTIC_ONLY"
    assert q["pearson_correlation"]>.999
    assert q["wave2d_calibration_approved"] is False


def test_production_stages_do_not_use_unreviewed_nist_multiplet():
    from pathlib import Path
    from scorpio_pipe.stages import lsf_model, wavesolution
    for stage in (lsf_model,wavesolution):
        source=Path(stage.__file__).read_text(encoding="utf-8")
        assert "lsf_neon_multiplet_diagnostic" not in source
    assert red_lineid_review()["auto_replace_production_lsf"] is False



def test_trace_alignment_keeps_nan_edges_without_silent_extrapolation():
    wave,signal=_three_neon()
    signal[:7]=np.nan
    signal[-7:]=np.nan
    fit=fit_original_arc_multiplet(
        wave,signal,(5330.7775,5341.0938,5343.2834)
    )
    assert fit["status"]=="DIAGNOSTIC_ONLY"
    signal[74]=np.nan
    invalid=fit_original_arc_multiplet(
        wave,signal,(5330.7775,5341.0938,5343.2834)
    )
    assert invalid["status"]=="REJECT"
    assert invalid["reason"]=="nonfinite_original_fit_window"
