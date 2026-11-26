"""Diagnostic only: independent original-lamp profile hypothesis tests."""
import numpy as np
import pytest
from scorpio_pipe.lsf_profile_systematics import (
    fit_neon_pair_hypothesis, evaluate_neon_pair_systematics,
)


def _pair(*, sigma=2.7, qcurve=0.0, amp_b=500.0, seed=17):
    rng=np.random.default_rng(seed)
    x=np.arange(250.,dtype=float)
    wave=5000.0+0.86*x
    obs=(
        1100+0.08*(x-130)+qcurve*(x-130)**2
        +900*np.exp(-0.5*((x-110.2)/sigma)**2)
        +amp_b*np.exp(-0.5*((x-123.2)/sigma)**2)
        +rng.normal(0,2.,x.size)
    )
    return wave,obs,(wave[110],wave[123])


def test_recovers_known_pair_width_never_approves_science():
    w,f,lab=_pair()
    q=fit_neon_pair_hypothesis(w,f,lab,half_window_px=16)
    assert q["status"]=="DIAGNOSTIC_FIT",q
    assert abs(q["fwhm_A"]-2.354820045*2.7*0.86)<0.1
    assert q["science_lsf_validated"] is False
    assert q["auto_apply_allowed"] is False


def test_comparison_requires_three_windows_and_both_continua():
    w,f,lab=_pair(qcurve=.03)
    q=evaluate_neon_pair_systematics(
        w,f,lab,half_windows_px=(12,16,22)
    )
    assert q["status"]=="DIAGNOSTIC_ONLY",q
    assert q["n_trials"]==6
    assert q["n_accepted"]>=4
    assert len(q["continuum_model_differences"])>=2
    assert q["uncertainty_status"]=="MODEL_SENSITIVITY_ONLY_NOT_STATISTICAL_CI"
    assert q["auto_apply_allowed"] is False


def test_missing_component_rejected_not_one_line_fake_width():
    w,f,lab=_pair(amp_b=0.)
    q=fit_neon_pair_hypothesis(w,f,lab,min_component_peak=90.)
    assert q["status"]=="REJECT"


def test_detector_edge_rejected():
    w,f,lab=_pair()
    q=fit_neon_pair_hypothesis(w,f,(w[228],w[244]))
    assert q["status"]=="REJECT"
    assert q["reason"]=="detector_edge_truncation"


def test_wavelength_grid_and_lab_pair_must_be_ordered():
    w,f,lab=_pair()
    with pytest.raises(ValueError,match="increasing wavelength"):
        fit_neon_pair_hypothesis(w[::-1],f,lab)
    with pytest.raises(ValueError,match="ordered"):
        fit_neon_pair_hypothesis(w,f,(lab[1],lab[0]))


def test_unsafe_hypothesis_grid_cannot_run():
    w,f,lab=_pair()
    with pytest.raises(ValueError,match="At least three"):
        evaluate_neon_pair_systematics(w,f,lab,half_windows_px=(11,13))
    with pytest.raises(ValueError,match="Both linear"):
        evaluate_neon_pair_systematics(w,f,lab,continuum_degrees=(1,))


def test_production_pipeline_never_imports_or_applies_diagnostic_profile():
    from pathlib import Path
    from scorpio_pipe.stages import lsf_model
    text=Path(lsf_model.__file__).read_text(encoding="utf-8")
    assert "lsf_profile_systematics" not in text
    assert "fit_neon_pair_hypothesis(" not in text
    assert "science_lsf_validated" in text
