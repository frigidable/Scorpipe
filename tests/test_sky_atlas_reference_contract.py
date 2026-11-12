"""Independent sky-atlas probes must remain diagnostic and source-typed."""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.sky_atlas_reference import (
    SkyReference, fit_blended_sky_shift, group_blended_lines, evaluate_lsf_sensitivity,
    read_uves_airglow_csv,
)


def _lab_doublet():
    return [
        SkyReference(5197.902, "laboratory_air", "NI_lab"),
        SkyReference(5200.257, "laboratory_air", "NI_lab"),
    ]


def _synthetic_spectrum(shift=0.22, seed=3, third_amp=0.0):
    rng = np.random.default_rng(seed)
    wave = np.linspace(5188.0, 5211.0, 201)
    sigma = 3.1/2.354820045
    sky = (4.0 + 0.02*(wave-5199.0)
           + 45.0*np.exp(-0.5*((wave-5197.902-shift)/sigma)**2)
           + 30.0*np.exp(-0.5*((wave-5200.257-shift)/sigma)**2)
           + third_amp*np.exp(-0.5*((wave-5202.979-shift)/sigma)**2))
    sky += rng.normal(0.0, 0.7, wave.size)
    return wave, sky


def test_paranal_reference_is_measured_not_laboratory(tmp_path):
    path = tmp_path / "UVES_airglow.csv"
    path.write_text(
        "source_table,lambda_air_A,peak_INT,FWHM_A\n"
        "gident_580L,5200.285645,146.71915,0.12109\n"
        "gident_580L,5197.928223,252.81952,0.12188\n"
        "gident_580L,5202.979004,28.07875,0.12004\n",
        encoding="utf-8",
    )
    ref = read_uves_airglow_csv(path)
    assert len(ref) == 3
    assert ref[0].wavelength_air_A == pytest.approx(5197.928223)
    assert all(r.reference_kind == "uves_observed_centroid" for r in ref)
    assert all(r.source_id == "Hanuschik2003_UVES_airglow" for r in ref)
    assert len(group_blended_lines(ref,instrument_fwhm_A=3.5)) == 1
    assert len(group_blended_lines(ref,instrument_fwhm_A=1.0)) == 3
    wave,sky = _synthetic_spectrum()
    result = fit_blended_sky_shift(
        wave,sky,0.7,ref,wavelength_frame="air",instrument_fwhm_A=3.1
    )
    assert result["reference_is_lab_standard"] is False
    assert result["auto_apply_allowed"] is False
    assert result["science_wavelength_zero_point_approved"] is False


def test_synthetic_lab_doublet_recovers_shift_without_fixed_peak_ratios():
    wave,sky = _synthetic_spectrum()
    result = fit_blended_sky_shift(
        wave,sky,0.7,_lab_doublet(),
        wavelength_frame="air",instrument_fwhm_A=3.1,
    )
    assert result["status"] == "PROVISIONAL",result
    assert abs(result["shift_A"] - 0.22) < 0.07
    assert result["n_sky_lines"] == 2
    assert result["detection_delta_chi2"] > 100
    assert result["reference_is_lab_standard"] is True
    assert result["auto_apply_allowed"] is False
    a, b = result["peak_amplitudes_free"]
    assert a > b > 0


def test_two_exposures_have_distinct_diagnostic_shifts():
    wave, early = _synthetic_spectrum(shift=0.15,seed=8)
    _, late = _synthetic_spectrum(shift=0.38,seed=9)
    a = fit_blended_sky_shift(
        wave,early,0.7,_lab_doublet(),
        wavelength_frame="air",instrument_fwhm_A=3.1,
    )
    b = fit_blended_sky_shift(
        wave,late,0.7,_lab_doublet(),
        wavelength_frame="air",instrument_fwhm_A=3.1,
    )
    assert a["status"] == b["status"] == "PROVISIONAL"
    assert abs((b["shift_A"] - a["shift_A"]) - 0.23) < 0.09


def test_air_vacuum_mismatch_is_fatal():
    wave, sky = _synthetic_spectrum()
    with pytest.raises(ValueError, match="air Angstrom"):
        fit_blended_sky_shift(
            wave,sky,0.7,_lab_doublet(),
            wavelength_frame="vacuum",instrument_fwhm_A=3.1,
        )


def test_continuum_only_is_not_a_diagnostic_wavelength_detection():
    wave = np.linspace(5188.0, 5211.0, 201)
    y = 4.0 + 0.03*(wave-5199.0)
    result = fit_blended_sky_shift(
        wave,y,0.7,_lab_doublet(),
        wavelength_frame="air",instrument_fwhm_A=3.1,
    )
    assert result["status"] == "UNRELIABLE"
    assert result["detection_delta_chi2"] < 1e-5
    assert not result["auto_apply_allowed"]


def test_reject_malformed_catalogue_and_invalid_noise(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text(
        "source_table,lambda_air_A,peak_INT,FWHM_A\n"
        "gident_580L,5200.2,20.0,-1.0\n",encoding="utf-8",
    )
    with pytest.raises(ValueError,match="Nonphysical"):
        read_uves_airglow_csv(p)
    wave,sky = _synthetic_spectrum()
    with pytest.raises(ValueError,match="Too few"):
        fit_blended_sky_shift(
            wave,sky,np.full(wave.size,-1.0),_lab_doublet(),
            wavelength_frame="air",instrument_fwhm_A=3.1,
        )


def test_mixed_observed_and_laboratory_reference_is_rejected():
    wave, sky = _synthetic_spectrum()
    mixed = _lab_doublet() + [
        SkyReference(5202.979, "uves_observed_centroid", "UVES_2003")
    ]
    with pytest.raises(ValueError, match="Cannot mix laboratory"):
        fit_blended_sky_shift(
            wave, sky, 0.7, mixed,
            wavelength_frame="air", instrument_fwhm_A=3.1,
        )


def test_dense_unresolved_catalogue_is_not_used_as_nine_independent_anchors():
    wave,sky = _synthetic_spectrum()
    dense = [
        SkyReference(5195.0 + i*0.35, "uves_observed_centroid", "UVES_2003")
        for i in range(9)
    ]
    with pytest.raises(ValueError,match="eight components"):
        fit_blended_sky_shift(
            wave,sky,0.7,dense,
            wavelength_frame="air",instrument_fwhm_A=3.1,
        )



def test_unresolved_ni_doublet_is_not_absolute_wavelength_certification():
    w, f = _synthetic_spectrum()
    result = fit_blended_sky_shift(
        w, f, 0.7, _lab_doublet(),
        wavelength_frame="air", instrument_fwhm_A=5.6,
    )
    assert result["unresolved_reference_blend"] is True
    assert result["nearest_reference_separation_A"] == pytest.approx(2.355)
    assert result["absolute_shift_deblend_identifiability"] == (
        "UNRESOLVED_COMPONENT_AMPLITUDE_DEGENERACY"
    )
    assert result["science_wavelength_zero_point_approved"] is False
    assert result["auto_apply_allowed"] is False


def test_lsf_sensitivity_runs_three_models_without_approving_correction():
    w, f = _synthetic_spectrum(shift=0.24)
    summary = evaluate_lsf_sensitivity(
        w, f, 0.7, _lab_doublet(), wavelength_frame="air",
        instrumental_fwhm_trials_A=[2.8, 3.1, 3.4],
        max_shift_spread_A=0.2,
    )
    assert summary["n_fwhm_trials"] == 3
    assert len(summary["shift_trial_values_A"]) == 3
    assert np.isfinite(summary["shift_spread_A"])
    assert summary["status"] in {"UNSTABLE", "PROVISIONAL"}
    assert summary["science_wavelength_zero_point_approved"] is False
    assert summary["auto_apply_allowed"] is False


def test_lsf_uncertainty_budget_cannot_be_skipped_with_one_width():
    w, f = _synthetic_spectrum()
    with pytest.raises(ValueError, match=">=3 distinct"):
        evaluate_lsf_sensitivity(
            w, f, 0.7, _lab_doublet(), wavelength_frame="air",
            instrumental_fwhm_trials_A=[3.1],
        )
    with pytest.raises(ValueError, match=">=3 distinct"):
        evaluate_lsf_sensitivity(
            w, f, 0.7, _lab_doublet(), wavelength_frame="air",
            instrumental_fwhm_trials_A=[3.1, 3.1, 3.1],
        )
