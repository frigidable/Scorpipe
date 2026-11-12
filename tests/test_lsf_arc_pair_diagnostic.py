"""A close Ne pair can be fitted without treating blended wings as one line."""
import numpy as np
import pytest

from scorpio_pipe.lsf_arc_pair import fit_arc_pair_diagnostic


def _make_pair(shift=0.3, sig=2.8, b=550., seed=111):
    xx=np.arange(200,dtype=float)
    wave=5000. + 0.85*xx
    rng=np.random.default_rng(seed)
    sci=(
        900. + 0.1*(xx-70.)
        + 900.*np.exp(-0.5*((xx-65.-shift)/sig)**2)
        + b*np.exp(-0.5*((xx-77.-shift)/sig)**2)
        + rng.normal(0., 2., xx.size)
    )
    return wave,sci


def test_two_gaussians_recover_shared_lsf_with_independent_amplitudes():
    wave,sci=_make_pair()
    out=fit_arc_pair_diagnostic(wave,sci,(wave[65],wave[77]))
    assert out["status"]=="DIAGNOSTIC_ONLY",out
    assert abs(out["fwhm_A"] - 2.354820045*2.8*0.85)<0.08
    assert abs(out["common_shift_px"]-0.3)<0.04
    assert out["component_amplitudes_free"][0]>out["component_amplitudes_free"][1]
    assert out["science_lsf_validated"] is False
    assert out["safe_to_apply_to_sky_or_science"] is False
    assert out["wave_ref"]=="air"


def test_two_original_lamps_distinguish_shift_from_width():
    wave,a=_make_pair(shift=0.08,seed=88)
    _,b=_make_pair(shift=0.25,seed=99)
    first=fit_arc_pair_diagnostic(wave,a,(wave[65],wave[77]))
    second=fit_arc_pair_diagnostic(wave,b,(wave[65],wave[77]))
    assert first["status"]==second["status"]=="DIAGNOSTIC_ONLY"
    assert abs((second["common_shift_px"]-first["common_shift_px"])-0.17)<0.05
    assert abs(first["fwhm_A"]-second["fwhm_A"])<0.08


def test_weak_second_component_is_rejected_instead_of_being_ignored():
    wave,sci=_make_pair(b=0.)
    result=fit_arc_pair_diagnostic(
        wave,sci,(wave[65],wave[77]),min_peak_amplitude=60.,
    )
    assert result["status"]=="REJECT"


def test_close_line_centers_not_independent_enough_to_identify_two_widths():
    wave,sci=_make_pair()
    q=fit_arc_pair_diagnostic(wave,sci,(wave[65],wave[68]))
    assert q["status"]=="REJECT"
    assert q["reason"]=="pair_centers_unresolved_for_stable_fit"


def test_edge_and_invalid_wave_grid_are_fatal_or_rejected():
    wave,sci=_make_pair()
    q=fit_arc_pair_diagnostic(wave,sci,(wave[180],wave[193]))
    assert q["status"]=="REJECT"
    assert q["reason"]=="insufficient_detector_edge_support"
    with pytest.raises(ValueError,match="increasing air"):
        fit_arc_pair_diagnostic(wave[::-1],sci,(wave[65],wave[77]))
    with pytest.raises(ValueError,match="ordered"):
        fit_arc_pair_diagnostic(wave,sci,(wave[77],wave[65]))


def test_diagnostic_pair_does_not_automatically_change_lsf_model():
    from pathlib import Path
    from scorpio_pipe.stages import lsf_model

    code=Path(lsf_model.__file__).read_text(encoding="utf-8")
    assert "fit_arc_pair_diagnostic(" not in code
    assert "science_lsf_validated" in code
