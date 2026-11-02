"""A10: integrated Halpha+[NII] and [SII] windows are diagnostic blends."""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.science.lines import DEFAULT_LINES, measure_lines
from scorpio_pipe.science.reddening import balmer_decrement_summary


def _data():
    wave=np.arange(4800.,6810.,.25)
    flux=np.ones_like(wave)
    var=np.full_like(wave,.0025)
    def gauss(center,amplitude,sigma=1.5):
        return amplitude*np.exp(-.5*((wave-center)/sigma)**2)
    flux+=gauss(4861.33,20.)
    flux+=gauss(6562.80,30.)
    flux+=gauss(6548.05,9.)
    flux+=gauss(6583.45,27.)
    flux+=gauss(5006.84,18.)
    flux+=gauss(6716.44,7.)
    flux+=gauss(6730.82,8.)
    return wave,flux,var


def _measure(select=None):
    wave,flux,var=_data()
    return measure_lines(wave,flux,var,
         calsys_rel=None,calsys_tell=None,calsys_abs=None,
         lines=DEFAULT_LINES if select is None else select)


def test_halpha_and_nii_are_flagged_not_component_fluxes():
    results={r["name"]:r for r in _measure()}
    ha=results["Halpha"]
    assert ha["detected"] is True
    assert ha["blend_flag"] is True
    assert "[NII]6548" in ha["blend_partners"]
    assert "[NII]6583" in ha["blend_partners"]
    assert ha["flux_interpretation"]=="BLENDED_DIAGNOSTIC_ONLY"
    assert ha["blend_safe_for_physical_ratios"] is False
    assert results["[NII]6548"]["blend_safe_for_physical_ratios"] is False


def test_sii_doublet_is_not_claimed_as_two_independently_measured_fluxes():
    results={r["name"]:r for r in _measure()}
    for name, partner in (("[SII]6716","[SII]6731"),
                          ("[SII]6731","[SII]6716")):
        row=results[name]
        assert partner in row["blend_partners"]
        assert row["blend_safe_for_physical_ratios"] is False


def test_isolated_oiii_can_be_reported_as_diagnostic_flux():
    results={r["name"]:r for r in _measure()}
    row=results["[OIII]5007"]
    assert row["detected"]
    assert row["blend_flag"] is False
    assert row["blend_partners"]==[]
    assert row["blend_safe_for_physical_ratios"] is True


def test_halpha_alone_still_considers_unrequested_nii_catalog():
    only=[s for s in DEFAULT_LINES if s.name=="Halpha"]
    result=_measure(select=only)[0]
    assert result["name"]=="Halpha"
    assert set(result["blend_partners"]) >= {"[NII]6548","[NII]6583"}
    assert result["blend_safe_for_physical_ratios"] is False


def test_balmer_decrement_rejected_for_blended_halpha():
    result={r["name"]:r for r in _measure()}
    assert result["Halpha"]["detected"] is True
    assert result["Hbeta"]["detected"] is True
    assert balmer_decrement_summary(result) is None


def test_independently_fitted_fluxes_can_be_used_for_balmer_decrement():
    # Legacy external fitted-line results do not carry a negative eligibility
    # flag; only the new window-integral measurement explicitly vetoes blends.
    fitted={"Halpha":{"flux":8.58,"detected":True,"blend_safe_for_physical_ratios":True},
            "Hbeta":{"flux":3.,"detected":True,"blend_safe_for_physical_ratios":True}}
    summary=balmer_decrement_summary(fitted)
    assert summary is not None
    assert summary["line_ratio_Ha_Hb"] == pytest.approx(2.86)


def test_blend_status_independent_of_requested_air_vacuum_medium():
    select=[s for s in DEFAULT_LINES if s.name=="Halpha"]
    w,f,v=_data()
    result=measure_lines(w,f,v, calsys_rel=None,calsys_tell=None,calsys_abs=None,
                         lines=select,wave_ref="vacuum")[0]
    assert "[NII]6548" in result["blend_partners"]
    assert result["blend_safe_for_physical_ratios"] is False
