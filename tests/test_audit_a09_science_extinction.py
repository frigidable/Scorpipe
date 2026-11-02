"""A09: independently correct standard and science atmospheric extinction."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.calib import fluxcal as fc
from scorpio_pipe.stages import fluxcal_apply as stage


def _spec_and_sens(tmp_path, *, x=1.4):
    wave = 5000. + np.arange(24, dtype=float)
    k = .17 + .002 * (wave - 5000.)
    response = np.full(len(wave), 4.)
    physical = 30. + .1 * (wave-5000.)
    # The standard response is already corrected to above atmosphere.
    counts = physical * response * 10.**(-.4*k*x)
    header = fits.Header()
    header["BUNIT"] = "e-/s"
    header["WAVEREF"] = "air"
    header["WAVEUNIT"] = "Angstrom"
    header["REBNORM"] = "COUNT_V1"

    def make_spec(name, signal, variance):
        return fits.BinTableHDU.from_columns([
            fits.Column(name="LAMBDA", format="D", array=wave),
            fits.Column(name=f"FLUX_{name}", format="D", array=signal),
            fits.Column(name=f"VAR_{name}", format="D", array=variance),
            fits.Column(name=f"MASK_{name}", format="J", array=np.zeros(len(wave), dtype=np.int32)),
            fits.Column(name=f"SKY_{name}", format="D", array=signal*.1),
            fits.Column(name=f"NPIX_{name}", format="I", array=np.full(len(wave),3,dtype=np.int16)),
        ], name="SPEC_"+name)

    src = tmp_path/"science.fits"
    fits.HDUList([
        fits.PrimaryHDU(header=header),
        make_spec("TRACE", counts, np.ones_like(wave)*4.),
        make_spec("FIXED", counts*1.2, np.ones_like(wave)*9.),
    ]).writeto(src)
    sh = fits.Header()
    sh["WAVEREF"] = "air"
    sh["INBUNIT"] = "e-/s"
    sh["INREBNM"] = "COUNT_V1"
    sh["INWUNIT"] = "Angstrom"
    sh["INSTEP"] = 1.
    sen = tmp_path/"sens.fits"
    fits.HDUList([
        fits.PrimaryHDU(header=sh),
        fits.BinTableHDU.from_columns([
            fits.Column(name="WAVELENGTH", format="D", array=wave),
            fits.Column(name="RESPONSE_ABS", format="D", array=response),
            fits.Column(name="RESPONSE_UNC", format="D", array=np.zeros_like(wave)),
        ],name="SENSFUNC")
    ]).writeto(sen)
    curve = {"airmass": x, "provenance": "header", "wavelength_A": np.array([4900.,5100.]),
             "k_lambda": np.array([-.03,.37])} # => k = .17 at 5000 A
    stdqc = {"safe_for":"absolute_candidate"}
    sensqc = {"extinction_applied":True, "airmass_provenance":"header"}
    return src,sen,wave,k,counts,physical,curve,stdqc,sensqc


def test_science_airmass_corrects_trace_fixed_variances_and_sky(tmp_path):
    src,sen,w,k,counts,physical,curve,stdqc,sensqc = _spec_and_sens(tmp_path)
    result = fc.apply_fluxcal_to_spec1d(
        src,sensfunc_path=sen,standard_qc=stdqc,sensfunc_qc=sensqc,science_extinction=curve,
    )
    np.testing.assert_allclose(result["flux"],physical,rtol=1e-12)
    np.testing.assert_allclose(result["fixed"]["flux"],physical*1.2,rtol=1e-12)
    expected_factor = 10.**(.4*k*1.4)
    np.testing.assert_allclose(result["science_extinction_factor"],expected_factor,rtol=1e-12)
    np.testing.assert_allclose(result["var"],4./16.*expected_factor**2,rtol=1e-12)
    np.testing.assert_allclose(result["fixed"]["var"],9./16.*expected_factor**2,rtol=1e-12)
    np.testing.assert_allclose(result["sky"],counts*.1/4.*expected_factor,rtol=1e-12)
    assert result["science_extinction_applied"] is True
    assert result["science_airmass"] == 1.4
    assert result["status"] == "ABSOLUTE_TRUSTED"


def test_missing_science_airmass_never_claims_absolute_trusted(tmp_path):
    src,sen,w,k,counts,physical,curve,stdqc,sensqc = _spec_and_sens(tmp_path)
    result=fc.apply_fluxcal_to_spec1d(src,sensfunc_path=sen,standard_qc=stdqc,sensfunc_qc=sensqc)
    np.testing.assert_allclose(result["flux"], counts/4.)
    assert result["status"] == "ABSOLUTE_APPROX"
    assert result["science_extinction_applied"] is False


def test_unverified_science_airmass_is_applied_but_not_absolute_trusted(tmp_path):
    src,sen,w,k,counts,physical,curve,stdqc,sensqc = _spec_and_sens(tmp_path)
    curve["provenance"] = "manual"
    result=fc.apply_fluxcal_to_spec1d(src,sensfunc_path=sen,standard_qc=stdqc,sensfunc_qc=sensqc,science_extinction=curve)
    np.testing.assert_allclose(result["flux"],physical,rtol=1e-12)
    assert result["status"]=="ABSOLUTE_APPROX"


@pytest.mark.parametrize("bad_x",[0.5,6.,float("nan")])
def test_invalid_science_airmass_fails_closed(tmp_path,bad_x):
    src,sen,w,k,counts,physical,curve,stdqc,sensqc = _spec_and_sens(tmp_path)
    curve["airmass"]=bad_x
    with pytest.raises(ValueError,match="Science airmass"):
        fc.apply_fluxcal_to_spec1d(src,sensfunc_path=sen,standard_qc=stdqc,sensfunc_qc=sensqc,science_extinction=curve)


def test_curve_without_full_wavelength_coverage_fails_closed(tmp_path):
    src,sen,w,k,counts,physical,curve,stdqc,sensqc = _spec_and_sens(tmp_path)
    curve["wavelength_A"] = np.array([5005.,5010.])
    curve["k_lambda"] = np.array([.18,.19])
    with pytest.raises(ValueError,match="does not cover"):
        fc.apply_fluxcal_to_spec1d(src,sensfunc_path=sen,standard_qc=stdqc,sensfunc_qc=sensqc,science_extinction=curve)


def test_science_correction_rejected_if_standard_was_not_deextincted(tmp_path):
    src,sen,w,k,counts,physical,curve,stdqc,sensqc = _spec_and_sens(tmp_path)
    sensqc["extinction_applied"] = False
    with pytest.raises(ValueError,match="without corrected standard"):
        fc.apply_fluxcal_to_spec1d(src,sensfunc_path=sen,standard_qc=stdqc,sensfunc_qc=sensqc,science_extinction=curve)


def test_single_exposure_raw_provenance_model_and_multi_frame_refusal(tmp_path,monkeypatch):
    raw=tmp_path/"obj_raw.fits"
    fits.PrimaryHDU(data=np.zeros((3,4),dtype=np.float32)).writeto(raw)
    ext=tmp_path/"atmos.csv"
    ext.write_text("wavelength_A,k_lambda\n4900,0.2\n5100,0.25\n",encoding="utf-8")
    import scorpio_pipe.instruments as instruments
    import scorpio_pipe.calib.common as common
    meta=SimpleNamespace(airmass=1.3,provenance_class=lambda key:"header")
    monkeypatch.setattr(instruments,"parse_frame_meta",lambda *_a,**_kw:meta)
    monkeypatch.setattr(common,"reference_path",lambda *_a,**_kw:ext)
    cfg={"frames":{"obj":[str(raw)]}}
    model=stage._single_science_airmass_model(cfg)
    assert model["airmass"]==1.3
    assert model["provenance"]=="header"
    assert len(model["wavelength_A"])==2
    cfg["frames"]["obj"].append(str(raw))
    assert stage._single_science_airmass_model(cfg) is None


def test_atmospheric_correction_consistent_with_written_2d_and_fits(tmp_path,monkeypatch):
    src,sen,w,k,counts,physical,curve,stdqc,sensqc = _spec_and_sens(tmp_path)
    result=fc.apply_fluxcal_to_spec1d(src,sensfunc_path=sen,standard_qc=stdqc,sensfunc_qc=sensqc,science_extinction=curve)
    from scorpio_pipe.io.mef import WaveGrid, write_sci_var_mask
    stack=tmp_path/"stack2d.fits"
    write_sci_var_mask(stack,np.tile(counts,(4,1)),var=np.full((4,len(w)),4.),mask=np.zeros((4,len(w)),dtype=np.uint16),
                       grid=WaveGrid(float(w[0]),1.,len(w),unit="Angstrom",wave_ref="air"),
                       header=fits.Header({"BUNIT":"e-/s"}))
    monkeypatch.setattr(fc,"PRODUCT_CATALOG",SimpleNamespace(best_path=lambda *_a,**_kw:stack))
    outputs=fc.write_fluxcal_products({"work_dir":str(tmp_path)},tmp_path/"out",spec_payload=result,
      sensfunc_path=sen,telluric_path=None,reference_context_id="test-a09",science_spec_path=src,standard_qc=stdqc)
    with fits.open(outputs["fluxcal_spec1d_fits"]) as h:
        assert h[0].header["ATMCORR"] is True
        assert h[0].header["ATMXSCI"]==pytest.approx(1.4)
        np.testing.assert_allclose(h["SPEC_TRACE"].data["FLUX_TRACE"],physical,rtol=1e-6)
    with fits.open(outputs["fluxcal_2d_fits"]) as h:
        assert h[0].header["ATMCORR"] is True
        np.testing.assert_allclose(h["SCI"].data,np.tile(physical,(4,1)),rtol=2e-6)
        np.testing.assert_allclose(h["VAR"].data,np.tile(4./16.*10.**(.8*k*1.4),(4,1)),rtol=2e-6)
