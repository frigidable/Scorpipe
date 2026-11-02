"""A11b: detector arc sigma must be in Å even when lambda_map is in nm."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.stages.lsf_model import (
    _lambda_map_as_angstrom, run_lsf_model,
)
from scorpio_pipe.wavesol_paths import wavesol_dir


@pytest.mark.parametrize("native,scale", [("Angstrom", 1.), ("nm", 10.)])
def test_wavelength_map_native_unit_conversion(native, scale):
    hdr=fits.Header()
    hdr["WAVEUNIT"]=native
    arr=np.array([[501.2,502.3]]) if native=="nm" else np.array([[5012.,5023.]])
    converted, unit, factor=_lambda_map_as_angstrom(arr,hdr)
    assert unit==native
    assert factor==scale
    np.testing.assert_allclose(converted,[[5012.,5023.]])


@pytest.mark.parametrize("unit", ["", "meters", "micron", "Hz"])
def test_unknown_lsf_wavelength_unit_fails_closed(unit):
    hdr=fits.Header()
    if unit:
        hdr["WAVEUNIT"]=unit
    with pytest.raises(ValueError,match="explicit supported"):
        _lambda_map_as_angstrom(np.full((2,3),5000.),hdr)


def _run(tmp_path, *, unit):
    work=tmp_path / unit
    cfg={"work_dir":str(work),
         "lsf_model":{"fit_half_window_px":7,"min_line_snr":3.,
                      "min_line_measures":6}}
    wdir=wavesol_dir(cfg,run_root=work)
    wdir.mkdir(parents=True,exist_ok=True)
    ny,nx=16,155
    lam_A=5000.+.8*np.arange(nx)[None,:]+.01*np.arange(ny)[:,None]
    raw=lam_A/10. if unit=="nm" else lam_A
    hdr=fits.Header()
    hdr["WAVEUNIT"]=unit
    hdr["WAVEREF"]="air"
    fits.PrimaryHDU(data=np.asarray(raw,dtype=np.float32),header=hdr).writeto(wdir/"lambda_map.fits")
    rng=np.random.default_rng(841)
    arc=rng.normal(0.,.12,(ny,nx))
    lines=[5030.,5060.,5090.]
    for iy in range(ny):
        for lam in lines:
            x=(lam-(5000.+.01*iy))/.8
            arc[iy,:]+=125.*np.exp(-.5*((np.arange(nx)-x)/1.5)**2)
    ah=fits.Header()
    ah["DISPERSE"]="VPHG1200@540"
    ah["SLITWID"]=1.
    fits.PrimaryHDU(data=arc.astype(np.float32),header=ah).writeto(wdir/"superneon.fits")
    (wdir/"hand_pairs.txt").write_text("".join(f"{(l-5000.)/.8:.2f} {l:.3f}\n" for l in lines))
    output=run_lsf_model(cfg)
    with fits.open(output["lsf_map"]) as h:
        sigma=h["SCI"].data.copy()
        ph=h[0].header.copy()
    model=json.loads((work/"lsf_model"/"lsf_model.json").read_text())
    qc=json.loads((work/"lsf_model"/"lsf_qc.json").read_text())
    return sigma,ph,model,qc


def test_nm_and_angstrom_detector_lsf_are_physically_identical(tmp_path):
    sig_a,h_a,model_a,qc_a=_run(tmp_path,unit="Angstrom")
    sig_nm,h_nm,model_nm,qc_nm=_run(tmp_path,unit="nm")
    assert model_a["source"]==model_nm["source"]=="arc_measured"
    assert model_nm["instrument"]["wave_unit"]=="Angstrom"
    assert model_nm["instrument"]["native_wave_unit"]=="nm"
    assert model_nm["instrument"]["native_to_A_factor"]==10.
    assert h_nm["WAVEUNIT"]=="Angstrom"
    assert h_nm["CUNIT1"]=="Angstrom"
    assert h_nm["NATWUNIT"]=="nm"
    assert h_nm["LSFUNIT"]=="sigma_A"
    assert h_nm["LSFQUAL"] is False
    assert h_nm["LSFGEOM"]=="DETECTOR"
    assert qc_nm["science_1d_lsf_qualified"] is False
    np.testing.assert_allclose(sig_nm,sig_a,rtol=.02,atol=.025)
    assert .4 < float(np.nanmedian(sig_nm)) < 2.
    np.testing.assert_allclose(model_nm["summary_1d"]["lambda_A"],
                               model_a["summary_1d"]["lambda_A"],rtol=1e-5,atol=.015)
    np.testing.assert_allclose(model_nm["summary_1d"]["sigma_A"],
                               model_a["summary_1d"]["sigma_A"],rtol=.02,atol=.025)


def test_lsfgate_rejects_invalid_wavelength_map(tmp_path):
    with pytest.raises(ValueError,match="positive physical support"):
        _lambda_map_as_angstrom(np.full((2,4),-1.),fits.Header({"WAVEUNIT":"nm"}))
