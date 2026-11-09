"""Regression: long-slit ARC LSF must not use a truncated ±6 px moment."""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.lsf_arc_profile import (
    fit_gaussian_arc_profile,
    isolated_arc_centers_in_pixels,
)


def test_arc_gaussian_recovers_true_fwhm_not_4_angstrom():
    x = np.arange(210,dtype=float)
    sigma = 5.8/0.87/2.354820045
    rng=np.random.default_rng(1234)
    profile=(
        950 + 0.4*(x-80)
        + 1800*np.exp(-0.5*((x-80.24)/sigma)**2)
        + rng.normal(0,2.5,x.size)
    )
    fit, meta = fit_gaussian_arc_profile(
        profile,center_guess_px=80.,half_window_px=12
    )
    assert meta["reason"]=="accepted",meta
    assert abs(fit*2.354820045*0.87-5.8) < 0.09


def test_tight_arc_window_rejected_instead_of_faking_fwhm():
    x=np.arange(210,dtype=float)
    sigma=5.8/0.87/2.354820045
    profile=950+1800*np.exp(-0.5*((x-80.)/sigma)**2)
    fit,meta=fit_gaussian_arc_profile(
        profile,center_guess_px=80.,half_window_px=6
    )
    assert fit is None
    assert meta["reason"]=="window_truncates_profile_wings"


def test_close_manual_arc_references_are_not_fitted_as_isolated():
    positions=np.asarray([20.,30.,90.,128.])
    assert isolated_arc_centers_in_pixels(
        positions,half_window_px=11
    ).tolist()==[False,False,True,True]


def test_noisy_flat_arc_not_accepted():
    rng=np.random.default_rng(15)
    profile=950+rng.normal(0,30,210)
    sig,meta=fit_gaussian_arc_profile(
        profile,center_guess_px=80.,half_window_px=11,min_snr=12
    )
    assert sig is None


def test_malformed_input_guards_and_empty_catalogue():
    with pytest.raises(ValueError):
        fit_gaussian_arc_profile(
            np.zeros((2,30)),center_guess_px=10.,half_window_px=11
        )
    with pytest.raises(ValueError):
        isolated_arc_centers_in_pixels(
            np.asarray([1.,np.nan]),half_window_px=11
        )
    assert isolated_arc_centers_in_pixels(
        np.array([]),half_window_px=11
    ).size==0


def test_measured_2d_arc_lsf_matches_synthetic_without_moment_bias():
    pytest.importorskip("astropy.io.fits")
    from scorpio_pipe.stages.lsf_model import _build_measured_lsf_map

    ny,nx=72,280
    x=np.arange(nx,dtype=float)[None,:]
    y=np.arange(ny,dtype=float)[:,None]
    lam_map=4700.+0.87*x+0.02*(y-36.)
    lines=np.asarray([4725.,4760.,4795.,4830.,4865.,4900.])
    rng=np.random.default_rng(33)
    arc=300.+rng.normal(0,1.6,(ny,nx))
    for line in lines:
        center=(line-4700.-0.02*(np.arange(ny)-36.))/0.87
        arc+=1100*np.exp(-0.5*((x-center[:,None])/2.8)**2)
    lsf,meta=_build_measured_lsf_map(
        arc,lam_map,lines,fit_half_window_px=12,
        min_snr=4.,poly_order_y=2,min_line_measures=8,
        expected_arc_fwhm_A=5.8,
    )
    assert lsf is not None
    assert meta["status"]=="measured"
    assert meta["fit_method"]=="gaussian_plus_linear_background_isolated"
    assert meta["n_lines_modeled"]>=5
    median_fwhm=2.354820045*float(np.nanmedian(lsf))
    assert abs(median_fwhm-(2.354820045*2.8*0.87))<0.25
    assert meta["row_step_px"]>=1


def test_lsf_stage_has_no_silent_science_approval():
    from pathlib import Path
    from scorpio_pipe.stages import lsf_model

    code=Path(lsf_model.__file__).read_text(encoding="utf-8")
    assert "gaussian_plus_linear_background_isolated" in code
    assert 'science_lsf_validated": False' in code
    assert "fit_half_window_px\", 12" in code
    assert "rejected_blends" in code



def test_lsf_stage_refuses_nominal_fallback_without_explicit_optin(tmp_path):
    fits=pytest.importorskip("astropy.io.fits")
    from scorpio_pipe.stages.lsf_model import run_lsf_model
    from scorpio_pipe.workspace_paths import stage_dir
    from pathlib import Path
    import json

    work_dir=tmp_path/"night"
    wave_dir=stage_dir(work_dir,"wavesol")
    wave_dir.mkdir(parents=True)
    ny,nx=40,180
    wave=np.broadcast_to(
        5000.0+0.87*np.arange(nx,dtype=float),
        (ny,nx),
    ).copy()
    # No actual ARC emission, so a nominal width would be fictional.
    arc=np.full((ny,nx),1000.0)
    fits.PrimaryHDU(
        wave.astype(np.float32),
        header=fits.Header({"WAVEREF":"air","WAVEUNIT":"Angstrom"}),
    ).writeto(wave_dir/"lambda_map.fits")
    fits.PrimaryHDU(
        arc.astype(np.float32),
        header=fits.Header({"DISPERSE":"VPHG1200B","SLITWID":1.2}),
    ).writeto(wave_dir/"superneon.fits")
    (wave_dir/"hand_pairs.txt").write_text(
        "20 5017.4\n60 5052.2\n100 5087.0\n140 5121.8\n",
        encoding="utf-8",
    )
    cfg={"work_dir":str(work_dir),"lsf_model":{"min_line_snr":4.0}}
    out_dir=work_dir/"lsf_model"
    with pytest.raises(RuntimeError, match="LSF ARC_PROFILE_QC REJECT"):
        run_lsf_model(cfg,out_dir=out_dir)
    rejected=out_dir/"lsf_profile_qc_reject.json"
    assert rejected.is_file()
    diag=json.loads(rejected.read_text(encoding="utf-8"))
    assert diag["status"]=="REJECT"
    assert diag["manual_fallback_allowed"] is False
    assert not (out_dir/"lsf_map.fits").exists()
