"""Actual synthetic SkySub source FITS variance/component group admission.

These tests use the real SkySub entrypoint with a synthetic CCD and explicit
FlatField-style component planes; no observational science data are touched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.calib.sky_masterbias_source_audit import (
    _component_source, audit_sky_masterbias_group_sources,
)
from scorpio_pipe.stages.sky_sub import run_sky_sub
from scorpio_pipe.work_layout import ensure_work_layout
from scorpio_pipe.workspace_paths import stage_dir


def _night(work: Path, *, gid="NIGHT_BIAS_GROUP", bvar=4.0, components=True):
    layout = ensure_work_layout(work)
    layout.raw.mkdir(parents=True, exist_ok=True)
    ny, nx = 48, 128
    y = np.arange(ny)[:, None]
    x = np.arange(nx)[None, :]
    wavesol = stage_dir(work, "wavesol")
    wavesol.mkdir(parents=True, exist_ok=True)
    lam = np.repeat(5000. + 2.*x, ny, axis=0)
    fits.writeto(wavesol/"lambda_map.fits", lam.astype("f4"), overwrite=True)
    rng = np.random.default_rng(42)
    skyline = sum(65.*np.exp(-.5*((x-c)/1.3)**2) for c in (30., 60., 92.))
    obj = 35.*np.exp(-.5*((y-24.)/2.0)**2)*(10.+.03*x)
    sci = (20. + skyline + obj + rng.normal(0., 2., (ny, nx))).astype("f4")
    source = layout.raw / "obj_0001.fits"
    if components:
        header = fits.Header()
        header["SCORPUM"] = "ELECTRON"
        header["BUNIT"] = "e-"
        header["BIASSUB"] = True
        header["FLATCOR"] = True
        header["BVARINC"] = True
        header["SB_GID"] = gid
        header["VCMPGID"] = gid
        header["VCMPVER"] = 1
        header["VCMPQ"] = "LOCAL"
        bsens = np.full((ny, nx), -.5, dtype="f4")
        bm = np.full((ny, nx), bvar, dtype="f4")
        vb = bsens**2 * bm
        var = np.full((ny, nx), 5.0, dtype="f4")
        vr = var - vb
        assert np.all(vr >= 0)
        hdus = [
            fits.PrimaryHDU(header=header),
            fits.ImageHDU(data=sci, name="SCI"),
            fits.ImageHDU(data=var, name="VAR"),
            fits.ImageHDU(data=np.zeros((ny, nx), dtype="u2"), name="MASK"),
        ]
        for key, values, unit in (
            ("VBIAS", vb, "e-^2"), ("VREST", vr, "e-^2"),
            ("BSENS", bsens, "e-/ADU"), ("BVARADU", bm, "ADU^2"),
        ):
            eh = fits.ImageHDU(data=values.astype("f4"), name=key)
            eh.header["BUNIT"] = unit
            hdus.append(eh)
        fits.HDUList(hdus).writeto(source, overwrite=True)
    else:
        fits.writeto(source, sci, overwrite=True)

    cfg = {
        "work_dir": str(work),
        "frames": {"obj": [str(source)]},
        "roi": {
            "obj_y1": 22, "obj_y2": 26,
            "sky_y1": 2, "sky_y2": 10,
            "sky2_y1": 37, "sky2_y2": 45,
            "units": "px",
        },
        "sky_sub": {
            "primary_method": "kelson_raw",
            "save_png": False,
            "covariance_capture": {"enabled": True},
            "kelson_raw": {"knot_step_A": 4., "object_eating_warn": -1.},
        },
    }
    run_sky_sub(cfg)
    stage = stage_dir(work, "sky")
    manifest = stage/"obj_0001_kelson_frozen_fit_lineage_v1.json"
    assert manifest.is_file()
    return source, manifest, stage/"obj_0001_skysub_raw.fits"


def test_actual_stage_input_components_are_verified_without_covariance_release(tmp_path):
    source, manifest, out = _night(tmp_path/"one")
    report = audit_sky_masterbias_group_sources([manifest])
    assert out.exists()
    assert report["status"] == "HOLD_PRE_SKY_VARIANCE_COMPONENTS_PRESENT_BUT_NOT_PROPAGATED"
    assert report["file_and_fit_lineage_verified"]
    assert report["n_sources_without_verified_component_planes"] == 0
    assert report["shared_group_bias_variance_maps_consistent"] is True
    assert report["masterbias_group_members"] == {"NIGHT_BIAS_GROUP": [0]}
    assert report["source_entries"][0]["source_status"] == (
        "VERIFIED_FLATFIELD_COMPONENTS_LOCAL_ONLY"
    )
    assert report["source_entries"][0]["source_fits"] == str(source.resolve())
    assert report["science_uncertainty_qualified"] is False
    assert report["production_variance_release_allowed"] is False
    assert "VREST_shared_MasterFlat_and_other_cross_exposure_correlations" in (
        report["unbound_provenance"]
    )
    with fits.open(out) as hdul:
        assert hdul[0].header["VCMPQ"] == "SKY_LOST"
        assert hdul[0].header["SKYVQUAL"] == "INCOMP"
        assert "VBIAS" not in hdul


def test_same_group_different_variance_maps_rejected_after_two_sky_runs(tmp_path):
    _, first, _ = _night(tmp_path/"first", gid="SHARED", bvar=4.0)
    _, second, _ = _night(tmp_path/"second", gid="SHARED", bvar=9.0)
    with pytest.raises(ValueError, match="inconsistent BVARADU maps"):
        audit_sky_masterbias_group_sources([first, second])
    ok = audit_sky_masterbias_group_sources([first])
    assert ok["n_exposures"] == 1


def test_missing_flatfield_components_yield_explicit_hold(tmp_path):
    _, manifest, _ = _night(tmp_path/"legacy", components=False)
    report = audit_sky_masterbias_group_sources([manifest])
    assert report["status"] == "HOLD_INCOMPLETE_PRE_SKY_VARIANCE_LINEAGE"
    assert report["n_sources_without_verified_component_planes"] == 1
    assert report["masterbias_group_members"] == {}
    assert report["science_uncertainty_qualified"] is False


@pytest.mark.parametrize("damage,message", [
    ("component_unit", "physical BUNIT"),
    ("vrest", "VBIAS \\+ VREST"),
    ("bias_response", "Signed BSENS"),
    ("group_id", "group"),
    ("units", "unit"),
    ("flatcor", "calibration state"),
    ("version", "Unqualified source"),
    ("missing_component", "Incomplete declared"),
])
def test_component_schema_fails_closed_on_invalid_physics(tmp_path, damage, message):
    # Low-cost direct component test: integrity binding to a Stage FITS is
    # separately verified in the real SkySub and four-file lineage tests.
    source, _, _ = _night(tmp_path/"original")
    corrupt = tmp_path/("modified_"+damage+".fits")
    with fits.open(source, memmap=False) as hdul:
        if damage == "component_unit":
            hdul["BSENS"].header["BUNIT"] = "ADU"
        elif damage == "vrest":
            hdul["VREST"].data[2, 3] += 2.
        elif damage == "bias_response":
            hdul["BSENS"].data[2, 3] *= 2.
        elif damage == "group_id":
            hdul[0].header["VCMPGID"] = "OTHER"
        elif damage == "units":
            hdul[0].header["BUNIT"] = "ADU"
        elif damage == "flatcor":
            hdul[0].header["FLATCOR"] = False
        elif damage == "version":
            hdul[0].header["VCMPVER"] = 9
        elif damage == "missing_component":
            hdul.pop(hdul.index_of("BVARADU"))
        hdul.writeto(corrupt)
    with pytest.raises(ValueError, match=message):
        _component_source(corrupt, max_pixels=1_000_000)


def test_duplicate_same_source_exposures_are_not_independent(tmp_path):
    _, manifest, _ = _night(tmp_path/"one")
    with pytest.raises(ValueError, match="Repeated SkySub source"):
        audit_sky_masterbias_group_sources([manifest, manifest])


def test_invalid_audit_budgets_are_rejected_before_file_io(tmp_path):
    with pytest.raises(ValueError, match="budget"):
        audit_sky_masterbias_group_sources([tmp_path/"notfound"], max_exposures=0)
    with pytest.raises(ValueError, match="bounded"):
        audit_sky_masterbias_group_sources([])
