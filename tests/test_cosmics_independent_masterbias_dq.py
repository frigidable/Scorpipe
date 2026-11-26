"""Independent indexed-bias CR detection: real producer, controlled CR engine."""
from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from scorpio_pipe import maskbits
from scorpio_pipe.cosmics_biasvar import read_clean_biasvar_sidecar
from scorpio_pipe.cosmics_dq import read_clean_dq_sidecar
from scorpio_pipe.stages import calib, cosmics, flatfield, sky_sub
from tests.golden.mini_night import make_mini_night


def _night(tmp_path):
    mn = make_mini_night(tmp_path)
    for i, path in enumerate(mn.cfg["frames"]["bias"]):
        with fits.open(path, mode="update") as hdul:
            hdul[0].data += 1000.0
            hdul[0].data[0, 0] = 0.0
            hdul[0].data[0, 1] = 1000.0 * i
    for path in mn.cfg["frames"]["obj"]:
        with fits.open(path, mode="update") as hdul:
            hdul[0].data += 1000.0
            hdul[0].data[0, 0] = 0.0
            hdul[0].data[0, 2] = 0.0
            hdul[0].header["BIASSUB"] = False
            if "SIMBZERO" in hdul[0].header:
                del hdul[0].header["SIMBZERO"]
            del hdul[0].header["RDNOISE"]
    mn.cfg["calib"].update(bias_policy="strict", bias_combine="median",
                           bias_gross_reject_adu=50.0, bias_mask_persistent_zero=True)
    mn.cfg["cosmics"] = dict(method="la_cosmic_independent", apply_to=["obj"],
                             bias_subtract=True, preserve_manual=False,
                             mask_raw_zero=True, save_png=False, save_mask_fits=True,
                             la_protect_lines=False, la_dilate=0)
    stage = mn.work_dir / "03_bias"
    calib.build_superbias(mn.cfg, run_root=mn.work_dir, stage_dir=stage,
                         allow_legacy_mirrors=False)
    idx = json.loads((stage / "master_bias_index.json").read_text(encoding="utf8"))
    group = idx["groups"][0]
    assert group["variance"]["bad_support_pixels"] == 1
    assert group["variance"]["persistent_zero_pixels"] == 1
    return mn, idx, group


def test_independent_cosmics_propagates_mask_before_detection_and_to_consumers(tmp_path, monkeypatch):
    mn, idx, group = _night(tmp_path)
    hdr = fits.getheader(mn.cfg["frames"]["obj"][0])
    with pytest.raises(RuntimeError, match="masked persistent raw zeros"):
        calib.resolve_master_bias(mn.cfg, hdr, policy_override="strict")
    selection = calib.resolve_master_bias(
        mn.cfg, hdr, policy_override="strict", accept_masked_dq=True,
    )
    master = fits.getdata(selection.sci_path)
    mdq = fits.getdata(selection.dq_path)
    var = fits.getdata(selection.var_path)
    assert np.count_nonzero(mdq) == 2
    seen = []

    def fake_detect(img, **kw):
        seen.append(kw["inmask"].copy())
        assert kw["gain"] == pytest.approx(0.62)
        assert kw["readnoise"] == pytest.approx(selection.rdnoise_e)
        mask = np.zeros(img.shape, dtype=bool)
        mask[0, 0] = mask[5, 6] = True
        clean = img.copy()
        clean[0, 0] = clean[5, 6] = -123.0
        clean[5, 7] = -999.0  # unclassified alteration must not leak
        return mask, clean

    monkeypatch.setitem(sys.modules, "astroscrappy",
                        types.SimpleNamespace(detect_cosmics=fake_detect))
    out = mn.work_dir / "05_cosmics"
    cosmics.clean_cosmics(mn.cfg, out_dir=out)
    assert len(seen) == 2
    assert all(m[0, :3].all() for m in seen)
    for path in mn.cfg["frames"]["obj"]:
        from pathlib import Path

        raw = fits.getdata(path)
        clean = out / "obj" / "clean" / f"{Path(path).stem}_clean.fits"
        data, h = fits.getdata(clean, header=True)
        dq = read_clean_dq_sidecar(clean, h, data.shape)
        assert h["BIASSUB"] and h["SBDQCNT"] == 2
        assert h["SB_GID"] == selection.gid
        assert h["RDNOISE"] == pytest.approx(selection.rdnoise_e)
        assert np.all((dq & mdq) == mdq)
        assert all(int(x) & int(maskbits.BADPIX) for x in dq[0, :3])
        assert not int(dq[0, 0]) & int(maskbits.COSMIC)
        assert int(dq[5, 6]) & int(maskbits.COSMIC)
        assert np.array_equal(data[dq == 0], (raw - master)[dq == 0])
        assert data[0, 0] == (raw - master)[0, 0]
        assert np.array_equal(read_clean_biasvar_sidecar(clean, h, data.shape), var)
        assert np.array_equal(flatfield._read_sci_var_mask(clean)[2], dq)
        assert np.array_equal(sky_sub._try_load_mask_for_clean(clean, clean_header=h), dq)
    coverage = fits.getdata(out / "obj" / "coverage.fits")
    assert np.all(coverage[0, :3] == 0)
    assert coverage[5, 6] == 0
    assert coverage[5, 7] == 2


@pytest.mark.parametrize("damage", ["missing", "count", "gid", "shape"])
def test_masked_masterbias_resolution_fails_on_damaged_dq(tmp_path, damage):
    mn, idx, group = _night(tmp_path)
    path = mn.work_dir / group["files"]["dq"]
    if damage == "missing":
        path.unlink()
    else:
        with fits.open(path, mode="update") as hdul:
            if damage == "count":
                hdul[0].data[0, 0] = 0
            elif damage == "gid":
                hdul[0].header["SB_GID"] = "wrong"
            else:
                hdul[0].data = hdul[0].data[:2].copy()
    with pytest.raises((RuntimeError, OSError)):
        calib.resolve_master_bias(
            mn.cfg, fits.getheader(mn.cfg["frames"]["obj"][0]),
            policy_override="strict", accept_masked_dq=True,
        )


def test_independent_requires_no_manual_rewrite(tmp_path):
    mn, _, _ = _night(tmp_path)
    mn.cfg["cosmics"]["preserve_manual"] = True
    with pytest.raises(ValueError, match="preserve_manual=false"):
        cosmics.clean_cosmics(mn.cfg, out_dir=mn.work_dir / "05_cosmics")


def test_masterbias_dq_marker_requires_typed_sidecar(tmp_path):
    header = fits.Header({"SBDQVER": 1, "SBDQCNT": 2, "SB_GID": "group"})
    with pytest.raises(ValueError, match="qualified DQ_V1"):
        read_clean_dq_sidecar(tmp_path / "a_clean.fits", header, (8, 8))
