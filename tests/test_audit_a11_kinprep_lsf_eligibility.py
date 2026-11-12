"""A11: arc-detector LSF must not be advertised as final 1D instrumental LSF."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.stages.kinematics_prep import (
    _load_arc_lsf_proxy,
    run_kinematics_prep,
)
from scorpio_pipe.workspace_paths import stage_dir


def _science_fixture(tmp_path, *, unit="Angstrom", medium="air",
                     arc_medium="air", arc_unit="Angstrom",
                     source="arc_measured", proxy_wave=None, proxy_sig=None):
    work = tmp_path / "work"
    edir = stage_dir(work, "extract1d")
    edir.mkdir(parents=True, exist_ok=True)
    lsfd = work / "lsf_model"
    lsfd.mkdir(parents=True, exist_ok=True)
    wave_A = np.array([5001., 5002., 5003., 5004., 5005., 5006., 5007.])
    values = wave_A / 10. if unit == "nm" else wave_A
    hdr = fits.Header()
    hdr["WAVEREF"] = medium
    hdr["WAVEUNIT"] = unit
    def ext(which):
        cols = [
            fits.Column(name="LAMBDA", format="D", array=values),
            fits.Column(name=f"FLUX_{which}", format="D", array=np.arange(7.) + 10.),
            fits.Column(name=f"VAR_{which}", format="D", array=np.ones(7)),
            fits.Column(name=f"MASK_{which}", format="J", array=np.zeros(7, dtype=np.int32)),
            fits.Column(name=f"SKY_{which}", format="D", array=np.zeros(7)),
            fits.Column(name=f"NPIX_{which}", format="I", array=np.full(7, 3, dtype=np.int16)),
        ]
        return fits.BinTableHDU.from_columns(cols, name=f"SPEC_{which}")
    fits.HDUList([
        fits.PrimaryHDU(header=hdr), ext("TRACE"), ext("FIXED")
    ]).writeto(edir / "spec1d.fits")
    (edir / "trace.json").write_text('{"trace_model":{"method":"test"},"aperture":{"half_width":2}}')
    lsf = lsfd / "lsf_model.json"
    lsf.write_text(json.dumps({
        "source": source,
        "instrument": {"wave_ref": arc_medium, "wave_unit": arc_unit},
        "summary_1d": {
            "lambda_A": [5002., 5003., 5004., 5005.] if proxy_wave is None else proxy_wave,
            "sigma_A": [1.2, 1.3, 1.4, 1.5] if proxy_sig is None else proxy_sig,
        },
    }))
    return {"work_dir": str(work)}, lsf


@pytest.mark.parametrize("unit", ["Angstrom", "nm"])
def test_kinprep_arc_proxy_is_never_a_qualified_science_lsf(tmp_path, unit):
    cfg, _ = _science_fixture(tmp_path, unit=unit)
    out = run_kinematics_prep(cfg)
    assert out["ok"] is True
    assert out["lsf_qualified"] is False
    assert out["sigma_inst_A_median"] is None
    assert out["sigma_arc_proxy_A_median"] == pytest.approx(1.35)
    p = Path(out["spec_kin_fits"])
    with fits.open(p) as h:
        ph=h[0].header
        assert ph["LSFQUAL"] is False
        assert ph["LSFSTAT"] == "ARC_PROXY_ONLY"
        assert "SIGINST" not in ph and "FWHMINST" not in ph
        table=h["SPEC_KIN"].data
        assert np.all(np.isnan(table["SIGMA_INST_A"]))
        assert np.all(np.isnan(table["SIGMA_INST_KMS"]))
        np.testing.assert_allclose(table["SIGMA_ARC_PROXY_A"][1:5],
                                   [1.2,1.3,1.4,1.5])
        assert np.isnan(table["SIGMA_ARC_PROXY_A"][0])
        assert np.isnan(table["SIGMA_ARC_PROXY_A"][-1])
        expected=299792.458*table["SIGMA_ARC_PROXY_A"][1:5] / np.array([5002.,5003.,5004.,5005.])
        np.testing.assert_allclose(table["SIGMA_ARC_PROXY_KMS"][1:5], expected)
        np.testing.assert_allclose(table["FLUX"],np.arange(7.)+10.)
        np.testing.assert_allclose(table["VAR"],np.ones(7))
    notes=(p.parent / "README.md").read_text()
    assert "NOT ELIGIBLE" in notes
    assert "detector-arc diagnostics" in notes
    prov=json.loads((p.parent / "lsf_provenance.json").read_text())
    assert prov["effective_1d_lsf_qualified"] is False
    assert prov["no_proxy_edge_extrapolation"] is True
    assert prov["arc_proxy_is_science_lsf"] is False
    assert prov["n_proxy_covered"] == 4
    assert prov["arc_proxy_source"] == "arc_measured"
    done=json.loads((p.parent / "done.json").read_text())
    assert done["lsf_qualified"] is False


def test_arc_lsf_medium_mismatch_fails_closed_for_proxy(tmp_path):
    cfg, _ = _science_fixture(tmp_path, medium="air", arc_medium="vacuum")
    out = run_kinematics_prep(cfg)
    assert out["lsf_status"] == "ARC_MEDIUM_MISMATCH"
    with fits.open(out["spec_kin_fits"]) as h:
        assert np.isnan(h["SPEC_KIN"].data["SIGMA_ARC_PROXY_A"]).all()
        assert np.isnan(h["SPEC_KIN"].data["SIGMA_INST_A"]).all()


def test_arc_lsf_unit_nm_previously_mislabeled_as_angstrom_is_rejected(tmp_path):
    cfg, _ = _science_fixture(tmp_path, arc_unit="nm")
    out = run_kinematics_prep(cfg)
    assert out["lsf_status"] == "ARC_UNIT_UNVERIFIED"
    with fits.open(out["spec_kin_fits"]) as h:
        assert np.isnan(h["SPEC_KIN"].data["SIGMA_ARC_PROXY_A"]).all()


def test_missing_lsf_still_exports_valid_spectrum_without_fake_sigmas(tmp_path):
    cfg, lsf = _science_fixture(tmp_path)
    lsf.unlink()
    out = run_kinematics_prep(cfg)
    assert out["lsf_status"] == "NO_ARC_PROXY"
    with fits.open(out["spec_kin_fits"]) as h:
        t=h["SPEC_KIN"].data
        assert np.isnan(t["SIGMA_INST_A"]).all()
        assert np.isnan(t["SIGMA_ARC_PROXY_A"]).all()


@pytest.mark.parametrize("wav,sig", [
    ([5001.,5003.,5002.],[1.,2.,3.]),
    ([5001.,5002.,5002.],[1.,2.,3.]),
    ([5001.,5002.,5003.],[1.,-2.,3.]),
    ([5001.,5002.,5003.],[1.,float("nan"),3.]),
    ([5001.,5002.,5003.],[1.,2.]),
])
def test_invalid_arc_resolution_grid_is_not_exported(tmp_path, wav, sig):
    cfg, path = _science_fixture(tmp_path, proxy_wave=wav, proxy_sig=sig)
    lam, sigma, meta = _load_arc_lsf_proxy(path)
    assert lam.size==0 and sigma.size==0
    assert meta["status"] == "ARC_PROXY_INVALID"
    out=run_kinematics_prep(cfg)
    assert out["lsf_qualified"] is False


def test_manual_fallback_arc_lsf_remains_diagnostic_only(tmp_path):
    cfg, _ = _science_fixture(tmp_path, source="manual_fallback")
    out=run_kinematics_prep(cfg)
    assert out["lsf_status"]=="ARC_PROXY_ONLY"
    assert out["lsf_qualified"] is False
    with fits.open(out["spec_kin_fits"]) as h:
        assert np.isnan(h["SPEC_KIN"].data["SIGMA_INST_A"]).all()
        assert np.isfinite(h["SPEC_KIN"].data["SIGMA_ARC_PROXY_A"]).any()


def test_native_nm_converted_by_a11b_remains_unqualified_arc_proxy(tmp_path):
    cfg, path = _science_fixture(tmp_path, arc_unit="Angstrom")
    source = json.loads(path.read_text(encoding="utf-8"))
    source["instrument"]["native_wave_unit"] = "nm"
    path.write_text(json.dumps(source), encoding="utf-8")
    out = run_kinematics_prep(cfg)
    assert out["lsf_status"] == "ARC_PROXY_ONLY"
    assert out["lsf_qualified"] is False
    with fits.open(out["spec_kin_fits"], memmap=False) as hdul:
        assert hdul[0].header["LSFQUAL"] is False
        assert np.isnan(hdul["SPEC_KIN"].data["SIGMA_INST_A"]).all()
        assert np.isfinite(hdul["SPEC_KIN"].data["SIGMA_ARC_PROXY_A"]).any()
