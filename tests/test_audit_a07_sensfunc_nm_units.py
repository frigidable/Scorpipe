"""A07: Å reference libraries must be mapped into nm standard-star axes."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.calib import sensfunc as sf
from scorpio_pipe.calib.fluxcal import _load_sensfunc


def _arrange_nm_standard(tmp_path, monkeypatch, *, unit="nm"):
    wave_nm = np.arange(400., 540.2, .2)
    lam_a = wave_nm * 10.
    response = 12. * np.exp(.0002 * (lam_a - 4700.))
    ref_flux = 2. + .0001 * (lam_a - 4700.)
    extinction_k = .2 + .00005 * (lam_a - 4000.)
    airmass = 1.25
    obs = ref_flux * response / (10.**(.4 * extinction_k * airmass))
    hdr = fits.Header()
    hdr["WAVEREF"] = "air"
    hdr["WAVEUNIT"] = unit
    hdr["REBNORM"] = "COUNT_V1"
    hdr["BUNIT"] = "e-/s"
    mask = np.zeros(len(wave_nm), dtype=np.uint16)
    mask[(wave_nm >= 500.) & (wave_nm <= 501.)] = 1
    standard = dict(
        wave=wave_nm, flux=obs, var=(obs / 100.)**2,
        mask=mask, header=hdr, waveref="air", bunit="e-/s",
    )
    monkeypatch.setattr(sf, "read_spec1d", lambda *_a, **_kw: standard)
    monkeypatch.setattr(sf, "ensure_refctx", lambda _cfg: {"context_id": "test-a07"})
    ref_path = tmp_path / "ref.csv"
    grid = np.arange(3800., 5602., 2.)
    np.savetxt(ref_path, np.column_stack([grid, 2. + .0001*(grid-4700.)]),
               header="wavelength_A,flux", comments="", delimiter=",")
    monkeypatch.setattr(
        sf, "_load_standard_reference",
        lambda *_a, **_kw: ({}, ref_path, "f_lambda", "air"),
    )
    ext_path = tmp_path / "ext.csv"
    np.savetxt(ext_path, np.column_stack([grid, .2+.00005*(grid-4000.)]),
               header="wavelength_A,k_lambda", comments="", delimiter=",")
    tell_path = tmp_path / "tell.csv"
    tell_path.write_text("lambda_min_A,lambda_max_A,label\n5080,5100,tell\n")
    def ref_lookup(_cfg, *, exact=None, prefix=None):
        if prefix == "extinction_curve.":
            return ext_path
        if exact == "telluric_mask.default":
            return tell_path
        return None
    monkeypatch.setattr(sf, "reference_path", ref_lookup)
    monkeypatch.setattr(sf, "reference_metadata", lambda *_a, **_kw: {})
    raw_path = tmp_path / "standard_raw.fits"
    fits.PrimaryHDU(data=np.zeros((2,2), dtype=np.float32)).writeto(raw_path)
    monkeypatch.setattr(sf, "parse_frame_meta", lambda *_a, **_kw: SimpleNamespace(
        airmass=airmass,
        provenance_class=lambda _k: "header",
        provenance_for=lambda _k: "header",
    ))
    qc = {"selected_standard": {"object_name": "synthetic", "path": str(raw_path)}}
    return standard, response, qc


def test_nm_reference_extinction_and_line_masks_round_trip_in_fits(tmp_path, monkeypatch):
    standard, response, qc = _arrange_nm_standard(tmp_path, monkeypatch)
    result = sf.build_sensfunc(
        {}, standard_spec_path=tmp_path / "ignored.fits", standard_qc=qc,
    )
    w = standard["wave"]
    assert result["wave_unit"] == "nm"
    assert result["input_waveunit"] == "nm"
    assert result["qc"]["reference_to_spectral_factor"] == .1
    assert result["qc"]["extinction_applied"] is True
    assert result["qc"]["model_evaluation_points"] == len(w)
    assert result["qc"]["fit_points"] < len(w)
    assert np.all(result["feature_mask"][(w >= 484.) & (w <= 489.)] == 1)
    assert np.all(result["telluric_mask"][(w >= 508.) & (w <= 510.)] == 1)
    assert np.all(result["fit_mask"][(w >= 500.) & (w <= 501.)] == 0)
    np.testing.assert_allclose(result["raw_ratio"], response, rtol=1.e-9)
    np.testing.assert_allclose(result["response_abs"], response, rtol=.002)

    outputs = sf.write_sensfunc_products(tmp_path / "sens", result)
    with fits.open(outputs["sensfunc_table_fits"]) as h:
        assert h[0].header["WAVEUNIT"] == "nm"
        assert h[0].header["INWUNIT"] == "nm"
        assert h[0].header["INSTEP"] == pytest.approx(.2)
        np.testing.assert_allclose(h["SENSFUNC"].data["WAVELENGTH"], w)
        np.testing.assert_allclose(h["SENSFUNC"].data["RESPONSE_ABS"],
                                   response, rtol=.002)
    loaded = _load_sensfunc(outputs["sensfunc_table_fits"])
    assert loaded["input_waveunit"] == "nm"
    np.testing.assert_allclose(loaded["wave"], w)
    np.testing.assert_allclose(loaded["response_abs"], response, rtol=.002)


def test_unknown_spectral_axis_unit_fails_before_reference_loading(tmp_path, monkeypatch):
    standard, _, qc = _arrange_nm_standard(tmp_path, monkeypatch, unit="micron")
    with pytest.raises(ValueError, match="Unsupported standard spectral wavelength unit"):
        sf.build_sensfunc({}, standard_spec_path="ignored", standard_qc=qc)
