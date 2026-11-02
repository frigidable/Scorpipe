"""Regression for audit finding A01: sensitivity fit/evaluation grid identity.

Masked spectral bins are absent from the training sample, but remain present
in the evaluated response and its FITS product on the original wavelength grid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.calib import sensfunc as sf


def _synthetic_standard(monkeypatch, *, invalid_everything=False):
    wave = 4000. + 2.0 * np.arange(701, dtype=float)
    ref_wave = 3950. + 2.0 * np.arange(751, dtype=float)
    ref_flux = np.full_like(ref_wave, 2.0)
    expected_response = 10.0 * np.exp(2.0e-4 * (wave - 4700.))
    flux = 2.0 * expected_response
    var = np.full_like(wave, 0.04)
    mask = np.zeros(wave.size, dtype=np.uint16)
    mask[(wave >= 5000.) & (wave <= 5060.)] = 2
    var[np.flatnonzero((wave >= 5130.) & (wave <= 5140.))] = -1.
    if invalid_everything:
        mask[:] = 2

    header = fits.Header()
    header["BUNIT"] = "e-/s"
    header["WAVEREF"] = "air"
    header["WAVEUNIT"] = "Angstrom"
    header["REBNORM"] = "COUNT_V1"
    observed = {
        "wave": wave, "flux": flux, "var": var, "mask": mask,
        "header": header, "waveref": "air", "bunit": "e-/s",
    }
    monkeypatch.setattr(sf, "read_spec1d", lambda *_args, **_kwargs: observed)
    monkeypatch.setattr(sf, "ensure_refctx", lambda _cfg: {"context_id": "test-a01"})
    monkeypatch.setattr(sf, "_load_standard_reference",
                        lambda *_args, **_kwargs: ({}, Path("standard.csv"), "f_lambda", "air"))
    monkeypatch.setattr(sf, "read_small_csv", lambda _path: {
        "wavelength_A": ref_wave, "flux": ref_flux,
    })
    monkeypatch.setattr(sf, "reference_path", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sf, "reference_metadata", lambda *_args, **_kwargs: {})
    return observed, expected_response


def test_masked_fit_evaluates_full_response_and_fits_columns(tmp_path, monkeypatch):
    observed, expected = _synthetic_standard(monkeypatch)
    payload = sf.build_sensfunc({}, standard_spec_path=tmp_path / "ignored.fits")
    wave = observed["wave"]
    assert payload["qc"]["fit_points"] < wave.size
    assert payload["qc"]["fit_points"] > 8
    assert payload["qc"]["model_evaluation_points"] == wave.size
    assert np.any(payload["feature_mask"] != 0)
    assert np.any(payload["fit_mask"] == 0)
    assert np.all(payload["fit_mask"][(wave >= 5000.) & (wave <= 5060.)] == 0)
    assert np.all(payload["fit_mask"][(wave >= 5130.) & (wave <= 5140.)] == 0)

    for key in ("wave", "raw_ratio", "response_abs", "response_rel",
                "response_unc", "fit_mask", "feature_mask",
                "telluric_mask", "low_sn_mask"):
        assert np.asarray(payload[key]).shape == wave.shape, key
    np.testing.assert_allclose(payload["response_abs"], expected, rtol=2e-3, atol=0)
    assert np.isfinite(payload["response_abs"]).all()

    outputs = sf.write_sensfunc_products(tmp_path / "results", payload)
    with fits.open(outputs["sensfunc_table_fits"]) as hdul:
        table = hdul["SENSFUNC"].data
        assert len(table) == len(wave)
        np.testing.assert_allclose(table["WAVELENGTH"], wave, rtol=0, atol=1e-8)
        np.testing.assert_allclose(table["RESPONSE_ABS"], expected, rtol=2e-3)
        assert int(np.sum(table["FIT_MASK"])) == payload["qc"]["fit_points"]


def test_below_eight_usable_standard_pixels_is_fail_closed(tmp_path, monkeypatch):
    _synthetic_standard(monkeypatch, invalid_everything=True)
    with pytest.raises(ValueError, match="only 0 valid standard-star pixels"):
        sf.build_sensfunc({}, standard_spec_path=tmp_path / "ignored.fits")


def test_spline_evaluation_grid_is_independent_of_training_grid():
    fit_wave = np.linspace(4500., 5200., 51)
    full_wave = np.linspace(4400., 5300., 131)
    y = np.log(5.0) + 2.5e-4 * (fit_wave - 4900.)
    result, meta = sf._robust_spline_fit(
        fit_wave, y, w=np.ones_like(fit_wave), eval_x=full_wave
    )
    assert result.shape == full_wave.shape
    assert meta["method"] in {"UnivariateSpline", "polyfit"}
    np.testing.assert_allclose(result, np.log(5.0) + 2.5e-4 * (full_wave - 4900.),
                               rtol=1e-6, atol=1e-6)


def test_invalid_evaluation_grid_is_rejected():
    wave = np.arange(4500., 4600., 1.)
    with pytest.raises(ValueError, match="finite 1D"):
        sf._robust_spline_fit(wave, np.ones_like(wave),
                              eval_x=np.array([4500., np.nan]))
