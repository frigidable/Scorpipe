"""Full-grid Sensfunc fit regression: stellar/telluric masks must not shrink the output."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

import scorpio_pipe.calib.sensfunc as sf


def test_sensfunc_fits_on_good_pixels_and_publishes_full_grid(tmp_path, monkeypatch):
    wave = np.arange(4000.0, 5101.0, 1.0)
    standard_flux = np.full_like(wave, 2.0)
    observed_counts = 5.0 * standard_flux
    variance = np.full_like(wave, 0.01)
    mask = np.zeros_like(wave, dtype=np.uint16)
    mask[500] = 2
    header = fits.Header()
    header["WAVEREF"] = "air"
    header["WAVEUNIT"] = "Angstrom"
    header["BUNIT"] = "e-/s"
    header["REBNORM"] = "COUNT_V1"

    def fake_read_spec1d(_path, *, mode="trace"):
        return {
            "wave": wave, "flux": observed_counts, "var": variance,
            "mask": mask, "header": header, "waveref": "air",
            "bunit": "e-/s",
        }

    monkeypatch.setattr(sf, "read_spec1d", fake_read_spec1d)
    monkeypatch.setattr(sf, "ensure_refctx", lambda cfg: {"context_id": "test-context"})
    monkeypatch.setattr(sf, "_load_standard_reference", lambda cfg, selected_name: (
        {}, tmp_path / "reference.csv", "f_lambda", "air"
    ))
    monkeypatch.setattr(sf, "read_small_csv", lambda path: {
        "wavelength_A": wave, "flux": standard_flux,
    })
    monkeypatch.setattr(sf, "reference_path", lambda cfg, **kw: None)
    monkeypatch.setattr(sf, "reference_metadata", lambda cfg, **kw: {})

    payload = sf.build_sensfunc({}, standard_spec_path=tmp_path / "standard.fits")
    assert payload["wave"].shape == (1101,)
    assert payload["response_abs"].shape == wave.shape
    assert payload["response_rel"].shape == wave.shape
    assert payload["response_unc"].shape == wave.shape
    assert payload["fit_mask"].shape == wave.shape
    assert np.count_nonzero(payload["fit_mask"] == 0) > 50
    assert payload["fit_mask"][500] == 0
    np.testing.assert_allclose(payload["response_abs"], 5.0, rtol=0, atol=1e-4)

    products = sf.write_sensfunc_products(tmp_path / "products", payload)
    with fits.open(products["sensfunc_table_fits"], memmap=False) as hdul:
        data = hdul["SENSFUNC"].data
        assert len(data) == len(wave)
        np.testing.assert_allclose(data["RESPONSE_ABS"], 5.0, atol=1e-4)
        assert np.count_nonzero(data["FIT_MASK"] == 0) > 50


def test_sensfunc_refuses_fabricated_constant_from_insufficient_fit_points():
    with pytest.raises(ValueError, match="at least 8 valid"):
        sf._robust_spline_fit(
            np.arange(7, dtype=float),
            np.ones(7, dtype=float),
            x_eval=np.arange(20, dtype=float),
        )


def test_sensfunc_direct_fit_respects_full_evaluation_grid():
    x = np.linspace(4000.0, 5000.0, 100)
    good = np.ones(x.size, dtype=bool)
    good[45:65] = False
    model, meta = sf._robust_spline_fit(
        x[good], np.log(np.full(np.count_nonzero(good), 5.0)),
        x_eval=x,
    )
    assert meta["method"] in {"UnivariateSpline", "polyfit"}
    assert model.shape == x.shape
    np.testing.assert_allclose(np.exp(model), 5., atol=1e-4)
