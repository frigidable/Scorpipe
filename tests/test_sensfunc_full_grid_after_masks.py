"""Regression: Sensfunc must evaluate its fit on the complete wavelength grid.

Previously, _robust_spline_fit was called with wave[fit_mask] and returned
only that many model samples. The caller combined these with full-wave masks
and passed them to an interpolation with mismatched x/y lengths.
"""

from pathlib import Path

import numpy as np
from astropy.io import fits

from scorpio_pipe.calib import sensfunc


def test_masked_training_points_evaluate_on_full_grid():
    wave = np.arange(4300.0, 5400.0, 1.0)
    fit_mask = ~(
        ((wave >= 4320.0) & (wave <= 4370.0))
        | ((wave >= 4840.0) & (wave <= 4890.0))
    )
    assert np.count_nonzero(~fit_mask) == 102

    x_fit = wave[fit_mask]
    y_fit = np.log(1.7 + 0.00002 * (x_fit - 4300.0))
    model_log, fit_meta = sensfunc._robust_spline_fit(
        x_fit, y_fit, w=np.ones_like(x_fit), evaluate_at=wave
    )

    assert model_log.shape == wave.shape
    assert np.isfinite(model_log).all()
    np.testing.assert_allclose(
        np.exp(model_log[fit_mask]), np.exp(y_fit), rtol=3e-3
    )
    assert fit_meta["method"] in {"UnivariateSpline", "polyfit"}


def test_build_sensfunc_handles_realistic_balmer_mask(monkeypatch):
    wave = np.arange(4300.0, 5400.0, 1.0)
    flux = 100.0 + 0.015 * (wave - 4300.0)
    var = np.full(wave.shape, 4.0)

    def read_spec(_path, *, mode):
        assert mode == "trace"
        return {
            "wave": wave,
            "flux": flux,
            "var": var,
            "mask": np.zeros(wave.shape, dtype=np.int32),
            "header": fits.Header({
                "REBNORM": "COUNT_V1",
                "WAVEREF": "air",
                "WAVEUNIT": "Angstrom",
            }),
            "waveref": "air",
            "bunit": "e-/Angstrom/s",
        }

    def reference_table(_path):
        return {
            "wavelength_A": np.array([4000.0, 4500.0, 5000.0, 5600.0]),
            "flux": np.array([2.0, 1.95, 1.85, 1.65]),
        }

    monkeypatch.setattr(sensfunc, "read_spec1d", read_spec)
    monkeypatch.setattr(sensfunc, "ensure_refctx", lambda cfg: {"context_id": "synthetic"})
    monkeypatch.setattr(
        sensfunc, "_load_standard_reference",
        lambda cfg, selected_name=None: (
            {}, Path("synthetic_reference.csv"), "f_lambda", "air"
        ),
    )
    monkeypatch.setattr(sensfunc, "read_small_csv", reference_table)
    monkeypatch.setattr(sensfunc, "reference_path", lambda *a, **kw: None)
    monkeypatch.setattr(sensfunc, "reference_metadata", lambda *a, **kw: {})

    payload = sensfunc.build_sensfunc(
        {"fluxcal": {"sensfunc_min_snr": 5.0}},
        standard_spec_path=Path("synthetic_standard.fits"),
    )

    assert payload["response_abs"].shape == wave.shape
    assert payload["response_unc"].shape == wave.shape
    assert payload["fit_mask"].shape == wave.shape
    assert np.isfinite(payload["response_abs"]).all()
    assert np.any(payload["fit_mask"] == 0)
    assert np.any(payload["fit_mask"] == 1)
    # Masked stellar absorption intervals must not collapse the output grid.
    balmer_mask = (wave >= 4840.0) & (wave <= 4890.0)
    assert np.all(payload["fit_mask"][balmer_mask] == 0)
    assert np.isfinite(payload["response_abs"][balmer_mask]).all()
