"""A01 consolidation regression: nonpositive variance cannot train Sensfunc.

The existing full-grid response fix already exists on vNext; this tests the
remaining science-safe guard against invalid standard-star variance.
"""
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.calib import sensfunc


def _mock_standard(monkeypatch, *, all_variances_bad: bool):
    wave = 4300.0 + np.arange(1101, dtype=float)
    var = np.full_like(wave, 4.0)
    if all_variances_bad:
        var[:] = 0.0
    else:
        var[100] = 0.0
        var[150] = -1.0
        var[200] = np.inf

    hdr = fits.Header()
    hdr["REBNORM"] = "COUNT_V1"
    hdr["WAVEREF"] = "air"
    hdr["WAVEUNIT"] = "Angstrom"
    spectrum = {
        "wave": wave,
        "flux": 100.0 + 0.005 * (wave - 4300.0),
        "var": var,
        "mask": np.zeros_like(wave, dtype=np.uint16),
        "header": hdr,
        "waveref": "air",
        "bunit": "e-/s",
    }
    monkeypatch.setattr(sensfunc, "read_spec1d", lambda *_args, **_kw: spectrum)
    monkeypatch.setattr(sensfunc, "ensure_refctx", lambda _cfg: {"context_id": "a01-synthetic"})
    monkeypatch.setattr(
        sensfunc, "_load_standard_reference",
        lambda *_args, **_kw: ({}, Path("synthetic.csv"), "f_lambda", "air"),
    )
    monkeypatch.setattr(
        sensfunc, "read_small_csv",
        lambda _path: {
            "wavelength_A": np.array([4000.0, 6000.0]),
            "flux": np.array([2.0, 2.0]),
        },
    )
    monkeypatch.setattr(sensfunc, "reference_path", lambda *_args, **_kw: None)
    monkeypatch.setattr(sensfunc, "reference_metadata", lambda *_args, **_kw: {})
    return wave


def test_sensfunc_excludes_nonpositive_variances_from_full_grid(monkeypatch):
    wave = _mock_standard(monkeypatch, all_variances_bad=False)
    result = sensfunc.build_sensfunc({}, standard_spec_path=Path("synthetic.fits"))

    assert result["response_abs"].shape == wave.shape
    assert result["fit_mask"].shape == wave.shape
    assert np.all(np.isfinite(result["response_abs"]))
    assert np.all(result["response_abs"] > 0)
    for pixel in (100, 150, 200):
        assert result["fit_mask"][pixel] == 0

    qc = result["qc"]
    assert qc["model_evaluation_points"] == wave.size
    assert qc["fit_points"] == int(np.sum(result["fit_mask"]))
    assert qc["fit_wavelength_min"] <= qc["fit_wavelength_max"]


def test_sensfunc_all_zero_variances_fail_closed(monkeypatch):
    _mock_standard(monkeypatch, all_variances_bad=True)
    with pytest.raises(ValueError, match="requires at least 8 valid"):
        sensfunc.build_sensfunc({}, standard_spec_path=Path("synthetic.fits"))
