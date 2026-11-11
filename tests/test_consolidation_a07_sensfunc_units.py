"""A07 regression: Angstrom calibration references on nm standard spectra.

The numerical standard flux density stays per Angstrom; only wavelength
coordinates (reference, extinction, Balmer and telluric) change units.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.calib import sensfunc as sf
from scorpio_pipe.calib.fluxcal import _load_sensfunc


def _prepare_standard(tmp_path, monkeypatch, *, waveunit="nm"):
    wave_nm = 400.0 + 0.2 * np.arange(701, dtype=float)
    wave_A = wave_nm * 10.0
    ref_flux = 2.0 + 0.0001 * (wave_A - 4700.0)
    response = 12.0 * np.exp(0.0002 * (wave_A - 4700.0))
    extinction_k = 0.2 + 0.00005 * (wave_A - 4000.0)
    airmass = 1.25
    counts = ref_flux * response / (10.0 ** (0.4 * extinction_k * airmass))

    hdr = fits.Header({"WAVEREF": "air", "REBNORM": "COUNT_V1", "BUNIT": "e-/s"})
    if waveunit is not None:
        hdr["WAVEUNIT"] = waveunit
    standard = {
        "wave": wave_nm,
        "flux": counts,
        "var": np.full(wave_nm.shape, 0.04),
        "mask": np.zeros(wave_nm.shape, dtype=np.uint16),
        "header": hdr,
        "waveref": "air",
        "bunit": "e-/s",
    }
    standard["mask"][(wave_nm >= 500.0) & (wave_nm <= 501.0)] = 2
    raw = tmp_path / "original_standard.fits"
    fits.PrimaryHDU().writeto(raw)
    extinction_file = tmp_path / "extinction.csv"
    extinction_file.write_text("dummy", encoding="utf-8")
    telluric_file = tmp_path / "telluric.csv"
    telluric_file.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(sf, "read_spec1d", lambda *_args, **_kw: standard)
    monkeypatch.setattr(sf, "ensure_refctx", lambda _cfg: {"context_id": "a07-synthetic"})
    monkeypatch.setattr(
        sf, "_load_standard_reference",
        lambda *_args, **_kw: ({}, Path("reference.csv"), "f_lambda", "air"),
    )

    def read_reference(path):
        if Path(path) == extinction_file:
            return {
                "wavelength_A": np.array([3900.0, 5500.0]),
                "k_lambda": np.array([0.195, 0.275]),
            }
        return {
            "wavelength_A": np.array([3900.0, 5500.0]),
            "flux": np.array([1.92, 2.08]),
        }

    monkeypatch.setattr(sf, "read_small_csv", read_reference)

    def ref_path(_cfg, *, prefix=None, exact=None):
        if prefix == "extinction_curve.":
            return extinction_file
        if exact == "telluric_mask.default":
            return telluric_file
        return None

    monkeypatch.setattr(sf, "reference_path", ref_path)
    monkeypatch.setattr(sf, "reference_metadata", lambda *_a, **_kw: {})
    monkeypatch.setattr(
        sf, "read_interval_csv",
        lambda _p: [(5200.0, 5210.0, "telluric")],
    )
    monkeypatch.setattr(
        sf, "parse_frame_meta",
        lambda *_a, **_kw: SimpleNamespace(
            airmass=airmass,
            provenance_for=lambda _k: "header",
            provenance_class=lambda _k: "header",
        ),
    )
    return standard, response, raw


def test_nm_standard_uses_angstrom_reference_masks_and_extinction(tmp_path, monkeypatch):
    spectrum, expected, raw = _prepare_standard(tmp_path, monkeypatch)
    payload = sf.build_sensfunc(
        {}, standard_spec_path=tmp_path / "observed.fits",
        standard_qc={"selected_standard": {"path": str(raw)}},
    )
    wave = spectrum["wave"]
    assert payload["wave_unit"] == "nm"
    assert payload["qc"]["response_wavelength_unit"] == "nm"
    assert payload["qc"]["reference_to_spectral_factor"] == 0.1
    assert payload["qc"]["extinction_applied"] is True
    assert payload["flux_unit"] == "erg/s/cm2/Angstrom"
    assert payload["response_abs"].shape == wave.shape
    assert np.all(payload["fit_mask"][(wave >= 484.0) & (wave <= 489.0)] == 0)
    assert np.all(payload["telluric_mask"][(wave >= 520.0) & (wave <= 521.0)] == 1)
    assert np.all(payload["fit_mask"][(wave >= 520.0) & (wave <= 521.0)] == 0)
    np.testing.assert_allclose(payload["response_abs"], expected, rtol=5e-3)

    outputs = sf.write_sensfunc_products(tmp_path / "out", payload)
    with fits.open(outputs["sensfunc_table_fits"]) as hdul:
        assert hdul[0].header["WAVEUNIT"] == "nm"
        assert hdul[0].header["INWUNIT"] == "nm"
        assert len(hdul["SENSFUNC"].data) == len(wave)
        np.testing.assert_allclose(hdul["SENSFUNC"].data["WAVELENGTH"], wave)
    loaded = _load_sensfunc(outputs["sensfunc_table_fits"])
    np.testing.assert_allclose(loaded["wave"], wave)
    assert loaded["input_waveunit"] == "nm"


@pytest.mark.parametrize("unit,error", [
    ("micron", "Unsupported standard spectral wavelength unit"),
    (None, "COUNT_V1 standard requires explicit wavelength-axis units"),
])
def test_unrecognized_count_v1_native_units_fail_closed(tmp_path, monkeypatch, unit, error):
    _prepare_standard(tmp_path, monkeypatch, waveunit=unit)
    with pytest.raises(ValueError, match=error):
        sf.build_sensfunc({}, standard_spec_path=tmp_path / "observed.fits")
