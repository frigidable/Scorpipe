"""A05: line catalogue and integrated flux use Angstrom wavelength coordinates.

Flux density calibrated per Angstrom must *not* be rescaled when the input
spectral coordinate is stored in nm.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.stages import line_measure as stage


@pytest.mark.parametrize("unit,factor", [
    ("Angstrom", 1.0), ("angstroms", 1.0),
    ("nm", 10.0), ("nanometer", 10.0),
])
def test_native_science_axis_to_angstrom(unit, factor):
    base = np.array([500.0, 500.5, 501.0])
    wave, native, status = stage._science_wavelength_angstrom(
        base, fits.Header({"WAVEUNIT": unit, "REBNORM": "COUNT_V1"})
    )
    np.testing.assert_allclose(wave, base * factor)
    assert native in {"Angstrom", "nm"}
    assert status == "EXPLICIT"


@pytest.mark.parametrize("unit", ["Hz", "micron", "air/vacuum"])
def test_unknown_axis_units_fail_closed(unit):
    with pytest.raises(ValueError, match="Unsupported LineMeasure wavelength unit"):
        stage._science_wavelength_angstrom(
            np.arange(500.0, 510.0),
            fits.Header({"WAVEUNIT": unit, "REBNORM": "COUNT_V1"}),
        )


def test_missing_count_v1_unit_is_not_implicitly_angstrom():
    with pytest.raises(ValueError, match="requires explicit WAVEUNIT"):
        stage._science_wavelength_angstrom(
            np.arange(500.0, 510.0),
            fits.Header({"REBNORM": "COUNT_V1"}),
        )


def test_legacy_missing_axis_unit_is_explicitly_unqualified():
    wave, native, status = stage._science_wavelength_angstrom(
        np.array([5000.0, 5001.0]),
        fits.Header({"REBNORM": "LEGACY"}),
    )
    np.testing.assert_array_equal(wave, [5000.0, 5001.0])
    assert native == "ASSUMED_ANGSTROM"
    assert status == "LEGACY_ASSUMED"


@pytest.mark.parametrize("axis", [
    [500.0, 500.0], [500.5, 500.0], [500.0, float("nan")],
])
def test_nonmonotonic_or_invalid_coordinates_fail(axis):
    with pytest.raises(ValueError, match="positive increasing"):
        stage._science_wavelength_angstrom(
            np.asarray(axis), fits.Header({"WAVEUNIT": "nm"}),
        )


@pytest.mark.parametrize("unit,scale", [("nm", 0.1), ("Angstrom", 1.0)])
def test_stage_measures_equivalent_native_nm_and_angstrom_axes(
    tmp_path, monkeypatch, unit, scale,
):
    wave_A = np.linspace(4940.0, 5020.0, 401)
    wave = wave_A * scale
    injected_flux = 1.0 + 30.0 * np.exp(-0.5 * ((wave_A - 5000.0) / 0.8) ** 2)
    variance = np.full_like(wave, 0.01)
    header = fits.Header({
        "WAVEREF": "air", "WAVEUNIT": unit,
        "REBNORM": "COUNT_V1", "BUNIT": "erg/s/cm2/Angstrom",
    })
    data = {
        "wave": wave, "flux": injected_flux, "var": variance,
        "mask": np.zeros_like(wave, dtype=np.uint16),
        "header": header, "dq1d_version": 1,
        "frac_used": np.ones_like(wave),
    }
    fake_spec = tmp_path / unit / "science.fits"
    fake_calsys = tmp_path / unit / "calsys.fits"
    fake_flux_qc = tmp_path / unit / "flux_qc.json"
    fake_paths = {
        "fluxcal_spec1d_fits": fake_spec,
        "calibration_systematics_fits": fake_calsys,
        "fluxcal_qc_json": fake_flux_qc,
    }
    monkeypatch.setattr(
        stage, "PRODUCT_CATALOG",
        SimpleNamespace(best_path=lambda _cfg, key, **_kw: fake_paths[key]),
    )
    monkeypatch.setattr(stage, "read_spec1d", lambda *_a, **_kw: data)
    zeros = np.zeros_like(wave)
    monkeypatch.setattr(stage, "_read_calsys",
                        lambda *_a, **_kw: {"rel": zeros, "tell": zeros, "abs": zeros})
    capture = {}

    def fake_lines(science_wave, flux, var, **kw):
        capture["wave"] = np.asarray(science_wave).copy()
        capture["flux"] = np.asarray(flux).copy()
        capture["waveref"] = kw["waveref"]
        return [{"name": "Hbeta", "detected": False, "reason": "synthetic_diagnostic"}]

    monkeypatch.setattr(stage, "measure_lines", fake_lines)
    monkeypatch.setattr(stage, "balmer_decrement_summary", lambda _rows: None)
    monkeypatch.setattr(stage, "write_lines_csv",
                        lambda path, _rows: Path(path).write_text("name\n", encoding="utf-8") and Path(path))
    monkeypatch.setattr(stage, "plot_line_measurements_png",
                        lambda path, *_args: Path(path).write_bytes(b"png") and Path(path))
    monkeypatch.setattr(stage, "write_done_json", lambda **_kw: None)

    out = tmp_path / unit / "line_measure"
    result = stage.run_line_measure({}, run_root=tmp_path / unit, stage_dir_path=out)
    assert result is not None
    np.testing.assert_allclose(capture["wave"], wave_A, rtol=0, atol=1e-8)
    np.testing.assert_allclose(capture["flux"], injected_flux, rtol=0, atol=0)
    assert capture["waveref"] == "air"
    report = json.loads((out / "science_lines.json").read_text(encoding="utf-8"))
    assert report["measurement_wavelength_unit"] == "Angstrom"
    assert report["input_wavelength_unit"] == unit
    assert report["coordinate_qualification"] == "EXPLICIT"
