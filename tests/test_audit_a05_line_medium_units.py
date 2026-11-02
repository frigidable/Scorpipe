"""A05: AIR rest lines are not numerically equal to VACUUM wavelengths."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.airvac import air_to_vac
from scorpio_pipe.science.lines import LineSpec, measure_line
from scorpio_pipe.stages import line_measure as stage

LINE = LineSpec("test_line", 5000., 6., (-22., -12.), (12., 22.))


def _spectrum(center, step=.20):
    wave = np.arange(4955., 5045., step, dtype=float)
    flux = np.ones_like(wave) + 25. * np.exp(-.5 * ((wave - center) / .6)**2)
    var = np.ones_like(wave) * .01
    return wave, flux, var


def _measure(wave, flux, var, *, medium):
    return measure_line(wave, flux, var, spec=LINE, wave_ref=medium,
                        calsys_rel=None, calsys_tell=None, calsys_abs=None)


def test_air_line_at_air_rest_is_zero_velocity():
    wave, flux, var = _spectrum(5000.)
    measured = _measure(wave, flux, var, medium="air")
    assert measured["detected"]
    assert measured["expected_wavelength"] == pytest.approx(5000.)
    assert abs(measured["velocity_kms"]) < .2


def test_vacuum_line_at_vacuum_rest_is_zero_velocity():
    vac_rest = float(air_to_vac(np.array([5000.]))[0])
    assert 5001. < vac_rest < 5002.
    wave, flux, var = _spectrum(vac_rest)
    measured = _measure(wave, flux, var, medium="vacuum")
    assert measured["detected"]
    assert measured["wave_ref"] == "vacuum"
    assert measured["rest_wavelength_in_grid_A"] == pytest.approx(vac_rest)
    assert measured["expected_wavelength"] == pytest.approx(vac_rest)
    assert abs(measured["velocity_kms"]) < .2


def test_vacuum_line_at_redshift_uses_vacuum_then_redshift():
    z = .002
    vac_rest = float(air_to_vac(np.array([5000.]))[0])
    wave, flux, var = _spectrum(vac_rest * (1.+z))
    measured = measure_line(
        wave, flux, var, spec=LINE, redshift=z, wave_ref="vacuum",
        calsys_rel=None, calsys_tell=None, calsys_abs=None,
    )
    assert measured["detected"]
    assert measured["expected_wavelength"] == pytest.approx(vac_rest * (1.+z))
    assert abs(measured["centroid"]-vac_rest*(1.+z)) < .2


def test_science_line_medium_rejects_unknown_label():
    w, f, v = _spectrum(5000.)
    with pytest.raises(ValueError, match="medium is invalid"):
        _measure(w, f, v, medium="unknown")


def _fake_stage(monkeypatch, tmp_path, *, wave, flux, var, header):
    spec = {
        "wave": wave, "flux": flux, "var": var,
        "mask": np.zeros_like(wave, dtype=np.uint16),
        "header": header,
    }
    monkeypatch.setattr(stage, "read_spec1d", lambda *_args, **_kwargs: spec)
    monkeypatch.setattr(stage, "DEFAULT_LINES", (LINE,))
    monkeypatch.setattr(stage, "PRODUCT_CATALOG", SimpleNamespace(
        best_path=lambda _cfg, key, *, run_root: tmp_path / (key+".fits")
    ))
    return tmp_path / "stage"


def test_nm_vacuum_stage_measures_in_angstrom_and_correct_medium(monkeypatch, tmp_path):
    center = float(air_to_vac(np.asarray([5000.]))[0])
    wave_a, flux, var = _spectrum(center)
    hdr = fits.Header()
    hdr["WAVEREF"] = "vacuum"
    hdr["WAVEUNIT"] = "nm"
    hdr["REBNORM"] = "COUNT_V1"
    path = _fake_stage(monkeypatch, tmp_path, wave=wave_a/10.,
                       flux=flux, var=var, header=hdr)
    stage.run_line_measure({"science": {"redshift": 0.}}, run_root=tmp_path, stage_dir_path=path)
    payload = json.loads((path/"science_lines.json").read_text())
    assert payload["wavelength_input_unit"] == "nm"
    assert payload["measurement_wavelength_unit"] == "Angstrom"
    assert payload["coordinate_qualification"] == "EXPLICIT"
    row = payload["lines"][0]
    assert row["detected"]
    assert abs(row["velocity_kms"]) < .2
    assert row["wave_ref"] == "vacuum"


def test_count_v1_unknown_medium_is_rejected_by_stage(monkeypatch, tmp_path):
    wave, flux, var = _spectrum(5000.)
    hdr = fits.Header()
    hdr["WAVEUNIT"] = "Angstrom"
    hdr["REBNORM"] = "COUNT_V1"
    path = _fake_stage(monkeypatch, tmp_path, wave=wave,
                       flux=flux, var=var, header=hdr)
    with pytest.raises(ValueError, match="require declared WAVEREF"):
        stage.run_line_measure({}, run_root=tmp_path, stage_dir_path=path)


def test_science_stage_rejects_malformed_wavelength_unit(monkeypatch, tmp_path):
    wave, flux, var = _spectrum(5000.)
    hdr = fits.Header()
    hdr["WAVEREF"] = "air"
    hdr["WAVEUNIT"] = "meters"
    path = _fake_stage(monkeypatch, tmp_path, wave=wave,
                       flux=flux, var=var, header=hdr)
    with pytest.raises(ValueError, match="Unsupported"):
        stage.run_line_measure({}, run_root=tmp_path, stage_dir_path=path)
