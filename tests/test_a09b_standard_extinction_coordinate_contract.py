"""A09b: standard-star atmospheric law must use native air coordinates."""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.calib.common import convert_wavelength_convention
from scorpio_pipe.calib.sensfunc import _interpolate_standard_extinction_air


def test_standard_vacuum_nm_samples_recover_native_air_extinction():
    air = np.arange(5000.0, 5011.0)
    vacuum_nm = convert_wavelength_convention(
        air, from_ref="air", to_ref="vacuum",
    ) / 10.0
    curve = np.array([4900.0, 5100.0])
    k = np.array([0.31, 0.11])
    got = _interpolate_standard_extinction_air(
        vacuum_nm, wave_unit="nm", wave_ref="vacuum",
        extinction_wave_A=curve, k_lambda=k,
    )
    expected = np.interp(air, curve, k)
    np.testing.assert_allclose(got, expected, rtol=2e-9, atol=1e-11)


def test_air_angstrom_samples_interpolate_without_conversion():
    wave = 5000.0 + np.arange(5, dtype=float)
    got = _interpolate_standard_extinction_air(
        wave, wave_unit="Angstrom", wave_ref="air",
        extinction_wave_A=np.array([4900.0, 5100.0]),
        k_lambda=np.array([0.3, 0.1]),
    )
    np.testing.assert_allclose(got, 0.2 - (wave - 5000) / 1000.0)


@pytest.mark.parametrize("curve_wave,curve_k", [
    ([5000.0, 5000.0, 5100.0], [0.2, 0.2, 0.1]),
    ([5100.0, 5000.0], [0.1, 0.2]),
    ([5000.0, 5100.0], [0.2, -0.1]),
    ([5000.0, 5100.0], [0.2, float("nan")]),
])
def test_invalid_tabulated_extinction_curve_fails_closed(curve_wave, curve_k):
    with pytest.raises(ValueError, match="Invalid atmospheric extinction reference"):
        _interpolate_standard_extinction_air(
            np.arange(5020.0, 5025.0),
            wave_unit="Angstrom", wave_ref="air",
            extinction_wave_A=np.asarray(curve_wave),
            k_lambda=np.asarray(curve_k),
        )


def test_missing_extinction_wavelength_coverage_fails_instead_of_median_padding():
    with pytest.raises(ValueError, match="does not cover"):
        _interpolate_standard_extinction_air(
            np.arange(5020.0, 5025.0), wave_unit="Angstrom",
            wave_ref="air",
            extinction_wave_A=np.array([5021.0, 5023.0]),
            k_lambda=np.array([0.2, 0.19]),
        )


@pytest.mark.parametrize("medium,unit", [
    ("vac", "nm"),
    ("vacuum", "meter"),
    ("", "Angstrom"),
])
def test_undeclared_or_invalid_medium_units_block_standard_transfer(medium, unit):
    with pytest.raises(ValueError, match="explicit"):
        _interpolate_standard_extinction_air(
            np.arange(5020.0, 5025.0), wave_unit=unit,
            wave_ref=medium,
            extinction_wave_A=np.array([5000.0, 5100.0]),
            k_lambda=np.array([0.2, 0.1]),
        )



def _build_sensfunc_fixture(tmp_path, monkeypatch, *, curve_min=4900.0, curve_max=5100.0,
                            header_comments=True):
    from astropy.io import fits
    from types import SimpleNamespace
    from scorpio_pipe.calib import sensfunc as sf

    raw = tmp_path / "raw_std.fits"
    fits.PrimaryHDU().writeto(raw)
    observed_wave_nm = 500.0 + np.arange(24, dtype=float) * 0.1
    header = fits.Header({
        "WAVEUNIT": "nm",
        "WAVEREF": "vacuum",
        "REBNORM": "COUNT_V1",
        "BUNIT": "e-/s",
    })
    spec = {
        "wave": observed_wave_nm,
        "flux": np.full_like(observed_wave_nm, 100.0),
        "var": np.ones_like(observed_wave_nm),
        "mask": np.zeros_like(observed_wave_nm, dtype=np.uint16),
        "header": header,
        "waveref": "vacuum",
        "bunit": "e-/s",
    }
    ref = tmp_path / "reference.csv"
    ref.write_text(
        "wavelength_A,flux\n4800,10\n5200,10\n", encoding="utf-8",
    )
    ext = tmp_path / "extinction.csv"
    comments = (
        "# units: mag_per_airmass\n# wavelength_convention: air\n"
        if header_comments else ""
    )
    ext.write_text(
        comments + "wavelength_A,k_lambda\n"
        + f"{curve_min},0.30\n{curve_max},0.15\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sf, "read_spec1d", lambda *_a, **_kw: spec)
    monkeypatch.setattr(sf, "ensure_refctx", lambda *_a, **_kw: {"context_id": "test"})
    monkeypatch.setattr(
        sf, "_load_standard_reference",
        lambda *_a, **_kw: ({}, ref, "f_lambda", "air"),
    )
    monkeypatch.setattr(
        sf, "reference_path",
        lambda _cfg, *, exact=None, prefix=None: (
            ext if prefix and prefix.startswith("extinction_curve.") else None
        ),
    )
    monkeypatch.setattr(
        sf, "reference_metadata",
        lambda _cfg, *, exact=None, prefix=None: (
            {"wavelength_convention": "air", "units": "mag_per_airmass"}
            if prefix and prefix.startswith("extinction_curve.") else {}
        ),
    )
    monkeypatch.setattr(
        sf, "parse_frame_meta",
        lambda *_a, **_kw: SimpleNamespace(
            airmass=1.4,
            provenance_class=lambda key: "header",
            provenance_for=lambda key: "header",
        ),
    )
    qc = {"selected_standard": {"object_name": "test", "path": str(raw)}}
    return sf, qc


def test_full_sensfunc_invokes_native_air_extinction_with_vacuum_nm_standard(
    tmp_path, monkeypatch,
):
    sf, qc = _build_sensfunc_fixture(tmp_path, monkeypatch)
    result = sf.build_sensfunc(
        {}, standard_spec_path=tmp_path / "unused_spec.fits", standard_qc=qc,
    )
    assert result["qc"]["extinction_applied"] is True
    assert result["qc"]["fit_points"] >= 8
    assert np.isfinite(result["response_abs"]).all()
    assert np.all(result["response_abs"] > 0)


def test_full_sensfunc_refuses_extinction_extrapolation(
    tmp_path, monkeypatch,
):
    sf, qc = _build_sensfunc_fixture(
        tmp_path, monkeypatch, curve_min=5005.0, curve_max=5010.0,
    )
    with pytest.raises(ValueError, match="does not cover standard"):
        sf.build_sensfunc(
            {}, standard_spec_path=tmp_path / "unused_spec.fits", standard_qc=qc,
        )


def test_full_sensfunc_requires_source_file_unit_declarations(
    tmp_path, monkeypatch,
):
    sf, qc = _build_sensfunc_fixture(
        tmp_path, monkeypatch, header_comments=False,
    )
    with pytest.raises(ValueError, match="must declare air / mag_per_airmass"):
        sf.build_sensfunc(
            {}, standard_spec_path=tmp_path / "unused_spec.fits", standard_qc=qc,
        )
