"""A11b: ARC LSF must be invariant to nm/Angstrom lambda-map storage.

This is an engineering test on synthetic ARC profiles. It does not qualify
the detector-geometry LSF for rectified or extracted science spectra.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.stages.lsf_model import (
    _lambda_map_as_angstrom, run_lsf_model,
)
from scorpio_pipe.wavesol_paths import wavesol_dir


@pytest.mark.parametrize("unit,factor", [("Angstrom", 1.0), ("nm", 10.0)])
def test_native_lambda_map_unit_conversion(unit, factor):
    original = np.array([[5012.0, 5023.0]]) / factor
    header = fits.Header({"WAVEUNIT": unit})
    out, native, measured_factor = _lambda_map_as_angstrom(original, header)
    np.testing.assert_allclose(out, [[5012.0, 5023.0]])
    assert native == unit
    assert measured_factor == factor


@pytest.mark.parametrize("unit", ["", "micron", "Hz", "meters"])
def test_unsupported_native_unit_fails_closed(unit):
    hdr = fits.Header()
    if unit:
        hdr["WAVEUNIT"] = unit
    with pytest.raises(ValueError, match="explicit wavelength-map unit"):
        _lambda_map_as_angstrom(np.full((2, 3), 5000.0), hdr)


def test_nonpositive_lambda_map_fails_closed():
    with pytest.raises(ValueError, match="positive physical support"):
        _lambda_map_as_angstrom(
            np.full((2, 3), -5000.0),
            fits.Header({"WAVEUNIT": "Angstrom"}),
        )


def _run_synthetic_arc(tmp_path, *, unit):
    work = tmp_path / unit
    cfg = {
        "work_dir": str(work),
        "lsf_model": {
            "fit_half_window_px": 8,
            "min_line_snr": 3.0,
            "min_line_measures": 6,
            "expected_arc_fwhm_A": 2.83,
        },
    }
    wdir = wavesol_dir(cfg, run_root=work)
    wdir.mkdir(parents=True, exist_ok=True)

    ny, nx = 16, 155
    lam_A = 5000.0 + 0.8 * np.arange(nx)[None, :] + 0.01 * np.arange(ny)[:, None]
    native = lam_A / 10.0 if unit == "nm" else lam_A
    hdr = fits.Header()
    hdr["WAVEUNIT"] = unit
    hdr["CUNIT1"] = unit
    hdr["WAVEREF"] = "air"
    fits.PrimaryHDU(data=np.asarray(native, dtype=np.float32), header=hdr).writeto(
        wdir / "lambda_map.fits"
    )

    rng = np.random.default_rng(841)
    arc = rng.normal(0.0, 0.12, size=(ny, nx))
    lines = [5030.0, 5060.0, 5090.0]
    for iy in range(ny):
        for lam in lines:
            centroid_px = (lam - (5000.0 + 0.01 * iy)) / 0.8
            arc[iy, :] += 125.0 * np.exp(
                -0.5 * ((np.arange(nx) - centroid_px) / 1.5) ** 2
            )
    ah = fits.Header()
    ah["DISPERSE"] = "VPHG1200@540"
    ah["SLITWID"] = 1.0
    fits.PrimaryHDU(data=arc.astype(np.float32), header=ah).writeto(
        wdir / "superneon.fits"
    )
    (wdir / "hand_pairs.txt").write_text(
        "".join(f"{(lam - 5000.0) / 0.8:.2f} {lam:.3f}\n" for lam in lines),
        encoding="utf-8",
    )

    result = run_lsf_model(cfg)
    with fits.open(result["lsf_map"]) as hdul:
        width_map = np.array(hdul["SCI"].data, copy=True)
        out_header = hdul[0].header.copy()
    with open(work / "lsf_model" / "lsf_model.json", encoding="utf-8") as f:
        model = json.load(f)
    with open(work / "lsf_model" / "lsf_qc.json", encoding="utf-8") as f:
        qc = json.load(f)
    return width_map, out_header, model, qc


def test_nm_and_angstrom_produce_same_measured_arc_lsf(tmp_path):
    sigma_A, header_A, model_A, qc_A = _run_synthetic_arc(
        tmp_path, unit="Angstrom"
    )
    sigma_nm, header_nm, model_nm, qc_nm = _run_synthetic_arc(
        tmp_path, unit="nm"
    )
    assert model_nm["source"] == model_A["source"] == "arc_measured"
    assert model_nm["instrument"]["native_wave_unit"] == "nm"
    assert model_nm["instrument"]["native_to_A_factor"] == 10.0
    assert model_nm["instrument"]["wave_unit"] == "Angstrom"
    assert header_nm["WAVEUNIT"] == "Angstrom"
    assert header_nm["CUNIT1"] == "Angstrom"
    assert header_nm["NATWUNIT"] == "nm"
    assert header_nm["LSFGEOM"] == "DETECTOR"
    assert header_nm["LSFSCI"] is False
    assert header_nm["LSFQUAL"] is False
    assert qc_nm["science_lsf_validated"] is False
    np.testing.assert_allclose(sigma_nm, sigma_A, rtol=0.02, atol=0.025)
    assert 0.4 < float(np.nanmedian(sigma_nm)) < 2.0
    np.testing.assert_allclose(
        model_nm["summary_1d"]["lambda_A"],
        model_A["summary_1d"]["lambda_A"],
        rtol=1e-5, atol=0.015,
    )
    np.testing.assert_allclose(
        model_nm["summary_1d"]["sigma_A"],
        model_A["summary_1d"]["sigma_A"],
        rtol=0.02, atol=0.025,
    )
