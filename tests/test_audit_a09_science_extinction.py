"""A09: independent standard/science atmospheric transfer with safe provenance."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.calib import fluxcal as fc
from scorpio_pipe.calib.common import convert_wavelength_convention
from scorpio_pipe.stages import fluxcal_apply as stage
from scorpio_pipe.workspace_paths import stage_dir


def _spec_and_sens(tmp_path, *, x=1.4):
    wave = 5000.0 + np.arange(24, dtype=float)
    k = 0.17 + 0.002 * (wave - 5000.0)
    response = np.full(wave.size, 4.0)
    physical = 30.0 + 0.1 * (wave - 5000.0)
    counts = physical * response * 10.0 ** (-0.4 * k * x)
    header = fits.Header({
        "BUNIT": "e-/s",
        "WAVEREF": "air",
        "WAVEUNIT": "Angstrom",
        "REBNORM": "COUNT_V1",
    })

    def make_spec(name, signal, variance):
        return fits.BinTableHDU.from_columns([
            fits.Column(name="LAMBDA", format="D", array=wave),
            fits.Column(name=f"FLUX_{name}", format="D", array=signal),
            fits.Column(name=f"VAR_{name}", format="D", array=variance),
            fits.Column(
                name=f"MASK_{name}", format="J",
                array=np.zeros(wave.size, dtype=np.int32),
            ),
            fits.Column(name=f"SKY_{name}", format="D", array=signal * 0.1),
            fits.Column(
                name=f"NPIX_{name}", format="I",
                array=np.full(wave.size, 3, dtype=np.int16),
            ),
        ], name="SPEC_" + name)

    source = tmp_path / "science.fits"
    fits.HDUList([
        fits.PrimaryHDU(header=header),
        make_spec("TRACE", counts, np.full_like(wave, 4.0)),
        make_spec("FIXED", counts * 1.2, np.full_like(wave, 9.0)),
    ]).writeto(source)

    sh = fits.Header({
        "WAVEREF": "air", "INBUNIT": "e-/s",
        "INREBNM": "COUNT_V1", "INWUNIT": "Angstrom", "INSTEP": 1.0,
    })
    sens = tmp_path / "sens.fits"
    fits.HDUList([
        fits.PrimaryHDU(header=sh),
        fits.BinTableHDU.from_columns([
            fits.Column(name="WAVELENGTH", format="D", array=wave),
            fits.Column(name="RESPONSE_ABS", format="D", array=response),
            fits.Column(
                name="RESPONSE_UNC", format="D",
                array=np.zeros_like(wave),
            ),
        ], name="SENSFUNC"),
    ]).writeto(sens)
    curve = {
        "airmass": x, "provenance": "header",
        "wavelength_ref": "air",
        "wavelength_A": np.array([4950.0, 5100.0]),
        "k_lambda": np.array([0.07, 0.37]),
    }
    standard_qc = {"safe_for": "absolute_candidate"}
    sens_qc = {"extinction_applied": True, "airmass_provenance": "header"}
    return source, sens, wave, k, counts, physical, curve, standard_qc, sens_qc


def test_science_airmass_corrects_trace_fixed_variances_and_sky(tmp_path):
    source, sens, wave, k, counts, physical, curve, std, sq = _spec_and_sens(tmp_path)
    result = fc.apply_fluxcal_to_spec1d(
        source, sensfunc_path=sens, standard_qc=std, sensfunc_qc=sq,
        science_extinction=curve,
    )
    factor = 10.0 ** (0.4 * k * curve["airmass"])
    np.testing.assert_allclose(result["science_extinction_factor"], factor, rtol=1e-12)
    np.testing.assert_allclose(result["flux"], physical, rtol=1e-12)
    np.testing.assert_allclose(result["fixed"]["flux"], physical * 1.2, rtol=1e-12)
    np.testing.assert_allclose(result["var"], 4.0 / 16.0 * factor**2, rtol=1e-12)
    np.testing.assert_allclose(result["fixed"]["var"], 9.0 / 16.0 * factor**2, rtol=1e-12)
    np.testing.assert_allclose(result["sky"], counts * 0.1 / 4.0 * factor, rtol=1e-12)
    assert result["science_extinction_applied"] is True
    assert result["science_airmass"] == pytest.approx(1.4)
    # Full science qualification requires additional calibration validation.
    assert result["status"] == "ABSOLUTE_APPROX"


def test_uncorrected_science_remains_absolute_approx(tmp_path):
    source, sens, wave, k, counts, physical, curve, std, sq = _spec_and_sens(tmp_path)
    result = fc.apply_fluxcal_to_spec1d(
        source, sensfunc_path=sens, standard_qc=std, sensfunc_qc=sq,
    )
    assert result["science_extinction_applied"] is False
    assert result["status"] == "ABSOLUTE_APPROX"
    np.testing.assert_allclose(result["flux"], counts / 4.0)


@pytest.mark.parametrize("bad_x", [0.5, 6.0, float("nan")])
def test_invalid_airmass_is_not_silently_coerced(tmp_path, bad_x):
    source, sens, wave, k, counts, physical, curve, std, sq = _spec_and_sens(tmp_path)
    curve["airmass"] = bad_x
    with pytest.raises(ValueError, match="Science airmass"):
        fc.apply_fluxcal_to_spec1d(
            source, sensfunc_path=sens, standard_qc=std,
            sensfunc_qc=sq, science_extinction=curve,
        )


def test_extinction_curve_must_fully_cover_science_wavelengths(tmp_path):
    source, sens, wave, k, counts, physical, curve, std, sq = _spec_and_sens(tmp_path)
    curve["wavelength_A"] = np.array([5005.0, 5010.0])
    curve["k_lambda"] = np.array([0.18, 0.19])
    with pytest.raises(ValueError, match="does not cover"):
        fc.apply_fluxcal_to_spec1d(
            source, sensfunc_path=sens, standard_qc=std,
            sensfunc_qc=sq, science_extinction=curve,
        )


def test_science_correction_requires_corrected_standard(tmp_path):
    source, sens, wave, k, counts, physical, curve, std, sq = _spec_and_sens(tmp_path)
    sq["extinction_applied"] = False
    with pytest.raises(ValueError, match="separately corrected standard"):
        fc.apply_fluxcal_to_spec1d(
            source, sensfunc_path=sens, standard_qc=std,
            sensfunc_qc=sq, science_extinction=curve,
        )


def test_vacuum_nm_axis_uses_air_angstrom_curve_without_mixing_units():
    wavelengths_air = np.arange(5000.0, 5011.0)
    wavelengths_vac_nm = convert_wavelength_convention(
        wavelengths_air, from_ref="air", to_ref="vacuum",
    ) / 10.0
    curve = {
        "airmass": 1.3,
        "wavelength_ref": "air",
        "wavelength_A": np.array([4950.0, 5100.0]),
        "k_lambda": np.array([0.07, 0.37]),
    }
    got = fc._science_extinction_factor(
        wavelengths_vac_nm,
        fits.Header({"WAVEREF": "vacuum", "WAVEUNIT": "nm"}),
        curve,
    )
    expected = 10.0 ** (0.4 * 1.3 * (0.17 + 0.002 * (wavelengths_air - 5000.0)))
    np.testing.assert_allclose(got, expected, rtol=1e-9)


def test_undeclared_reference_medium_is_not_assumed():
    h = fits.Header({"WAVEREF": "air", "WAVEUNIT": "Angstrom"})
    curve = {
        "airmass": 1.3,
        "wavelength_A": np.array([4900.0, 5100.0]),
        "k_lambda": np.array([0.1, 0.2]),
    }
    with pytest.raises(ValueError, match="declared AIR"):
        fc._science_extinction_factor(np.arange(5000.0, 5005.0), h, curve)


def _lineage_fixture(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    raw = work / "s22800607.fts"
    fits.PrimaryHDU().writeto(raw)
    sd = stage_dir(work, "stack2d")
    sd.mkdir(parents=True, exist_ok=True)
    stack_path = sd / "stack2d.fits"
    stack_path.write_bytes(b"synthetic report-linked path")
    src = work / "spec1d.fits"
    fits.PrimaryHDU(header=fits.Header({"INPUT2D": str(stack_path)})).writeto(src)
    report = {
        "ok": True,
        "n_inputs": 1,
        "inputs": [{
            "file": "s22800607_skysub.fits",
            "path": str(work / "linearize" / "s22800607_skysub.fits"),
            "rejected": False,
        }],
        "outputs": {"stack2d_fits": str(stack_path)},
    }
    (sd / "stack2d_done.json").write_text(json.dumps(report), encoding="utf-8")
    atmosphere = work / "extinction.csv"
    atmosphere.write_text(
        "wavelength_A,k_lambda\n4900,0.2\n5100,0.25\n", encoding="utf-8",
    )
    import scorpio_pipe.instruments as inst
    import scorpio_pipe.calib.common as common
    monkeypatch.setattr(
        inst, "parse_frame_meta",
        lambda *_a, **_kw: SimpleNamespace(
            airmass=1.3, provenance_class=lambda key: "header",
        ),
    )
    monkeypatch.setattr(
        common, "reference_path", lambda *_a, **_kw: atmosphere,
    )
    cfg = {"work_dir": str(work), "frames": {"obj": [str(raw)]}}
    return cfg, src, work, sd


def test_provenance_gated_single_exposure_admission(tmp_path, monkeypatch):
    cfg, src, work, sd = _lineage_fixture(tmp_path, monkeypatch)
    model = stage._single_science_airmass_model(
        cfg, science_spec_path=src, run_root=work,
    )
    assert model["airmass"] == pytest.approx(1.3)
    assert model["provenance"] == "header"
    assert model["lineage_qualification"] == "SINGLE_STACK_STEM_MATCH_ONLY"
    assert model["wavelength_ref"] == "air"
    assert model["wavelength_A"].shape == (2,)


def test_multiple_raw_science_frames_cannot_use_mean_airmass(tmp_path, monkeypatch):
    cfg, src, work, sd = _lineage_fixture(tmp_path, monkeypatch)
    cfg["frames"]["obj"].append(cfg["frames"]["obj"][0])
    assert stage._single_science_airmass_model(
        cfg, science_spec_path=src, run_root=work,
    ) is None


def test_stack_report_and_extract_source_must_match(tmp_path, monkeypatch):
    cfg, src, work, sd = _lineage_fixture(tmp_path, monkeypatch)
    with fits.open(src, mode="update") as h:
        h[0].header["INPUT2D"] = str(work / "different.fits")
        h.flush()
    assert stage._single_science_airmass_model(
        cfg, science_spec_path=src, run_root=work,
    ) is None


def test_mismatched_raw_stem_is_not_accepted(tmp_path, monkeypatch):
    cfg, src, work, sd = _lineage_fixture(tmp_path, monkeypatch)
    bad = sd / "stack2d_done.json"
    payload = json.loads(bad.read_text(encoding="utf-8"))
    payload["inputs"][0]["file"] = "other_skysub.fits"
    payload["inputs"][0]["path"] = str(work / "linearize" / "other_skysub.fits")
    bad.write_text(json.dumps(payload), encoding="utf-8")
    assert stage._single_science_airmass_model(
        cfg, science_spec_path=src, run_root=work,
    ) is None


def test_written_2d_1d_products_have_consistent_atmospheric_transfer(tmp_path, monkeypatch):
    source, sens, wave, k, counts, physical, curve, std, sq = _spec_and_sens(tmp_path)
    result = fc.apply_fluxcal_to_spec1d(
        source, sensfunc_path=sens, standard_qc=std, sensfunc_qc=sq,
        science_extinction=curve,
    )
    from scorpio_pipe.io.mef import WaveGrid, write_sci_var_mask
    stack = tmp_path / "stack2d.fits"
    write_sci_var_mask(
        stack, np.tile(counts, (4, 1)),
        var=np.full((4, wave.size), 4.0),
        mask=np.zeros((4, wave.size), dtype=np.uint16),
        grid=WaveGrid(float(wave[0]), 1.0, wave.size, unit="Angstrom", wave_ref="air"),
        header=fits.Header({"BUNIT": "e-/s"}),
    )
    monkeypatch.setattr(
        fc, "PRODUCT_CATALOG",
        SimpleNamespace(best_path=lambda *_a, **_kw: stack),
    )
    outputs = fc.write_fluxcal_products(
        {"work_dir": str(tmp_path)},
        tmp_path / "out",
        spec_payload=result, sensfunc_path=sens,
        telluric_path=None, reference_context_id="a09-test",
        science_spec_path=source, standard_qc=std,
    )
    with fits.open(outputs["fluxcal_spec1d_fits"], memmap=False) as h:
        assert h[0].header["ATMCORR"] is True
        assert h[0].header["ATMXSCI"] == pytest.approx(1.4)
        assert h[0].header["ATMPROV"] == "header"
        np.testing.assert_allclose(
            h["SPEC_TRACE"].data["FLUX_TRACE"], physical, rtol=1e-6,
        )
        np.testing.assert_allclose(
            h["SPEC_FIXED"].data["FLUX_FIXED"], physical * 1.2, rtol=1e-6,
        )
    with fits.open(outputs["fluxcal_2d_fits"], memmap=False) as h:
        assert h[0].header["ATMCORR"] is True
        np.testing.assert_allclose(h["SCI"].data, np.tile(physical, (4, 1)), rtol=2e-6)
        np.testing.assert_allclose(
            h["VAR"].data,
            np.tile(4.0 / 16.0 * 10.0 ** (0.8 * k * 1.4), (4, 1)),
            rtol=2e-6,
        )
    qc = json.loads(Path(outputs["fluxcal_qc_json"]).read_text("utf-8"))
    assert qc["science_extinction_applied"] is True
    assert qc["science_airmass"] == pytest.approx(1.4)
    assert qc["status"] == "ABSOLUTE_APPROX"
