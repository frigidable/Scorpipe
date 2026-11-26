"""A10: a fixed window integral is not a deblended line-component flux."""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.calib.common import air_to_vacuum
from scorpio_pipe.science.lines import (
    DEFAULT_LINES, LineSpec, _window_overlap_partners, measure_lines,
)
from scorpio_pipe.science.reddening import balmer_decrement_summary


def _data():
    wave = np.arange(4800.0, 6810.0, 0.25)
    flux = np.ones_like(wave)
    var = np.full_like(wave, 0.0025)
    for center, amplitude in (
        (4861.33, 20.0), (6562.80, 30.0),
        (6548.05, 9.0), (6583.45, 27.0),
        (5006.84, 18.0), (6716.44, 7.0), (6730.82, 8.0),
    ):
        flux += amplitude * np.exp(-0.5 * ((wave - center) / 1.5) ** 2)
    return wave, flux, var


def _rows(select=None, **kwargs):
    wave, flux, var = _data()
    return measure_lines(
        wave, flux, var,
        calsys_rel=None, calsys_tell=None, calsys_abs=None,
        lines=DEFAULT_LINES if select is None else select,
        **kwargs,
    )


def test_halpha_nii_integration_windows_are_blended_not_component_flux():
    measured = {row["name"]: row for row in _rows()}
    ha = measured["Halpha"]
    assert ha["detected"] is True
    assert ha["blend_flag"] is True
    assert set(ha["blend_partners"]) >= {"[NII]6548", "[NII]6583"}
    assert ha["flux_interpretation"] == "BLENDED_DIAGNOSTIC_ONLY"
    assert ha["component_flux_qualified"] is False
    assert ha["flux_kind"] == "window_integral_not_deblended"
    for name in ("[NII]6548", "[NII]6583"):
        assert measured[name]["blend_flag"] is True
        assert measured[name]["component_flux_qualified"] is False


def test_sii_doublet_windows_are_not_independent_component_fluxes():
    measured = {row["name"]: row for row in _rows()}
    for name, partner in (
        ("[SII]6716", "[SII]6731"),
        ("[SII]6731", "[SII]6716"),
    ):
        row = measured[name]
        assert partner in row["blend_partners"]
        assert row["blend_flag"] is True
        assert row["blend_warning"] is True
        assert row["component_flux_qualified"] is False
        assert row["flux_kind"] == "window_integral_not_deblended"


def test_unblended_window_is_still_only_diagnostic():
    measured = {row["name"]: row for row in _rows()}
    oi = measured["[OIII]5007"]
    assert oi["detected"] is True
    assert oi["blend_partners"] == []
    assert oi["blend_flag"] is False
    assert oi["blend_window_overlap_free"] is True
    # Lack of catalogued overlap is NOT evidence of qualified line flux.
    assert oi["component_flux_qualified"] is False
    assert oi["flux_interpretation"] == "ISOLATED_WINDOW_DIAGNOSTIC"


def test_requesting_halpha_alone_still_checks_unrequested_nii_catalogue():
    selected = (next(s for s in DEFAULT_LINES if s.name == "Halpha"),)
    row = _rows(select=selected)[0]
    assert row["name"] == "Halpha"
    assert set(row["blend_partners"]) >= {"[NII]6548", "[NII]6583"}
    assert row["component_flux_qualified"] is False


@pytest.mark.parametrize("medium", ["air", "vacuum"])
def test_blend_classification_respects_declared_wavelength_medium(medium):
    halpha = next(s for s in DEFAULT_LINES if s.name == "Halpha")
    partners = _window_overlap_partners(
        halpha, DEFAULT_LINES, redshift=0.015, waveref=medium,
    )
    assert set(partners) >= {"[NII]6548", "[NII]6583"}
    assert "[SII]6716" not in partners


def test_one_custom_requested_line_is_compared_to_default_catalogue():
    custom = LineSpec("Halpha_custom", 6562.8, 18.0, (-60., -25.), (25., 60.))
    row = _rows(select=(custom,))[0]
    assert "Halpha" in row["blend_partners"]
    assert "[NII]6583" in row["blend_partners"]


def test_vacuum_observed_spectrum_retains_blend_classification():
    wave, flux, var = _data()
    vac_wave = air_to_vacuum(wave)
    rows = measure_lines(
        vac_wave, flux, var,
        calsys_rel=None, calsys_tell=None, calsys_abs=None,
        lines=tuple(s for s in DEFAULT_LINES if s.name == "Halpha"),
        waveref="vacuum",
    )
    assert len(rows) == 1
    assert "[NII]6583" in rows[0]["blend_partners"]


def test_balmer_decrement_rejects_numerical_window_fluxes():
    measured = {row["name"]: row for row in _rows()}
    assert measured["Halpha"]["detected"] is True
    assert measured["Hbeta"]["detected"] is True
    assert balmer_decrement_summary(measured) is None


def test_independently_deblended_external_line_fluxes_are_not_blocked():
    external = {
        "Halpha": {
            "flux": 8.58, "detected": True, "blend_warning": False,
            "blend_flag": False, "component_flux_qualified": True,
        },
        "Hbeta": {
            "flux": 3.0, "detected": True, "blend_warning": False,
            "blend_flag": False, "component_flux_qualified": True,
        },
    }
    result = balmer_decrement_summary(external)
    assert result is not None
    assert result["line_ratio_Ha_Hb"] == pytest.approx(2.86)


def test_explicit_unqualified_component_vetoes_physical_ratio():
    external = {
        "Halpha": {"flux": 8.58, "detected": True, "component_flux_qualified": False},
        "Hbeta": {"flux": 3.0, "detected": True, "component_flux_qualified": True},
    }
    assert balmer_decrement_summary(external) is None


@pytest.mark.parametrize("z", [float("nan"), -1.0, float("inf")])
def test_invalid_redshift_refused_in_blend_catalogue(z):
    halpha = next(s for s in DEFAULT_LINES if s.name == "Halpha")
    with pytest.raises(ValueError, match="redshift"):
        _window_overlap_partners(halpha, DEFAULT_LINES, redshift=z, waveref="air")


def test_unmeasurable_line_has_overlap_diagnostic_but_no_claim_of_integral():
    select = (next(s for s in DEFAULT_LINES if s.name == "Halpha"),)
    wave = np.arange(4820.0, 4910.0, 0.5)
    flux = np.ones_like(wave)
    var = np.ones_like(wave)
    row = measure_lines(
        wave, flux, var,
        calsys_rel=None, calsys_tell=None, calsys_abs=None,
        lines=select,
    )[0]
    assert row["detected"] is False
    assert row["reason"] == "insufficient_sampling"
    assert row["fit_method"] == "NOT_MEASURED"
    assert row["flux_interpretation"] == "UNMEASURED"
    assert "[NII]6583" in row["blend_partners"]


def test_real_stage_keeps_blended_candidates_separate_from_formal_detections(
    tmp_path, monkeypatch,
):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from astropy.io import fits
    from scorpio_pipe.stages import line_measure as stage

    wave, flux, var = _data()
    spec = {
        "wave": wave, "flux": flux, "var": var,
        "mask": np.zeros(wave.size, dtype=np.uint16),
        "header": fits.Header({
            "WAVEREF": "air", "WAVEUNIT": "Angstrom",
            "REBNORM": "COUNT_V1", "STKVARQ": "COND_ONLY",
        }),
        "dq1d_version": None,
    }
    paths = {
        "fluxcal_spec1d_fits": tmp_path / "science.fits",
        "calibration_systematics_fits": tmp_path / "sys.fits",
        "fluxcal_qc_json": tmp_path / "fluxcal.json",
    }
    monkeypatch.setattr(
        stage, "PRODUCT_CATALOG",
        SimpleNamespace(best_path=lambda _cfg, key, **_kw: paths[key]),
    )
    monkeypatch.setattr(stage, "read_spec1d", lambda *_a, **_kw: spec)
    z = np.zeros_like(wave)
    monkeypatch.setattr(
        stage, "_read_calsys",
        lambda *_a, **_kw: {"rel": z, "tell": z, "abs": z},
    )

    def preview(path, *_args, **_kwargs):
        p = Path(path)
        p.write_bytes(b"synthetic")
        return p

    monkeypatch.setattr(stage, "plot_line_measurements_png", preview)
    monkeypatch.setattr(stage, "write_done_json", lambda **_kw: None)

    out = tmp_path / "line_measure"
    stage.run_line_measure({}, run_root=tmp_path, stage_dir_path=out)
    payload = json.loads((out / "science_lines.json").read_text(encoding="utf-8"))
    qc = json.loads((out / "line_measure_plan.json").read_text(encoding="utf-8"))["qc"]
    rows = {x["name"]: x for x in payload["lines"]}

    assert payload["window_flux_qualification"] == "DIAGNOSTIC_ONLY_NOT_DEBLENDED"
    assert payload["n_blended_window_candidates"] >= 5
    assert payload["n_overlap_free_window_candidates"] >= 1
    assert rows["Halpha"]["conditional_detection_candidate"] is True
    assert rows["Halpha"]["detected"] is False
    assert rows["Halpha"]["component_flux_qualified"] is False
    assert rows["[SII]6716"]["blend_flag"] is True
    assert qc["n_blended_window_detections"] == 0
    assert qc["n_blended_window_candidates"] >= 5
    assert qc["n_lines_detected"] == 0
    assert qc["window_flux_qualification"] == "DIAGNOSTIC_ONLY_NOT_DEBLENDED"
