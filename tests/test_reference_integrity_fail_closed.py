"""Reference integrity and fail-closed calibration gates (science P0).

These tests validate physical trust boundaries, not only stable numerical output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import scorpio_pipe.calib.sensfunc as sf
import scorpio_pipe.calib.telluric as tl
import scorpio_pipe.calib.fluxcal as fc
import scorpio_pipe.stages.standard_prepare as sp
from scorpio_pipe.calib.common import read_small_csv


def _packaged_resources() -> Path:
    return Path(sf.__file__).resolve().parents[1] / "resources"


def test_bundled_four_point_standard_is_not_certified(monkeypatch):
    lib = _packaged_resources() / "standards" / "library.yaml"
    monkeypatch.setattr(sf, "reference_path", lambda cfg, **kw: lib)
    with pytest.raises(ValueError, match="UNVALIDATED_STANDARD_REFERENCE"):
        sf._load_standard_reference({}, selected_name="BD+28 4211")


@pytest.mark.parametrize("identity", [None, "", "Feige 110", "HZ 44"])
def test_sensfunc_refuses_missing_or_unmatched_standard_identity(identity, tmp_path, monkeypatch):
    lib = tmp_path / "standards" / "library.yaml"
    lib.parent.mkdir()
    lib.write_text(
        "standards:\n  bd284211:\n"
        "    display_name: BD+28 4211\n"
        "    table: standards/reference.csv\n"
        "    scientific_status: validated\n"
        "    source: CALSPEC-pinned-example\n"
        "    units: f_lambda\n"
        "    wavelength_convention: air\n",
        encoding="utf-8",
    )
    (lib.parent / "reference.csv").write_text(
        "wavelength_A,flux\n4000,1e-13\n5000,1e-13\n", encoding="utf-8"
    )
    monkeypatch.setattr(sf, "reference_path", lambda cfg, **kw: lib)
    with pytest.raises(ValueError, match="STANDARD_IDENTITY_MISSING|UNKNOWN_SPECTROPHOTOMETRIC_STANDARD"):
        sf._load_standard_reference({}, selected_name=identity)


def test_validated_identified_standard_resolves_exact_named_table(tmp_path, monkeypatch):
    root = tmp_path / "references"
    d = root / "standards"
    d.mkdir(parents=True)
    lib = d / "library.yaml"
    lib.write_text(
        "standards:\n  science_std:\n"
        "    display_name: Science Standard\n"
        "    table: standards/verified.csv\n"
        "    scientific_status: validated\n"
        "    source: pinned-CALSPEC-example\n"
        "    units: f_lambda\n"
        "    wavelength_convention: vacuum\n",
        encoding="utf-8",
    )
    table = d / "verified.csv"
    table.write_text("wavelength_A,flux\n4000,1e-13\n5000,1e-13\n", encoding="utf-8")
    monkeypatch.setattr(sf, "reference_path", lambda cfg, **kw: lib)
    _library, selected, units, medium = sf._load_standard_reference(
        {}, selected_name="Science Standard"
    )
    assert selected == table.resolve()
    assert units == "f_lambda"
    assert medium == "vacuum"


def test_missing_named_table_does_not_fallback_to_unrelated_context_reference(
    tmp_path, monkeypatch
):
    root = tmp_path / "standards"
    root.mkdir()
    lib = root / "library.yaml"
    lib.write_text(
        "standards:\n  science_std:\n"
        "    display_name: Science Standard\n"
        "    table: standards/does-not-exist.csv\n"
        "    scientific_status: validated\n"
        "    source: provenance-checked\n"
        "    units: f_lambda\n"
        "    wavelength_convention: air\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sf, "reference_path", lambda cfg, **kw: tmp_path / "other.csv" if kw.get("prefix") else lib)
    with pytest.raises(FileNotFoundError, match="STANDARD_REFERENCE_TABLE_MISSING"):
        sf._load_standard_reference({}, selected_name="Science Standard")


def test_diagnostic_telluric_has_unity_transmission_between_oxygen_bands(monkeypatch):
    resources = _packaged_resources()
    mask = resources / "telluric" / "default_mask.csv"
    template = resources / "telluric" / "default_template.csv"
    wave = np.arange(6750., 7751.)
    spec = {
        "wave": wave,
        "flux": np.full(wave.size, 50.),
        "var": np.full(wave.size, 1.),
        "waveref": "air",
    }
    monkeypatch.setattr(tl, "read_spec1d", lambda *args, **kw: spec)
    monkeypatch.setattr(tl, "ensure_refctx", lambda cfg: {"context_id": "test"})
    monkeypatch.setattr(
        tl, "reference_path",
        lambda cfg, **kw: mask if kw.get("exact") == "telluric_mask.default"
        else template if kw.get("exact") == "telluric_template.default" else None,
    )
    monkeypatch.setattr(tl, "reference_metadata", lambda cfg, **kw: {})
    payload = tl.build_telluric_model({}, standard_spec_path="mock.fits")
    gap = (wave >= 7100.) & (wave <= 7450.)
    np.testing.assert_array_equal(payload["transmission"][gap], 1.)
    np.testing.assert_array_equal(payload["confidence"][gap], 0.)
    np.testing.assert_array_equal(payload["model_mask"][gap], 0)
    assert payload["qc"]["science_qualified"] is False
    assert payload["qc"]["template_applied"] is False
    assert payload["qc"]["template_ignored"] is True


def test_diagnostic_telluric_fits_cannot_be_applied_to_science(tmp_path):
    from scorpio_pipe.calib.common import write_bintable_fits
    model = write_bintable_fits(
        tmp_path / "telluric.fits",
        primary_header=fits.Header({"TELQUAL": "DIAGNOSTIC", "WAVEREF": "air"}),
        extname="TELLURIC_MODEL",
        columns={
            "WAVELENGTH": np.array([5000., 5001., 5002.]),
            "TRANSMISSION": np.array([0.8, 0.9, 1.]),
            "CONFIDENCE": np.array([1., 1., 1.]),
        },
    )
    assert fc._load_telluric(model)["quality"] == "DIAGNOSTIC"


def test_requested_masterflat_missing_aborts_before_rectification(tmp_path, monkeypatch):
    raw = tmp_path / "standard.fits"
    raw.write_bytes(b"dummy")
    monkeypatch.setattr(sp, "_try_existing_rectified", lambda *args, **kw: None)
    monkeypatch.setattr(sp, "_resolve_superflat_path", lambda *args, **kw: tmp_path / "absent.fits")
    with pytest.raises(FileNotFoundError, match="STANDARD_FLATFIELD_REQUIRED"):
        sp._prepare_rectified_standard(
            {"fluxcal": {"standard_apply_flat": True}},
            run_root=tmp_path,
            stage_dir_path=tmp_path / "prepare",
            raw_path=raw,
        )


def test_diagnostic_telluric_application_rejected_for_actual_science_spectrum(tmp_path):
    """The numerical science API must reject tagged diagnostic transmission."""
    from scorpio_pipe.calib.common import write_bintable_fits

    wave = np.array([5000., 5001., 5002.])
    sci = tmp_path / "science.fits"
    fits.HDUList([
        fits.PrimaryHDU(header=fits.Header({"BUNIT": "ADU", "WAVEREF": "air"})),
        fits.BinTableHDU.from_columns([
            fits.Column(name="LAMBDA", format="D", array=wave),
            fits.Column(name="FLUX_TRACE", format="E", array=np.array([10., 10., 10.])),
            fits.Column(name="VAR_TRACE", format="E", array=np.array([1., 1., 1.])),
            fits.Column(name="MASK_TRACE", format="J", array=np.array([0, 0, 0])),
        ], name="SPEC_TRACE"),
    ]).writeto(sci)

    sens = write_bintable_fits(
        tmp_path / "sens.fits",
        primary_header=fits.Header({"WAVEREF": "air"}),
        extname="SENSFUNC",
        columns={
            "WAVELENGTH": wave,
            "RESPONSE_ABS": np.full(3, 2.),
            "RESPONSE_UNC": np.full(3, 0.02),
        },
    )
    telluric = write_bintable_fits(
        tmp_path / "telluric.fits",
        primary_header=fits.Header({"TELQUAL": "DIAGNOSTIC", "WAVEREF": "air"}),
        extname="TELLURIC_MODEL",
        columns={
            "WAVELENGTH": wave,
            "TRANSMISSION": np.array([0.75, 0.8, 0.9]),
            "CONFIDENCE": np.array([0.3, 0.3, 0.3]),
        },
    )

    with pytest.raises(ValueError, match="UNQUALIFIED_TELLURIC_MODEL"):
        fc.apply_fluxcal_to_spec1d(
            sci, sensfunc_path=sens, telluric_path=telluric
        )


def test_requested_masterflat_processing_failure_aborts(tmp_path, monkeypatch):
    raw = tmp_path / "standard.fits"
    raw.write_bytes(b"dummy")
    master = tmp_path / "superflat.fits"
    master.write_bytes(b"dummy")
    monkeypatch.setattr(sp, "_try_existing_rectified", lambda *a, **kw: None)
    monkeypatch.setattr(sp, "_resolve_superflat_path", lambda *a, **kw: master)

    def fail_flat(*a, **kw):
        raise RuntimeError("injected detector-flat failure")

    monkeypatch.setattr(sp, "apply_flat", fail_flat)
    with pytest.raises(RuntimeError, match="STANDARD_FLATFIELD_FAILED") as ei:
        sp._prepare_rectified_standard(
            {"fluxcal": {"standard_apply_flat": True}},
            run_root=tmp_path,
            stage_dir_path=tmp_path / "prepare",
            raw_path=raw,
        )
    assert isinstance(ei.value.__cause__, RuntimeError)
