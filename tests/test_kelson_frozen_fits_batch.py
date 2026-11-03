"""Real synthetic-night batch fidelity: read-only, fail-closed, JSON CLI."""
from __future__ import annotations

import json

from astropy.io import fits

from scorpio_pipe.calib.sky_fit_batch import audit_sky_directory, main
from tests.test_kelson_frozen_capture import _science_night


def test_actual_night_batch_fidelity_all_exposures_pass(tmp_path, capsys):
    stage, _ = _science_night(tmp_path / "night", capture=True)
    expected = [p.read_bytes() for p in sorted(stage.glob("*.fits"))]
    report = audit_sky_directory(stage)
    assert report["status"] == "PASS"
    assert report["n_kelson"] == 1
    assert report["n_passed"] == 1
    assert report["n_failed"] == 0
    assert report["per_exposure"][0]["audit"]["status"] == "MATCH"
    assert not report["science_uncertainty_qualified"]

    assert main([str(stage)]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["status"] == "PASS"
    assert emitted["n_passed"] == 1
    assert [p.read_bytes() for p in sorted(stage.glob("*.fits"))] == expected


def test_real_night_missing_capture_explicit_failure(tmp_path, capsys):
    stage, science = _science_night(tmp_path / "night", capture=False)
    report = audit_sky_directory(stage)
    assert report["status"] == "FAIL"
    assert report["n_kelson"] == 1
    assert report["n_failed"] == 1
    assert "No valid current Kelson capture linkage" in report["per_exposure"][0]["error"]
    assert main([str(stage)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "FAIL"


def test_real_night_missing_archive_fails_closed(tmp_path):
    stage, science = _science_night(tmp_path / "night", capture=True)
    hdr = fits.getheader(science)
    (stage / hdr["SKYFRZP"]).unlink()
    report = audit_sky_directory(stage)
    assert report["status"] == "FAIL"
    assert report["n_failed"] == 1
    assert "NPZ missing" in report["per_exposure"][0]["error"]


def test_real_night_corrupt_model_fails_without_modifying_input(tmp_path):
    stage, science = _science_night(tmp_path / "night", capture=True)
    model = stage / "obj_0001_skymodel_raw.fits"
    with fits.open(model, mode="update", memmap=False) as hdul:
        hdul["SCI"].data[0, 0] += 5.
    snapshot = model.read_bytes()
    report = audit_sky_directory(stage)
    assert report["status"] == "FAIL"
    assert "do not reproduce" in report["per_exposure"][0]["error"]
    assert model.read_bytes() == snapshot


def test_real_night_rejects_archive_path_traversal(tmp_path):
    stage, science = _science_night(tmp_path / "night", capture=True)
    with fits.open(science, mode="update", memmap=False) as hdul:
        hdul[0].header["SKYFRZP"] = "../other/evil_kelson_frozen_fit_v1.npz"
    report = audit_sky_directory(stage)
    assert report["status"] == "FAIL"
    assert "linkage" in report["per_exposure"][0]["error"]


def test_no_input_sky_stage_exits_nonzero_with_structured_report(tmp_path, capsys):
    empty = tmp_path / "empty_sky"
    empty.mkdir()
    assert main([str(empty)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "FAIL"
    assert report["science_uncertainty_qualified"] is False
    assert "No raw SkySub FITS" in report["error"]
