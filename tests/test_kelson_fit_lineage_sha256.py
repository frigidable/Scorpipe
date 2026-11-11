"""Exact-file provenance audit for the ACTUAL production Kelson solver.

The fixtures are synthetic observing nights created by the real SkySub
entrypoint. Success is archive identity, NOT a scientific VAR qualification.
"""
from __future__ import annotations

import json

import pytest
from astropy.io import fits

from scorpio_pipe.calib.sky_fit_lineage import audit_frozen_kelson_fit_lineage
from scorpio_pipe.covariance_operator_capture import sha256_file
from tests.test_kelson_frozen_capture import _science_night


def _linked_night(tmp_path):
    stage, residual = _science_night(tmp_path / "night", capture=True)
    stem = "obj_0001"
    manifest = stage / f"{stem}_kelson_frozen_fit_lineage_v1.json"
    assert manifest.is_file()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return stage, residual, manifest, data


def test_production_sky_capture_links_actual_input_solver_and_both_fits(tmp_path):
    stage, residual, manifest, data = _linked_night(tmp_path)
    assert data["schema"] == "SCORPIPE_KELSON_FROZEN_FIT_LINEAGE_V1"
    assert data["native_detector_covariance_qualified"] is False
    assert data["publication_variance_release_allowed"] is False
    assert data["records"]["skysub_raw"]["path"] == str(residual.resolve())
    assert data["records"]["selected_sky_input"]["path"].endswith("obj_0001.fits")
    report = audit_frozen_kelson_fit_lineage(manifest)
    assert report["status"] == "HOLD_UNBOUND_DETECTOR_AND_SKY_COVARIANCE"
    assert report["file_hashes_verified"] and report["frozen_model_replayed"]
    assert report["fidelity"]["status"] == "MATCH"
    assert report["science_uncertainty_qualified"] is False
    assert report["production_variance_release_allowed"] is False
    done = json.loads((stage / "done.json").read_text(encoding="utf-8"))
    assert "frozen_covariance_lineage" in json.dumps(done)
    with fits.open(residual, memmap=False) as hdul:
        assert hdul[0].header["SKYFRZQ"] == "FIT_ONLY"
        assert hdul[0].header["SKYVQUAL"] == "INCOMP"


@pytest.mark.parametrize("kind", [
    "selected_sky_input", "frozen_fit", "skymodel_raw", "skysub_raw",
])
def test_any_mutated_science_or_fit_bytes_fail_sha256_admission(tmp_path, kind):
    _, _, manifest, data = _linked_night(tmp_path)
    from pathlib import Path
    path = Path(data["records"][kind]["path"])
    # Keep file size unchanged: this must be a cryptographic check,
    # not a trivial size guard. The FITS header might become invalid;
    # hashes must fail before the attempted FITS parse.
    with path.open("r+b") as stream:
        first = stream.read(1)
        stream.seek(0)
        stream.write(bytes([first[0] ^ 1]))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        audit_frozen_kelson_fit_lineage(manifest)


def test_rehashed_modified_model_still_fails_independent_fit_replay(tmp_path):
    _, _, manifest, data = _linked_night(tmp_path)
    from pathlib import Path
    model = Path(data["records"]["skymodel_raw"]["path"])
    with fits.open(model, memmap=False) as hdul:
        hdul["SCI"].data[3, 20] += 4.
        hdul.writeto(model, overwrite=True)
    # SHA can be recomputed by a malicious/stale metadata writer; replay
    # the frozen coefficients against FITS instead of trusting JSON alone.
    data["records"]["skymodel_raw"]["sha256"] = sha256_file(model)
    data["records"]["skymodel_raw"]["size_bytes"] = model.stat().st_size
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="do not reproduce"):
        audit_frozen_kelson_fit_lineage(manifest)


def test_source_identity_cannot_be_rebound_to_wrong_name_or_directory(tmp_path):
    _, _, manifest, data = _linked_night(tmp_path)
    from pathlib import Path
    other = tmp_path / "other_sky_model.fits"
    old = Path(data["records"]["skymodel_raw"]["path"])
    other.write_bytes(old.read_bytes())
    data["records"]["skymodel_raw"]["path"] = str(other.resolve())
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="stage output directory"):
        audit_frozen_kelson_fit_lineage(manifest, check_fit_against_products=False)


def test_failed_optional_lineage_write_preserves_science_and_warns(tmp_path, monkeypatch):
    import scorpio_pipe.calib.sky_fit_lineage as lineage

    def broken(*args, **kwargs):
        raise RuntimeError("controlled provenance disk error")

    monkeypatch.setattr(lineage, "write_frozen_kelson_fit_lineage", broken)
    stage, residual = _science_night(tmp_path / "night", capture=True)
    assert residual.exists()
    with fits.open(residual) as hdul:
        assert hdul[0].header["SKYFRZQ"] == "FIT_ONLY"
        assert hdul[0].header["SKYVQUAL"] == "INCOMP"
    assert not (stage / "obj_0001_kelson_frozen_fit_lineage_v1.json").exists()
    done = json.loads((stage / "done.json").read_text(encoding="utf-8"))
    assert "SKY_FROZEN_LINEAGE_FAILED" in json.dumps(done)
    assert "LINEAGE_BIND_FAILED" in json.dumps(done)


def test_default_disabled_sky_capture_has_no_lineage(tmp_path):
    stage, residual = _science_night(tmp_path / "night", capture=False)
    assert residual.exists()
    assert not (stage / "obj_0001_kelson_frozen_fit_lineage_v1.json").exists()
