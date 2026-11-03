"""Independent read-only fidelity checks on actually captured SkySub FITS."""
from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.calib.sky_fit_audit import audit_frozen_kelson_products
from tests.test_kelson_frozen_capture import _science_night


def _actual_paths(tmp_path):
    stage, skysub = _science_night(tmp_path / "night", capture=True)
    model = stage / "obj_0001_skymodel_raw.fits"
    with fits.open(skysub) as hdul:
        filename = str(hdul[0].header["SKYFRZP"])
    snapshot = stage / filename
    assert snapshot.exists() and model.exists()
    return snapshot, model, skysub


def test_recovered_real_fit_matches_sky_model_and_actual_lsqr_residual(tmp_path):
    frozen, model, skysub = _actual_paths(tmp_path)
    before = [p.read_bytes() for p in (frozen, model, skysub)]
    audit = audit_frozen_kelson_products(frozen, model, skysub)
    assert audit["status"] == "MATCH"
    assert audit["n_selected_fit_pixels"] >= 64
    assert audit["n_native_pixels"] == 48 * 128
    assert audit["model_max_allowed_fraction"] <= 1.
    assert audit["lsqr_residual_relative_difference"] < .01
    assert not audit["science_uncertainty_qualified"]
    assert audit["lsqr_iterations"] >= 0
    assert np.isfinite(audit["lsqr_condition_estimate"])
    assert [p.read_bytes() for p in (frozen, model, skysub)] == before


def test_real_fit_audit_result_independent_of_row_chunk_size(tmp_path):
    frozen, model, skysub = _actual_paths(tmp_path)
    one = audit_frozen_kelson_products(frozen, model, skysub, chunk_rows=1)
    many = audit_frozen_kelson_products(frozen, model, skysub, chunk_rows=48)
    np.testing.assert_allclose(
        [one["model_max_abs"], one["model_rms"]],
        [many["model_max_abs"], many["model_rms"]],
        atol=1e-10, rtol=1e-8,
    )
    assert one["lsqr_measured_residual_norm"] == pytest.approx(
        many["lsqr_measured_residual_norm"]
    )


def test_real_fit_audit_rejects_modified_sky_model(tmp_path):
    frozen, model, skysub = _actual_paths(tmp_path)
    corrupted = tmp_path / "corrupt_model.fits"
    with fits.open(model, memmap=False) as hdul:
        hdul["SCI"].data[3, 20] += 2.
        hdul.writeto(corrupted)
    with pytest.raises(ValueError, match="do not reproduce"):
        audit_frozen_kelson_products(frozen, corrupted, skysub)


def test_real_fit_audit_rejects_modified_sky_residual(tmp_path):
    frozen, model, skysub = _actual_paths(tmp_path)
    corrupted = tmp_path / "corrupt_skysub.fits"
    with fits.open(skysub, memmap=False) as hdul:
        hdul["SCI"].data += 25.
        hdul.writeto(corrupted)
    with pytest.raises(ValueError, match="inconsistent with archived LSQR"):
        audit_frozen_kelson_products(frozen, model, corrupted)


def test_real_fit_audit_rejects_mismatched_fits_archive_link(tmp_path):
    frozen, model, skysub = _actual_paths(tmp_path)
    alias = tmp_path / "different_snapshot.npz"
    alias.write_bytes(frozen.read_bytes())
    with pytest.raises(ValueError, match="identity cards"):
        audit_frozen_kelson_products(alias, model, skysub)


@pytest.mark.parametrize("key,value", [
    ("model_atol", -1.),
    ("model_rtol", np.nan),
    ("residual_rtol", -1.),
    ("chunk_rows", 0),
])
def test_fidelity_audit_rejects_invalid_tolerance_controls(tmp_path, key, value):
    # Reject malformed audit controls even before opening source files.
    with pytest.raises(ValueError):
        audit_frozen_kelson_products(
            tmp_path / "not_found.npz",
            tmp_path / "not_found_model.fits",
            tmp_path / "not_found_residual.fits",
            **{key: value},
        )
