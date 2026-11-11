"""Actual COUNT_V1 worker-validity replay, with fail-closed lineage checks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.calib.count_v1_capture import (
    audit_count_v1_capture,
    write_count_v1_capture,
)
from scorpio_pipe.covariance_operator_capture import sha256_file
from scorpio_pipe.stages.linearize import _write_mef
from tests.test_p13c_linearize_integrated_counts import _fixture


def _captured(tmp_path: Path, *, density=False, normalize=False, cleanup=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    output = _fixture(
        tmp_path, bunit="e-", gain=2., dw=.5 if density else 1.,
        density=density, normalize=normalize, capture=True,
    )
    assert output["count_v1_actual_valid"] is not None
    source = tmp_path / "electron.fits"
    rect = tmp_path / "object_skysub.fits"
    dw = .5 if density else 1.
    _write_mef(
        rect, output["rect"], output["hdr"], output["rect_var"],
        output["rect_mask"], output["rect_cov"],
        wave0=4999., dw=dw, unit="Angstrom", wave_ref="air",
    )
    dest = tmp_path / "object_count_v1_actual_validity_v1.json"
    lam = np.broadcast_to(
        5000. + 2.*np.arange(12, dtype=np.float64), (4, 12)
    ).copy()
    write_count_v1_capture(
        manifest_path=dest,
        source_fits=source, output_fits=rect,
        actual_valid_mask=output["count_v1_actual_valid"],
        effective_lambda_map=lam,
        wave0=4999., dw=dw, nlam=output["rect"].shape[1],
        output_density=density, normalize_exptime=normalize,
        exptime_s=100., cleanup_executed=cleanup,
    )
    return source, rect, dest


@pytest.mark.parametrize("density,normalize", [
    (False, False), (True, False), (False, True), (True, True),
])
def test_actual_worker_mask_and_count_v1_replay_matches_fits(tmp_path, density, normalize):
    _source, _out, manifest = _captured(
        tmp_path, density=density, normalize=normalize,
    )
    got = audit_count_v1_capture(manifest)
    assert got["hashes_verified"] and got["count_v1_replayed"]
    assert got["max_delta_sci"] < 1e-4
    assert got["max_delta_var"] < 1e-4
    assert got["max_delta_cov"] < 1e-4
    assert got["status"] == "HOLD_REPLAYED_COUNT_V1_WITHOUT_SKY_TO_1D_ADMISSION"
    assert got["science_uncertainty_qualified"] is False
    assert got["production_variance_release_allowed"] is False
    assert got["sky_lineage"] == "UNBOUND"


@pytest.mark.parametrize("which", ["source", "output", "archive"])
def test_changed_source_or_output_or_capture_is_detected_by_sha(tmp_path, which):
    source, output, manifest = _captured(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    target = {"source": source, "output": output,
              "archive": Path(data["archive"]["path"])}[which]
    with target.open("ab") as out:
        out.write(b"unauthorised after capture")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        audit_count_v1_capture(manifest)


def test_rehashed_false_valid_mask_cannot_reproduce_true_science(tmp_path):
    _source, _output, manifest = _captured(tmp_path)
    d = json.loads(manifest.read_text(encoding="utf-8"))
    archive = Path(d["archive"]["path"])
    with np.load(archive, allow_pickle=False) as payload:
        valid = payload["valid"].copy()
        lam = payload["lambda_map"].copy()
    valid[:, 5] = False
    np.savez_compressed(archive, valid=valid, lambda_map=lam)
    d["archive"]["sha256"] = sha256_file(archive)
    manifest.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError, match="output SCI differs"):
        audit_count_v1_capture(manifest)


def test_cleanup_and_unknown_sky_source_must_not_be_qualified(tmp_path):
    _source, _output, manifest = _captured(tmp_path, cleanup=True)
    r = audit_count_v1_capture(manifest)
    assert r["status"] == "HOLD_UNCAPTURED_POST_LINEARIZE_SKY_CLEANUP"
    assert not r["count_v1_replayed"]
    assert r["production_variance_release_allowed"] is False


def test_missing_sky_provenance_cannot_be_faked_by_other_exposure(tmp_path):
    _source, _output, manifest = _captured(tmp_path / "linear")
    from tests.test_kelson_frozen_capture import _science_night
    stage, _ = _science_night(tmp_path / "other_sky", capture=True)
    other = stage / "obj_0001_kelson_frozen_fit_lineage_v1.json"
    assert other.is_file()
    with pytest.raises(ValueError, match="not the captured Kelson SkySub output"):
        audit_count_v1_capture(manifest, sky_lineage_manifest=other)


def test_invalid_budget_and_mask_dtypes_are_rejected_before_output(tmp_path):
    source, output, manifest = _captured(tmp_path)
    with pytest.raises(ValueError, match="byte budget"):
        audit_count_v1_capture(manifest, max_bytes=8)
    with pytest.raises(ValueError, match="geometry"):
        write_count_v1_capture(
            manifest_path=tmp_path/"bad.json",
            source_fits=source, output_fits=output,
            actual_valid_mask=np.ones((4, 12), dtype=np.float32),
            effective_lambda_map=np.zeros((4, 12), dtype=float),
            wave0=4999., dw=1., nlam=24, output_density=False,
            normalize_exptime=False, exptime_s=100.,
            cleanup_executed=False,
        )
