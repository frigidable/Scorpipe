"""Read-only P1.4e1 admission: check actual P1.4d2 sidecar lineage.

The fixtures deliberately use inert files with FITS-like names: this tests
cryptographic coupling and geometry, not observational FITS qualification.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scorpio_pipe.calib.frozen_covariance_admission import (
    audit_frozen_operator_sidecar_pair,
)
from scorpio_pipe.covariance_operator_capture import (
    StackOperatorCapture,
    save_extraction_operators,
    sha256_file,
)


@pytest.fixture()
def linked_pair(tmp_path: Path):
    stack_dir = tmp_path / "stack"
    extract_dir = tmp_path / "extract"
    stack_dir.mkdir()
    extract_dir.mkdir()
    a = tmp_path / "a_skysub.fits"
    b = tmp_path / "b_skysub.fits"
    stack = stack_dir / "stack2d.fits"
    spec = extract_dir / "spec1d.fits"
    for path, payload in (
        (a, b"synthetic source A"),
        (b, b"synthetic source B"),
        (stack, b"stacked science fixture"),
        (spec, b"one-dimensional science fixture"),
    ):
        path.write_bytes(payload)

    capture = StackOperatorCapture(stack_dir, [a, b], ny=3, nlam=4)
    capture.write(0, 0, 3, np.full((3, 4), .3))
    capture.write(1, 0, 3, np.full((3, 4), .7))
    stack_manifest = capture.finalize(
        stack_fits=stack,
        meta={"y_shifts": [0., 0.], "resampling_contract": "COUNT_V1"},
    )
    trace = np.zeros((3, 4))
    fixed = np.zeros((3, 4))
    trace[1] = 1.
    fixed[2] = 1.
    extract_manifest = save_extraction_operators(
        extract_dir, input_fits=stack, spec_fits=spec,
        E_trace=trace, E_fixed=fixed,
        metadata={"extraction_method": "boxcar", "input_mode": "stack2d"},
    )
    return stack_manifest, extract_manifest, (a, b, stack, spec)


def _update_manifest(path: Path, callback):
    data = json.loads(path.read_text(encoding="utf-8"))
    callback(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_identical_fits_stage_lineage_is_verifiable_but_not_science_qualified(linked_pair):
    stack_manifest, extract_manifest, _ = linked_pair
    report = audit_frozen_operator_sidecar_pair(stack_manifest, extract_manifest)
    assert report["status"] == "HOLD_UNBOUND_PRE_SKY_OPERATOR_LINEAGE"
    assert report["integrity_verified"] is True
    assert report["linked_stack_extract_fits"] is True
    assert report["shape_y_lambda"] == [3, 4]
    assert report["n_stack_exposures"] == 2
    assert report["science_uncertainty_qualified"] is False
    assert report["production_variance_release_allowed"] is False
    missing = report["required_but_not_bound_by_this_audit"]
    assert "exact_production_Kelson_final_fit_mask_basis_weights" in missing
    assert "complete_pre_sky_independent_detector_variance" in missing
    assert "signed_shared_MasterBias_sensitivities_and_group_identity" in missing


@pytest.mark.parametrize("what", ["science", "source", "stack_weights", "extract_weights"])
def test_tampered_capture_or_original_file_fails_closed(linked_pair, what):
    stack_manifest, extract_manifest, (a, _b, stack, _spec) = linked_pair
    s = json.loads(stack_manifest.read_text(encoding="utf-8"))
    e = json.loads(extract_manifest.read_text(encoding="utf-8"))
    victim = {
        "science": stack,
        "source": a,
        "stack_weights": stack_manifest.parent / s["inputs"][0]["weights_file"],
        "extract_weights": extract_manifest.parent / e["weights_file"],
    }[what]
    with victim.open("ab") as fd:
        fd.write(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        audit_frozen_operator_sidecar_pair(stack_manifest, extract_manifest)


def test_exact_stack_to_extract_input_identity_is_required(linked_pair, tmp_path):
    stack_manifest, extract_manifest, (_a, _b, stack, _spec) = linked_pair
    unrelated = tmp_path / "unrelated.fits"
    unrelated.write_bytes(b"different data")

    def change(data):
        data["input"]["path"] = str(unrelated)
        data["input"]["sha256"] = sha256_file(unrelated)

    _update_manifest(extract_manifest, change)
    with pytest.raises(ValueError, match="not the captured Stack2D output"):
        audit_frozen_operator_sidecar_pair(stack_manifest, extract_manifest)


def test_modified_rehashed_non_normalized_stack_weights_rejected(linked_pair):
    stack_manifest, extract_manifest, _ = linked_pair
    data = json.loads(stack_manifest.read_text(encoding="utf-8"))
    file = stack_manifest.parent / data["inputs"][0]["weights_file"]
    arr = np.load(file, allow_pickle=False).copy()
    arr[0, 1] = .9
    np.save(file, arr)

    def change(manifest):
        manifest["inputs"][0]["weights_sha256"] = sha256_file(file)

    _update_manifest(stack_manifest, change)
    with pytest.raises(ValueError, match="not normalized"):
        audit_frozen_operator_sidecar_pair(stack_manifest, extract_manifest)


def test_rehashed_stack_gap_cannot_be_used_by_extraction(linked_pair):
    stack_manifest, extract_manifest, _ = linked_pair
    metadata = json.loads(stack_manifest.read_text(encoding="utf-8"))
    for item in metadata["inputs"]:
        file = stack_manifest.parent / item["weights_file"]
        arr = np.load(file, allow_pickle=False).copy()
        arr[:, 2] = 0.
        np.save(file, arr)
        item["weights_sha256"] = sha256_file(file)
    stack_manifest.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="no stack support"):
        audit_frozen_operator_sidecar_pair(stack_manifest, extract_manifest)


def test_stage_diagnostic_cannot_bypass_size_limits(linked_pair):
    stack_manifest, extract_manifest, _ = linked_pair
    with pytest.raises(ValueError, match="byte budget"):
        audit_frozen_operator_sidecar_pair(
            stack_manifest, extract_manifest, max_stack_weight_bytes=16,
        )
    with pytest.raises(ValueError, match="byte budget"):
        audit_frozen_operator_sidecar_pair(
            stack_manifest, extract_manifest, max_extract_weight_bytes=16,
        )


def test_invalid_or_misdeclared_metadata_fails_closed(linked_pair):
    stack_manifest, extract_manifest, _ = linked_pair
    _update_manifest(stack_manifest, lambda data: data.update(
        {"schema": "UNTRUSTED_FROZEN_OPERATOR"}
    ))
    with pytest.raises(ValueError, match="schema"):
        audit_frozen_operator_sidecar_pair(stack_manifest, extract_manifest)


def test_spatial_gap_cannot_be_hidden_by_support_elsewhere_in_column(linked_pair):
    stack_manifest, extract_manifest, _ = linked_pair
    metadata = json.loads(stack_manifest.read_text(encoding="utf-8"))
    # Only TRACE row 1 loses support at wavelength 2; rows 0 and 2 are
    # still covered. A wavelength-only mask would miss this invalid map.
    for item in metadata["inputs"]:
        file = stack_manifest.parent / item["weights_file"]
        arr = np.load(file, allow_pickle=False).copy()
        arr[1, 2] = 0.
        np.save(file, arr)
        item["weights_sha256"] = sha256_file(file)
    stack_manifest.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="spatial/spectral pixel with no stack support"):
        audit_frozen_operator_sidecar_pair(stack_manifest, extract_manifest)


def test_rebound_capture_input_index_is_rejected(linked_pair):
    stack_manifest, extract_manifest, _ = linked_pair
    _update_manifest(stack_manifest, lambda obj: obj["inputs"][0].update({"index": 1}))
    with pytest.raises(ValueError, match="input order"):
        audit_frozen_operator_sidecar_pair(stack_manifest, extract_manifest)
