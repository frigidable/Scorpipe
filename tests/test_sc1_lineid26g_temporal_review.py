"""Read-only N2146sat LineID26c NPZ native-source metadata and proxy tests."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "qa" / "sc1_lineid26g_temporal_review.py"
)
SPEC = importlib.util.spec_from_file_location("sc1_lineid26g_temporal_review", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def _write_npz(path, cube, *, names=None, dx=None, second_cube=False):
    if names is None:
        names = MOD.FILENAMES
    if dx is None:
        dx = np.arange(-18, 19, dtype=np.int32)
    arrays = {
        "raw_counts": cube,
        "original_filename": np.array(names),
        "native_dx": np.array(dx),
        "x_native_centre_float_prior": np.zeros((7, 1032), dtype=float),
        "x_native_centre_integer": np.zeros((7, 1032), dtype=np.int32),
    }
    if second_cube:
        arrays["other_counts"] = cube
    np.savez(path, **arrays)


def _case(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    old = tmp_path / "saved_work"
    old.mkdir()
    saved = old / "config.yaml"
    saved.write_text("immutable original config", encoding="utf-8")
    names = {
        "sky": ["s22800604.fts"],
        "obj": [f"s228006{i:02d}.fts" for i in range(7, 13)],
        "bias": [f"bias{i:02d}.fts" for i in range(20)],
        "flat": [f"flat{i:02d}.fts" for i in range(3)],
        "neon": [f"neon{i:02d}.fts" for i in range(2)],
    }
    role_root = tmp_path / "role_contract"
    role_root.mkdir()
    role = role_root / "role_contract.json"
    role.write_text(json.dumps({
        "schema": "SC1_N2146SAT_OFFSET_SKY_ROLE_CONTRACT_V1",
        "status": "ROLE_CONTRACT_RESOLVED_NO_SCIENCE_EXECUTION",
        "source_config": str(saved),
        "resolved_roles": {
            kind: [str(raw / n) for n in group]
            for kind, group in names.items()
        },
        "source_raw_manifest_sha256": "abc123",
        "sky_subtraction_approved": False,
        "stage_execution_authorized": False,
    }), encoding="utf-8")
    x = np.arange(37)
    cube = np.broadcast_to(50 + 0.05 * x, (7, 1032, 37)).copy()
    for idx, amplitude in enumerate([5, 8, 7, 6, 5, 4, 3]):
        cube[idx, :, 13:24] += amplitude
    archive = tmp_path / "source_archive"
    archive.mkdir()
    npz = archive / "lineid26c_ni5199_native_counts.npz"
    _write_npz(npz, cube)
    return npz, role, saved, raw, cube


def _prepare(npz, role, out, **kwargs):
    return MOD.prepare(
        npz, role, out, expected_npz_sha256=MOD.sha256_file(npz), **kwargs
    )


def test_core_proxy_rate_metadata_and_readonly_original(tmp_path):
    npz, role, saved, raw, _ = _case(tmp_path)
    outfile = tmp_path / "external" / "new_review.json"
    old_config = saved.read_bytes()
    old_npz = npz.read_bytes()
    old_role = role.read_bytes()
    result = _prepare(npz, role, outfile)
    assert result["status"] == "RAW_FEATURE_TIME_TRANSFER_DIAGNOSTIC_ONLY"
    assert len(result["comparisons"]) == 14
    off, first = result["comparisons"][:2]
    assert off["median_line_core_proxy_raw_counts"] == pytest.approx(55)
    assert first["median_line_core_proxy_raw_counts"] == pytest.approx(88)
    assert first["rate_ratio_relative_to_OFF04"] == pytest.approx(
        (88 / 1200) / (55 / 600)
    )
    assert result["source_cube_key"] == "raw_counts"
    assert result["verified_embedded_original_filename"] == MOD.FILENAMES
    assert result["verified_embedded_native_dx"] == list(range(-18, 19))
    assert result["frame_axis_basis"] == (
        "NPZ_embedded_original_filename_verified_against_explicit_role_contract"
    )
    assert result["source_hash_pinned"] is True
    assert result["stages_executed"] is False
    for name in (
        "source_spatial_mask_approved",
        "independent_atomic_line_id_approved",
        "time_transfer_coefficient_approved",
        "sky_subtraction_approved",
        "masterbias_physical_model_validated",
        "wavelength_lsf_validated",
        "covariance_qualified",
    ):
        assert result[name] is False
    assert saved.read_bytes() == old_config
    assert role.read_bytes() == old_role
    assert npz.read_bytes() == old_npz
    assert outfile.exists()
    with pytest.raises(FileExistsError, match="overwrite"):
        _prepare(npz, role, outfile)


def test_missing_embedded_original_filename_is_fail_closed(tmp_path):
    npz, role, _, _, cube = _case(tmp_path)
    np.savez(npz, raw_counts=cube, native_dx=np.arange(-18, 19))
    with pytest.raises(ValueError, match="lacks required frame/dx metadata"):
        _prepare(npz, role, tmp_path / "external" / "out.json")


def test_wrong_role_contract_is_rejected_before_calculation(tmp_path):
    npz, role, _, _, _ = _case(tmp_path)
    content = json.loads(role.read_text(encoding="utf-8"))
    content["resolved_roles"]["sky"] = content["resolved_roles"]["obj"][:1]
    role.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="Wrong OFF/ON"):
        _prepare(npz, role, tmp_path / "external" / "out.json")


def test_ambiguous_cube_requires_explicit_key(tmp_path):
    npz, role, _, _, cube = _case(tmp_path)
    _write_npz(npz, cube, second_cube=True)
    with pytest.raises(ValueError, match="No unique"):
        _prepare(npz, role, tmp_path / "external" / "wrong.json")
    result = _prepare(
        npz, role, tmp_path / "external" / "good.json", key="raw_counts"
    )
    assert result["source_cube_key"] == "raw_counts"


def test_nonfinite_pixels_and_protected_output_paths(tmp_path):
    npz, role, saved, raw, cube = _case(tmp_path)
    for out in (
        saved.parent / "bad.json", raw / "bad.json",
        npz.parent / "bad.json", role.parent / "bad.json",
    ):
        with pytest.raises(ValueError, match="prior source"):
            _prepare(npz, role, out)
        assert not out.exists()
    cube[2, 100, 20] = np.nan
    _write_npz(npz, cube)
    with pytest.raises(ValueError, match="Nonfinite"):
        _prepare(npz, role, tmp_path / "external" / "bad.json")


def test_negative_line_core_residual_is_not_clipped():
    x = np.arange(37, dtype=float)
    cube = np.broadcast_to(100 + 0.3 * x, (7, 1032, 37)).copy()
    cube[..., 13:24] -= 2
    assert np.allclose(MOD.row_feature_proxy(cube), -22.0)


def test_embedded_frame_order_cannot_be_overridden_by_manual_assertion(tmp_path):
    npz, role, _, _, cube = _case(tmp_path)
    names = list(MOD.FILENAMES)
    names[0], names[1] = names[1], names[0]
    _write_npz(npz, cube, names=names)
    with pytest.raises(ValueError, match="original_filename does not match"):
        _prepare(
            npz, role, tmp_path / "external" / "out.json",
            asserted_order=True,
        )


def test_wrong_native_dx_rejected(tmp_path):
    npz, role, _, _, cube = _case(tmp_path)
    _write_npz(npz, cube, dx=np.arange(37, dtype=np.int32))
    with pytest.raises(ValueError, match="native_dx"):
        _prepare(npz, role, tmp_path / "external" / "out.json")


def test_external_source_sha_is_mandatory(tmp_path):
    npz, role, _, _, _ = _case(tmp_path)
    target = tmp_path / "external" / "out.json"
    for digest in (None, "0" * 64, MOD.sha256_file(npz)[:-1]):
        with pytest.raises(ValueError, match="Pinned"):
            MOD.prepare(npz, role, target, expected_npz_sha256=digest)
    assert not target.exists()


def test_metadata_access_and_provenance(tmp_path):
    npz, _, _, _, _ = _case(tmp_path)
    labels, offsets = MOD.verified_archive_axis(npz)
    assert labels == MOD.FILENAMES
    assert offsets == list(range(-18, 19))
    inventory = MOD.npz_inventory(npz)
    assert inventory["raw_counts"]["shape"] == [7, 1032, 37]
    assert inventory["original_filename"]["shape"] == [7]
    assert inventory["native_dx"]["dtype"] == "int32"


def test_original_filename_rejects_duplicates_or_unknown_object(tmp_path):
    npz, role, _, _, cube = _case(tmp_path)
    names = list(MOD.FILENAMES)
    names[-1] = names[-2]
    _write_npz(npz, cube, names=names)
    with pytest.raises(ValueError, match="original_filename does not match"):
        _prepare(npz, role, tmp_path / "external" / "out.json")
