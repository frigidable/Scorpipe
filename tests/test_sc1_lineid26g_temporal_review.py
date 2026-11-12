"""Regressions for read-only N2146sat LineID26c native 5199 review."""
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
    role = tmp_path / "role_contract.json"
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
    npz = tmp_path / "lineid26c_fullslit_raw_native.npz"
    np.savez(npz, samples=cube)
    return npz, role, saved, raw, cube


def test_core_proxy_linear_background_and_temporal_rate(tmp_path):
    npz, role, saved, raw, cube = _case(tmp_path)
    outfile = tmp_path / "external" / "new_review.json"
    old_bytes = saved.read_bytes()
    result = MOD.prepare(npz, role, outfile, asserted_order=True)
    assert result["status"] == "RAW_FEATURE_TIME_TRANSFER_DIAGNOSTIC_ONLY"
    assert len(result["comparisons"]) == 14
    off, first = result["comparisons"][:2]
    assert off["median_line_core_proxy_raw_counts"] == pytest.approx(55)
    assert first["median_line_core_proxy_raw_counts"] == pytest.approx(88)
    assert first["rate_ratio_relative_to_OFF04"] == pytest.approx(
        (88 / 1200) / (55 / 600)
    )
    assert result["source_cube_key"] == "samples"
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
    assert saved.read_bytes() == old_bytes
    assert outfile.exists()
    with pytest.raises(FileExistsError):
        MOD.prepare(npz, role, outfile, asserted_order=True)


def test_missing_frame_axis_assertion_fails_closed(tmp_path):
    npz, role, _, _, _ = _case(tmp_path)
    with pytest.raises(ValueError, match="frame axis"):
        MOD.prepare(npz, role, tmp_path / "unused.json")


def test_wrong_role_contract_fails_closed(tmp_path):
    npz, role, _, _, _ = _case(tmp_path)
    content = json.loads(role.read_text(encoding="utf-8"))
    content["resolved_roles"]["sky"] = content["resolved_roles"]["obj"][:1]
    role.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="Wrong OFF/ON"):
        MOD.prepare(npz, role, tmp_path / "unused.json", asserted_order=True)


def test_ambiguous_npz_requires_explicit_key(tmp_path):
    npz, role, _, _, cube = _case(tmp_path)
    np.savez(npz, a=cube, b=cube)
    with pytest.raises(ValueError, match="No unique"):
        MOD.prepare(npz, role, tmp_path / "unused.json", asserted_order=True)
    result = MOD.prepare(
        npz, role, tmp_path / "new.json", key="a", asserted_order=True,
    )
    assert result["source_cube_key"] == "a"


def test_nonfinite_pixel_and_unsafe_report_path(tmp_path):
    npz, role, saved, raw, cube = _case(tmp_path)
    with pytest.raises(ValueError, match="prior source"):
        MOD.prepare(npz, role, saved.parent / "report.json", asserted_order=True)
    with pytest.raises(ValueError, match="prior source"):
        MOD.prepare(npz, role, raw / "report.json", asserted_order=True)
    cube[2, 100, 20] = np.nan
    np.savez(npz, samples=cube)
    with pytest.raises(ValueError, match="Nonfinite"):
        MOD.prepare(npz, role, tmp_path / "new.json", asserted_order=True)


def test_negative_residual_preserved_without_clipping():
    x = np.arange(37, dtype=float)
    cube = np.broadcast_to(100 + 0.3 * x, (7, 1032, 37)).copy()
    cube[..., 13:24] -= 2
    assert np.allclose(MOD.row_feature_proxy(cube), -22.0)
