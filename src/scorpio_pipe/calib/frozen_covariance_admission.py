"""Read-only admission audit for P1.4d2 Stack2D/Extract1D sidecar pairing.

This does NOT instantiate the full pre-sky covariance operator. It checks
that two captured final-stage frozen operators belong to the same science
product and have not changed since capture. Unrecorded upstream operators
and source lineage remain a strict scientific HOLD.
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from scorpio_pipe.covariance_operator_capture import SCHEMA, sha256_file


_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_UNBOUND = (
    "exact_production_Kelson_final_fit_mask_basis_weights",
    "per_exposure_COUNT_V1_native_row_plans_and_input_validity",
    "complete_pre_sky_independent_detector_variance",
    "signed_shared_MasterBias_sensitivities_and_group_identity",
    "source_2d_coordinate_and_alignment_operators",
    "fluxcal_transfer_and_wavelength_reference_lineage",
    "data_derived_selection_and_profile_sampling_uncertainty",
)


def _manifest(path: Path, stage: str) -> tuple[Path, dict[str, Any], tuple[int, int]]:
    p = Path(path).resolve(strict=True)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError("Unknown frozen-operator capture schema")
    if data.get("stage") != stage or data.get("status") != "CONDITIONAL_FROZEN_OPERATOR_ONLY":
        raise ValueError("Frozen-operator stage/status is invalid")
    if data.get("weights_dtype") != "float64":
        raise ValueError("Frozen-operator dtype must be float64")
    raw = data.get("shape_y_lambda")
    if (not isinstance(raw, list) or len(raw) != 2
            or any(type(x) is not int or x <= 0 for x in raw)):
        raise ValueError("Frozen-operator spatial/spectral shape is invalid")
    return p, data, (raw[0], raw[1])


def _artifact(record: Any) -> tuple[Path, str]:
    if not isinstance(record, dict):
        raise ValueError("Captured artifact record is missing")
    digest = record.get("sha256")
    value = record.get("path")
    if (not isinstance(digest, str) or not _HEX_SHA256.fullmatch(digest)
            or not isinstance(value, str)):
        raise ValueError("Captured artifact path/hash is invalid")
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise ValueError("Captured science artifact needs an existing absolute path")
    if sha256_file(path) != digest:
        raise ValueError(f"Captured science artifact SHA256 mismatch: {path}")
    return path.resolve(strict=True), digest


def _sidecar(root: Path, name: Any, digest: Any, *, expected: str) -> Path:
    if not isinstance(name, str) or name != expected:
        raise ValueError("Frozen operator file identity is invalid")
    if not isinstance(digest, str) or not _HEX_SHA256.fullmatch(digest):
        raise ValueError("Frozen operator SHA256 identity is invalid")
    path = root / name
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("Frozen operator sidecar missing or escapes capture directory")
    if sha256_file(path) != digest:
        raise ValueError(f"Frozen operator sidecar SHA256 mismatch: {name}")
    return path


def audit_frozen_operator_sidecar_pair(
    stack_manifest: Path,
    extract_manifest: Path,
    *,
    max_stack_weight_bytes: int = 2_147_483_648,
    max_extract_weight_bytes: int = 536_870_912,
) -> dict[str, Any]:
    """Verify P1.4d2 captured stage lineage; always keep publication on HOLD.

    This is an intentionally opt-in file audit. It never trusts the manifest
    alone: all referenced source/output science files and captured weights
    are re-hashed, and normalized stack weights are independently rechecked.
    Absolute FITS paths are required to prevent ambiguous relative relinking.
    """
    sp, s, shape = _manifest(stack_manifest, "stack2d")
    ep, e, eshape = _manifest(extract_manifest, "extract1d")
    if shape != eshape or s.get("weights_normalized") is not True:
        raise ValueError("Stack and extraction geometry/normalization mismatch")
    ny, nlam = shape
    inputs = s.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("Frozen stack must enumerate all input exposures")
    if (type(max_stack_weight_bytes) is not int or max_stack_weight_bytes <= 0
            or type(max_extract_weight_bytes) is not int or max_extract_weight_bytes <= 0):
        raise ValueError("Frozen operator audit byte limits must be positive integers")
    if len(inputs)*ny*nlam*8 > max_stack_weight_bytes:
        raise ValueError("Frozen stack operator exceeds audit byte budget")
    if 2*ny*nlam*8 > max_extract_weight_bytes:
        raise ValueError("Frozen extraction operator exceeds audit byte budget")

    stack_path, stack_sha = _artifact(s.get("output"))
    extract_input_path, extract_input_sha = _artifact(e.get("input"))
    extract_path, _ = _artifact(e.get("output"))
    if stack_sha != extract_input_sha or stack_path != extract_input_path:
        raise ValueError("Extract1D input is not the captured Stack2D output")
    # Bind weights to the captured exposures, including their exact order.
    arrays = []
    try:
        for i, rec in enumerate(inputs):
            if not isinstance(rec, dict) or type(rec.get("index")) is not int or rec["index"] != i:
                raise ValueError("Captured stack input order/index is invalid")
            _artifact(rec)
            name = f"stack_weight_{i:04d}.npy"
            file = _sidecar(sp.parent, rec.get("weights_file"), rec.get("weights_sha256"), expected=name)
            if file.stat().st_size > ny*nlam*8 + 4096:
                raise ValueError("Frozen stack weight sidecar is oversized")
            arr = np.load(file, mmap_mode="r", allow_pickle=False)
            if arr.shape != shape or arr.dtype != np.float64:
                raise ValueError("Captured stack weight geometry/dtype mismatch")
            arrays.append(arr)
        for y0 in range(0, ny, 16):
            y1 = min(ny, y0 + 16)
            total = np.zeros((y1-y0, nlam), dtype=np.float64)
            for arr in arrays:
                block = np.asarray(arr[y0:y1], dtype=np.float64)
                if not np.isfinite(block).all() or np.any(block < 0):
                    raise ValueError("Nonfinite or negative captured stack coefficients")
                total += block
            supported = total > 0
            if np.any(~np.isclose(total[supported], 1., rtol=2e-6, atol=2e-8)):
                raise ValueError("Captured stack coefficients are not normalized")
            # Extraction coefficients cannot claim a contributor in empty pixels.
            if y0 == 0:
                support_summary = np.zeros(nlam, dtype=bool)
            support_summary |= np.any(supported, axis=0)
    finally:
        arrays.clear()

    name = e.get("weights_file")
    archive = _sidecar(ep.parent, name, e.get("weights_sha256"), expected="extraction_weights.npz")
    if archive.stat().st_size > max_extract_weight_bytes + 4096:
        raise ValueError("Frozen extraction archive exceeds audit byte budget")
    with zipfile.ZipFile(archive) as z:
        if sum(info.file_size for info in z.infolist()) > max_extract_weight_bytes + 4096:
            raise ValueError("Frozen extraction archive inflates beyond budget")
    with np.load(archive, allow_pickle=False) as data:
        if set(data.files) != {"E_trace", "E_fixed"}:
            raise ValueError("Both TRACE and FIXED captured operators required")
        for k in ("E_trace", "E_fixed"):
            arr = data[k]
            if arr.shape != shape or arr.dtype != np.float64 or not np.isfinite(arr).all():
                raise ValueError("Captured extraction coefficients have invalid geometry")
            if np.any(np.any(arr != 0, axis=0) & ~support_summary):
                raise ValueError("Extraction uses wavelength column with no stack support")
    # The frozen operators alone do not prove the original CCD fits,
    # spectral mapping, source group assignment or actual captured sky fit.
    return {
        "status": "HOLD_UNBOUND_PRE_SKY_OPERATOR_LINEAGE",
        "integrity_verified": True,
        "linked_stack_extract_fits": True,
        "stack_manifest": str(sp),
        "extract_manifest": str(ep),
        "stack_output_sha256": stack_sha,
        "extract_output_path": str(extract_path),
        "shape_y_lambda": [ny, nlam],
        "n_stack_exposures": len(inputs),
        "required_but_not_bound_by_this_audit": list(_UNBOUND),
        "science_uncertainty_qualified": False,
        "production_variance_release_allowed": False,
    }
