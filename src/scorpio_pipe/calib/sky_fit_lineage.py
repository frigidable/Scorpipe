"""Hash-linked provenance of an actually fitted Kelson sky solution.

Purely opt-in bookkeeping: no change to SCI/VAR/MASK, no covariance release.
The selected SkySub input may be cosmics-cleaned, unit-converted by
downstream input adapters, or a RAW frame: SHA256 binds FILE identity, not
the mathematical transformation to sky-fit input arrays.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from astropy.io import fits

from scorpio_pipe.covariance_operator_capture import sha256_file


_SCHEMA = "SCORPIPE_KELSON_FROZEN_FIT_LINEAGE_V1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_KEYS = ("selected_sky_input", "frozen_fit", "skymodel_raw", "skysub_raw")


def _file_record(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Kelson provenance input is not a file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def write_frozen_kelson_fit_lineage(
    manifest_path: Path,
    *,
    selected_sky_input: Path,
    frozen_fit: Path,
    skymodel_raw: Path,
    skysub_raw: Path,
) -> Path:
    """Bind ACTUAL SkySub source, snapshot and both output FITS, atomically.

    The writer runs only after both science FITS have been written. It
    deliberately does NOT claim that the input path is native CCD raw data,
    or that the frozen finite-iteration LSQR fit is an exact WLS derivative.
    """
    root = Path(manifest_path)
    root.parent.mkdir(parents=True, exist_ok=True)
    records = {key: _file_record(path) for key, path in (
        ("selected_sky_input", selected_sky_input),
        ("frozen_fit", frozen_fit),
        ("skymodel_raw", skymodel_raw),
        ("skysub_raw", skysub_raw),
    )}
    if len({x["path"] for x in records.values()}) != len(records):
        raise ValueError("Kelson provenance sources and products must be distinct")
    parent = root.parent.resolve()
    for key in ("frozen_fit", "skymodel_raw", "skysub_raw"):
        if Path(records[key]["path"]).parent != parent:
            raise ValueError(f"Kelson {key} must belong to the selected SkySub stage")
    payload = {
        "schema": _SCHEMA,
        "status": "FIT_ONLY_NOT_SCIENCE_COVARIANCE",
        "records": records,
        "source_transform_lineage": "NOT_FULLY_CAPTURED",
        "native_detector_covariance_qualified": False,
        "publication_variance_release_allowed": False,
    }
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=root.parent,
            prefix="." + root.stem + ".", suffix=".tmp", delete=False,
        ) as handle:
            temp = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, root)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    return root


def audit_frozen_kelson_fit_lineage(
    manifest_path: Path,
    *,
    check_fit_against_products: bool = True,
) -> dict[str, Any]:
    """Fail closed on mismatched science/snapshot bytes or model identity.

    File SHA256 evidence cannot validate upstream units, independent and
    shared MasterBias terms, LSQR selection sensitivity, or final 1D LSF.
    """
    manifest = Path(manifest_path).resolve(strict=True)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if (not isinstance(payload, dict)
            or payload.get("schema") != _SCHEMA
            or payload.get("status") != "FIT_ONLY_NOT_SCIENCE_COVARIANCE"
            or payload.get("source_transform_lineage") != "NOT_FULLY_CAPTURED"
            or payload.get("native_detector_covariance_qualified") is not False
            or payload.get("publication_variance_release_allowed") is not False):
        raise ValueError("Unknown/unqualified Kelson provenance schema or status")
    records = payload.get("records")
    if not isinstance(records, dict) or set(records) != set(_KEYS):
        raise ValueError("Kelson provenance file set is incomplete")
    verified: dict[str, Path] = {}
    for key in _KEYS:
        rec = records[key]
        if not isinstance(rec, dict) or set(rec) != {"path", "sha256", "size_bytes"}:
            raise ValueError("Kelson provenance artifact record invalid")
        name, digest, size = rec.get("path"), rec.get("sha256"), rec.get("size_bytes")
        if (not isinstance(name, str) or not Path(name).is_absolute()
                or not isinstance(digest, str) or _SHA256.fullmatch(digest) is None
                or type(size) is not int or size < 0):
            raise ValueError("Kelson artifact identifier invalid")
        path = Path(name).resolve(strict=True)
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Kelson artifact size/shape changed: {key}")
        if sha256_file(path) != digest:
            raise ValueError(f"Kelson artifact SHA256 mismatch: {key}")
        verified[key] = path
    if len(set(verified.values())) != len(verified):
        raise ValueError("Kelson provenance artifact identities not distinct")
    root = manifest.parent
    for key in ("frozen_fit", "skymodel_raw", "skysub_raw"):
        if verified[key].parent != root:
            raise ValueError(f"Kelson {key} no longer belongs to stage output directory")
    name = verified["frozen_fit"].name
    if (not name.endswith("_kelson_frozen_fit_v1.npz")
            or verified["skymodel_raw"].name != name.replace(
                "_kelson_frozen_fit_v1.npz", "_skymodel_raw.fits"
            ) or verified["skysub_raw"].name != name.replace(
                "_kelson_frozen_fit_v1.npz", "_skysub_raw.fits"
            )):
        raise ValueError("Kelson product naming lineage is inconsistent")
    for key in ("skymodel_raw", "skysub_raw"):
        with fits.open(verified[key], memmap=False) as hdul:
            hdr = hdul[0].header
            if (str(hdr.get("SKYMD") or "") != "kelson_raw"
                    or hdr.get("SKYFRZP") != name
                    or hdr.get("SKYFRZV") != 1
                    or hdr.get("SKYFRZQ") != "FIT_ONLY"):
                raise ValueError("Kelson FITS archive identity cards disagree")
    fidelity: dict[str, Any] | None = None
    if check_fit_against_products:
        from scorpio_pipe.calib.sky_fit_audit import audit_frozen_kelson_products
        fidelity = audit_frozen_kelson_products(
            verified["frozen_fit"],
            verified["skymodel_raw"],
            verified["skysub_raw"],
        )
        if fidelity.get("status") != "MATCH":
            raise ValueError("Kelson frozen fit does not replay actual sky-model FITS")
    return {
        "status": "HOLD_UNBOUND_DETECTOR_AND_SKY_COVARIANCE",
        "file_hashes_verified": True,
        "frozen_fit_fits_identity_verified": True,
        "frozen_model_replayed": bool(check_fit_against_products),
        "selected_sky_input": str(verified["selected_sky_input"]),
        "fit_archive": str(verified["frozen_fit"]),
        "model_fits": str(verified["skymodel_raw"]),
        "residual_fits": str(verified["skysub_raw"]),
        "fidelity": fidelity,
        "unbound_provenance": [
            "native_detector_input_transform",
            "pre_sky_independent_and_shared_masterbias_noise",
            "complete_COUNT_V1_stack_1D_lineage",
            "adaptive_clipping_and_extraction_profile_uncertainty",
        ],
        "science_uncertainty_qualified": False,
        "production_variance_release_allowed": False,
    }
