"""Optional exact COUNT_V1 validity capture and read-only science replay.

This binds one actual Linearize input/output pair to the effective
wavelength array, final pre-rebin valid pixels and output normalization.
It does not include adaptive cleanup/fluxcal or establish a full
SkySub-to-1D covariance; publication/science flags remain false.
"""
from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from scorpio_pipe.covariance_operator_capture import sha256_file

SCHEMA = "SCORPIPE_COUNT_V1_ACTUAL_VALIDITY_V1"


def _record(path: Path) -> dict[str, str]:
    p = Path(path).resolve(strict=True)
    if not p.is_file():
        raise ValueError("COUNT_V1 artifact is not a file")
    return {"path": str(p), "sha256": sha256_file(p)}


def write_count_v1_capture(
    *,
    manifest_path: Path,
    source_fits: Path,
    output_fits: Path,
    actual_valid_mask: np.ndarray,
    effective_lambda_map: np.ndarray,
    wave0: float,
    dw: float,
    nlam: int,
    output_density: bool,
    normalize_exptime: bool,
    exptime_s: float,
    cleanup_executed: bool,
    max_bytes: int = 536_870_912,
) -> Path:
    """Capture the actual worker's Boolean mask; never infer it from DQ.

    An array recorded *after* masks are assembled avoids guessing about
    extra cosmic-mask overlays or detector validity flags. The source may
    have already been corrected: this writer does not identify RAW CCD.
    """
    mask = np.asarray(actual_valid_mask)
    lam = np.asarray(effective_lambda_map, dtype=np.float64)
    if (mask.dtype != np.dtype(bool) or mask.ndim != 2
            or lam.shape != mask.shape or min(mask.shape) < 1
            or type(nlam) is not int or nlam < 1
            or not np.isfinite(wave0) or not np.isfinite(dw) or dw <= 0
            or not np.isfinite(exptime_s) or (normalize_exptime and exptime_s <= 0)):
        raise ValueError("COUNT_V1 capture geometry/normalization is invalid")
    payload_bytes = int(mask.size*(1+8))
    if type(max_bytes) is not int or max_bytes <= 0 or payload_bytes > max_bytes:
        raise ValueError("COUNT_V1 capture exceeds explicit byte budget")
    dest = Path(manifest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    archive = dest.with_suffix(".npz")
    if archive == Path(source_fits) or archive == Path(output_fits):
        raise ValueError("COUNT_V1 archive cannot overwrite FITS")
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=str(dest.parent),
            prefix="."+dest.stem+".", suffix=".npz", delete=False,
        ) as fh:
            temp = Path(fh.name)
            np.savez_compressed(fh, valid=mask, lambda_map=lam)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp, archive)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    data = {
        "schema": SCHEMA,
        "status": "FROZEN_REPLAY_ONLY",
        "source": _record(source_fits),
        "output": _record(output_fits),
        "archive": _record(archive),
        "shape_native_yx": list(mask.shape),
        "grid": {"wave0": float(wave0), "dw": float(dw),
                 "nlam": int(nlam)},
        "output_density": bool(output_density),
        "normalize_exptime": bool(normalize_exptime),
        "exptime_s": float(exptime_s),
        "cleanup_executed": bool(cleanup_executed),
        "sky_lineage": "UNBOUND",
        "science_uncertainty_qualified": False,
        "production_variance_release_allowed": False,
    }
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(dest.parent),
            prefix="."+dest.stem+".", suffix=".json.tmp", delete=False,
        ) as fh:
            tmp = Path(fh.name)
            json.dump(data, fh, indent=2, sort_keys=True, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return dest


def audit_count_v1_capture(
    manifest_path: Path, *, max_bytes: int = 536_870_912,
    sky_lineage_manifest: Path | None = None,
) -> dict[str, Any]:
    """Reproduce exact COUNT_V1 SCI/VAR/COV for supported electron sources.

    The numerical replay is disabled for cleanup and non-electron inputs.
    No archive content is trusted without hash and source/output linkage.
    """
    from scorpio_pipe.stages.linearize import (
        _build_linearize_row_plan, _apply_linearize_row_plan,
        _apply_linearize_row_plan_var,
    )

    path = Path(manifest_path).resolve(strict=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("schema") != SCHEMA or data.get("status") != "FROZEN_REPLAY_ONLY"
            or data.get("science_uncertainty_qualified") is not False
            or data.get("production_variance_release_allowed") is not False):
        raise ValueError("Unknown COUNT_V1 validity capture contract")
    checked = {}
    for name in ("source", "output", "archive"):
        record = data.get(name)
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError("COUNT_V1 artifact provenance missing")
        file = Path(record["path"])
        if not file.is_absolute() or not file.is_file():
            raise ValueError("COUNT_V1 artifact path is missing/relative")
        if sha256_file(file) != record["sha256"]:
            raise ValueError(f"COUNT_V1 {name} SHA256 mismatch")
        checked[name] = file.resolve()
    if checked["archive"].parent != path.parent or checked["archive"] != path.with_suffix(".npz"):
        raise ValueError("COUNT_V1 sidecar archive identity mismatch")
    shape = data.get("shape_native_yx")
    if not isinstance(shape, list) or len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError("COUNT_V1 native geometry invalid")
    if type(max_bytes) is not int or max_bytes <= 0 or shape[0]*shape[1]*9 > max_bytes:
        raise ValueError("COUNT_V1 audit byte budget exceeded")
    with zipfile.ZipFile(checked["archive"]) as zipped:
        if sum(item.file_size for item in zipped.infolist()) > max_bytes + 4096:
            raise ValueError("COUNT_V1 archive decompression exceeds budget")
    with np.load(checked["archive"], allow_pickle=False) as z:
        if set(z.files) != {"valid", "lambda_map"}:
            raise ValueError("COUNT_V1 actual-validity archive schema mismatch")
        mask = z["valid"]
        lam = z["lambda_map"]
    if (mask.shape != tuple(shape) or mask.dtype != np.dtype(bool)
            or lam.shape != mask.shape or lam.dtype != np.dtype("float64")):
        raise ValueError("COUNT_V1 captured validity/wavelength geometry invalid")
    sky_lineage_state = "UNBOUND"
    if sky_lineage_manifest is not None:
        from scorpio_pipe.calib.sky_fit_lineage import audit_frozen_kelson_fit_lineage
        linked = audit_frozen_kelson_fit_lineage(
            sky_lineage_manifest, check_fit_against_products=True,
        )
        if Path(linked["residual_fits"]).resolve() != checked["source"]:
            raise ValueError("Linearize source is not the captured Kelson SkySub output")
        sky_lineage_state = "HASH_AND_FIT_REPLAY_VERIFIED"
    grid = data["grid"]
    wave0, dw, nlam = float(grid["wave0"]), float(grid["dw"]), grid["nlam"]
    if (not np.isfinite(wave0) or not np.isfinite(dw) or dw <= 0
            or type(nlam) is not int or nlam <= 0):
        raise ValueError("COUNT_V1 grid values invalid")
    with fits.open(checked["source"], memmap=False) as source, fits.open(
        checked["output"], memmap=False
    ) as output:
        if not all(k in source for k in ("SCI", "VAR", "MASK")):
            raise ValueError("COUNT_V1 replay requires exact source SCI/VAR/MASK")
        if not all(k in output for k in ("SCI", "VAR", "COV")):
            raise ValueError("COUNT_V1 output missing SCI/VAR/COV")
        sci = np.asarray(source["SCI"].data, dtype=np.float64)
        var = np.asarray(source["VAR"].data, dtype=np.float64)
        hdr = source[0].header
        out_hdr = output[0].header
        if (sci.shape != tuple(shape) or var.shape != sci.shape):
            raise ValueError("COUNT_V1 source FITS geometry mismatch")
        if (out_hdr.get("REBNORM") != "COUNT_V1" or
                output["SCI"].data.shape != (shape[0], nlam)):
            raise ValueError("COUNT_V1 output geometry/header mismatch")
        if (str(hdr.get("BUNIT") or "") != "e-"
                or str(hdr.get("SCORPUM") or "") != "ELECTRON"):
            return {
                "status": "HOLD_UNQUALIFIED_SOURCE_UNIT_TRANSFORM",
                "hashes_verified": True, "count_v1_replayed": False,
                "science_uncertainty_qualified": False,
                "production_variance_release_allowed": False,
            }
        if data["cleanup_executed"]:
            return {
                "status": "HOLD_UNCAPTURED_POST_LINEARIZE_SKY_CLEANUP",
                "hashes_verified": True, "count_v1_replayed": False,
                "science_uncertainty_qualified": False,
                "production_variance_release_allowed": False,
            }
        factor = 1.0 if not data["output_density"] else 1.0/dw
        if data["normalize_exptime"]:
            exptime = float(data["exptime_s"])
            if exptime <= 0 or not np.isfinite(exptime):
                raise ValueError("COUNT_V1 invalid normalization exposure time")
            factor /= exptime
        sci_out = np.asarray(output["SCI"].data)
        var_out = np.asarray(output["VAR"].data)
        cov_out = np.asarray(output["COV"].data)
        edges = wave0 + dw*np.arange(nlam+1, dtype=np.float64)
        max_delta_sci = max_delta_var = max_delta_cov = 0.0
        for row in range(shape[0]):
            plan = _build_linearize_row_plan(lam[row], edges)
            if plan is None:
                expected_sci = np.zeros(nlam)
                expected_var = np.zeros(nlam)
                expected_cov = np.zeros(nlam)
            else:
                expected_sci, expected_cov = _apply_linearize_row_plan(
                    sci[row], plan, valid_mask=mask[row],
                )
                expected_var = _apply_linearize_row_plan_var(
                    var[row], plan, valid_mask=mask[row],
                )
            expected_sci *= factor
            expected_var *= factor**2
            for actual, expected, name in (
                (sci_out[row], expected_sci, "SCI"),
                (var_out[row], expected_var, "VAR"),
                (cov_out[row], expected_cov, "COV"),
            ):
                if not np.allclose(actual, expected, rtol=2e-5, atol=2e-4, equal_nan=False):
                    raise ValueError(f"COUNT_V1 output {name} differs from frozen operator replay")
            max_delta_sci = max(max_delta_sci, float(np.max(np.abs(sci_out[row]-expected_sci))))
            max_delta_var = max(max_delta_var, float(np.max(np.abs(var_out[row]-expected_var))))
            max_delta_cov = max(max_delta_cov, float(np.max(np.abs(cov_out[row]-expected_cov))))
    return {
        "status": "HOLD_REPLAYED_COUNT_V1_WITHOUT_SKY_TO_1D_ADMISSION",
        "hashes_verified": True, "count_v1_replayed": True,
        "max_delta_sci": max_delta_sci, "max_delta_var": max_delta_var,
        "max_delta_cov": max_delta_cov,
        "sky_lineage": sky_lineage_state,
        "science_uncertainty_qualified": False,
        "production_variance_release_allowed": False,
    }
