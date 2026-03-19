"""Product validators for the data model contract.

These validators are intended to be called at pipeline stage boundaries.
They must be deterministic and must *not* silently accept ambiguous products.

Hard-fail policy
----------------
Violations raise :class:`ProductContractError`.

This module is the new "law". Legacy modules (e.g. :mod:`scorpio_pipe.boundary_contract`)
may re-export these symbols for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scorpio_pipe.contracts.calibration import (
    CalibrationProductKind,
    CalibrationStatus,
    calibration_semantics_summary,
    normalize_calibration_status,
    telluric_semantics_required,
    validate_calibration_claim,
)
from scorpio_pipe.contracts.data_model import DATA_MODEL_VERSION, get_data_model_version
from scorpio_pipe.contracts import maskbits
from scorpio_pipe.contracts.units import (
    infer_sci_bunit_from_unit_model,
    is_recognized_sci_bunit,
    is_recognized_var_bunit,
    parse_bunit,
    var_bunit_from_sci_bunit,
)


class ProductContractError(RuntimeError):
    """Stage-scoped contract violation."""

    def __init__(self, *, stage: str, path: str, code: str, message: str) -> None:
        self.stage = str(stage)
        self.path = str(path)
        self.code = str(code)
        self.message = str(message)
        super().__init__(f"[{self.stage}] {self.code}: {self.message} ({self.path})")


def _has_wavelength_wcs(hdr: Any) -> bool:
    try:
        ctype1 = str(hdr.get("CTYPE1", "") or "").upper()
        if ctype1.startswith("WAVE") or ctype1.startswith("AWAV"):
            return True
        if "SCORP_L0" in hdr and "SCORP_DL" in hdr and "SCORP_NL" in hdr:
            return True
    except Exception:
        return False
    return False


def _require_data_model(stage: str, path: Path, hdr0: Any) -> None:
    v = get_data_model_version(hdr0)
    if v is None:
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code="DATA_MODEL_MISSING",
            message="Missing SCORPDMV (data model version)",
        )
    if int(v) != int(DATA_MODEL_VERSION):
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code="DATA_MODEL_MISMATCH",
            message=f"SCORPDMV mismatch: file={v} pipeline={DATA_MODEL_VERSION}",
        )


def _require_noise_provenance(stage: str, path: Path, hdr0: Any) -> None:
    # Compact <=8-char cards are preferred.
    has_gain = hdr0.get("GAIN") is not None or hdr0.get("SCORPGN") is not None
    has_rn = hdr0.get("RDNOISE") is not None or hdr0.get("SCORPRN") is not None
    has_src = hdr0.get("NOISRC") is not None or hdr0.get("SCORPNS") is not None

    if not (has_gain and has_rn and has_src):
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code="NOISE_PROV_MISSING",
            message="Missing noise provenance (need gain, read-noise and source: GAIN/RDNOISE/NOISRC or SCORPGN/SCORPRN/SCORPNS)",
        )


def _require_mask_schema(stage: str, path: Path, hdr0: Any) -> None:
    mkv = hdr0.get("SCORPMKV", None)
    if mkv is None:
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code="MASK_SCHEMA_MISSING",
            message="Missing SCORPMKV (mask schema version)",
        )
    try:
        mkv_i = int(mkv)
    except Exception:
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code="MASK_SCHEMA_INVALID",
            message=f"Invalid SCORPMKV={mkv!r} (expected int)",
        )

    if mkv_i != int(maskbits.MASK_SCHEMA_VERSION):
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code="MASK_SCHEMA_MISMATCH",
            message=f"MASK schema version mismatch: file={mkv_i} pipeline={int(maskbits.MASK_SCHEMA_VERSION)}",
        )

    for i in range(0, 10):
        k = f"SCORPMB{i}"
        if k not in hdr0:
            raise ProductContractError(
                stage=stage,
                path=str(path),
                code="MASK_SCHEMA_INCOMPLETE",
                message=f"Missing {k} in mask schema header cards",
            )


def _require_waveref(stage: str, path: Path, hdr0: Any) -> None:
    if not _has_wavelength_wcs(hdr0):
        return
    ref = str(hdr0.get("WAVEREF", "") or "").strip().lower()
    if ref in {"air", "vacuum", "vac", "vacuo", "unknown"}:
        # unknown is allowed only for early debugging products; but if WAVE axis exists,
        # downstream expects a physical reference. Treat unknown as missing.
        if ref == "unknown":
            raise ProductContractError(
                stage=stage,
                path=str(path),
                code="WAVEREF_MISSING",
                message="Wavelength medium is not declared (expected WAVEREF=air or vacuum)",
            )
        return
    raise ProductContractError(
        stage=stage,
        path=str(path),
        code="WAVEREF_MISSING",
        message="Wavelength medium is not declared (expected WAVEREF=air or vacuum)",
    )


def _validate_plane_arrays(stage: str, path: Path, sci: np.ndarray, var: np.ndarray, mask: np.ndarray) -> None:
    if sci.shape != var.shape or sci.shape != mask.shape:
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code="SHAPE_MISMATCH",
            message=f"SCI/VAR/MASK shape mismatch: sci={sci.shape} var={var.shape} mask={mask.shape}",
        )
    if not np.issubdtype(sci.dtype, np.floating):
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code="DTYPE",
            message=f"SCI dtype must be float, got {sci.dtype}",
        )
    if not np.issubdtype(var.dtype, np.floating):
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code="DTYPE",
            message=f"VAR dtype must be float, got {var.dtype}",
        )
    if mask.dtype != np.uint16 and mask.dtype != np.uint32:
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code="DTYPE",
            message=f"MASK dtype must be uint16/uint32, got {mask.dtype}",
        )

    # Hard contract: no NaNs or negative variance.
    if not np.isfinite(sci).all():
        raise ProductContractError(stage=stage, path=str(path), code="SCI_NONFINITE", message="SCI contains NaN/Inf")
    if not np.isfinite(var).all():
        raise ProductContractError(stage=stage, path=str(path), code="VAR_NONFINITE", message="VAR contains NaN/Inf")
    if np.any(var < 0):
        raise ProductContractError(stage=stage, path=str(path), code="VAR_NEGATIVE", message="VAR contains negative values")


def _require_bunit(stage: str, path: Path, sci_hdr: Any, var_hdr: Any, hdr0: Any) -> None:
    bunit_sci = str(sci_hdr.get("BUNIT", "") or "").strip()
    bunit_var = str(var_hdr.get("BUNIT", "") or "").strip()

    if not bunit_sci:
        # fallback: primary may still carry BUNIT (legacy writer)
        bunit_sci = str(hdr0.get("BUNIT", "") or "").strip()

    if not bunit_sci:
        # last resort: infer from SCORPUM
        bunit_sci = infer_sci_bunit_from_unit_model(str(hdr0.get("SCORPUM", "") or "")) or ""

    if not bunit_sci:
        raise ProductContractError(stage=stage, path=str(path), code="BUNIT_MISSING", message="Missing BUNIT for SCI")

    if not is_recognized_sci_bunit(bunit_sci):
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code="BUNIT_INVALID",
            message=f"Unrecognized SCI BUNIT={bunit_sci!r}",
        )

    if not bunit_var:
        raise ProductContractError(stage=stage, path=str(path), code="BUNIT_MISSING", message="Missing BUNIT for VAR")

    if not is_recognized_var_bunit(bunit_var):
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code="BUNIT_INVALID",
            message=f"Unrecognized VAR BUNIT={bunit_var!r}",
        )

    # Consistency check (family + /s): compare against deterministic expected value.
    expected = var_bunit_from_sci_bunit(bunit_sci)
    fam_exp = parse_bunit(expected)
    fam_var = parse_bunit(bunit_var)

    if fam_exp.base != fam_var.base or fam_exp.per_second != fam_var.per_second:
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code="BUNIT_MISMATCH",
            message=f"VAR BUNIT={bunit_var!r} is inconsistent with SCI BUNIT={bunit_sci!r} (expected family like {expected!r})",
        )


def _require_flux_like_unit(stage: str, path: Path, unit_value: str, *, header_key: str) -> None:
    fam = parse_bunit(unit_value)
    if fam.base != "physical_flux":
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code="FLUXUNIT_INVALID",
            message=f"{header_key} must declare a physical flux unit, got {unit_value!r}",
        )


def _require_calibration_semantics(stage: str, path: Path, hdr0: Any, *, product_kind: str | CalibrationProductKind) -> None:
    try:
        status = validate_calibration_claim(hdr0.get("CALSTAT", None), product_kind=product_kind)
    except Exception as e:
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code="CALSTAT_INVALID",
            message=f"Missing/invalid CALSTAT: {e}",
        )

    flux_unit = str(hdr0.get("FLUXUNIT", "") or hdr0.get("BUNIT", "")).strip()
    if not flux_unit:
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code="FLUXUNIT_MISSING",
            message="Missing FLUXUNIT/BUNIT in calibrated product",
        )
    _require_flux_like_unit(stage, path, flux_unit, header_key="FLUXUNIT")

    primary_bunit = str(hdr0.get("BUNIT", "") or "").strip()
    if primary_bunit:
        _require_flux_like_unit(stage, path, primary_bunit, header_key="BUNIT")
        fam_flux = parse_bunit(flux_unit)
        fam_bunit = parse_bunit(primary_bunit)
        if fam_flux.base != fam_bunit.base or fam_flux.per_second != fam_bunit.per_second:
            raise ProductContractError(
                stage=stage,
                path=str(path),
                code="FLUXUNIT_MISMATCH",
                message=f"FLUXUNIT={flux_unit!r} is inconsistent with BUNIT={primary_bunit!r}",
            )

    refctid = str(hdr0.get("REFCTID", "") or "").strip()
    if not refctid:
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code="REFCTID_MISSING",
            message="Missing REFCTID link to reference context",
        )

    sensfunc = str(hdr0.get("SENSFUNC", "") or "").strip()
    if not sensfunc:
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code="SENSFUNC_MISSING",
            message="Missing SENSFUNC reference in calibrated product",
        )

    telluric = str(hdr0.get("TELLURIC", "") or "").strip()
    if telluric_semantics_required(status) and not telluric:
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code="TELLURIC_MISSING",
            message="Missing TELLURIC reference for telluric-aware calibrated product",
        )


def validate_mef2d_product(path: str | Path, *, stage: str) -> None:
    """Validate a 2D MEF product: SCI/VAR/MASK."""

    p = Path(path)
    try:
        from astropy.io import fits  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ProductContractError(stage=stage, path=str(p), code="IO_ERROR", message=str(e))

    try:
        with fits.open(p, memmap=False) as hdul:
            hdr0 = hdul[0].header

            if "SCI" not in hdul:
                raise ProductContractError(stage=stage, path=str(p), code="SCI_MISSING", message="Missing SCI extension")
            if "VAR" not in hdul:
                raise ProductContractError(stage=stage, path=str(p), code="VAR_MISSING", message="Missing VAR extension")
            if "MASK" not in hdul:
                raise ProductContractError(stage=stage, path=str(p), code="MASK_MISSING", message="Missing MASK extension")

            sci = np.asarray(hdul["SCI"].data)
            var = np.asarray(hdul["VAR"].data)
            msk = np.asarray(hdul["MASK"].data)

            sci_hdr = hdul["SCI"].header
            var_hdr = hdul["VAR"].header

        _validate_plane_arrays(stage, p, sci, var, msk)
        _require_data_model(stage, p, hdr0)
        _require_noise_provenance(stage, p, hdr0)
        _require_mask_schema(stage, p, hdr0)
        _require_waveref(stage, p, hdr0)
        _require_bunit(stage, p, sci_hdr, var_hdr, hdr0)

    except ProductContractError:
        raise
    except Exception as e:
        raise ProductContractError(stage=stage, path=str(p), code="MEF_CONTRACT", message=str(e))


def validate_spec1d_product(path: str | Path, *, stage: str) -> None:
    """Validate a 1D spectrum product."""

    p = Path(path)
    try:
        from astropy.io import fits  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ProductContractError(stage=stage, path=str(p), code="IO_ERROR", message=str(e))

    def _require_cols(hdu, cols: list[str]) -> None:
        names = {str(n).upper() for n in (hdu.columns.names or [])}
        miss = [c for c in cols if c.upper() not in names]
        if miss:
            raise ValueError(f"Missing required columns: {', '.join(miss)}")

    try:
        with fits.open(p, memmap=False) as hdul:
            hdr0 = hdul[0].header
            _require_data_model(stage, p, hdr0)

            if "SPEC_TRACE" not in hdul or "SPEC_FIXED" not in hdul:
                raise ValueError("Missing required extensions SPEC_TRACE and/or SPEC_FIXED")

            ht = hdul["SPEC_TRACE"]
            hf = hdul["SPEC_FIXED"]
            _require_cols(ht, ["LAMBDA", "FLUX_TRACE", "VAR_TRACE", "MASK_TRACE"])
            _require_cols(hf, ["LAMBDA", "FLUX_FIXED", "VAR_FIXED", "MASK_FIXED"])

            lam = np.asarray(ht.data["LAMBDA"], dtype=float)
            var_t = np.asarray(ht.data["VAR_TRACE"], dtype=float)
            var_f = np.asarray(hf.data["VAR_FIXED"], dtype=float)

            if not np.all(np.isfinite(lam)):
                raise ValueError("Non-finite wavelength values")
            if np.any(var_t < 0) or np.any(var_f < 0):
                raise ValueError("Negative variance values")

            ref = str(hdr0.get("WAVEREF", "") or "").strip().lower()
            if ref not in {"air", "vacuum", "vac", "vacuo"}:
                raise ValueError("Missing/invalid WAVEREF in primary header")

    except ProductContractError:
        raise
    except Exception as e:
        raise ProductContractError(stage=stage, path=str(p), code="SPEC1D_CONTRACT", message=str(e))



def validate_calibrated_mef2d_product(path: str | Path, *, stage: str) -> None:
    p = Path(path)
    validate_mef2d_product(p, stage=stage)
    try:
        from astropy.io import fits  # type: ignore
        with fits.open(p, memmap=False) as hdul:
            hdr0 = hdul[0].header
            sci_hdr = hdul["SCI"].header
            var_hdr = hdul["VAR"].header
        _require_calibration_semantics(stage, p, hdr0, product_kind=CalibrationProductKind.MEF2D)
        sci_bunit = str(sci_hdr.get("BUNIT", "") or hdr0.get("BUNIT", "")).strip()
        _require_flux_like_unit(stage, p, sci_bunit, header_key="SCI.BUNIT")
        expected_var = var_bunit_from_sci_bunit(sci_bunit)
        fam_exp = parse_bunit(expected_var)
        fam_var = parse_bunit(str(var_hdr.get("BUNIT", "") or ""))
        if fam_exp.base != fam_var.base or fam_exp.per_second != fam_var.per_second:
            raise ProductContractError(
                stage=stage,
                path=str(p),
                code="BUNIT_MISMATCH",
                message=f"VAR BUNIT={str(var_hdr.get('BUNIT', '') or '')!r} is inconsistent with calibrated SCI BUNIT={sci_bunit!r}",
            )
    except ProductContractError:
        raise
    except Exception as e:
        raise ProductContractError(stage=stage, path=str(p), code="CALIBRATED_MEF_CONTRACT", message=str(e))


def validate_calibrated_spec1d_product(path: str | Path, *, stage: str) -> None:
    p = Path(path)
    validate_spec1d_product(p, stage=stage)
    try:
        from astropy.io import fits  # type: ignore
        with fits.open(p, memmap=False) as hdul:
            hdr0 = hdul[0].header
        _require_calibration_semantics(stage, p, hdr0, product_kind=CalibrationProductKind.SPEC1D)
        primary_bunit = str(hdr0.get("BUNIT", "") or hdr0.get("FLUXUNIT", "")).strip()
        _require_flux_like_unit(stage, p, primary_bunit, header_key="BUNIT")
    except ProductContractError:
        raise
    except Exception as e:
        raise ProductContractError(stage=stage, path=str(p), code="CALIBRATED_SPEC1D_CONTRACT", message=str(e))


def validate_lambda_map_product(path: str | Path, *, stage: str) -> None:
    """Validate wavesolution ``lambda_map.fits`` lookup product.

    ``lambda_map.fits`` is a 2D lookup image of wavelength (λ) for each detector
    pixel. It is not a standard SCI/VAR/MASK MEF, so it has its own minimal
    contract.

    Required header cards (Primary HDU)
    ---------------------------------
    - ``SCORPVER``: pipeline version provenance
    - ``WAVEUNIT`` (or ``CUNIT1``): wavelength unit string (e.g. ``Angstrom``)
    - ``WAVEREF``: ``air`` or ``vacuum``

    Required data properties
    ------------------------
    - 2D array
    - finite values (no NaN/Inf)
    """

    pp = Path(path)
    if not pp.is_file():
        raise ProductContractError(
            stage=stage,
            path=str(pp),
            code="MISSING_PRODUCT",
            message="lambda_map.fits not found",
        )

    try:
        from astropy.io import fits  # type: ignore

        with fits.open(pp, memmap=False) as hdul:
            if not hdul or hdul[0].data is None:
                raise ValueError("Missing primary image")
            data = np.asarray(hdul[0].data, dtype=float)
            if data.ndim != 2:
                raise ValueError("lambda_map must be a 2D image")
            if not np.all(np.isfinite(data)):
                raise ValueError("Non-finite values in lambda_map")

            hdr0 = hdul[0].header
            if not str(hdr0.get("SCORPVER", "") or "").strip():
                raise ValueError("Missing SCORPVER provenance")

            wave_unit = str(hdr0.get("WAVEUNIT", "") or hdr0.get("CUNIT1", "") or "").strip()
            if not wave_unit:
                raise ValueError("Missing WAVEUNIT/CUNIT1")

            cunit1 = str(hdr0.get("CUNIT1", "") or "").strip()
            if cunit1 and cunit1 != wave_unit:
                raise ValueError("CUNIT1 does not match WAVEUNIT")

            ctype1 = str(hdr0.get("CTYPE1", "") or "").upper().strip()
            if ctype1 and not (ctype1.startswith("WAVE") or ctype1.startswith("AWAV")):
                raise ValueError("CTYPE1 must indicate wavelength axis")

            waveref = str(hdr0.get("WAVEREF", "") or hdr0.get("LAMREF", "") or "").strip().lower()
            if waveref not in {"air", "vacuum"}:
                raise ValueError("Missing/invalid WAVEREF")

    except ProductContractError:
        raise
    except Exception as e:
        raise ProductContractError(stage=stage, path=str(pp), code="LAMBDA_MAP_CONTRACT", message=str(e))


def validate_product(path: str | Path, *, stage: str) -> None:
    """Auto-detect and validate a product."""

    p = Path(path)
    if p.suffix.lower() != ".fits":
        return

    # Heuristic: spec1d contains tables.
    name = p.name.lower()

    if name.endswith("lambda_map.fits"):
        validate_lambda_map_product(p, stage=stage)
        return

    if name.endswith("fluxcal_spec1d.fits"):
        validate_calibrated_spec1d_product(p, stage=stage)
        return

    if name.endswith("spec1d.fits") or name.startswith("spec1d"):
        validate_spec1d_product(p, stage=stage)
        return

    if name.endswith("fluxcal_2d.fits"):
        validate_calibrated_mef2d_product(p, stage=stage)
        return

    validate_mef2d_product(p, stage=stage)


def validate_task_outputs(paths: Iterable[str | Path], *, stage: str) -> None:
    for x in paths:
        validate_product(x, stage=stage)
# ----------------------------------------------------------------------------
# Calibration contracts (P0-ARCH-09)
# ----------------------------------------------------------------------------


def _read_primary_2d(path: Path, *, stage: str, code_prefix: str) -> tuple[np.ndarray, Any]:
    """Read primary data array + header (2D) with strict checks."""

    try:
        from astropy.io import fits  # type: ignore
    except Exception as e:
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code=f"{code_prefix}_fits_import_error",
            message=f"Failed to import astropy.io.fits: {e}",
        ) from e

    try:
        data, hdr = fits.getdata(path, header=True, memmap=False)
    except Exception as e:
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code=f"{code_prefix}_unreadable",
            message=f"Cannot read FITS: {e}",
        ) from e

    if data is None:
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code=f"{code_prefix}_empty",
            message="FITS contains no data",
        )

    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code=f"{code_prefix}_ndim",
            message=f"Expected 2D image, got ndim={arr.ndim}",
        )

    return arr, hdr


def validate_calib_bias_triplet(stage_dir: str | Path, *, stage: str) -> None:
    """Validate MasterBias triplet outputs (SCI + VAR + DQ).

    Contract kind: calib_bias_triplet

    Expected files (in the stage directory):
      - master_bias.fits
      - master_bias_var.fits
      - master_bias_dq.fits

    DoD goals:
      - shapes agree
      - finite arrays
      - VAR >= 0
      - DQ integer mask
      - minimal provenance cards present
    """

    sd = Path(stage_dir).expanduser().resolve()
    sci_p = (sd / "master_bias.fits").resolve()
    var_p = (sd / "master_bias_var.fits").resolve()
    dq_p = (sd / "master_bias_dq.fits").resolve()

    if not sci_p.exists():
        raise ProductContractError(stage=stage, path=str(sci_p), code="bias_missing", message="Missing master_bias.fits")
    if not var_p.exists():
        raise ProductContractError(stage=stage, path=str(var_p), code="bias_var_missing", message="Missing master_bias_var.fits")
    if not dq_p.exists():
        raise ProductContractError(stage=stage, path=str(dq_p), code="bias_dq_missing", message="Missing master_bias_dq.fits")

    sci, hsci = _read_primary_2d(sci_p, stage=stage, code_prefix="bias")
    var, _hvar = _read_primary_2d(var_p, stage=stage, code_prefix="bias_var")
    dq, _hdq = _read_primary_2d(dq_p, stage=stage, code_prefix="bias_dq")

    if var.shape != sci.shape:
        raise ProductContractError(
            stage=stage,
            path=str(var_p),
            code="bias_var_shape_mismatch",
            message=f"VAR shape mismatch: sci={tuple(sci.shape)} var={tuple(var.shape)}",
        )
    if dq.shape != sci.shape:
        raise ProductContractError(
            stage=stage,
            path=str(dq_p),
            code="bias_dq_shape_mismatch",
            message=f"DQ shape mismatch: sci={tuple(sci.shape)} dq={tuple(dq.shape)}",
        )

    # SCI and VAR must be finite.
    if not np.isfinite(sci).all():
        raise ProductContractError(stage=stage, path=str(sci_p), code="bias_has_nan", message="master_bias contains NaN/Inf")
    if not np.isfinite(var).all():
        raise ProductContractError(stage=stage, path=str(var_p), code="bias_var_has_nan", message="master_bias_var contains NaN/Inf")
    if np.any(var < 0):
        raise ProductContractError(stage=stage, path=str(var_p), code="bias_var_negative", message="master_bias_var contains negative values")

    # DQ must be integer-like.
    if not np.issubdtype(dq.dtype, np.integer):
        raise ProductContractError(
            stage=stage,
            path=str(dq_p),
            code="bias_dq_dtype",
            message=f"master_bias_dq must be integer mask, got dtype={dq.dtype}",
        )

    # Minimal provenance (superbias stamps these cards).
    has_nbias = hsci.get("NBIAS") is not None
    has_gid = hsci.get("SB_GID") is not None
    has_meth = hsci.get("SB_METH") is not None

    if not (has_nbias and has_gid and has_meth):
        raise ProductContractError(
            stage=stage,
            path=str(sci_p),
            code="bias_provenance_missing",
            message="Missing superbias provenance cards (need NBIAS, SB_GID, SB_METH)",
        )


def validate_calib_flat(stage_dir: str | Path, *, stage: str) -> None:
    """Validate superflat output.

    Contract kind: calib_flat

    Expected file:
      - superflat.fits

    Sanity:
      - readable 2D image
      - finite
      - not degenerate (median positive, non-zero spread)
      - minimal provenance: superbias selection present
    """

    sd = Path(stage_dir).expanduser().resolve()
    p = (sd / "superflat.fits").resolve()
    if not p.exists():
        raise ProductContractError(stage=stage, path=str(p), code="flat_missing", message="Missing superflat.fits")

    img, hdr = _read_primary_2d(p, stage=stage, code_prefix="flat")

    if not np.isfinite(img).all():
        raise ProductContractError(stage=stage, path=str(p), code="flat_has_nan", message="superflat contains NaN/Inf")

    med = float(np.median(img))
    if not np.isfinite(med) or med <= 0:
        raise ProductContractError(stage=stage, path=str(p), code="flat_bad_median", message=f"superflat median is not positive (median={med})")

    # Minimal provenance: superflat stamps superbias selection.
    has_sb = (hdr.get("SB_FULL") is not None) or (hdr.get("SB_PATH") is not None)
    if not has_sb:
        raise ProductContractError(
            stage=stage,
            path=str(p),
            code="flat_provenance_missing",
            message="Missing superbias provenance in header (expected SB_FULL or SB_PATH)",
        )


def validate_calib_neon(stage_dir: str | Path, *, stage: str) -> None:
    """Validate superneon output.

    Contract kind: calib_neon

    Expected file:
      - superneon.fits

    Sanity:
      - readable 2D image
      - finite
      - not empty/constant
      - minimal QC header cards
    """

    sd = Path(stage_dir).expanduser().resolve()
    p = (sd / "superneon.fits").resolve()
    if not p.exists():
        raise ProductContractError(stage=stage, path=str(p), code="neon_missing", message="Missing superneon.fits")

    img, hdr = _read_primary_2d(p, stage=stage, code_prefix="neon")

    if not np.isfinite(img).all():
        raise ProductContractError(stage=stage, path=str(p), code="neon_has_nan", message="superneon contains NaN/Inf")

    # Must not be completely constant.
    spread = float(np.nanpercentile(img, 99) - np.nanpercentile(img, 1))
    if not np.isfinite(spread) or spread == 0.0:
        raise ProductContractError(stage=stage, path=str(p), code="neon_empty", message="superneon appears empty/constant")

    # Minimal QC cards written by superneon stage.
    if hdr.get("NPK") is None:
        raise ProductContractError(stage=stage, path=str(p), code="neon_header_missing", message="Missing NPK (peak count) header card")



def _read_bintable(path: Path, *, stage: str, code_prefix: str) -> tuple[Any, Any]:
    try:
        from astropy.io import fits  # type: ignore
    except Exception as e:
        raise ProductContractError(stage=stage, path=str(path), code=f"{code_prefix}_fits_import_error", message=str(e))
    try:
        with fits.open(path, memmap=False) as hdul:
            if len(hdul) < 2 or not isinstance(hdul[1], fits.BinTableHDU):
                raise ValueError('missing binary table extension')
            return hdul[0].header, hdul[1].data
    except ProductContractError:
        raise
    except Exception as e:
        raise ProductContractError(stage=stage, path=str(path), code=f"{code_prefix}_unreadable", message=str(e))


def validate_sensfunc_product(path: str | Path, *, stage: str) -> None:
    p = Path(path)
    hdr, tab = _read_bintable(p, stage=stage, code_prefix='sensfunc')
    need = {'WAVELENGTH', 'RESPONSE_ABS', 'RESPONSE_REL', 'RESPONSE_UNC', 'FIT_MASK'}
    names = set(getattr(tab, 'names', []) or [])
    miss = sorted(need - names)
    if miss:
        raise ProductContractError(stage=stage, path=str(p), code='SENSFUNC_COLUMNS', message=f'Missing columns: {miss}')
    w = np.asarray(tab['WAVELENGTH'], dtype=float)
    r = np.asarray(tab['RESPONSE_ABS'], dtype=float)
    u = np.asarray(tab['RESPONSE_UNC'], dtype=float)
    if w.ndim != 1 or w.size < 4:
        raise ProductContractError(stage=stage, path=str(p), code='SENSFUNC_SHAPE', message='Need 1D wavelength grid with >=4 samples')
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(r)) or not np.all(np.isfinite(u)):
        raise ProductContractError(stage=stage, path=str(p), code='SENSFUNC_NONFINITE', message='Non-finite values in sensfunc table')
    if np.any(r <= 0) or np.any(u < 0):
        raise ProductContractError(stage=stage, path=str(p), code='SENSFUNC_RANGE', message='Invalid sensfunc response/uncertainty range')
    _require_waveref(stage, p, hdr)
    if not str(hdr.get('REFCTID', '') or '').strip():
        raise ProductContractError(stage=stage, path=str(p), code='SENSFUNC_REFCTID', message='Missing REFCTID')


def validate_telluric_model_product(path: str | Path, *, stage: str) -> None:
    p = Path(path)
    hdr, tab = _read_bintable(p, stage=stage, code_prefix='telluric')
    need = {'WAVELENGTH', 'TRANSMISSION', 'CONFIDENCE', 'MODEL_MASK'}
    names = set(getattr(tab, 'names', []) or [])
    miss = sorted(need - names)
    if miss:
        raise ProductContractError(stage=stage, path=str(p), code='TELLURIC_COLUMNS', message=f'Missing columns: {miss}')
    t = np.asarray(tab['TRANSMISSION'], dtype=float)
    c = np.asarray(tab['CONFIDENCE'], dtype=float)
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(c)):
        raise ProductContractError(stage=stage, path=str(p), code='TELLURIC_NONFINITE', message='Non-finite telluric arrays')
    if np.any(t <= 0) or np.any(t > 1.5) or np.any(c < 0) or np.any(c > 1.0):
        raise ProductContractError(stage=stage, path=str(p), code='TELLURIC_RANGE', message='Telluric transmission/confidence out of range')
    _require_waveref(stage, p, hdr)
    if not str(hdr.get('REFCTID', '') or '').strip():
        raise ProductContractError(stage=stage, path=str(p), code='TELLURIC_REFCTID', message='Missing REFCTID')


def validate_calibration_systematics_product(path: str | Path, *, stage: str) -> None:
    p = Path(path)
    try:
        from astropy.io import fits  # type: ignore
        with fits.open(p, memmap=False) as hdul:
            hdr0 = hdul[0].header
            names = {h.name for h in hdul[1:]}
            need = {'WAVE', 'CALSYS_REL', 'CALSYS_TELL', 'CALSYS_ABS', 'CALSYS_TOTAL'}
            miss = sorted(need - names)
            if miss:
                raise ProductContractError(stage=stage, path=str(p), code='CALSYS_EXTENSIONS', message=f'Missing HDUs: {miss}')
            if not str(hdr0.get('REFCTID', '') or '').strip():
                raise ProductContractError(stage=stage, path=str(p), code='CALSYS_REFCTID', message='Missing REFCTID')
            if not str(hdr0.get('WAVEREF', '') or '').strip().lower() in {'air', 'vacuum'}:
                raise ProductContractError(stage=stage, path=str(p), code='CALSYS_WAVEREF', message='Missing/invalid WAVEREF')
            try:
                validate_calibration_claim(hdr0.get('CALSTAT', None), product_kind=CalibrationProductKind.SPEC1D)
            except Exception as e:
                raise ProductContractError(stage=stage, path=str(p), code='CALSYS_CALSTAT', message=str(e))
            sysunit = str(hdr0.get('SYSUNIT', '') or '').strip().lower()
            if sysunit not in {'fraction', 'relative'}:
                raise ProductContractError(stage=stage, path=str(p), code='CALSYS_SYSUNIT', message='SYSUNIT must be fraction/relative')
            wave = np.asarray(hdul['WAVE'].data, dtype=float)
            rel = np.asarray(hdul['CALSYS_REL'].data, dtype=float)
            tel = np.asarray(hdul['CALSYS_TELL'].data, dtype=float)
            abs_ = np.asarray(hdul['CALSYS_ABS'].data, dtype=float)
            tot = np.asarray(hdul['CALSYS_TOTAL'].data, dtype=float)
            if not np.all(np.isfinite(wave)) or not np.all(np.isfinite(rel)) or not np.all(np.isfinite(tel)) or not np.all(np.isfinite(abs_)) or not np.all(np.isfinite(tot)):
                raise ProductContractError(stage=stage, path=str(p), code='CALSYS_NONFINITE', message='Calibration systematics contain NaN/Inf')
            if np.any(rel < 0) or np.any(tel < 0) or np.any(abs_ < 0) or np.any(tot < 0):
                raise ProductContractError(stage=stage, path=str(p), code='CALSYS_NEGATIVE', message='Calibration systematics must be non-negative')
            if rel.shape != tel.shape or rel.shape != abs_.shape or rel.shape != tot.shape or rel.shape != wave.shape:
                raise ProductContractError(stage=stage, path=str(p), code='CALSYS_SHAPE', message='Calibration systematic layers shape mismatch')
            expected = np.sqrt(rel * rel + tel * tel + abs_ * abs_)
            if np.nanmax(np.abs(expected - tot)) > 5e-4:
                raise ProductContractError(stage=stage, path=str(p), code='CALSYS_TOTAL_MISMATCH', message='CALSYS_TOTAL must equal quadrature(rel,tell,abs)')
    except ProductContractError:
        raise
    except Exception as e:
        raise ProductContractError(stage=stage, path=str(p), code='CALSYS_UNREADABLE', message=str(e))
