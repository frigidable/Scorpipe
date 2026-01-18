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

    if name.endswith("spec1d.fits") or name.startswith("spec1d"):
        validate_spec1d_product(p, stage=stage)
        return

    validate_mef2d_product(p, stage=stage)


def validate_task_outputs(paths: Iterable[str | Path], *, stage: str) -> None:
    for x in paths:
        validate_product(x, stage=stage)