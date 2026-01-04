"""Boundary contract enforcement (P0-PROV-001).

The pipeline moves *products* between stages. To keep science honest and the UX
predictable, we validate products at stage boundaries.

Contract checks (hard fail)
---------------------------
2D MEF products (SCI/VAR/MASK):
* Shapes consistent; SCI finite; VAR >= 0; MASK dtype is uint16.
* Explicit unit tags (SCORPUM) and noise provenance (SCORPGN/SCORPRN).
* If a wavelength axis exists, WAVEREF must be declared once (air/vacuum).

1D spec products (spec1d.fits):
* Arrays/columns lengths consistent; finite wavelength; VAR >= 0.
* WAVEREF declared.

All errors are raised as :class:`ProductContractError` with a stage-scoped,
actionable message.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ProductContractError(RuntimeError):
    stage: str
    path: str
    code: str
    message: str

    def __str__(self) -> str:  # pragma: no cover
        return f"[{self.stage}] {self.code}: {self.message} ({self.path})"


def _has_wavelength_wcs(hdr: Any) -> bool:
    try:
        ctype1 = str(hdr.get("CTYPE1", "") or "").upper()
        if ctype1.startswith("WAVE") or ctype1.startswith("AWAV"):
            return True
        # MEF WaveGrid cards used by scorpio_pipe
        if "SCORP_L0" in hdr and "SCORP_DL" in hdr and "SCORP_NL" in hdr:
            return True
    except Exception:
        return False
    return False


def _require_units_and_noise(stage: str, path: Path, hdr: Any) -> None:
    unit = str(hdr.get("SCORPUM", "") or "").strip().upper()
    if unit not in {"ADU", "ELECTRON"}:
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code="UNITS_MISSING",
            message="Missing/invalid SCORPUM unit model (expected ADU or ELECTRON)",
        )
    if hdr.get("SCORPGN") is None or hdr.get("SCORPRN") is None:
        raise ProductContractError(
            stage=stage,
            path=str(path),
            code="NOISE_PROV_MISSING",
            message="Missing noise provenance (expected SCORPGN gain and SCORPRN read-noise)",
        )


def _require_waveref(stage: str, path: Path, hdr: Any) -> None:
    if not _has_wavelength_wcs(hdr):
        return
    ref = str(hdr.get("WAVEREF", "") or "").strip().lower()
    if ref in {"air", "vacuum"}:
        return
    # tolerate common shorthand
    if ref in {"vac", "vacuo"}:
        return
    raise ProductContractError(
        stage=stage,
        path=str(path),
        code="WAVEREF_MISSING",
        message="Wavelength medium is not declared (expected WAVEREF=air or vacuum)",
    )


def validate_mef_product(path: str | Path, *, stage: str) -> None:
    """Validate a 2D MEF product."""

    p = Path(path)
    try:
        from scorpio_pipe.io.mef import read_sci_var_mask
    except Exception as e:  # pragma: no cover
        raise ProductContractError(stage=stage, path=str(p), code="IO_ERROR", message=str(e))

    try:
        sci, var, mask, hdr = read_sci_var_mask(p, validate=True)
    except Exception as e:
        # preserve original error text, but make it stage-scoped.
        raise ProductContractError(stage=stage, path=str(p), code="SHAPE/FINITE/VAR/MASK", message=str(e))

    # Require all core planes.
    if var is None:
        raise ProductContractError(
            stage=stage,
            path=str(p),
            code="VAR_MISSING",
            message="VAR extension is missing (2D products must carry variance)",
        )
    if mask is None:
        raise ProductContractError(
            stage=stage,
            path=str(p),
            code="MASK_MISSING",
            message="MASK extension is missing (2D products must carry mask bits)",
        )

    # Header-level checks.
    _require_units_and_noise(stage, p, hdr)
    _require_waveref(stage, p, hdr)


def validate_spec1d_product(path: str | Path, *, stage: str) -> None:
    """Validate a 1D spec1d FITS product.

    Current schema (extract1d):
      - SPEC_TRACE table: LAMBDA, FLUX_TRACE, VAR_TRACE, MASK_TRACE
      - SPEC_FIXED table: LAMBDA, FLUX_FIXED, VAR_FIXED, MASK_FIXED
      - Primary header MUST declare WAVEREF once λ exists.
    """

    p = Path(path)
    try:
        from astropy.io import fits  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ProductContractError(stage=stage, path=str(p), code="IO_ERROR", message=str(e))

    def _require_cols(hdu, cols: list[str]) -> None:
        names = {str(n).upper() for n in (hdu.columns.names or [])}
        miss = [c for c in cols if c.upper() not in names]
        if miss:
            raise ValueError(f"Missing required columns: {', '.join(miss)}")

    try:
        with fits.open(p) as hdul:
            hdr0 = hdul[0].header
            if "SPEC_TRACE" not in hdul or "SPEC_FIXED" not in hdul:
                raise ValueError("Missing required extensions SPEC_TRACE and/or SPEC_FIXED")

            ht = hdul["SPEC_TRACE"]
            hf = hdul["SPEC_FIXED"]
            _require_cols(ht, ["LAMBDA", "FLUX_TRACE", "VAR_TRACE", "MASK_TRACE"])
            _require_cols(hf, ["LAMBDA", "FLUX_FIXED", "VAR_FIXED", "MASK_FIXED"])

            lam = ht.data["LAMBDA"]
            flux_t = ht.data["FLUX_TRACE"]
            var_t = ht.data["VAR_TRACE"]
            mask_t = ht.data["MASK_TRACE"]

            flux_f = hf.data["FLUX_FIXED"]
            var_f = hf.data["VAR_FIXED"]
            mask_f = hf.data["MASK_FIXED"]

            # Shapes / finiteness
            for name, arr in {
                "LAMBDA": lam,
                "FLUX_TRACE": flux_t,
                "VAR_TRACE": var_t,
                "FLUX_FIXED": flux_f,
                "VAR_FIXED": var_f,
            }.items():
                if arr is None:
                    raise ValueError(f"Missing {name}")
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"Non-finite values in {name}")

            if np.any(var_t < 0) or np.any(var_f < 0):
                raise ValueError("Negative variance values")

            # Mask dtype sanity
            if mask_t is None or mask_f is None:
                raise ValueError("Missing MASK_* columns")

            ref = str(hdr0.get("WAVEREF", "") or "").strip().lower()
            if ref not in {"air", "vacuum", "vac", "vacuo"}:
                raise ValueError("Missing/invalid WAVEREF in primary header")
    except Exception as e:
        raise ProductContractError(stage=stage, path=str(p), code="SPEC1D_CONTRACT", message=str(e))


def validate_products(paths: Iterable[str | Path], *, stage: str) -> None:
    """Validate a list of products (auto-detect MEF vs spec1d)."""

    for x in paths:
        p = Path(x)
        if not p.exists():
            raise ProductContractError(stage=stage, path=str(p), code="MISSING_OUTPUT", message="Output file not found")
        if p.suffix.lower() != ".fits":
            continue
        # Spec1D is a small, table-like FITS
        if p.name.lower().endswith("spec1d.fits") or p.name.lower() == "spec1d.fits":
            validate_spec1d_product(p, stage=stage)
        else:
            validate_mef_product(p, stage=stage)
