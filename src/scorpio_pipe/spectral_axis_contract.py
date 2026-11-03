"""Science-grade spectral-axis compatibility for rectified SCORPIO spectra.

Matching numerical wavelength grids is necessary but NOT sufficient:
air (FITS AWAV) and vacuum (FITS WAVE) wavelength scales, or different
spectral velocity reference frames, must not be averaged directly.

Authority: Greisen et al. (2006), FITS WCS Paper III; FITS Standard §8.4.1.

Historical anonymous minimal test arrays may have no spectroscopic metadata.
They remain tagged UNQUAL, never silently promoted to air/TOPOCENT.
Production SCORPIO files must have a complete spectral-axis contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from astropy.io import fits


_VALID_FRAMES = frozenset({
    "TOPOCENT", "GEOCENTR", "BARYCENT", "HELIOCEN",
    "LSRK", "LSRD", "GALACTOC", "LOCALGRP",
    "CMBDIPOL", "SOURCE",
})


@dataclass(frozen=True)
class SpectralAxisContract:
    medium: str
    ctype: str
    specsys: str
    flexure_applied: bool | None
    qualified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "wavelength_medium": self.medium,
            "ctype1": self.ctype,
            "specsys": self.specsys,
            "flexure_applied": self.flexure_applied,
            "validation": "QUALIFIED" if self.qualified else "UNQUAL",
        }


def read_spectral_axis_contract(
    hdr: fits.Header,
    *,
    source: str = "spectrum",
) -> SpectralAxisContract:
    """Read physical spectral semantics without guessing missing metadata."""
    instrument = str(hdr.get("INSTRUME") or "").strip().upper()
    production = instrument.startswith("SCORPIO")

    medium = str(hdr.get("WAVEREF") or "").strip().lower()
    if medium and medium not in {"air", "vacuum"}:
        raise ValueError(
            f"{source}: invalid WAVEREF {medium!r}; require air or vacuum"
        )
    if not medium:
        medium = "UNKNOWN"

    ctype = str(hdr.get("CTYPE1") or "").strip().upper()
    ctype_kind = ctype.split("-", 1)[0]
    if ctype_kind and ctype_kind not in {"AWAV", "WAVE"}:
        # A wavelength-calibrated Stack2D input must not have a pixel or
        # frequency-only spectral axis, even when a numeric grid is readable.
        raise ValueError(
            f"{source}: unsupported spectral CTYPE1={ctype!r}; "
            "Stack2D requires FITS WAVE or AWAV"
        )
    if not ctype_kind:
        ctype_kind = "UNKNOWN"

    medium_from_ctype = {"AWAV": "air", "WAVE": "vacuum"}
    if (
        medium != "UNKNOWN" and ctype_kind != "UNKNOWN"
        and medium_from_ctype[ctype_kind] != medium
    ):
        raise ValueError(
            f"{source}: spectral CTYPE1={ctype!r} contradicts "
            f"WAVEREF={medium!r}"
        )

    specsys = str(hdr.get("SPECSYS") or "").strip().upper()
    if specsys and specsys not in _VALID_FRAMES:
        raise ValueError(
            f"{source}: unknown FITS spectral frame SPECSYS={specsys!r}"
        )
    if not specsys:
        specsys = "UNKNOWN"

    flexure_applied: bool | None = None
    if production and "DLAMAP" not in hdr:
        raise ValueError(
            f"{source}: SCORPIO Stack2D requires explicit DLAMAP "
            "flexure-correction provenance"
        )
    if "DLAMAP" in hdr:
        val = hdr["DLAMAP"]
        if not isinstance(val, bool):
            raise ValueError(
                f"{source}: DLAMAP must be a FITS logical, got {val!r}"
            )
        flexure_applied = bool(val)
        if flexure_applied:
            flag = str(hdr.get("FLXFLAG") or "").strip().upper()
            if flag == "UNVERIFIED":
                raise ValueError(
                    f"{source}: DLAMAP=True contradicts unverified flexure"
                )
            if flag != "OK":
                raise ValueError(
                    f"{source}: DLAMAP=True requires independently "
                    "verified FLXFLAG=OK"
                )
            try:
                delta = float(hdr["DLAM_A"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{source}: applied flexure requires finite DLAM_A"
                ) from exc
            from math import isfinite

            if not isfinite(delta):
                raise ValueError(
                    f"{source}: applied flexure requires finite DLAM_A"
                )

    qualified = (
        medium != "UNKNOWN"
        and ctype_kind != "UNKNOWN"
        and specsys != "UNKNOWN"
    )
    if production and not qualified:
        raise ValueError(
            f"{source}: SCORPIO Stack2D requires explicit WAVEREF, "
            "CTYPE1 and SPECSYS; missing metadata cannot be inferred"
        )
    # An explicitly identified medium or frame must never be paired with
    # unknown complementary semantics: otherwise partially written FITS
    # headers could accidentally pass the same-grid comparison.
    partially_specified = any((
        medium != "UNKNOWN", ctype_kind != "UNKNOWN",
        specsys != "UNKNOWN",
    ))
    if partially_specified and not qualified:
        raise ValueError(
            f"{source}: partial spectral WCS is not valid for Stack2D; "
            "provide WAVEREF, CTYPE1 and SPECSYS together"
        )

    return SpectralAxisContract(
        medium=medium, ctype=ctype_kind, specsys=specsys,
        flexure_applied=flexure_applied,
        qualified=qualified,
    )


def validate_same_spectral_frame(
    reference: SpectralAxisContract,
    current: SpectralAxisContract,
    *,
    source: str,
) -> None:
    """Reject physically different grids even with identical numbers."""
    if (reference.medium, reference.ctype, reference.specsys) != (
        current.medium, current.ctype, current.specsys,
    ):
        raise ValueError(
            f"{source}: incompatible spectral coordinate contracts: "
            f"reference {reference.to_dict()}, current {current.to_dict()}. "
            "Rectify both spectra to an explicitly validated common frame."
        )
    # A missing DLAMAP indicator means unqualified flexure provenance.
    # That differs from a known-applied shift. This specifically prevents
    # a corrected frame from being stacked with an unidentified one.
    if reference.flexure_applied != current.flexure_applied:
        raise ValueError(
            f"{source}: incompatible flexure correction provenance: "
            f"reference DLAMAP={reference.flexure_applied}, "
            f"input DLAMAP={current.flexure_applied}"
        )
