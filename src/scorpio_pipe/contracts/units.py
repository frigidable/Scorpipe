"""Unit rules for the data model contract.

This module provides:
- Normalization + recognition of SCI/VAR units in FITS BUNIT.
- Deterministic conversion of SCI BUNIT -> expected VAR BUNIT.

We keep the allowed vocabulary intentionally small, but accept common aliases
(e.g. "e-" / "electron").
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnitFamily:
    """Parsed unit family for BUNIT."""

    base: str  # 'electron'|'adu'|'physical_flux'|'unknown'
    per_second: bool


def _norm(s: str) -> str:
    return str(s or "").strip().replace(" ", "").lower()


def parse_bunit(bunit: str) -> UnitFamily:
    """Parse a FITS BUNIT string into a compact family.

    Examples
    --------
    - "e-" -> (electron, False)
    - "electron/s" -> (electron, True)
    - "ADU" -> (adu, False)
    """

    u = _norm(bunit)
    if not u:
        return UnitFamily(base="unknown", per_second=False)

    per_second = False
    if "/s" in u or "s-1" in u:
        per_second = True
        u = u.replace("/s", "").replace("s-1", "")

    # Strip exponent-like suffixes for family detection.
    u2 = u.replace("^2", "").replace("**2", "").replace("2", "")

    if "adu" in u2:
        return UnitFamily(base="adu", per_second=per_second)

    if "electron" in u2 or u2 in {"e", "e-", "e_", "elec", "el"} or "e-" in u2:
        return UnitFamily(base="electron", per_second=per_second)

    if any(tok in u for tok in ("erg", "flam", "jy", "w/m", "photon")):
        return UnitFamily(base="physical_flux", per_second=per_second)

    return UnitFamily(base="unknown", per_second=per_second)


def is_recognized_sci_bunit(bunit: str) -> bool:
    return parse_bunit(bunit).base in {"electron", "adu", "physical_flux"}


def is_recognized_var_bunit(bunit: str) -> bool:
    # We accept the same families as SCI; the exponent is checked separately.
    return parse_bunit(bunit).base in {"electron", "adu", "physical_flux"}


def var_bunit_from_sci_bunit(bunit_sci: str) -> str:
    """Return expected VAR BUNIT from a SCI BUNIT.

    Policy
    ------
    We stamp explicit squared units:
    - electron  -> electron2
    - ADU       -> ADU2

    For rate units, append "/s2".
    """

    fam = parse_bunit(bunit_sci)
    base = fam.base
    # Preserve wavelength-density dimensions in a spectral SCI product.
    # A VAR of e-/Angstrom has units (e-/Angstrom)^2, not electron2.
    # Keep legacy compact labels for detector-integrated or pure-rate SCI.
    us = _norm(bunit_sci)
    if base in {"electron", "adu"} and any(
        tok in us for tok in ("/angstrom", "/nanometer", "/nm", "/pix", "/pixel")
    ):
        return f"({bunit_sci})^2"
    if base == "electron":
        out = "electron2"
    elif base == "adu":
        out = "ADU2"
    else:
        out = "unknown2"

    if base == "physical_flux":
        out = f"({bunit_sci})^2"
        return out

    if fam.per_second:
        out = f"{out}/s2"
    return out


def infer_sci_bunit_from_unit_model(unit_model: str | None) -> str | None:
    """Infer a SCI BUNIT from a SCORPUM-like unit model."""

    m = _norm(unit_model or "")
    if not m:
        return None
    if "adu" in m:
        return "ADU"
    if "electron" in m or m in {"e", "e-", "e_"}:
        return "e-"
    return None
