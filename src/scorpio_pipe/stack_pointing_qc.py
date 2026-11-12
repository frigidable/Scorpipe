"""Conservative sky-pointing/rotator admission for long-slit stacking.

A shared wavelength grid and OBJECT value never establish that two spectra
sample the same sky positions. Coordinates and rotator angles are measured
metadata, NOT a verified slit-on-sky astrometric registration. This module
never claims slit position angles from ROTANGLE.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _sexagesimal(value: Any, *, ra: bool) -> float | None:
    """Parse explicit sexagesimal coordinates; numeric RA without unit is unknown."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if ra:
            return None  # never guess degrees versus hours
        v = float(value)
        return v if math.isfinite(v) and -90 <= v <= 90 else None
    token = str(value).strip()
    if not token or token.upper() in {"NONE", "NAN", "UNKNOWN", "UNDEFINED"}:
        return None
    parts = token.replace("h", ":").replace("m", ":").replace("s", "").split(":")
    if len(parts) != 3:
        if ra:
            return None
        try:
            d = float(token)
        except (TypeError, ValueError):
            return None
        return d if math.isfinite(d) and -90 <= d <= 90 else None
    try:
        a, b, c = map(float, parts)
    except (TypeError, ValueError):
        return None
    if not all(map(math.isfinite, (a, b, c))):
        return None
    if not (0 <= b < 60 and 0 <= c < 60):
        return None
    if ra:
        if not 0 <= a < 24:
            return None
        return (a + b/60.0 + c/3600.0) * 15.0
    if not 0 <= abs(a) <= 90:
        return None
    sign = -1.0 if token.startswith("-") else 1.0
    dec = sign * (abs(a) + b/60.0 + c/3600.0)
    return dec if -90 <= dec <= 90 else None


def _header_coordinates(header: Mapping[str, Any]) -> tuple[float, float] | None:
    # Exact RA_DEG/DEC_DEG (when present) explicitly identify units.
    if "RA_DEG" in header and "DEC_DEG" in header:
        try:
            ra, dec = float(header["RA_DEG"]), float(header["DEC_DEG"])
            if math.isfinite(ra) and math.isfinite(dec) and -90 <= dec <= 90:
                return ra % 360.0, dec
        except (TypeError, ValueError):
            pass
    ra = _sexagesimal(header.get("RA"), ra=True)
    dec = _sexagesimal(header.get("DEC"), ra=False)
    return (ra, dec) if ra is not None and dec is not None else None


def _rotator(header: Mapping[str, Any]) -> float | None:
    try:
        v = float(header.get("ROTANGLE"))
        return v % 360.0 if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _separation_arcsec(a: tuple[float, float], b: tuple[float, float]) -> float:
    # Haversine retains numerical precision for subarcsecond offsets.
    ra1, dec1 = map(math.radians, a)
    ra2, dec2 = map(math.radians, b)
    dd = dec2 - dec1
    dra = (ra2 - ra1 + math.pi) % (2 * math.pi) - math.pi
    z = math.sin(dd/2.0)**2 + (
        math.cos(dec1)*math.cos(dec2)*math.sin(dra/2.0)**2
    )
    return math.degrees(2.0*math.asin(math.sqrt(min(1.0,max(0.0,z))))) * 3600.0


def check_stack_pointing(
    headers: Sequence[Mapping[str, Any]],
    *, names: Sequence[str] | None = None,
    max_pointing_sep_arcsec: float = 5.0,
    max_rotator_diff_deg: float = 1.0,
    require_complete_metadata: bool = False,
) -> dict[str, Any]:
    """Review candidate Stack2D frames against the first frame.

    REJECT: reliable metadata disagree beyond configured limits.
    UNVERIFIED: incomplete metadata; caller must not describe it as a
    scientifically approved spatial co-registration.
    PASS: header-level consistency only; tracing/registration is still needed.

    Explicit angles are mechanical ROTANGLE; this check does not infer
    celestial slit PA from them. Neither passing nor a y-shift alone proves
    identical sky coverage.
    """
    if not headers:
        raise ValueError("No headers supplied for Stack2D geometry check")
    if names is not None and len(names) != len(headers):
        raise ValueError("Stack2D geometry names/header count mismatch")
    for label, v in (
        ("max_pointing_sep_arcsec", max_pointing_sep_arcsec),
        ("max_rotator_diff_deg", max_rotator_diff_deg),
    ):
        if not math.isfinite(float(v)) or float(v) <= 0:
            raise ValueError(f"Invalid Stack2D geometry threshold: {label}")
    labels = list(names) if names is not None else [f"input_{i}" for i in range(len(headers))]
    ref_coord, ref_rot = _header_coordinates(headers[0]), _rotator(headers[0])
    measurements: list[dict[str, Any]] = []
    issues: list[str] = []
    unknown = False
    for i, (hdr, label) in enumerate(zip(headers, labels)):
        coord, rot = _header_coordinates(hdr), _rotator(hdr)
        sep = (_separation_arcsec(ref_coord, coord)
               if coord is not None and ref_coord is not None else None)
        dr = (abs((rot - ref_rot + 180.0) % 360.0 - 180.0)
              if rot is not None and ref_rot is not None else None)
        if i and (sep is None or dr is None):
            unknown = True
        if i and sep is not None and sep > max_pointing_sep_arcsec:
            issues.append(
                f"{label}: pointing separation {sep:.3f} arcsec exceeds "
                f"{max_pointing_sep_arcsec:g} arcsec"
            )
        if i and dr is not None and dr > max_rotator_diff_deg:
            issues.append(
                f"{label}: ROTANGLE difference {dr:.3f} deg exceeds "
                f"{max_rotator_diff_deg:g} deg (instrument rotation, not slit PA)"
            )
        measurements.append({
            "file": str(label),
            "ra_deg": coord[0] if coord else None,
            "dec_deg": coord[1] if coord else None,
            "rotangle_deg": rot,
            "pointing_sep_arcsec": sep,
            "rotangle_diff_deg": dr,
        })
    if require_complete_metadata and unknown:
        issues.append(
            "Missing or ambiguous pointing/rotator metadata: strict "
            "Stack2D science admission requires explicit coordinate metadata"
        )
    status = "REJECT" if issues else ("UNVERIFIED" if unknown else "PASS")
    return {
        "status": status,
        "science_spatially_verified": False,
        "reference": str(labels[0]),
        "limits": {
            "max_pointing_sep_arcsec": float(max_pointing_sep_arcsec),
            "max_rotator_diff_deg": float(max_rotator_diff_deg),
            "require_complete_metadata": bool(require_complete_metadata),
        },
        "measurements": measurements,
        "issues": issues,
        "interpretation": (
            "Header-level telescope pointing and ROTANGLE checks, not "
            "astrometric slit-registration or a derived on-sky slit PA."
        ),
    }
