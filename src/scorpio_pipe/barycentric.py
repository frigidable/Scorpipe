from __future__ import annotations

"""Barycentric velocity utilities.

This module provides an explicit, opt-in barycentric correction workflow.

We expose *velocity* as the primary product; applying the correction is left to
higher-level stages (e.g. stack2d) to keep behavior transparent.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # pragma: no cover
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time
    import astropy.units as u
except Exception:  # pragma: no cover
    EarthLocation = None  # type: ignore
    SkyCoord = None  # type: ignore
    Time = None  # type: ignore
    u = None  # type: ignore


# BTA (SAO RAS, Nizhny Arkhyz) approximate location.
# Lon 41°26′30″E, Lat +43°39′12″N, height 2070 m
_BTA_LON_DEG = 41.4416666667
_BTA_LAT_DEG = 43.6533333333
_BTA_HEIGHT_M = 2070.0


@dataclass(frozen=True)
class BarycentricResult:
    v_bary_mps: float
    doppler_factor: float  # 1 + v/c
    specsyst_in: str
    specsyst_out: str
    ok: bool
    message: str | None = None


def _parse_radec(header: Any) -> tuple[float, float] | None:
    ra = header.get("RA", header.get("OBJRA", None))
    dec = header.get("DEC", header.get("OBJDEC", None))
    if ra is None or dec is None:
        return None
    try:
        # If already numeric degrees.
        ra_f = float(ra)
        dec_f = float(dec)
        # Heuristic: RA in hours if <= 24 and DEC within [-90,90].
        if abs(ra_f) <= 24.0 and abs(dec_f) <= 90.0 and header.get("RADECSYS", "").strip().upper() != "FK5":
            # Some headers store RA in hours.
            return ra_f * 15.0, dec_f
        return ra_f, dec_f
    except Exception:
        # Try sexagesimal via SkyCoord.
        if SkyCoord is None:
            return None
        try:
            c = SkyCoord(str(ra), str(dec), unit=(u.hourangle, u.deg))
            return float(c.ra.deg), float(c.dec.deg)
        except Exception:
            try:
                c = SkyCoord(str(ra), str(dec), unit=(u.deg, u.deg))
                return float(c.ra.deg), float(c.dec.deg)
            except Exception:
                return None


def _parse_obstime(header: Any):
    mjd = header.get("MJD-OBS", header.get("MJD", None))
    if mjd is not None:
        try:
            return float(mjd)
        except Exception:
            pass
    dateobs = header.get("DATE-OBS", header.get("DATE", None))
    if dateobs is not None:
        return str(dateobs)
    return None


def compute_barycentric_velocity_mps(
    header: Any,
    *,
    location: str = "BTA",
) -> BarycentricResult:
    """Compute barycentric radial velocity correction [m/s] for the target.

    Requires astropy. Uses RA/DEC + observation time + observatory location.
    """
    if EarthLocation is None or SkyCoord is None or Time is None:  # pragma: no cover
        return BarycentricResult(
            v_bary_mps=float("nan"),
            doppler_factor=float("nan"),
            specsyst_in=str(header.get("SPECSYS", "TOPOCENT") or "TOPOCENT"),
            specsyst_out="BARYCENT",
            ok=False,
            message="astropy not available for barycentric correction",
        )

    radec = _parse_radec(header)
    if radec is None:
        return BarycentricResult(
            v_bary_mps=float("nan"),
            doppler_factor=float("nan"),
            specsyst_in=str(header.get("SPECSYS", "TOPOCENT") or "TOPOCENT"),
            specsyst_out="BARYCENT",
            ok=False,
            message="missing or unparseable RA/DEC in header",
        )
    ra_deg, dec_deg = radec

    t = _parse_obstime(header)
    if t is None:
        return BarycentricResult(
            v_bary_mps=float("nan"),
            doppler_factor=float("nan"),
            specsyst_in=str(header.get("SPECSYS", "TOPOCENT") or "TOPOCENT"),
            specsyst_out="BARYCENT",
            ok=False,
            message="missing MJD-OBS/DATE-OBS in header",
        )

    if location.upper() == "BTA":
        loc = EarthLocation.from_geodetic(_BTA_LON_DEG, _BTA_LAT_DEG, _BTA_HEIGHT_M)
    else:
        # Fallback: allow "lon,lat,height" string.
        try:
            lon, lat, h = [float(x) for x in str(location).split(",")]
            loc = EarthLocation.from_geodetic(lon, lat, h)
        except Exception:
            loc = EarthLocation.from_geodetic(_BTA_LON_DEG, _BTA_LAT_DEG, _BTA_HEIGHT_M)

    if isinstance(t, float):
        obstime = Time(t, format="mjd", scale="utc")
    else:
        obstime = Time(str(t), format="isot", scale="utc")

    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    try:
        v = target.radial_velocity_correction(obstime=obstime, location=loc)
        v_mps = float(v.to(u.m / u.s).value)
        c = 299792458.0
        dop = 1.0 + (v_mps / c)
        return BarycentricResult(
            v_bary_mps=v_mps,
            doppler_factor=dop,
            specsyst_in=str(header.get("SPECSYS", "TOPOCENT") or "TOPOCENT"),
            specsyst_out="BARYCENT",
            ok=True,
            message=None,
        )
    except Exception as e:
        return BarycentricResult(
            v_bary_mps=float("nan"),
            doppler_factor=float("nan"),
            specsyst_in=str(header.get("SPECSYS", "TOPOCENT") or "TOPOCENT"),
            specsyst_out="BARYCENT",
            ok=False,
            message=f"astropy barycentric failed: {e}",
        )


def apply_barycentric_to_wavelength_grid(
    lambda0: float,
    dlambda: float,
    *,
    doppler_factor: float,
) -> tuple[float, float]:
    """Apply barycentric correction to a *linear* wavelength grid.

    Uses λ_bary = λ_obs / doppler_factor (doppler_factor = 1 + v/c).
    """
    if not np.isfinite(doppler_factor) or doppler_factor == 0:
        return float(lambda0), float(dlambda)
    return float(lambda0) / float(doppler_factor), float(dlambda) / float(doppler_factor)
