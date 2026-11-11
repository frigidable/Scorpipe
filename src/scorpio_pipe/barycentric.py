from __future__ import annotations

"""Opt-in optical velocity corrections; no pipeline stage applies them implicitly.

Astropy vcorr is an optical correction: F = 1 + vcorr/c and lambda_out =
lambda_in * F (vacuum wavelengths). It is not a radio/relativistic velocity.
Computing a result never changes a header, array, file, or spectral frame.
"""

import warnings
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, solar_system_ephemeris
from astropy.time import Time
from astropy.utils import iers
from erfa import ErfaWarning

# Published SAO RAS BTA site; preserve the existing coordinates, not a site guess.
# https://www.sao.ru/Doc-en/Telescopes/bta/descrip.html
_BTA_LON_DEG = 41.4416666667
_BTA_LAT_DEG = 43.6533333333
_BTA_HEIGHT_M = 2070.0
_C_MPS = 299792458.0
_FRAMES = {"barycentric": "BARYCENT", "heliocentric": "HELIOCEN"}


@dataclass(frozen=True)
class BarycentricResult:
    # Legacy field/entrypoint names retained for compatibility; specsyst_out
    # distinguishes barycentric from heliocentric, never infer kind from name.
    v_bary_mps: float
    doppler_factor: float
    specsyst_in: str
    specsyst_out: str
    ok: bool
    message: str | None = None
    ra_deg: float | None = None
    dec_deg: float | None = None
    obstime_mjd_utc: float | None = None
    location_geocentric_m: tuple[float, float, float] | None = None
    ephemeris: str = "builtin"
    iers_source: str | None = None
    astropy_version: str | None = None
    velocity_definition: str = "optical"
    applied: bool = False
    coordinate_frame: str = "ICRS"


def _number(value: Any, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_, str, bytes))
        or np.ndim(value) != 0
        or isinstance(value, u.Quantity)
    ):
        raise ValueError(f"{name} must be a finite scalar in the documented units")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _parse_radec(header: Any, *, ra_unit: str | None = None) -> tuple[float, float]:
    ra = header.get("RA", header.get("OBJRA"))
    dec = header.get("DEC", header.get("OBJDEC"))
    if ra is None or dec is None:
        raise ValueError("Missing RA/DEC")
    if any(isinstance(value, (bool, np.bool_)) for value in (ra, dec)):
        raise ValueError("RA/DEC cannot be boolean")
    unit = ra_unit if ra_unit is not None else header.get("RAUNIT")
    if unit is None:
        # SCORPIO fixture convention: colon-separated RA is hours. Numeric RA
        # has no inferable unit, even when RADESYS/RADECSYS says FK5.
        if isinstance(ra, str) and ":" in ra:
            unit = "hourangle"
        else:
            raise ValueError(
                "Numeric RA is ambiguous: specify ra_unit or RAUNIT (deg/hourangle)"
            )
    aliases = {
        "deg": u.deg,
        "degree": u.deg,
        "degrees": u.deg,
        "hourangle": u.hourangle,
        "hour": u.hourangle,
        "hours": u.hourangle,
        "h": u.hourangle,
    }
    ra_u = aliases.get(str(unit).strip().lower())
    if ra_u is None:
        raise ValueError("RA unit must be deg or hourangle")
    dec_unit = str(header.get("DECUNIT", "deg")).strip().lower()
    if dec_unit not in {"deg", "degree", "degrees"}:
        raise ValueError("DECUNIT must be degrees")
    from astropy.coordinates import Angle

    r = Angle(ra, unit=ra_u).to_value(u.deg)
    d = Angle(dec, unit=u.deg).to_value(u.deg)
    if (
        not np.isfinite(r)
        or not np.isfinite(d)
        or not 0 <= r < 360
        or not -90 <= d <= 90
    ):
        raise ValueError("RA/DEC outside finite angular ranges")
    return float(r), float(d)


def _date_time(header: Any, key: str) -> Time:
    text = str(header[key]).strip()
    if "/" in text:
        instrument = str(header.get("INSTRUME", "")).upper().replace("-", "")
        if instrument not in {"SCORPIO1", "SCORPIO2"}:
            raise ValueError(
                "Ambiguous slash date outside the SCORPIO YYYY/DD/MM contract"
            )
        year, day, month = text.split("/")
        text = f"{year}-{month}-{day}"
    if "T" not in text:
        # TIME-OBS in SCORPIO fixtures is local time, not UTC.
        if key != "DATE-OBS" or not header.get("UT"):
            raise ValueError(f"{key} requires a UTC clock; only DATE-OBS may borrow UT")
        text += "T" + str(header["UT"]).strip()
    return Time(text, format="isot", scale="utc")


def _parse_obstime(header: Any, *, time_reference: str = "start") -> Time:
    if str(header.get("TIMESYS", "UTC")).strip().upper() != "UTC":
        raise ValueError("Only UTC observation metadata is supported")
    if time_reference not in {"start", "mid"}:
        raise ValueError("time_reference must be start or mid")

    def pair(mjd_key, date_key):
        times = []
        if header.get(mjd_key) is not None:
            times.append(
                Time(_number(header[mjd_key], mjd_key), format="mjd", scale="utc")
            )
        if header.get(date_key) is not None:
            times.append(_date_time(header, date_key))
        if len(times) == 2 and abs((times[0] - times[1]).to_value(u.s)) > 0.001:
            raise ValueError(f"Inconsistent {mjd_key}/{date_key}")
        return times[0] if times else None

    avg = pair("MJD-AVG", "DATE-AVG")
    start = pair("MJD-OBS", "DATE-OBS")
    duration = header.get("EXPTIME")
    if duration is not None:
        duration = _number(duration, "EXPTIME [s]")
        if duration < 0:
            raise ValueError("EXPTIME must be nonnegative")
    mid = None
    if start is not None:
        if time_reference == "mid":
            mid = start
        elif duration is not None:
            mid = start + duration / 2 * u.s
    if avg is not None:
        if mid is not None and abs((mid - avg).to_value(u.s)) > 0.001:
            raise ValueError("Average time disagrees with observation midpoint")
        return avg
    if mid is None:
        raise ValueError(
            "Need MJD-AVG/DATE-AVG or observation time with EXPTIME; no DATE creation-time fallback"
        )
    return mid


def _location(location: str | EarthLocation) -> EarthLocation:
    if isinstance(location, EarthLocation):
        if not location.isscalar:
            raise ValueError("Observer location must be scalar")
        lon, lat, h = location.to_geodetic()
        coords = (lon.to_value(u.deg), lat.to_value(u.deg), h.to_value(u.m))
    elif isinstance(location, str) and location.strip().upper() == "BTA":
        coords = (_BTA_LON_DEG, _BTA_LAT_DEG, _BTA_HEIGHT_M)
    else:
        try:
            coords = tuple(float(x) for x in location.split(","))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid observer location; expected BTA, EarthLocation or lon,lat,height [deg,deg,m]"
            ) from exc
    if len(coords) != 3 or not all(np.isfinite(coords)):
        raise ValueError("Invalid observer location coordinates")
    lon, lat, height = coords
    if not -180 <= lon <= 180 or not -90 <= lat <= 90 or not -500 <= height <= 100000:
        raise ValueError("Observer location outside terrestrial geodetic ranges")
    return EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, height * u.m)


def compute_barycentric_velocity_mps(
    header: Any,
    *,
    location: str | EarthLocation = "BTA",
    kind: str = "barycentric",
    ra_unit: str | None = None,
    time_reference: str = "start",
) -> BarycentricResult:
    """Compute only, from topocentric metadata, in SI m/s (optical convention).

    Numeric RA requires explicit deg/hourangle. DEC defaults to degrees.
    Missing RADESYS means ICRS only if EQUINOX is also absent; FK5 uses EQUINOX
    (default J2000). UTC start times require EXPTIME in seconds; explicit average
    timestamps take precedence if consistent. time_reference='mid' declares
    MJD-OBS/DATE-OBS already at midpoint. No DATE/MJD creation-time fallback.

    Uses builtin ERFA ephemerides and bundled IERS-B, without network or degraded
    Earth-orientation extrapolation. Outside the bundled table, returns ok=False.
    BTA coordinates are the published site, not a precision geodetic survey.
    """
    source = str(header.get("SPECSYS", "TOPOCENT")).strip().upper()
    destination = _FRAMES.get(kind, "UNKNOWN")
    try:
        if kind not in _FRAMES:
            raise ValueError("kind must be barycentric or heliocentric")
        if (
            source != "TOPOCENT"
            or str(header.get("SSYSOBS", "TOPOCENT")).strip().upper() != "TOPOCENT"
        ):
            raise ValueError(
                "Correction requires topocentric input; already corrected/unknown frames are refused"
            )
        ra, dec = _parse_radec(header, ra_unit=ra_unit)
        loc = _location(location)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ErfaWarning)
            warnings.simplefilter("error", iers.IERSWarning)
            obstime = _parse_obstime(header, time_reference=time_reference)
            frame = (
                str(header.get("RADESYS", header.get("RADECSYS", ""))).strip().upper()
            )
            if not frame and header.get("EQUINOX") is not None:
                raise ValueError("EQUINOX without RADESYS is ambiguous")
            if frame in {"", "ICRS"}:
                if frame == "ICRS" and header.get("EQUINOX") is not None:
                    raise ValueError("ICRS does not use EQUINOX")
                target = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
            elif frame == "FK5":
                equinox = _number(
                    header.get("EQUINOX", 2000.0), "EQUINOX [Julian year]"
                )
                target = SkyCoord(
                    ra * u.deg,
                    dec * u.deg,
                    frame="fk5",
                    equinox=Time(equinox, format="jyear"),
                ).icrs
            else:
                raise ValueError(
                    "Unsupported coordinate RADESYS; use explicit ICRS/FK5"
                )
            table = iers.IERS_B.open(iers.IERS_B_FILE)
            _, status = table.ut1_utc(obstime, return_status=True)
            if status < 0:
                raise ValueError("Observation time outside bundled IERS-B coverage")
            with (
                iers.conf.set_temp("auto_download", False),
                iers.earth_orientation_table.set(table),
                solar_system_ephemeris.set("builtin"),
            ):
                v = target.radial_velocity_correction(
                    kind=kind, obstime=obstime, location=loc
                ).to_value(u.m / u.s)
        v = _number(v, "vcorr [m/s]")
        factor = 1 + v / _C_MPS
        if abs(v) >= _C_MPS or not np.isfinite(factor) or factor <= 0:
            raise ValueError("Invalid optical correction factor")
        return BarycentricResult(
            v,
            factor,
            source,
            destination,
            True,
            ra_deg=float(target.ra.deg),
            dec_deg=float(target.dec.deg),
            obstime_mjd_utc=float(obstime.mjd),
            location_geocentric_m=tuple(float(x.to_value(u.m)) for x in loc.geocentric),
            iers_source="IERS_B bundled astropy-iers-data "
            + version("astropy-iers-data"),
            astropy_version=version("astropy"),
        )
    except (
        ValueError,
        TypeError,
        ArithmeticError,
        OSError,
        RuntimeError,
        iers.IERSWarning,
        ErfaWarning,
    ) as exc:
        return BarycentricResult(
            float("nan"), float("nan"), source, destination, False, str(exc)
        )


def apply_barycentric_to_wavelength_grid(
    lambda0: float,
    dlambda: float,
    *,
    doppler_factor: float,
    specsys_in: str = "TOPOCENT",
    specsys_out: str = "BARYCENT",
    already_applied: bool = False,
    wave_reference: str = "vacuum",
) -> tuple[float, float]:
    """Scale a linear VACUUM grid by the optical Doppler factor; no resampling.

    Returns new (origin, step), leaving input arrays/files/headers untouched.
    Bare numbers retain no history: callers must pass and propagate frame/state.
    already_applied or a non-topocentric input refuses repeated corrections.
    For numerical inversion pass 1/F, NOT 1-vcorr/c. This inverse-factor use does
    not itself update provenance or implement a general reference-frame graph.
    Air grids require a separate air/vacuum conversion (generally nonlinear).
    """
    # Preserve existing public error contracts while rejecting non-scalar inputs
    # and unit-bearing quantities (an undocumented vacuum-grid convention).
    try:
        factor = _number(doppler_factor, "doppler_factor [dimensionless]")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Barycentric Doppler factor must be finite and positive"
        ) from exc
    if factor <= 0:
        raise ValueError("Barycentric Doppler factor must be finite and positive")
    try:
        origin = _number(lambda0, "lambda0")
        step = _number(dlambda, "dlambda")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Barycentric wavelength grid must be finite and nondegenerate"
        ) from exc
    if origin <= 0 or step == 0:
        raise ValueError("Barycentric wavelength grid must be finite and nondegenerate")
    if already_applied or specsys_in != "TOPOCENT":
        raise ValueError("Correction already applied or input frame is not TOPOCENT")
    if specsys_out not in {"BARYCENT", "HELIOCEN"}:
        raise ValueError("Output frame must be explicit BARYCENT or HELIOCEN")
    if wave_reference != "vacuum":
        raise ValueError("Optical Doppler scaling requires vacuum wavelengths")
    out = origin * factor, step * factor
    if not all(np.isfinite(out)) or out[0] <= 0 or out[1] == 0:
        raise ValueError("Corrected grid is not finite/representable")
    return out
