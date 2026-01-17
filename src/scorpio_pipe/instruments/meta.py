from __future__ import annotations

"""Header Contract: normalize SCORPIO-1 and SCORPIO-2 FITS headers.

Why this exists
--------------
SCORPIO-1 and SCORPIO-2 headers are *semantically similar* but differ in:

* key naming (e.g. SC2 uses ``SLITWID`` reliably, SC1 may leave it blank)
* value formatting (dates, spacing, mixed cases)

For calibration autogrouping and strict matching we must extract a stable,
instrument-agnostic metadata blob and **fail loudly** when critical keys are
missing.

This module contains the shared dataclasses, normalization helpers, and the
public dispatcher :func:`parse_frame_meta`.
"""

from dataclasses import dataclass
import re
from datetime import datetime, timezone
from typing import Any, Mapping


# -----------------------------
# Errors
# -----------------------------


class HeaderContractError(RuntimeError):
    """Raised when a header violates the metadata contract.

    We do *not* silently guess values for fields that affect calibration/science
    matching. The error message is meant to be actionable.
    """

    def __init__(
        self,
        message: str,
        *,
        instrument: str | None = None,
        missing_keys: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ):
        self.instrument = instrument
        self.missing_keys = missing_keys or []
        self.context = context or {}
        base = message
        if self.missing_keys:
            base += f" | missing={self.missing_keys}"
        if self.context:
            # keep short & readable
            ctx = ", ".join(f"{k}={v!r}" for k, v in list(self.context.items())[:8])
            base += f" | ctx: {ctx}"
        super().__init__(base)


# -----------------------------
# Core dataclasses
# -----------------------------


@dataclass(frozen=True)
class ReadoutKey:
    """Readout configuration key used for matching."""

    node: str
    rate: float
    gain: float


@dataclass(frozen=True)
class FrameMeta:
    """Normalized, instrument-agnostic metadata extracted from FITS headers."""

    # Per task spec: internal instrument id
    instrument: str  # "SCORPIO1" | "SCORPIO2"

    mode: str  # "Spectra" | "Image" | ... (normalized)
    imagetyp: str  # "bias" | "flat" | "neon" | "obj" | ... (normalized)
    disperser: str  # e.g. "VPHG1200@540" (may be "" for imaging)

    slit_width_arcsec: float
    slit_pos: float | None

    binning_x: int
    binning_y: int

    naxis1: int
    naxis2: int

    readout_key: ReadoutKey
    date_time_utc: datetime
    object_name: str

    # P0-M: explicit lamp contract for wavelength work (recorded in provenance).
    lamp_raw: str = ""
    lamp_type: str = "Unknown"  # HeNeAr | Ne | Unknown
    lamp_source: str = "none"  # header | config | default | none

    @property
    def binning_key(self) -> str:
        return f"{self.binning_x}x{self.binning_y}"

    @property
    def slit_width_key(self) -> str:
        """Stable slit width string for grouping.

        SC2 headers often store values like 1.00019; for matching we need a
        stable representation. We round to 0.1" which matches typical slit
        sets (0.5, 0.7, 1.0, 1.2, ...).
        """

        v = round(float(self.slit_width_arcsec), 1)
        return f"{v:.1f}"

    @property
    def instrument_db_key(self) -> str:
        """Return the key used by the shipped instrument DB."""

        return "SCORPIO-2" if self.instrument == "SCORPIO2" else "SCORPIO"


# -----------------------------
# Normalization helpers
# -----------------------------


def norm_str(v: Any) -> str:
    """Trim and normalize a string-like header value."""

    if v is None:
        return ""
    s = str(v)
    # FITS string values are often padded to fixed width.
    s = s.strip()
    # Normalize inner whitespace (but keep @ and other separators).
    s = re.sub(r"\s+", " ", s)
    return s


def norm_upper_nospace(v: Any) -> str:
    return norm_str(v).replace(" ", "").upper()


def norm_lower(v: Any) -> str:
    return norm_str(v).lower()


def _get(h: Mapping[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in h:
            return h.get(k)
    return None


def parse_binning(h: Mapping[str, Any], *, instrument: str) -> tuple[int, int]:
    """Parse binning (x, y) from common SCORPIO header formats."""

    v = _get(h, "BINNING", "CCDSUM", "CCDBIN", "CCDBIN1")
    if v is not None:
        s = norm_upper_nospace(v).replace("X", "x")
        s = s.replace("X", "x")
        s = s.replace(" ", "")
        # common: 1X2 or 1x2
        if "x" in s:
            parts = [p for p in s.split("x") if p]
            if len(parts) == 2 and all(re.fullmatch(r"\d+", p) for p in parts):
                return int(parts[0]), int(parts[1])
        # alternate: "1 2"
        parts = [p for p in re.split(r"\s+", norm_str(v)) if p]
        if len(parts) == 2 and all(re.fullmatch(r"\d+", p) for p in parts):
            return int(parts[0]), int(parts[1])

    # explicit keywords (rare in SCORPIO headers, but safe)
    x = _get(h, "BINX", "XBIN", "XBINNING")
    y = _get(h, "BINY", "YBIN", "YBINNING")
    if x is not None and y is not None:
        try:
            return int(x), int(y)
        except Exception:
            pass

    raise HeaderContractError(
        "Missing/invalid binning", instrument=instrument, missing_keys=["BINNING"]
    )


def parse_int(h: Mapping[str, Any], key: str, *, instrument: str) -> int:
    v = _get(h, key)
    if v is None:
        raise HeaderContractError(
            f"Missing required header key {key}",
            instrument=instrument,
            missing_keys=[key],
        )
    try:
        return int(v)
    except Exception:
        raise HeaderContractError(
            f"Invalid int value for {key}: {v!r}",
            instrument=instrument,
            context={key: v},
        )


def parse_float(h: Mapping[str, Any], key: str, *, instrument: str) -> float:
    v = _get(h, key)
    if v is None:
        raise HeaderContractError(
            f"Missing required header key {key}",
            instrument=instrument,
            missing_keys=[key],
        )
    try:
        return float(v)
    except Exception:
        raise HeaderContractError(
            f"Invalid float value for {key}: {v!r}",
            instrument=instrument,
            context={key: v},
        )


def parse_date_obs(date_obs: str) -> tuple[int, int, int]:
    """Parse DATE-OBS into (year, month, day).

    SCORPIO headers often use the non-standard format ``YYYY/DD/MM`` (see the
    comments in instrument headers). We support:

    * YYYY/DD/MM
    * YYYY-MM-DD
    * YYYY-MM-DDThh:mm:ss (date part is used)
    """

    s = norm_str(date_obs)
    if not s:
        raise ValueError("empty DATE-OBS")

    # ISO-ish
    if "T" in s:
        s = s.split("T", 1)[0]

    if "/" in s:
        # SCORPIO convention: YYYY/DD/MM
        parts = [p for p in s.split("/") if p]
        if len(parts) == 3:
            y = int(parts[0])
            d = int(parts[1])
            m = int(parts[2])
            return y, m, d
    if "-" in s:
        parts = [p for p in s.split("-") if p]
        if len(parts) == 3:
            y = int(parts[0])
            m = int(parts[1])
            d = int(parts[2])
            return y, m, d

    # last-resort: try datetime.fromisoformat
    try:
        dt = datetime.fromisoformat(s)
        return dt.year, dt.month, dt.day
    except Exception as e:
        raise ValueError(f"Unrecognized DATE-OBS format: {date_obs!r}") from e


def parse_hms(hms: str) -> tuple[int, int, float]:
    s = norm_str(hms)
    if not s:
        raise ValueError("empty time")
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError(f"bad time: {hms!r}")
    hh = int(parts[0])
    mm = int(parts[1])
    ss = float(parts[2])
    return hh, mm, ss


def parse_datetime_utc(
    h: Mapping[str, Any], *, instrument: str, prefer_ut: bool = True
) -> datetime:
    """Return a timezone-aware UTC datetime.

    Priority:
    1) DATE-OBS + UT
    2) DATE-OBS + TIME-OBS
    3) DATE + UT (rare)
    """

    date_obs = _get(h, "DATE-OBS", "DATEOBS")
    date_fallback = _get(h, "DATE")

    ut = _get(h, "UT")
    time_obs = _get(h, "TIME-OBS", "TIMEOBS")

    date_src = date_obs or date_fallback
    if not date_src:
        raise HeaderContractError(
            "Missing DATE-OBS/DATE",
            instrument=instrument,
            missing_keys=["DATE-OBS"],
        )

    try:
        y, m, d = parse_date_obs(str(date_src))
    except Exception as e:
        raise HeaderContractError(
            "Invalid DATE-OBS/DATE",
            instrument=instrument,
            context={"DATE-OBS": date_src},
        ) from e

    time_src = None
    if prefer_ut and ut is not None and norm_str(ut):
        time_src = ut
    elif time_obs is not None and norm_str(time_obs):
        time_src = time_obs
    elif ut is not None and norm_str(ut):
        time_src = ut

    if not time_src:
        raise HeaderContractError(
            "Missing UT/TIME-OBS",
            instrument=instrument,
            missing_keys=["UT"],
        )

    try:
        hh, mm, ss = parse_hms(str(time_src))
    except Exception as e:
        raise HeaderContractError(
            "Invalid UT/TIME-OBS",
            instrument=instrument,
            context={"UT": time_src},
        ) from e

    whole = int(ss)
    micros = int(round((ss - whole) * 1_000_000))
    return datetime(y, m, d, hh, mm, whole, micros, tzinfo=timezone.utc)


def classify_imagetyp(v: Any) -> str:
    s = norm_lower(v)
    # collapse variants
    if s in {"object", "obj", "science"}:
        return "obj"
    if s.startswith("bias"):
        return "bias"
    if s.startswith("flat"):
        return "flat"
    if s.startswith("neon") or s.startswith("arc"):
        return "neon"
    return s


def parse_frame_meta(h: Mapping[str, Any], *, strict: bool = True) -> FrameMeta:
    """Dispatch to SCORPIO-1/SCORPIO-2 parser by inspecting ``INSTRUME``."""

    from .scorpio1 import Scorpio1HeaderParser
    from .scorpio2 import Scorpio2HeaderParser

    instr = norm_upper_nospace(_get(h, "INSTRUME", "INSTRUMENT") or "")
    if "SCORPIO-2" in instr or "SCORPIO2" in instr:
        return Scorpio2HeaderParser(strict=strict).parse(h)
    if "SCORPIO" in instr:
        return Scorpio1HeaderParser(strict=strict).parse(h)

    raise HeaderContractError(
        "Unknown instrument (cannot dispatch header parser)",
        instrument=instr or None,
        context={"INSTRUME": _get(h, "INSTRUME", "INSTRUMENT")},
    )
