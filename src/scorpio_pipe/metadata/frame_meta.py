from __future__ import annotations

"""FrameMeta and header normalization (P0-B2).

This module defines the **single source of truth** for normalized frame
metadata. Stages and association logic must not read raw FITS headers directly
for compatibility keys.

Design principles
-----------------
- Deterministic: one header + one policy -> one FrameMeta.
- No silent guessing: missing required keys raise :class:`HeaderContractError`.
- Traceable: every field records provenance (header vs fallback).

Notes
-----
We keep the core `FrameMeta` fields compatible with earlier versions to avoid
large-scale churn, and append new contract fields with safe defaults.
"""

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
import math
import re
from typing import Any, Mapping

from scorpio_pipe.lamp_contract import (
    LAMP_HENEAR,
    LAMP_NE,
    LAMP_UNKNOWN,
    infer_lamp_from_header,
)

from .policy import FieldRequirement, MetadataPolicy, default_metadata_policy
from .sources import FallbackSource, FallbackSources


class HeaderContractError(RuntimeError):
    """Raised when a header violates the metadata contract."""

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
            ctx = ", ".join(f"{k}={v!r}" for k, v in list(self.context.items())[:8])
            base += f" | ctx: {ctx}"
        super().__init__(base)


@dataclass(frozen=True)
class ReadoutKey:
    """Readout configuration key used for matching."""

    node: str
    rate: float
    gain: float
    rdnoise: float | None = None


@dataclass(frozen=True)
class FrameMeta:
    """Normalized, instrument-agnostic metadata extracted from FITS headers."""

    # --- Core fields used throughout the pipeline (kept stable for compatibility) ---
    instrument: str  # "SCORPIO1" | "SCORPIO2"

    mode: str
    imagetyp: str  # bias|flat|neon|obj|...
    disperser: str

    slit_width_arcsec: float
    slit_pos: float | None

    binning_x: int
    binning_y: int

    naxis1: int
    naxis2: int

    readout_key: ReadoutKey
    date_time_utc: datetime
    object_name: str

    # --- New contract fields (P0-B2) ---
    frame_id: str = ""
    exptime: float | None = None
    detector: str = ""
    slitmask: str = ""
    sperange: str | None = None

    # Calibration-grade metadata (P0.2)
    airmass: float | None = None
    zenith_distance_deg: float | None = None
    parallactic_angle_deg: float | None = None
    ra_deg: float | None = None
    dec_deg: float | None = None
    utc_mid_exposure: datetime | None = None
    observatory: str = ""
    site_name: str = ""
    seeing_fwhm_arcsec: float | None = None

    # Provenance and policy outcomes
    meta_provenance: dict[str, str] = field(default_factory=dict)
    meta_missing_optional: tuple[str, ...] = ()
    meta_fallback_used: dict[str, str] = field(default_factory=dict)

    # Lamp contract (P0-M)
    lamp_raw: str = ""
    lamp_type: str = "Unknown"  # HeNeAr | Ne | Unknown
    lamp_source: str = "none"  # header | config | default | none

    @property
    def binning_key(self) -> str:
        return f"{self.binning_x}x{self.binning_y}"

    @property
    def slit_width_key(self) -> str:
        """Stable slit width string for grouping."""

        try:
            v = float(self.slit_width_arcsec)
        except Exception:
            v = float("nan")
        if not math.isfinite(v):
            return "nan"
        v = round(v, 1)
        return f"{v:.1f}"

    @property
    def instrument_db_key(self) -> str:
        return "SCORPIO-2" if self.instrument == "SCORPIO2" else "SCORPIO"

    @property
    def grism(self) -> str:
        """Alias for `disperser` (common jargon in configs)."""

        return self.disperser

    @property
    def objtype(self) -> str:
        """Alias for the normalized frame type."""

        return self.imagetyp

    def with_frame_id(self, frame_id: str) -> "FrameMeta":
        return replace(self, frame_id=str(frame_id))

    def provenance_for(self, field: str) -> str:
        return str((self.meta_provenance or {}).get(str(field), "missing"))

    def provenance_class(self, field: str) -> str:
        prov = self.provenance_for(field)
        if prov.startswith("header:"):
            return "header"
        if prov.startswith("derived:"):
            return "derived"
        if prov.startswith("fallback:"):
            return "fallback"
        val = getattr(self, str(field), None)
        if val not in (None, ""):
            return "untracked"
        return "missing"

    def metadata_is_degraded(self, field: str) -> bool:
        return self.provenance_class(field) in {"fallback", "missing"}

    def calibration_metadata_summary(self) -> dict[str, Any]:
        keys = (
            "airmass",
            "zenith_distance_deg",
            "parallactic_angle_deg",
            "ra_deg",
            "dec_deg",
            "utc_mid_exposure",
            "observatory",
            "site_name",
            "seeing_fwhm_arcsec",
        )
        return {
            str(k): {
                "value": getattr(self, k),
                "provenance": self.provenance_for(k),
                "quality": self.provenance_class(k),
            }
            for k in keys
        }

    @classmethod
    def from_header(
        cls,
        header: Mapping[str, Any],
        *,
        strict: bool = True,
        policy: MetadataPolicy | None = None,
        fallback_sources: FallbackSources | None = None,
    ) -> "FrameMeta":
        """Construct a FrameMeta from a FITS header-like mapping."""

        # Delegate to the single normalizer entrypoint.
        return parse_frame_meta(
            header, strict=strict, policy=policy, fallback_sources=fallback_sources
        )


# -----------------------------
# Normalization helpers
# -----------------------------


def norm_str(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
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


def _find_key(h: Mapping[str, Any], *keys: str) -> str | None:
    for k in keys:
        if k in h:
            return k
    return None


def classify_imagetyp(v: Any) -> str:
    s = norm_lower(v)
    if s in {"object", "obj", "science"}:
        return "obj"
    if s.startswith("bias"):
        return "bias"
    if s.startswith("flat"):
        return "flat"
    if s.startswith("neon") or s.startswith("arc"):
        return "neon"
    return s


def parse_date_obs(date_obs: str) -> tuple[int, int, int]:
    s = norm_str(date_obs)
    if not s:
        raise ValueError("empty DATE-OBS")
    if "T" in s:
        s = s.split("T", 1)[0]
    if "/" in s:
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
    dt = datetime.fromisoformat(s)
    return dt.year, dt.month, dt.day


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


def _parse_angle_like(value: Any, *, is_ra: bool = False) -> float | None:
    raw = norm_str(value)
    if not raw:
        return None
    try:
        fv = float(raw)
        return fv * 15.0 if is_ra and abs(fv) <= 24.0 else fv
    except Exception:
        pass

    s = raw.lower().replace('h', ':').replace('d', ':').replace('m', ':').replace('s', '')
    s = re.sub(r'\s+', ':', s)
    parts = [p for p in s.split(':') if p]
    if len(parts) < 3:
        return None
    sign = -1.0 if str(parts[0]).strip().startswith('-') else 1.0
    vals = [abs(float(parts[0])), float(parts[1]), float(parts[2])]
    deg = vals[0] + vals[1] / 60.0 + vals[2] / 3600.0
    if is_ra:
        return deg * 15.0
    return sign * deg


def _parse_ra_deg(h: Mapping[str, Any]) -> tuple[float | None, str]:
    for k in ("RA_DEG", "OBJRA_DEG"):
        if k in h and norm_str(h.get(k)):
            try:
                return float(h.get(k)), f"header:{k}"
            except Exception:
                pass
    for k in ("RA", "OBJRA", "OBJ_RA"):
        if k in h and norm_str(h.get(k)):
            v = _parse_angle_like(h.get(k), is_ra=True)
            if v is not None:
                return float(v), f"header:{k}"
    return None, "missing"


def _parse_dec_deg(h: Mapping[str, Any]) -> tuple[float | None, str]:
    for k in ("DEC_DEG", "OBJDEC_DEG"):
        if k in h and norm_str(h.get(k)):
            try:
                return float(h.get(k)), f"header:{k}"
            except Exception:
                pass
    for k in ("DEC", "OBJDEC", "OBJ_DEC"):
        if k in h and norm_str(h.get(k)):
            v = _parse_angle_like(h.get(k), is_ra=False)
            if v is not None:
                return float(v), f"header:{k}"
    return None, "missing"


def _derive_zenith_distance_deg(airmass: float | None) -> tuple[float | None, str]:
    if airmass is None:
        return None, "missing"
    try:
        am = float(airmass)
    except Exception:
        return None, "missing"
    if not math.isfinite(am) or am < 1.0:
        return None, "missing"
    try:
        secz = max(1.0, am)
        zd = math.degrees(math.acos(1.0 / secz))
        return float(zd), "derived:airmass"
    except Exception:
        return None, "missing"


def _derive_mid_exposure_utc(dt_utc: datetime, exptime: float | None) -> tuple[datetime | None, str]:
    if exptime is None:
        return None, "missing"
    try:
        exp = float(exptime)
    except Exception:
        return None, "missing"
    if not math.isfinite(exp) or exp < 0:
        return None, "missing"
    return dt_utc + timedelta(seconds=0.5 * exp), "derived:date_time_utc+exptime"


def _extract_calibration_metadata(b: "_MetaBuilder", dt_utc: datetime, exptime: float | None) -> dict[str, Any]:
    airmass = b.get_float("airmass", ["AIRMASS", "AIRMAS"], default=None)
    zd = b.get_float("zenith_distance_deg", ["ZDIST", "ZD", "ZENITHD", "ZENITH"], default=None)
    parang = b.get_float("parallactic_angle_deg", ["PARANGLE", "PARANG", "PARALLAC"], default=None)
    seeing = b.get_float("seeing_fwhm_arcsec", ["SEEING", "SEEING1", "FWHM"], default=None)
    observatory = b.get_str("observatory", ["OBSERVAT", "OBSERVATORY", "TELESCOP"], default="")
    site_name = b.get_str("site_name", ["SITENAME", "SITE", "OBSERVAT", "OBSERVATORY"], default="")

    ra_deg, ra_src = _parse_ra_deg(b.h)
    dec_deg, dec_src = _parse_dec_deg(b.h)
    utc_mid, utc_mid_src = _derive_mid_exposure_utc(dt_utc, exptime)

    if zd is None:
        zd, zd_src = _derive_zenith_distance_deg(airmass)
    else:
        zd_src = b.prov.get("zenith_distance_deg", "missing")

    b.prov.setdefault("ra_deg", ra_src)
    b.prov.setdefault("dec_deg", dec_src)
    b.prov.setdefault("utc_mid_exposure", utc_mid_src)
    b.prov.setdefault("zenith_distance_deg", zd_src)
    b.prov.setdefault("airmass", b.prov.get("airmass", "missing"))
    b.prov.setdefault("parallactic_angle_deg", b.prov.get("parallactic_angle_deg", "missing"))
    b.prov.setdefault("seeing_fwhm_arcsec", b.prov.get("seeing_fwhm_arcsec", "missing"))
    b.prov.setdefault("observatory", b.prov.get("observatory", "missing"))
    b.prov.setdefault("site_name", b.prov.get("site_name", "missing"))

    if airmass is None:
        b.missing_optional.append("airmass")
    if zd is None:
        b.missing_optional.append("zenith_distance_deg")
    if parang is None:
        b.missing_optional.append("parallactic_angle_deg")
    if ra_deg is None:
        b.missing_optional.append("ra_deg")
    if dec_deg is None:
        b.missing_optional.append("dec_deg")
    if utc_mid is None:
        b.missing_optional.append("utc_mid_exposure")
    if not observatory:
        b.missing_optional.append("observatory")
    if not site_name:
        b.missing_optional.append("site_name")
    if seeing is None:
        b.missing_optional.append("seeing_fwhm_arcsec")

    return {
        "airmass": airmass,
        "zenith_distance_deg": zd,
        "parallactic_angle_deg": parang,
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "utc_mid_exposure": utc_mid,
        "observatory": observatory,
        "site_name": site_name,
        "seeing_fwhm_arcsec": seeing,
    }


class _MetaBuilder:
    """Helper that applies policy + records provenance."""

    def __init__(
        self,
        h: Mapping[str, Any],
        *,
        instrument: str,
        policy: MetadataPolicy,
        strict: bool,
        fallback: FallbackSources | None,
    ):
        self.h = h
        self.instrument = instrument
        self.policy = policy
        self.strict = bool(strict)
        self.fallback = fallback
        self.prov: dict[str, str] = {}
        self.missing_optional: list[str] = []
        self.fallback_used: dict[str, str] = {}

    def _req(self, field: str) -> FieldRequirement:
        return self.policy.requirement_for(field, strict=self.strict)

    def _missing(self, field: str, keys: list[str], *, hint: str | None = None) -> None:
        req = self._req(field)
        if req == FieldRequirement.OPTIONAL:
            self.missing_optional.append(field)
            return

        if req == FieldRequirement.FALLBACK_ALLOWED and self.fallback is not None:
            got = self.fallback.resolve(field, instrument=self.instrument)
            if got is not None:
                val, src = got
                src_s = f"fallback:{str(src.value).lower()}"
                self.prov[field] = src_s
                self.fallback_used[field] = src_s
                return

        exp = ", ".join(keys)
        msg = f"Missing required metadata field {field!r}. Expected header key(s): {exp}."
        if hint:
            msg += f" Hint: {hint}"
        raise HeaderContractError(msg, instrument=self.instrument, missing_keys=keys)

    def get_str(
        self,
        field: str,
        keys: list[str],
        *,
        norm=norm_str,
        allow_empty: bool = False,
        default: str = "",
        hint: str | None = None,
    ) -> str:
        k = _find_key(self.h, *keys)
        if k is not None:
            raw = self.h.get(k)
            s = norm(raw)
            if s or allow_empty:
                self.prov[field] = f"header:{k}"
                return s
        self._missing(field, keys, hint=hint)
        # Optional / fallback path
        if field in self.fallback_used and self.fallback is not None:
            got = self.fallback.resolve(field, instrument=self.instrument)
            if got is not None:
                val, _src = got
                return norm(val)
        return default

    def get_float(
        self,
        field: str,
        keys: list[str],
        *,
        allow_empty: bool = False,
        default: float | None = None,
        hint: str | None = None,
    ) -> float | None:
        k = _find_key(self.h, *keys)
        if k is not None:
            raw = self.h.get(k)
            if raw is None:
                pass
            else:
                s = norm_str(raw)
                if s or allow_empty:
                    try:
                        v = float(raw)
                        self.prov[field] = f"header:{k}"
                        return v
                    except Exception:
                        raise HeaderContractError(
                            f"Invalid float value for {k}: {raw!r}",
                            instrument=self.instrument,
                            context={k: raw},
                        )
        self._missing(field, keys, hint=hint)
        if field in self.fallback_used and self.fallback is not None:
            got = self.fallback.resolve(field, instrument=self.instrument)
            if got is not None:
                val, _src = got
                try:
                    return float(val)
                except Exception:
                    return default
        return default

    def get_int(self, field: str, keys: list[str], *, default: int | None = None, hint: str | None = None) -> int | None:
        k = _find_key(self.h, *keys)
        if k is not None:
            raw = self.h.get(k)
            if raw is not None and norm_str(raw):
                try:
                    v = int(raw)
                    self.prov[field] = f"header:{k}"
                    return v
                except Exception:
                    raise HeaderContractError(
                        f"Invalid int value for {k}: {raw!r}",
                        instrument=self.instrument,
                        context={k: raw},
                    )
        self._missing(field, keys, hint=hint)
        if field in self.fallback_used and self.fallback is not None:
            got = self.fallback.resolve(field, instrument=self.instrument)
            if got is not None:
                val, _src = got
                try:
                    return int(val)
                except Exception:
                    return default
        return default

    def parse_binning(self) -> tuple[int, int]:
        # Common SCORPIO variants.
        for k in ("BINNING", "CCDSUM", "CCDBIN", "CCDBIN1"):
            if k in self.h:
                raw = self.h.get(k)
                s = norm_upper_nospace(raw).replace("X", "x")
                if "x" in s:
                    parts = [p for p in s.split("x") if p]
                    if len(parts) == 2 and all(re.fullmatch(r"\d+", p) for p in parts):
                        self.prov["binning"] = f"header:{k}"
                        return int(parts[0]), int(parts[1])
                parts = [p for p in re.split(r"\s+", norm_str(raw)) if p]
                if len(parts) == 2 and all(re.fullmatch(r"\d+", p) for p in parts):
                    self.prov["binning"] = f"header:{k}"
                    return int(parts[0]), int(parts[1])

        xk = _find_key(self.h, "BINX", "XBIN", "XBINNING")
        yk = _find_key(self.h, "BINY", "YBIN", "YBINNING")
        if xk and yk:
            try:
                bx = int(self.h.get(xk))
                by = int(self.h.get(yk))
                self.prov["binning"] = f"header:{xk}+{yk}"
                return bx, by
            except Exception:
                pass

        self._missing("binning", ["BINNING"], hint="provide BINNING/CCDSUM/CCDBIN in the FITS header")
        # optional path
        return 1, 1

    def parse_datetime_utc(self) -> datetime:
        # DATE-OBS / DATE
        date_key = _find_key(self.h, "DATE-OBS", "DATEOBS", "DATE")
        if date_key is None:
            self._missing("date_time_utc", ["DATE-OBS"], hint="header must include DATE-OBS (or DATE)")
            return datetime(1970, 1, 1, tzinfo=timezone.utc)

        date_raw = self.h.get(date_key)
        # UT preferred, TIME-OBS fallback.
        time_key = None
        if _find_key(self.h, "UT") is not None and norm_str(self.h.get("UT")):
            time_key = "UT"
        else:
            time_key = _find_key(self.h, "TIME-OBS", "TIMEOBS", "UT")

        if time_key is None:
            self._missing("date_time_utc", ["UT"], hint="header must include UT or TIME-OBS")
            return datetime(1970, 1, 1, tzinfo=timezone.utc)

        time_raw = self.h.get(time_key)

        try:
            y, m, d = parse_date_obs(str(date_raw))
        except Exception as e:
            raise HeaderContractError(
                "Invalid DATE-OBS/DATE",
                instrument=self.instrument,
                context={date_key: date_raw},
            ) from e

        try:
            hh, mm, ss = parse_hms(str(time_raw))
        except Exception as e:
            raise HeaderContractError(
                "Invalid UT/TIME-OBS",
                instrument=self.instrument,
                context={time_key: time_raw},
            ) from e

        whole = int(ss)
        micros = int(round((ss - whole) * 1_000_000))
        self.prov["date_time_utc"] = f"header:{date_key}+{time_key}"
        return datetime(y, m, d, hh, mm, whole, micros, tzinfo=timezone.utc)


def _extract_sperange(h: Mapping[str, Any]) -> str | None:
    for k in ("SPERANGE", "SPERANG", "SPECRANGE", "SPECRANG", "SPEC_RNG"):
        if k in h and h.get(k) not in (None, ""):
            return str(h.get(k)).strip() or None
    return None


def _parse_slit_width_sc1(h: Mapping[str, Any], *, strict: bool) -> tuple[float, str]:
    # 1) Prefer real numeric value.
    for k in ("SLITWID", "SLIT", "SLITW"):
        if k in h and norm_str(h.get(k)):
            try:
                fv = float(h.get(k))
                if fv > 0:
                    return fv, f"header:{k}"
            except Exception:
                break

    # 2) FILTERS may contain "slit_1.2".
    filters = norm_lower(_get(h, "FILTERS") or "")
    m = re.search(r"\bslit[_-]?(\d+(?:\.\d+)?)\b", filters)
    if m:
        try:
            fv = float(m.group(1))
            if fv > 0:
                return fv, "derived:FILTERS"
        except Exception:
            pass

    if not strict:
        return float("nan"), "missing"

    raise HeaderContractError(
        "Cannot determine slit width for SCORPIO-1 (SLITWID empty and no slit_* in FILTERS)",
        instrument="SCORPIO1",
        missing_keys=["SLITWID", "FILTERS"],
        context={"SLITWID": _get(h, "SLITWID"), "FILTERS": _get(h, "FILTERS")},
    )


def _finalize_meta(
    *,
    b: _MetaBuilder,
    instrument: str,
    mode: str,
    imagetyp: str,
    disperser: str,
    slit_width: float,
    slit_pos: float | None,
    binx: int,
    biny: int,
    naxis1: int,
    naxis2: int,
    node: str,
    rate: float,
    gain: float,
    rdnoise: float | None,
    dt_utc: datetime,
    obj: str,
    exptime: float | None,
    detector: str,
    slitmask: str,
    sperange: str | None,
    airmass: float | None,
    zenith_distance_deg: float | None,
    parallactic_angle_deg: float | None,
    ra_deg: float | None,
    dec_deg: float | None,
    utc_mid_exposure: datetime | None,
    observatory: str,
    site_name: str,
    seeing_fwhm_arcsec: float | None,
) -> FrameMeta:
    # Lamp contract.
    lamp_raw, lamp_type = infer_lamp_from_header(b.h)
    lamp_source = "header" if lamp_type != LAMP_UNKNOWN else "none"
    if imagetyp == "neon" and mode.strip().lower().startswith("spect") and lamp_type in (LAMP_NE, LAMP_UNKNOWN):
        lamp_type = LAMP_HENEAR
        lamp_source = "default"

    prov = dict(b.prov)
    prov.setdefault("lamp_type", lamp_source)

    return FrameMeta(
        instrument=instrument,
        mode=mode,
        imagetyp=imagetyp,
        disperser=disperser,
        slit_width_arcsec=float(slit_width),
        slit_pos=slit_pos,
        binning_x=int(binx),
        binning_y=int(biny),
        naxis1=int(naxis1),
        naxis2=int(naxis2),
        readout_key=ReadoutKey(node=str(node), rate=float(rate), gain=float(gain), rdnoise=rdnoise),
        date_time_utc=dt_utc,
        object_name=obj,
        exptime=exptime,
        detector=detector,
        slitmask=slitmask,
        sperange=sperange,
        airmass=airmass,
        zenith_distance_deg=zenith_distance_deg,
        parallactic_angle_deg=parallactic_angle_deg,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        utc_mid_exposure=utc_mid_exposure,
        observatory=observatory,
        site_name=site_name,
        seeing_fwhm_arcsec=seeing_fwhm_arcsec,
        meta_provenance=prov,
        meta_missing_optional=tuple(dict.fromkeys(b.missing_optional)),
        meta_fallback_used=dict(b.fallback_used),
        lamp_raw=str(lamp_raw or ""),
        lamp_type=str(lamp_type or LAMP_UNKNOWN),
        lamp_source=str(lamp_source or "none"),
    )


def _parse_scorpio2(h: Mapping[str, Any], *, policy: MetadataPolicy, strict: bool, fallback: FallbackSources | None) -> FrameMeta:
    b = _MetaBuilder(h, instrument="SCORPIO2", policy=policy, strict=strict, fallback=fallback)

    mode = b.get_str("mode", ["MODE", "OBSMODE", "OBS_MODE"], hint="set MODE in the FITS header")
    imagetyp_raw = b.get_str("imagetyp", ["IMAGETYP", "IMTYPE", "OBSTYPE"], hint="set IMAGETYP in the FITS header")
    imagetyp = classify_imagetyp(imagetyp_raw)

    disperser = b.get_str(
        "disperser",
        ["DISPERSE", "GRISM", "GRATING"],
        allow_empty=True,
        hint="set DISPERSE/GRISM (blank is allowed for imaging)",
    )
    if strict and mode.strip().lower().startswith("spect") and not disperser:
        raise HeaderContractError(
            "Spectra frame has empty disperser",
            instrument="SCORPIO2",
            context={"MODE": mode, "DISPERSE": _get(h, "DISPERSE", "GRISM", "GRATING")},
        )

    slit_width = b.get_float("slit_width_arcsec", ["SLITWID"], hint="set SLITWID in the FITS header")
    if slit_width is None:
        slit_width = float("nan")
    if strict and mode.strip().lower().startswith("spect") and (not math.isfinite(float(slit_width)) or float(slit_width) <= 0):
        raise HeaderContractError(
            "Invalid SLITWID for spectra",
            instrument="SCORPIO2",
            context={"SLITWID": slit_width},
        )

    slit_pos = b.get_float("slit_pos", ["SLITPOS"], default=None)

    binx, biny = b.parse_binning()
    naxis1 = b.get_int("naxis1", ["NAXIS1"], hint="NAXIS1 must be present")
    naxis2 = b.get_int("naxis2", ["NAXIS2"], hint="NAXIS2 must be present")
    if naxis1 is None or naxis2 is None:
        naxis1 = int(naxis1 or 0)
        naxis2 = int(naxis2 or 0)

    node = b.get_str("node", ["NODE"], hint="set NODE (amplifier) in the FITS header")
    rate = b.get_float("rate", ["RATE"], hint="set RATE in the FITS header")
    gain = b.get_float("gain", ["GAIN", "EGAIN"], hint="set GAIN in the FITS header")
    rdnoise = b.get_float("rdnoise", ["RDNOISE", "READNOISE", "READNOIS", "RON", "RNOISE"], default=None)

    dt_utc = b.parse_datetime_utc()
    obj = b.get_str("object_name", ["OBJECT", "OBJNAME"], default="")

    exptime = b.get_float("exptime", ["EXPTIME", "EXPOSURE", "EXPT"], default=None)
    detector = b.get_str("detector", ["DETECTOR", "CCDNAME", "CCD"], default="")
    slitmask = b.get_str("slitmask", ["SLITMASK"], default="")
    sperange = _extract_sperange(h)
    if sperange is not None:
        b.prov.setdefault("sperange", "header:SPERANGE")

    calib_meta = _extract_calibration_metadata(b, dt_utc, exptime)

    return _finalize_meta(
        b=b,
        instrument="SCORPIO2",
        mode=mode,
        imagetyp=imagetyp,
        disperser=disperser,
        slit_width=float(slit_width),
        slit_pos=slit_pos,
        binx=binx,
        biny=biny,
        naxis1=int(naxis1),
        naxis2=int(naxis2),
        node=node,
        rate=float(rate or float("nan")),
        gain=float(gain or float("nan")),
        rdnoise=rdnoise,
        dt_utc=dt_utc,
        obj=obj,
        exptime=exptime,
        detector=detector,
        slitmask=slitmask,
        sperange=sperange,
        airmass=calib_meta["airmass"],
        zenith_distance_deg=calib_meta["zenith_distance_deg"],
        parallactic_angle_deg=calib_meta["parallactic_angle_deg"],
        ra_deg=calib_meta["ra_deg"],
        dec_deg=calib_meta["dec_deg"],
        utc_mid_exposure=calib_meta["utc_mid_exposure"],
        observatory=calib_meta["observatory"],
        site_name=calib_meta["site_name"],
        seeing_fwhm_arcsec=calib_meta["seeing_fwhm_arcsec"],
    )


def _parse_scorpio1(h: Mapping[str, Any], *, policy: MetadataPolicy, strict: bool, fallback: FallbackSources | None) -> FrameMeta:
    b = _MetaBuilder(h, instrument="SCORPIO1", policy=policy, strict=strict, fallback=fallback)

    mode = b.get_str("mode", ["MODE", "OBSMODE", "OBS_MODE"], hint="set MODE in the FITS header")
    imagetyp_raw = b.get_str("imagetyp", ["IMAGETYP", "IMTYPE", "OBSTYPE"], hint="set IMAGETYP in the FITS header")
    imagetyp = classify_imagetyp(imagetyp_raw)

    disperser = b.get_str(
        "disperser",
        ["DISPERSE", "GRISM", "GRATING"],
        allow_empty=True,
        hint="set DISPERSE/GRISM (blank is allowed for imaging)",
    )
    if strict and mode.strip().lower().startswith("spect") and not disperser:
        raise HeaderContractError(
            "Spectra frame has empty disperser",
            instrument="SCORPIO1",
            context={"MODE": mode, "DISPERSE": _get(h, "DISPERSE", "GRISM", "GRATING")},
        )

    slit_width, slit_src = _parse_slit_width_sc1(h, strict=strict)
    b.prov["slit_width_arcsec"] = slit_src

    slit_pos = b.get_float("slit_pos", ["SLITPOS"], default=None)

    binx, biny = b.parse_binning()
    naxis1 = b.get_int("naxis1", ["NAXIS1"], hint="NAXIS1 must be present")
    naxis2 = b.get_int("naxis2", ["NAXIS2"], hint="NAXIS2 must be present")
    if naxis1 is None or naxis2 is None:
        naxis1 = int(naxis1 or 0)
        naxis2 = int(naxis2 or 0)

    node = b.get_str("node", ["NODE"], hint="set NODE (amplifier) in the FITS header")
    rate = b.get_float("rate", ["RATE"], hint="set RATE in the FITS header")
    gain = b.get_float("gain", ["GAIN", "EGAIN"], hint="set GAIN in the FITS header")
    rdnoise = b.get_float("rdnoise", ["RDNOISE", "READNOISE", "READNOIS", "RON", "RNOISE"], default=None)

    dt_utc = b.parse_datetime_utc()
    obj = b.get_str("object_name", ["OBJECT", "OBJNAME"], default="")

    exptime = b.get_float("exptime", ["EXPTIME", "EXPOSURE", "EXPT"], default=None)
    detector = b.get_str("detector", ["DETECTOR", "CCDNAME", "CCD"], default="")
    slitmask = b.get_str("slitmask", ["SLITMASK"], default="")
    sperange = _extract_sperange(h)
    if sperange is not None:
        b.prov.setdefault("sperange", "header:SPERANGE")

    calib_meta = _extract_calibration_metadata(b, dt_utc, exptime)

    return _finalize_meta(
        b=b,
        instrument="SCORPIO1",
        mode=mode,
        imagetyp=imagetyp,
        disperser=disperser,
        slit_width=float(slit_width),
        slit_pos=slit_pos,
        binx=binx,
        biny=biny,
        naxis1=int(naxis1),
        naxis2=int(naxis2),
        node=node,
        rate=float(rate or float("nan")),
        gain=float(gain or float("nan")),
        rdnoise=rdnoise,
        dt_utc=dt_utc,
        obj=obj,
        exptime=exptime,
        detector=detector,
        slitmask=slitmask,
        sperange=sperange,
        airmass=calib_meta["airmass"],
        zenith_distance_deg=calib_meta["zenith_distance_deg"],
        parallactic_angle_deg=calib_meta["parallactic_angle_deg"],
        ra_deg=calib_meta["ra_deg"],
        dec_deg=calib_meta["dec_deg"],
        utc_mid_exposure=calib_meta["utc_mid_exposure"],
        observatory=calib_meta["observatory"],
        site_name=calib_meta["site_name"],
        seeing_fwhm_arcsec=calib_meta["seeing_fwhm_arcsec"],
    )



def parse_scorpio2_header(
    h: Mapping[str, Any],
    *,
    policy: MetadataPolicy,
    strict: bool = True,
    fallback: FallbackSources | None = None,
) -> FrameMeta:
    """Parse SCORPIO-2 header into FrameMeta (instrument-specific)."""
    return _parse_scorpio2(h, policy=policy, strict=strict, fallback=fallback)


def parse_scorpio1_header(
    h: Mapping[str, Any],
    *,
    policy: MetadataPolicy,
    strict: bool = True,
    fallback: FallbackSources | None = None,
) -> FrameMeta:
    """Parse SCORPIO (SCORPIO-1) header into FrameMeta (instrument-specific)."""
    return _parse_scorpio1(h, policy=policy, strict=strict, fallback=fallback)

def parse_frame_meta(
    h: Mapping[str, Any],
    *,
    strict: bool = True,
    policy: MetadataPolicy | None = None,
    fallback_sources: FallbackSources | None = None,
) -> FrameMeta:
    """Parse a FITS header-like mapping into :class:`FrameMeta`.

    Parameters
    ----------
    strict:
        If False, required fields are treated as optional (but recorded in
        :attr:`FrameMeta.meta_missing_optional`).
    policy:
        Missing-key policy. If not provided, :func:`default_metadata_policy` is
        used.
    fallback_sources:
        Optional explicit fallback values (config/table). Used only when policy
        allows it.
    """

    pol = policy or default_metadata_policy()

    # P0-ARCH-05.1: instrument adapters are resolved via the instrument catalog.
    # This keeps preflight/autoconfig and the metadata layer free of SCORPIO-specific if/else.
    from scorpio_pipe.catalog.instruments import resolve_adapter

    try:
        adapter = resolve_adapter(header=h)
    except KeyError as e:
        raise HeaderContractError(
            "Unknown instrument (cannot resolve adapter)",
            instrument=norm_upper_nospace(_get(h, "INSTRUME", "INSTRUMENT") or "") or None,
            context={"INSTRUME": _get(h, "INSTRUME", "INSTRUMENT")},
        ) from e

    # Adapter may accept policy/fallback (types live in scorpio_pipe.metadata).
    return adapter.parse_header(h, strict=strict, policy=pol, fallback_sources=fallback_sources)



__all__ = ["FrameMeta", "ReadoutKey", "HeaderContractError", "parse_frame_meta"]