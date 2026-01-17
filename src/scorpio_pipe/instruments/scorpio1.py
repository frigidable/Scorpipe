from __future__ import annotations

"""SCORPIO-1 header parser implementing the Header Contract."""

import re
from typing import Any, Mapping

from .meta import (
    FrameMeta,
    HeaderContractError,
    ReadoutKey,
    _get,
    classify_imagetyp,
    norm_lower,
    norm_str,
    norm_upper_nospace,
    parse_binning,
    parse_datetime_utc,
    parse_int,
)

from scorpio_pipe.lamp_contract import (
    LAMP_HENEAR,
    LAMP_NE,
    LAMP_UNKNOWN,
    infer_lamp_from_header,
)



def _parse_slit_width_arcsec(h: Mapping[str, Any], *, strict: bool) -> float:
    """SCORPIO-1 slit width logic.

    In many SCORPIO-1 headers ``SLITWID`` is present but empty. In this case we
    extract the slit from the ``FILTERS`` string, e.g. ``slit_1.2``.
    """

    # 1) Prefer a real numeric value if present.
    v = _get(h, "SLITWID", "SLIT", "SLITW")
    if v is not None and norm_str(v):
        try:
            fv = float(v)
            if fv > 0:
                return fv
        except Exception:
            # fall through to FILTERS
            pass

    # 2) FILTERS may contain: "slit_1.2", "slit_0.5 V", ...
    filters = norm_lower(_get(h, "FILTERS") or "")
    m = re.search(r"\bslit[_-]?(\d+(?:\.\d+)?)\b", filters)
    if m:
        try:
            fv = float(m.group(1))
            if fv > 0:
                return fv
        except Exception:
            pass

    if not strict:
        return float("nan")

    raise HeaderContractError(
        "Cannot determine slit width for SCORPIO-1 (SLITWID empty and no slit_* in FILTERS)",
        instrument="SCORPIO1",
        missing_keys=["SLITWID", "FILTERS"],
        context={
            "SLITWID": _get(h, "SLITWID"),
            "FILTERS": _get(h, "FILTERS"),
        },
    )


class Scorpio1HeaderParser:
    """Parse SCORPIO-1 FITS headers into :class:`~scorpio_pipe.instruments.FrameMeta`."""

    def __init__(self, *, strict: bool = True):
        self.strict = bool(strict)

    def parse(self, h: Mapping[str, Any]) -> FrameMeta:
        instr = "SCORPIO1"

        # Required: mode/imagetyp determine downstream grouping.
        mode_raw = _get(h, "MODE", "OBSMODE", "OBS_MODE")
        if mode_raw is None and self.strict:
            raise HeaderContractError(
                "Missing MODE", instrument=instr, missing_keys=["MODE"]
            )
        mode = norm_str(mode_raw)

        imagetyp_raw = _get(h, "IMAGETYP", "IMTYPE", "OBSTYPE")
        if imagetyp_raw is None and self.strict:
            raise HeaderContractError(
                "Missing IMAGETYP", instrument=instr, missing_keys=["IMAGETYP"]
            )
        imagetyp = classify_imagetyp(imagetyp_raw)

        # Disperser: must exist (even if blank) to prevent silent mismatch.
        if _get(h, "DISPERSE", "GRISM", "GRATING") is None and self.strict:
            raise HeaderContractError(
                "Missing disperser keyword (DISPERSE/GRISM/GRATING)",
                instrument=instr,
                missing_keys=["DISPERSE"],
            )
        disperser = norm_str(_get(h, "DISPERSE", "GRISM", "GRATING") or "")
        # If this is a spectra mode frame and disperser is empty -> contract violation.
        if self.strict and mode.strip().lower().startswith("spect") and not disperser:
            raise HeaderContractError(
                "Spectra frame has empty disperser",
                instrument=instr,
                context={"MODE": mode, "DISPERSE": _get(h, "DISPERSE")},
            )

        slit_width = _parse_slit_width_arcsec(h, strict=self.strict)

        # Optional slit position.
        slit_pos = None
        sp = _get(h, "SLITPOS")
        if sp is not None and norm_str(sp):
            try:
                slit_pos = float(sp)
            except Exception:
                slit_pos = None

        # Required: binning, frame size, readout.
        binx, biny = parse_binning(h, instrument=instr)
        naxis1 = parse_int(h, "NAXIS1", instrument=instr)
        naxis2 = parse_int(h, "NAXIS2", instrument=instr)

        node_raw = _get(h, "NODE")
        if node_raw is None and self.strict:
            raise HeaderContractError(
                "Missing NODE", instrument=instr, missing_keys=["NODE"]
            )
        node = norm_upper_nospace(node_raw)

        # RATE/GAIN influence readout behavior and must be explicit.
        rate_raw = _get(h, "RATE")
        gain_raw = _get(h, "GAIN")
        missing = [k for k, v in (("RATE", rate_raw), ("GAIN", gain_raw)) if v is None]
        if missing and self.strict:
            raise HeaderContractError(
                "Missing readout keyword(s)", instrument=instr, missing_keys=missing
            )
        try:
            rate = float(rate_raw) if rate_raw is not None else float("nan")
        except Exception:
            if self.strict:
                raise HeaderContractError(
                    "Invalid RATE", instrument=instr, context={"RATE": rate_raw}
                )
            rate = float("nan")
        try:
            gain = float(gain_raw) if gain_raw is not None else float("nan")
        except Exception:
            if self.strict:
                raise HeaderContractError(
                    "Invalid GAIN", instrument=instr, context={"GAIN": gain_raw}
                )
            gain = float("nan")

        # Time: prefer UT (explicitly labelled universal), fallback to DATE-OBS/TIME-OBS.
        dt_utc = parse_datetime_utc(h, instrument=instr, prefer_ut=True)

        obj = norm_str(_get(h, "OBJECT", "OBJNAME") or "")

        # P0-M: explicit lamp type (header inference + SCORPIO default).
        lamp_raw, lamp_type = infer_lamp_from_header(h)
        lamp_source = "header" if lamp_type != LAMP_UNKNOWN else "none"
        if imagetyp == "neon" and mode == "Spectra" and lamp_type in (LAMP_NE, LAMP_UNKNOWN):
            lamp_type = LAMP_HENEAR
            lamp_source = "default"

        return FrameMeta(
            instrument=instr,
            mode=mode,
            imagetyp=imagetyp,
            disperser=disperser,
            slit_width_arcsec=float(slit_width),
            slit_pos=slit_pos,
            binning_x=int(binx),
            binning_y=int(biny),
            naxis1=int(naxis1),
            naxis2=int(naxis2),
            readout_key=ReadoutKey(node=node, rate=float(rate), gain=float(gain)),
            date_time_utc=dt_utc,
            object_name=obj,
            lamp_raw=str(lamp_raw or ""),
            lamp_type=str(lamp_type or LAMP_UNKNOWN),
            lamp_source=str(lamp_source or "none"),
        )
