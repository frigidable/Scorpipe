from __future__ import annotations

"""SCORPIO-2 header parser implementing the Header Contract."""

from typing import Any, Mapping

from .meta import (
    FrameMeta,
    HeaderContractError,
    ReadoutKey,
    _get,
    classify_imagetyp,
    norm_str,
    norm_upper_nospace,
    parse_binning,
    parse_datetime_utc,
    parse_float,
    parse_int,
)


class Scorpio2HeaderParser:
    """Parse SCORPIO-2 FITS headers into :class:`~scorpio_pipe.instruments.FrameMeta`."""

    def __init__(self, *, strict: bool = True):
        self.strict = bool(strict)

    def parse(self, h: Mapping[str, Any]) -> FrameMeta:
        instr = "SCORPIO2"

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
        if self.strict and mode.strip().lower().startswith("spect") and not disperser:
            raise HeaderContractError(
                "Spectra frame has empty disperser",
                instrument=instr,
                context={"MODE": mode, "DISPERSE": _get(h, "DISPERSE")},
            )

        slit_width = parse_float(h, "SLITWID", instrument=instr)
        if self.strict and slit_width <= 0:
            raise HeaderContractError(
                "Invalid SLITWID (<=0)", instrument=instr, context={"SLITWID": slit_width}
            )

        slit_pos = None
        sp = _get(h, "SLITPOS")
        if sp is not None and norm_str(sp):
            try:
                slit_pos = float(sp)
            except Exception:
                slit_pos = None

        binx, biny = parse_binning(h, instrument=instr)
        naxis1 = parse_int(h, "NAXIS1", instrument=instr)
        naxis2 = parse_int(h, "NAXIS2", instrument=instr)

        node_raw = _get(h, "NODE")
        if node_raw is None and self.strict:
            raise HeaderContractError(
                "Missing NODE", instrument=instr, missing_keys=["NODE"]
            )
        node = norm_upper_nospace(node_raw)

        rate = parse_float(h, "RATE", instrument=instr)
        gain = parse_float(h, "GAIN", instrument=instr)

        dt_utc = parse_datetime_utc(h, instrument=instr, prefer_ut=True)

        obj = norm_str(_get(h, "OBJECT", "OBJNAME") or "")

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
        )
