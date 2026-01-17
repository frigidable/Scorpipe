from __future__ import annotations

"""Unit model utilities (pipeline internal standard: electrons).

Starting with v5.40.26 the pipeline treats **electrons** as the internal unit
for science arrays (SCI) and variance arrays (VAR). Raw inputs are typically
in ADU; we convert deterministically using the resolved CCD gain.

The conversion is purely linear:

    sci_e   = sci_adu * gain_e_per_adu
    var_e2  = var_adu2 * gain_e_per_adu**2

We also stamp explicit provenance keywords so downstream stages never need to
guess.

Keywords (primary header)
-------------------------
- BUNIT   : human readable unit for SCI ("ADU", "e-", "ADU/s", "e-/s", ...)
- SCORPUM : unit model for SCI ("ADU" or "ELECTRON")
- SCORPU0 : original unit for SCI before conversion (best-effort)
- SCORPGN : gain used [e-/ADU]
- SCORPRN : read noise used [e-]
- SCORPNS : source string (e.g. "header", "db", "override", ...)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from astropy.io import fits

from scorpio_pipe.noise_model import NoiseParams, resolve_noise_params, stamp_noise_keywords


class UnitModel(str, Enum):
    ADU = "ADU"
    ELECTRON = "ELECTRON"


@dataclass(frozen=True)
class UnitProvenance:
    original_unit: str
    unit_model_in: UnitModel
    unit_model_out: UnitModel
    gain_e_per_adu: float
    rdnoise_e: float
    noise_source: str


def _norm_unit_str(u: str) -> str:
    return str(u or "").strip().replace(" ", "").lower()


def split_rate_unit(unit: str) -> tuple[str, bool]:
    """Return (base_unit, per_second). Examples: 'ADU/s' -> ('ADU', True)."""
    u = _norm_unit_str(unit)
    if not u:
        return "", False
    if "/s" in u or "s-1" in u:
        base = u.replace("/s", "").replace("s-1", "").strip()
        return base.upper(), True
    return u.upper(), False


def infer_unit_model(hdr: fits.Header, *, default: UnitModel = UnitModel.ADU) -> UnitModel:
    """Best-effort infer the unit model of SCI data from header.

    Rules:
    - Prefer explicit SCORPUM when present.
    - Else look at BUNIT / SCIUNIT for 'adu' or 'e-' markers.
    - Else fall back to `default` (raw SCORPIO frames are ADU).
    """
    for key in ("SCORPUM", "SCORP_UM", "SCORPIOUM"):
        if key in hdr:
            v = _norm_unit_str(str(hdr.get(key)))
            if "electron" in v or v in ("e", "e-", "e-", "e-/s", "e/s"):
                return UnitModel.ELECTRON
            if "adu" in v:
                return UnitModel.ADU

    unit = str(hdr.get("BUNIT", hdr.get("SCIUNIT", "")) or "")
    u = _norm_unit_str(unit)
    if "adu" in u:
        return UnitModel.ADU
    if "electron" in u or "e-" in u or u in ("e", "e/s", "e-/s"):
        return UnitModel.ELECTRON
    return default


def _format_bunit(model: UnitModel, *, per_second: bool) -> str:
    if model == UnitModel.ELECTRON:
        return "e-/s" if per_second else "e-"
    return "ADU/s" if per_second else "ADU"


def stamp_unit_provenance(hdr: fits.Header, prov: UnitProvenance) -> fits.Header:
    """Stamp provenance keywords into `hdr` (in-place) and return it."""
    try:
        hdr["SCORPUM"] = (str(prov.unit_model_out.value), "SCI unit model (internal standard)")
    except Exception:
        pass
    try:
        hdr.setdefault("SCORPU0", str(prov.original_unit))
    except Exception:
        pass
    try:
        hdr["SCORPGN"] = (float(prov.gain_e_per_adu), "Gain used [e-/ADU]")
    except Exception:
        pass
    try:
        hdr["SCORPRN"] = (float(prov.rdnoise_e), "Read noise used [e-]")
    except Exception:
        pass
    try:
        hdr["SCORPNS"] = (str(prov.noise_source), "Noise params source")
    except Exception:
        pass
    return hdr


def ensure_electron_units(
    sci: np.ndarray,
    var: np.ndarray | None,
    hdr: fits.Header,
    *,
    cfg: dict | None = None,
    gain_override: float | None = None,
    rdnoise_override: float | None = None,
    bias_rn_est_adu: float | None = None,
    bias_rn_est_e: float | None = None,
    instrument_hint: str | None = None,
    require_gain: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, fits.Header, UnitProvenance, NoiseParams]:
    """Ensure SCI/VAR are in electrons and stamp provenance.

    Returns
    -------
    sci_e, var_e2, hdr_out, prov, noise_params
    """
    hdr_out = fits.Header(hdr)

    unit_in = str(hdr_out.get("BUNIT", hdr_out.get("SCIUNIT", "")) or "")
    base_in, per_sec_in = split_rate_unit(unit_in)

    model_in = infer_unit_model(hdr_out, default=UnitModel.ADU)

    params = resolve_noise_params(
        hdr_out,
        cfg=cfg,
        gain_override=gain_override,
        rdnoise_override=rdnoise_override,
        bias_rn_est_adu=bias_rn_est_adu,
        bias_rn_est_e=bias_rn_est_e,
        instrument_hint=instrument_hint,
        require_gain=require_gain if model_in == UnitModel.ADU else False,
    )
    gain = float(params.gain_e_per_adu)
    rn = float(params.rdnoise_e)

    sci = np.asarray(sci, dtype=np.float64)
    var_arr = None if var is None else np.asarray(var, dtype=np.float64)

    if model_in == UnitModel.ELECTRON:
        sci_e = sci
        var_e2 = var_arr
    else:
        sci_e = sci * gain
        var_e2 = None if var_arr is None else (var_arr * (gain ** 2))

    # Stamp explicit unit header.
    hdr_out["BUNIT"] = (_format_bunit(UnitModel.ELECTRON, per_second=per_sec_in), "Data unit (SCI)")
    prov = UnitProvenance(
        original_unit=str(unit_in) if unit_in else ("ADU" if model_in == UnitModel.ADU else "e-"),
        unit_model_in=model_in,
        unit_model_out=UnitModel.ELECTRON,
        gain_e_per_adu=gain,
        rdnoise_e=rn,
        noise_source=str(params.source),
    )
    stamp_unit_provenance(hdr_out, prov)
    # Compact <=8-char noise cards for interoperability.
    hdr_out = stamp_noise_keywords(hdr_out, params, overwrite=True)
    try:
        hdr_out.add_history(
            f"scorpio_pipe units: ELECTRON gain={gain:.6g} rdnoise_e={rn:.6g} rn_src={params.rn_src}"
        )
    except Exception:
        pass

    return (
        np.asarray(sci_e, dtype=np.float32),
        None if var_e2 is None else np.asarray(var_e2, dtype=np.float32),
        hdr_out,
        prov,
        params,
    )
