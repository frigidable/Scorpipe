from __future__ import annotations

"""Noise / variance helpers.

This module centralizes how the pipeline interprets CCD noise metadata.

We assume science arrays are stored in ADU by default. A simple, physically
interpretable variance model in electrons is:

    var_e  = max(signal_e, 0) + RN^2

Converting to ADU units (gain in e-/ADU) gives:

    var_adu2 = var_e / gain^2

Notes
-----
* For on-chip binning, read noise applies once per *binned* pixel.
* After deterministic operations (bias subtraction), variance is unchanged.
* For multiplicative flat-fielding (divide by flat), variance scales as 1/flat^2.
"""

from dataclasses import dataclass

import numpy as np
from astropy.io import fits


@dataclass(frozen=True)
class NoiseParams:
    gain_e_per_adu: float
    rdnoise_e: float
    source: str


def _parse_float(hdr: fits.Header, keys: tuple[str, ...]) -> float | None:
    for k in keys:
        if k in hdr:
            try:
                v = float(hdr[k])
                if np.isfinite(v):
                    return v
            except Exception:
                continue
    return None


def _default_rdnoise_sc2(rate_kpix_s: float | None) -> tuple[float, str]:
    """Heuristic defaults for SCORPIO-2 CCD261-84.

    Public docs report typical readout noise in the ~1.8–4.0 e- range and
    three readout rates: 65 / 185 / 335 kpixel/s (slow/normal/fast).

    We do *not* try to be overly precise without per-night controller
    characterization. The goal is: no silent fantasy variance.
    """
    if rate_kpix_s is None or not np.isfinite(rate_kpix_s):
        return 3.0, "default(sc2:typical)"
    if rate_kpix_s <= 100:
        return 2.2, "default(sc2:slow)"
    if rate_kpix_s <= 250:
        return 3.0, "default(sc2:normal)"
    return 4.0, "default(sc2:fast)"


def resolve_noise_params(
    hdr: fits.Header,
    *,
    gain_override: float | None = None,
    rdnoise_override: float | None = None,
    instrument_hint: str | None = None,
    require_gain: bool = False,
) -> NoiseParams:
    """Resolve gain/read-noise (in electrons) from header + optional overrides."""

    # 1) Gain
    if gain_override is not None:
        gain = float(gain_override)
        gain_src = "override"
    else:
        gain = _parse_float(hdr, ("GAIN", "EGAIN", "GAIN_E", "GAINE"))
        if gain is None or gain <= 0:
            if require_gain:
                raise ValueError(
                    "Missing CCD gain (e-/ADU). Provide a valid GAIN/EGAIN header card or set gain_override/gain_e_per_adu in the stage config."
                )
            # Conservative fallback for quicklook/legacy data.
            gain = 1.0
            gain_src = "default(1.0)"
        else:
            gain_src = "header"

    # 2) Read noise
    if rdnoise_override is not None:
        rn = float(rdnoise_override)
        rn_src = "override"
    else:
        rn = _parse_float(hdr, ("RDNOISE", "READNOIS", "RON", "RNOISE"))
        if rn is None or rn <= 0:
            instr = (instrument_hint or str(hdr.get("INSTRUME", "")) or "").upper()
            det = str(hdr.get("DETECTOR", "") or "").upper()
            rate = _parse_float(hdr, ("RATE", "READRATE", "RATERD"))
            if ("SCORPIO" in instr and "2" in instr) or ("CCD261" in det):
                rn, rn_src = _default_rdnoise_sc2(rate)
            else:
                rn = 3.0
                rn_src = "default(typical)"
        else:
            rn_src = "header"

    # 3) Basic sanity
    if not np.isfinite(gain) or gain <= 0:
        if require_gain:
            raise ValueError(
                "Invalid CCD gain (e-/ADU). Provide a valid GAIN/EGAIN header card or set gain_override/gain_e_per_adu in the stage config."
            )
        gain = 1.0
        gain_src = "default(1.0)"
    if not np.isfinite(rn) or rn < 0:
        rn = 3.0
        rn_src = "default(typical)"

    src = f"gain:{gain_src}; rn:{rn_src}"
    return NoiseParams(gain_e_per_adu=float(gain), rdnoise_e=float(rn), source=src)


def estimate_variance_adu2(
    data_adu: np.ndarray,
    hdr: fits.Header,
    *,
    gain_override: float | None = None,
    rdnoise_override: float | None = None,
    instrument_hint: str | None = None,
    require_gain: bool = False,
) -> tuple[np.ndarray, NoiseParams]:
    """Estimate per-pixel variance in ADU^2 from a simple CCD model."""

    params = resolve_noise_params(
        hdr,
        gain_override=gain_override,
        rdnoise_override=rdnoise_override,
        instrument_hint=instrument_hint,
        require_gain=require_gain,
    )
    gain = params.gain_e_per_adu
    rn = params.rdnoise_e

    # Convert to electrons, apply Poisson floor, add read noise.
    e = np.maximum(np.asarray(data_adu, dtype=np.float32) * gain, 0.0)
    var_e = e + float(rn) ** 2
    var_adu2 = var_e / (float(gain) ** 2)
    return np.asarray(var_adu2, dtype=np.float32), params


def estimate_variance_e2(
    sci_e: np.ndarray,
    *,
    rdnoise_e: float,
) -> np.ndarray:
    """Estimate per-pixel variance in electrons^2 from SCI in electrons.

    Model: Var[e-^2] = max(SCI_e, 0) + RN_e^2
    """
    sci = np.asarray(sci_e, dtype=np.float64)
    rn2 = float(rdnoise_e) ** 2
    return (np.maximum(sci, 0.0) + rn2).astype(np.float32)


def estimate_variance_auto(
    sci: np.ndarray,
    hdr: fits.Header,
    *,
    gain_override: float | None = None,
    rdnoise_override: float | None = None,
    unit_model: str | None = None,
    instrument_hint: str | None = None,
    require_gain: bool = False,
) -> tuple[np.ndarray, NoiseParams, str]:
    """Estimate variance for SCI, automatically handling ADU vs electrons.

    Returns (var, params, model) where model is "ADU" or "ELECTRON".
    """
    from scorpio_pipe.units_model import infer_unit_model, UnitModel

    h = fits.Header(hdr)
    model = infer_unit_model(h, default=UnitModel.ADU) if unit_model is None else UnitModel(str(unit_model).upper())
    params = resolve_noise_params(
        h,
        gain_override=gain_override,
        rdnoise_override=rdnoise_override,
        instrument_hint=instrument_hint,
        require_gain=require_gain if str(model.value).upper() == "ADU" else False,
    )
    if model == UnitModel.ELECTRON:
        var_e2 = estimate_variance_e2(sci, rdnoise_e=params.rdnoise_e)
        return var_e2, params, model.value

    # ADU model
    var_adu2, _params = estimate_variance_adu2(
        sci,
        h,
        gain_override=gain_override,
        rdnoise_override=rdnoise_override,
        instrument_hint=instrument_hint,
        require_gain=require_gain,
    )
    # `params` and `_params` are identical by construction; keep `params`.
    return var_adu2, params, model.value
