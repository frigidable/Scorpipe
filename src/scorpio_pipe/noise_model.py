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
) -> NoiseParams:
    """Resolve gain/read-noise (in electrons) from header + optional overrides."""

    # 1) Gain
    if gain_override is not None:
        gain = float(gain_override)
        gain_src = "override"
    else:
        gain = _parse_float(hdr, ("GAIN", "EGAIN", "GAIN_E", "GAINE"))
        if gain is None or gain <= 0:
            # Conservative fallback.
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
) -> tuple[np.ndarray, NoiseParams]:
    """Estimate per-pixel variance in ADU^2 from a simple CCD model."""

    params = resolve_noise_params(
        hdr,
        gain_override=gain_override,
        rdnoise_override=rdnoise_override,
        instrument_hint=instrument_hint,
    )
    gain = params.gain_e_per_adu
    rn = params.rdnoise_e

    # Convert to electrons, apply Poisson floor, add read noise.
    e = np.maximum(np.asarray(data_adu, dtype=np.float32) * gain, 0.0)
    var_e = e + float(rn) ** 2
    var_adu2 = var_e / (float(gain) ** 2)
    return np.asarray(var_adu2, dtype=np.float32), params
