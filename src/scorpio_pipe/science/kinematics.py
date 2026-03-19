from __future__ import annotations

import math

C_KMS = 299792.458


def velocity_from_centroid(rest_wavelength: float, observed_centroid: float) -> float | None:
    try:
        r = float(rest_wavelength)
        o = float(observed_centroid)
    except Exception:
        return None
    if not math.isfinite(r) or not math.isfinite(o) or r <= 0:
        return None
    return float(C_KMS * (o / r - 1.0))


def sigma_to_fwhm(sigma: float | None) -> float | None:
    if sigma is None:
        return None
    try:
        s = float(sigma)
    except Exception:
        return None
    if not math.isfinite(s) or s < 0:
        return None
    return float(2.354820045 * s)


def sigma_to_velocity_dispersion(rest_wavelength: float, sigma_angstrom: float | None) -> float | None:
    if sigma_angstrom is None:
        return None
    try:
        r = float(rest_wavelength)
        s = float(sigma_angstrom)
    except Exception:
        return None
    if not math.isfinite(r) or not math.isfinite(s) or r <= 0 or s < 0:
        return None
    return float(C_KMS * s / r)
