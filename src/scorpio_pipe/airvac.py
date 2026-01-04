from __future__ import annotations

"""Air/Vacuum wavelength conversion.

We implement the common Morton (2000)-style refractive index approximation used
widely in optical spectroscopy pipelines.

Input/Output wavelengths are in Angstrom.

Notes
-----
- The conversion is deterministic and reversible (via iteration for air->vac).
- This module is intentionally dependency-light (NumPy only).
"""

import numpy as np


def _to_array(x):
    return np.asarray(x, dtype=np.float64)


def refractive_index_air_morton2000(wave_vac_A: np.ndarray) -> np.ndarray:
    """Refractive index of air n(λ) for standard conditions (Morton 2000).

    Parameters
    ----------
    wave_vac_A : array
        Vacuum wavelength in Angstrom.

    Returns
    -------
    n : array
        Refractive index (dimensionless).
    """
    w = _to_array(wave_vac_A)
    # Convert to microns for the standard form (σ in µm^-1).
    wave_um = w * 1e-4
    sigma2 = (1.0 / wave_um) ** 2
    # Morton 2000 (similar to Edlén 1953/1966 style), in 1e-6.
    n_minus_1 = 1e-6 * (
        8342.13
        + 2406030.0 / (130.0 - sigma2)
        + 15997.0 / (38.9 - sigma2)
    )
    return 1.0 + n_minus_1


def vac_to_air(wave_vac_A):
    """Convert vacuum wavelength(s) to air wavelength(s) in Angstrom."""
    w = _to_array(wave_vac_A)
    n = refractive_index_air_morton2000(w)
    return w / n


def air_to_vac(wave_air_A, *, max_iter: int = 12, rtol: float = 1e-12):
    """Convert air wavelength(s) to vacuum wavelength(s) in Angstrom.

    Uses fixed-point iteration: w_vac = w_air * n(w_vac).
    """
    w_air = _to_array(wave_air_A)
    w_vac = np.array(w_air, copy=True, dtype=np.float64)
    for _ in range(max_iter):
        n = refractive_index_air_morton2000(w_vac)
        w_new = w_air * n
        if np.allclose(w_new, w_vac, rtol=rtol, atol=0.0):
            w_vac = w_new
            break
        w_vac = w_new
    return w_vac
