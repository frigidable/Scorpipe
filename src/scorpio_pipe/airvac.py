"""Air/Vacuum wavelength conversion.

Morton (2000), as published by VALD:
https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion

For vacuum wavelength v in Angstrom and s = 10000/v (inverse microns),
n = 1 + 0.0000834254 + 0.02406147/(130-s**2) + 0.00015998/(38.9-s**2).
This describes dry standard air at 15 C, 1 atm and 0.045% CO2. It is not
an atmospheric correction for the conditions at the telescope.

We support finite wavelengths v >= 2000 A (optical/IR); the corresponding
air lower bound is 2000/n(2000), approximately 1999.352 A. Complex values
and non-finite values, including masked-data NaN sentinels, raise ValueError:
callers must select valid wavelengths explicitly. Empty arrays are supported.
The inverse solves the same equation numerically, not a second approximation.
These NumPy-only helpers never read or change FITS metadata.
"""

from __future__ import annotations

import numpy as np

_MIN_VACUUM_A = 2000.0


def _to_array(x, *, minimum: float = _MIN_VACUUM_A):
    if np.iscomplexobj(x):
        raise ValueError("Wavelengths must be real, finite values in Angstrom")
    w = np.asarray(x, dtype=np.float64)
    if np.any(~np.isfinite(w)) or np.any(w < minimum):
        raise ValueError(f"Wavelengths must be finite and >= {minimum:.12g} Angstrom")
    return w


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
    sigma2 = (1e4 / w) ** 2
    n_minus_1 = (
        0.0000834254
        + 0.02406147 / (130.0 - sigma2)
        + 0.00015998 / (38.9 - sigma2)
    )
    return 1.0 + n_minus_1


def vac_to_air(wave_vac_A):
    """Convert vacuum wavelength(s) to air wavelength(s) in Angstrom."""
    w = _to_array(wave_vac_A)
    n = refractive_index_air_morton2000(w)
    return w / n


def air_to_vac(wave_air_A, *, max_iter: int = 12, rtol: float = 1e-12):
    """Convert air wavelength(s) to vacuum wavelength(s) in Angstrom.

    Uses fixed-point iteration: w_vac = w_air * n(w_vac). Invalid solver
    settings raise ValueError; failure to converge raises RuntimeError.
    """
    if not isinstance(max_iter, (int, np.integer)) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(rtol) or rtol <= 0:
        raise ValueError("rtol must be finite and positive")
    min_air = _MIN_VACUUM_A / refractive_index_air_morton2000(_MIN_VACUUM_A)
    w_air = _to_array(wave_air_A, minimum=float(min_air))
    # The initial guess must be within the vacuum domain even at its boundary.
    w_vac = np.maximum(w_air, _MIN_VACUUM_A)
    for _ in range(max_iter):
        n = refractive_index_air_morton2000(w_vac)
        w_new = w_air * n
        if np.allclose(w_new, w_vac, rtol=rtol, atol=0.0):
            return w_new
        w_vac = w_new
    raise RuntimeError(f"Air-to-vacuum inversion did not converge after {max_iter} iterations")
