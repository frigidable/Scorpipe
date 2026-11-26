"""Versioned FlatField MasterBias component contract (detector-local).

Do not conflate these diagonal marginal maps with a complete covariance
matrix. Pre-existing total VAR is retained unchanged. The 'REST' component
still contains MasterFlat and other errors, which may also be shared.
"""
from __future__ import annotations

import numpy as np


def decompose_flat_masterbias_variance(
    total_variance_e2: np.ndarray,
    bias_estimator_variance_adu2: np.ndarray,
    flat_response: np.ndarray,
    gain_e_per_adu: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (VBIAS[e-^2], VREST[e-^2], BSENS[e-/ADU]).

    Input MasterBias B is subtracted once from unrectified ADU science.
    At frozen flat F and gain G, dSCI_e/dB_adu = -G/F.
    The output per-pixel diagonal MasterBias contribution is
    VBIAS = (G/F)^2 VAR(B); VREST = VAR(total) - VBIAS.

    This is a *local* diagonal decomposition before SkySub and wavelength
    rectification. Flat errors are held fixed for the derivative.
    """
    total = np.asarray(total_variance_e2, dtype=np.float64)
    bias = np.asarray(bias_estimator_variance_adu2, dtype=np.float64)
    flat = np.asarray(flat_response, dtype=np.float64)
    if total.ndim != 2 or bias.shape != total.shape or flat.shape != total.shape:
        raise ValueError("MasterBias component inputs require identical 2D CCD geometry")
    gain = float(gain_e_per_adu)
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("MasterBias component gain must be positive and finite")
    if np.any(~np.isfinite(bias) | (bias < 0)):
        raise ValueError("MasterBias estimator variance must be finite and nonnegative")
    good_flat = np.isfinite(flat) & (flat != 0)
    if np.any(~np.isfinite(total) | (total < 0)):
        raise ValueError("FlatField output VAR must be finite and nonnegative")
    sensitivity = np.zeros_like(total)
    sensitivity[good_flat] = -gain / flat[good_flat]
    if np.any(~np.isfinite(sensitivity)):
        raise ValueError("Nonfinite MasterBias response through MasterFlat")
    vbias = np.square(sensitivity) * bias
    rest = total - vbias
    # Existing FlatField uses float32 intermediate VAR arithmetic; minor
    # cancellation is tolerated at the *same* order as its rounding.
    tol = 3e-5 * np.maximum(1.0, np.maximum(np.abs(total), np.abs(vbias)))
    if np.any(rest < -tol):
        raise ValueError("MasterBias component exceeds total FlatField variance")
    rest = np.maximum(rest, 0.)
    if not np.allclose(total, rest + vbias, rtol=3e-5, atol=3e-5):
        raise ValueError("FlatField component variance closure failed")
    return (
        vbias.astype(np.float32), rest.astype(np.float32),
        sensitivity.astype(np.float32),
    )
