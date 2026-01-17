"""Variance propagation utilities (P0-C2 VAR contract).

This module provides a small set of **deterministic** helpers to keep the
pipeline's VAR plane physically meaningful across stages.

Design goals
------------
- After each stage, VAR must be finite, non-negative, and same shape as SCI.
- Operations must propagate variance with standard error-propagation rules.
- Avoid introducing FITS keywords longer than 8 characters (handled elsewhere).

Notes
-----
We intentionally keep this module NumPy-only (plus optional Astropy headers) so
it is easy to unit-test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VarQC:
    """Lightweight QC bundle for sanity checks."""

    median_reduced_chi2: float | None
    n_used: int


def _as_float(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def ensure_var_contract(
    sci: np.ndarray,
    var: np.ndarray | None,
    *,
    mask: np.ndarray | None = None,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Return a VAR array that satisfies the basic contract.

    - shape == SCI
    - finite
    - >= 0

    If ``var`` is None, returns an array filled with ``fill_value``.
    """

    sci = np.asarray(sci)
    if var is None:
        out = np.full(sci.shape, float(fill_value), dtype=np.float32)
        return out

    v = _as_float(var)
    if v.shape != sci.shape:
        raise ValueError(f"VAR shape {v.shape} != SCI shape {sci.shape}")

    # Sanitize: non-finite or negative variance is invalid.
    bad = ~np.isfinite(v) | (v < 0)
    if np.any(bad):
        v = v.copy()
        v[bad] = float(fill_value)
        # If a mask is supplied, upstream callers may choose to mark these
        # pixels as NO_COVERAGE; we do not set mask bits here to avoid policy.

    return v


def estimate_ccd_variance(
    sci: np.ndarray,
    hdr: Any,
    *,
    gain_override: float | None = None,
    rdnoise_override: float | None = None,
    unit_model: str | None = None,
    instrument_hint: str | None = None,
    require_gain: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate variance from a simple CCD model.

    Model (electrons):  var_e = max(SCI_e, 0) + RN^2

    The function auto-detects whether SCI is in ADU or ELECTRON using the
    pipeline's unit model (SCORPUM) unless ``unit_model`` is provided.

    Returns
    -------
    var : np.ndarray
        Variance in the same unit-system as SCI squared (ADU^2 or e-^2).
    meta : dict
        Contains resolved gain/read-noise and a short provenance string.
    """

    from scorpio_pipe.noise_model import estimate_variance_auto

    var, params, model = estimate_variance_auto(
        sci,
        hdr,
        gain_override=gain_override,
        rdnoise_override=rdnoise_override,
        unit_model=unit_model,
        instrument_hint=instrument_hint,
        require_gain=require_gain,
    )
    meta = {
        "unit_model": str(model),
        "gain_e_per_adu": float(params.gain_e_per_adu),
        "rdnoise_e": float(params.rdnoise_e),
        "source": str(params.source),
    }
    return ensure_var_contract(np.asarray(sci), var), meta


def propagate_add(var_a: np.ndarray, var_b: np.ndarray) -> np.ndarray:
    """Variance for A + B (independent)."""
    a = _as_float(var_a)
    b = _as_float(var_b)
    if a.shape != b.shape:
        raise ValueError(f"VAR shapes differ: {a.shape} vs {b.shape}")
    return ensure_var_contract(a, a + b)


def propagate_sub(var_a: np.ndarray, var_b: np.ndarray) -> np.ndarray:
    """Variance for A - B (independent)."""
    return propagate_add(var_a, var_b)


def propagate_multiply(var: np.ndarray, factor: np.ndarray | float) -> np.ndarray:
    """Variance for SCI * factor (deterministic factor)."""
    v = _as_float(var)
    f = np.asarray(factor, dtype=np.float32)
    if f.ndim and f.shape != v.shape:
        raise ValueError(f"factor shape {f.shape} != VAR shape {v.shape}")
    return ensure_var_contract(v, v * (f ** 2))


def propagate_divide(
    sci: np.ndarray,
    var_sci: np.ndarray,
    divisor: np.ndarray,
    *,
    var_divisor: np.ndarray | None = None,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Variance for SCI / divisor.

    If ``var_divisor`` is provided, includes its contribution:

        Var(S/D) = Var(S)/D^2 + S^2 * Var(D) / D^4

    Pixels where |divisor| <= eps are treated as invalid: variance is set to 0.
    Caller should mark mask bits (e.g. BADPIX/NO_COVERAGE) as appropriate.
    """

    s = _as_float(sci)
    v = ensure_var_contract(s, var_sci)
    d = np.asarray(divisor, dtype=np.float32)
    if d.shape != s.shape:
        raise ValueError(f"divisor shape {d.shape} != SCI shape {s.shape}")

    safe = np.abs(d) > float(eps)
    out = np.zeros(s.shape, dtype=np.float32)
    out[safe] = v[safe] / (d[safe] ** 2)

    if var_divisor is not None:
        vd = ensure_var_contract(s, var_divisor)
        out[safe] = out[safe] + (s[safe] ** 2) * vd[safe] / (d[safe] ** 4)

    return ensure_var_contract(s, out)


def propagate_weighted_mean(
    vars: list[np.ndarray],
    weights: list[np.ndarray],
    *,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Variance of a weighted mean across frames.

    For output:
        y = sum_i (w_i * x_i) / sum_i w_i

    Assuming independent inputs:
        Var(y) = sum_i (w_i^2 * Var(x_i)) / (sum_i w_i)^2

    All arrays must have the same shape.
    """

    if not vars or not weights or len(vars) != len(weights):
        raise ValueError("vars and weights must be non-empty and same length")

    ref_shape = np.asarray(vars[0]).shape
    num = np.zeros(ref_shape, dtype=np.float64)
    den = np.zeros(ref_shape, dtype=np.float64)

    for v, w in zip(vars, weights):
        vv = _as_float(v)
        ww = np.asarray(w, dtype=np.float32)
        if vv.shape != ref_shape or ww.shape != ref_shape:
            raise ValueError("All vars/weights must share the same shape")
        num += (ww.astype(np.float64) ** 2) * vv.astype(np.float64)
        den += ww.astype(np.float64)

    safe = den > float(eps)
    out = np.zeros(ref_shape, dtype=np.float32)
    out[safe] = (num[safe] / (den[safe] ** 2)).astype(np.float32)

    return ensure_var_contract(np.zeros(ref_shape, dtype=np.float32), out)


def median_reduced_chi2(
    resid: np.ndarray,
    var: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    fatal_mask: int | None = None,
) -> VarQC:
    """Compute a robust reduced-chi^2 sanity metric.

    This is intended as a *sanity check* on background-like regions.

    Parameters
    ----------
    resid : np.ndarray
        Residual image (SCI - model) in the same units as SCI.
    var : np.ndarray
        Variance plane in the same units^2.
    mask : np.ndarray, optional
        uint16 mask plane.
    fatal_mask : int, optional
        Bits considered invalid. If None and mask is given, uses a conservative
        default (NO_COVERAGE|BADPIX|COSMIC|SATURATED|USER|REJECTED).

    Returns
    -------
    VarQC
        median_reduced_chi2 is None if not enough valid pixels.
    """

    r = _as_float(resid)
    v = ensure_var_contract(r, var)

    good = np.isfinite(r) & np.isfinite(v) & (v > 0)
    if mask is not None:
        from scorpio_pipe.io.mef import DEFAULT_FATAL_BITS

        fm = int(DEFAULT_FATAL_BITS if fatal_mask is None else fatal_mask)
        m = np.asarray(mask, dtype=np.uint16)
        good &= (m & np.uint16(fm)) == 0

    n = int(np.count_nonzero(good))
    if n < 10:
        return VarQC(median_reduced_chi2=None, n_used=n)

    chi2 = (r[good] ** 2) / v[good]
    # Robust: median instead of mean (less sensitive to unmodelled objects)
    try:
        med = float(np.median(chi2))
    except Exception:
        med = None

    return VarQC(median_reduced_chi2=med, n_used=n)
