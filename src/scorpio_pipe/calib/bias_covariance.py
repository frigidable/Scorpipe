"""Exact grouped-calibration fixed-weight variance for a declared scalar bias model.

Reference oracle only. The production pipeline does not yet carry separated
MasterBias covariance through FlatField, SkySub and Linearize. Never add this
model's result to a total VAR that already contains MasterBias uncertainty.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import numpy as np


def bias_group_diagnostics(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    ids = [str(r.get("bias_group_id") or "").strip() for r in records]
    present = [name for name in ids if name]
    counts = Counter(present)
    shared = sorted(name for name, count in counts.items() if count >= 2)
    missing = sum(not bool(name) for name in ids)
    if not records or missing:
        status = "UNKNOWN"
    elif shared:
        status = "SHARED"
    elif len(records) == 1:
        status = "SINGLE"
    else:
        status = "DISTINCT"
    included = [r.get("bias_var_included") is True for r in records]
    return {
        "status": status,
        "n_inputs": len(records),
        "n_groups_known": len(counts),
        "n_missing_group": missing,
        "shared_group_ids": shared,
        "mixed_groups": len(counts) > 1,
        "all_bias_var_included": bool(records) and all(included),
        "bias_var_included_count": sum(included),
        "unique_group": present[0] if present and not missing and len(counts) == 1 else None,
        "variance_qualified": False,
    }


def fixed_weight_group_variance(
    weights: np.ndarray,
    independent_variance: np.ndarray,
    sensitivity: np.ndarray,
    group_ids: Sequence[str],
    masterbias_variance: Mapping[str, np.ndarray | float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separate independent and shared calibration variance for fixed weights.

    This assumes independent scalar MasterBias errors between groups,
    with each exposure's response to that error specified explicitly.
    Rebinning mixtures require a full operator, not this scalar oracle.
    """
    w = np.asarray(weights, dtype=np.float64)
    vi = np.asarray(independent_variance, dtype=np.float64)
    deriv = np.asarray(sensitivity, dtype=np.float64)
    if w.ndim < 2 or w.shape != vi.shape or w.shape != deriv.shape:
        raise ValueError("Expected matching (nexp, spatial...) arrays")
    if len(group_ids) != w.shape[0] or not all(str(g).strip() for g in group_ids):
        raise ValueError("All exposures require explicit MasterBias groups")
    if np.any(~np.isfinite(w) | (w < 0)):
        raise ValueError("Weights must be finite nonnegative values")
    active = w > 0
    if np.any(active & (~np.isfinite(vi) | (vi < 0))):
        raise ValueError("Invalid independent variance on active inputs")
    if np.any(active & ~np.isfinite(deriv)):
        raise ValueError("Invalid MasterBias response on active inputs")
    sw = np.sum(w, axis=0)
    alpha = np.zeros_like(w)
    np.divide(w, sw, out=alpha, where=sw > 0)
    indep = np.sum(alpha**2 * np.where(active, vi, 0.), axis=0)
    common = np.zeros_like(indep)
    for gid in sorted(set(group_ids)):
        if gid not in masterbias_variance:
            raise ValueError(f"Missing MasterBias covariance for {gid!r}")
        calibration_var = np.asarray(masterbias_variance[gid], dtype=float)
        try:
            calibration_var = np.broadcast_to(calibration_var, indep.shape)
        except ValueError as exc:
            raise ValueError(f"MasterBias group {gid} shape mismatch") from exc
        if np.any(~np.isfinite(calibration_var) | (calibration_var < 0)):
            raise ValueError(f"Invalid MasterBias variance for {gid!r}")
        group = np.array([g == gid for g in group_ids], dtype=bool)
        group = group.reshape((-1,) + (1,) * (w.ndim - 1))
        response = np.sum(
            np.where(group & active, alpha * np.where(active, deriv, 0.), 0.),
            axis=0,
        )
        common += response**2 * calibration_var
    indep = np.where(sw > 0, indep, np.nan)
    common = np.where(sw > 0, common, np.nan)
    return indep, common, indep + common
