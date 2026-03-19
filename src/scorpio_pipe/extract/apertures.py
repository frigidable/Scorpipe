from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from scorpio_pipe.extract.core import boxcar_extract, estimate_sky_residual
from scorpio_pipe.maskbits import BADPIX, COSMIC, NO_COVERAGE, REJECTED, SATURATED, USER

_FATAL_BITS = int(NO_COVERAGE | BADPIX | COSMIC | SATURATED | USER | REJECTED)


def choose_standard_half_width(
    base_half_width: int,
    *,
    ny: int,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the wide-box aperture policy for standards.

    The default is intentionally conservative: capture flux first, optimize S/N
    later in diagnostics.
    """

    ccfg = cfg or {}
    factor = float(ccfg.get("wide_factor", 2.0) or 2.0)
    min_hw = int(ccfg.get("min_half_width", 8) or 8)
    max_hw = int(ccfg.get("max_half_width", max(12, ny // 3)) or max(12, ny // 3))
    selected = int(np.clip(int(np.ceil(max(1, int(base_half_width)) * factor)), min_hw, max_hw))
    return {
        "base_half_width": int(base_half_width),
        "wide_factor": float(factor),
        "min_half_width": int(min_hw),
        "max_half_width": int(max_hw),
        "selected_half_width": int(selected),
        "mode": "box_wide",
    }


def _total_positive_flux(flux_1d: np.ndarray) -> float:
    arr = np.asarray(flux_1d, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.sum(np.clip(arr, 0.0, np.inf)))


def enclosed_flux_curve(
    sci: np.ndarray,
    var: np.ndarray | None,
    mask: np.ndarray | None,
    y_center: np.ndarray,
    *,
    sky_windows: list[tuple[int, int]] | None,
    half_widths: Iterable[int],
    min_good_frac: float = 0.6,
) -> dict[str, Any]:
    """Estimate how close the chosen wide box is to a local flux plateau."""

    widths = sorted({max(1, int(w)) for w in half_widths})
    if not widths:
        widths = [4, 6, 8]
    sky_resid = estimate_sky_residual(
        sci,
        var,
        mask,
        sky_windows=sky_windows,
        fatal_bits=int(_FATAL_BITS),
    )
    totals: list[float] = []
    finite_fracs: list[float] = []
    for hw in widths:
        flux, _, _, _, npix = boxcar_extract(
            sci,
            var,
            mask,
            y_center,
            ap_hw=int(hw),
            fatal_bits=int(_FATAL_BITS),
            min_good_frac=float(min_good_frac),
            sky_resid=sky_resid,
            apply_local_sky=True,
        )
        totals.append(_total_positive_flux(flux))
        denom = max(1.0, float(2 * int(hw) + 1))
        finite_fracs.append(float(np.nanmedian(np.asarray(npix, dtype=float) / denom)))
    peak = max(totals) if totals else 0.0
    capture = [float(t / peak) if peak > 0 else 0.0 for t in totals]
    return {
        "half_widths": [int(w) for w in widths],
        "totals": [float(v) for v in totals],
        "capture_fraction": [float(v) for v in capture],
        "finite_fraction": [float(v) for v in finite_fracs],
        "plateau_total": float(peak),
    }


def estimate_contamination_proxy(
    sci: np.ndarray,
    *,
    y_center: float,
    sky_windows: list[tuple[int, int]] | None,
    selected_half_width: int,
) -> dict[str, Any]:
    """Measure whether a second spatial peak competes with the standard."""

    prof = np.nanmedian(np.asarray(sci, dtype=float), axis=1)
    if sky_windows:
        vals = []
        for a, b in sky_windows:
            sl = prof[max(0, int(a)) : min(prof.size, int(b) + 1)]
            sl = sl[np.isfinite(sl)]
            if sl.size:
                vals.append(float(np.median(sl)))
        if vals:
            prof = prof - float(np.median(vals))
    prof = np.where(np.isfinite(prof), prof, 0.0)
    yy = np.arange(prof.size, dtype=float)
    main_idx = int(np.clip(int(round(float(y_center))), 0, max(0, prof.size - 1)))
    main_peak = float(max(0.0, np.max(prof))) if prof.size else 0.0
    if prof.size == 0 or main_peak <= 0:
        return {"secondary_to_primary": 0.0, "secondary_y": None, "main_y": float(y_center)}
    keep = np.ones(prof.size, dtype=bool)
    hw = max(1, int(selected_half_width))
    keep[max(0, main_idx - hw) : min(prof.size, main_idx + hw + 1)] = False
    if not np.any(keep):
        return {"secondary_to_primary": 0.0, "secondary_y": None, "main_y": float(y_center)}
    sec_idx = int(np.argmax(np.where(keep, prof, -np.inf)))
    sec_peak = float(max(0.0, prof[sec_idx])) if np.isfinite(prof[sec_idx]) else 0.0
    ratio = float(sec_peak / main_peak) if main_peak > 0 else 0.0
    return {
        "secondary_to_primary": float(ratio),
        "secondary_y": float(yy[sec_idx]) if sec_peak > 0 else None,
        "main_y": float(yy[int(np.argmax(prof))]),
    }


__all__ = [
    "choose_standard_half_width",
    "enclosed_flux_curve",
    "estimate_contamination_proxy",
]
