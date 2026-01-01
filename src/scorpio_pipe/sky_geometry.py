"""Sky ROI / geometry helpers.

This module is intentionally GUI-independent.

It provides a single “source of truth” for defining:

- where the *object* is along the slit (``mask_obj_y``),
- where the *sky* windows are along the slit (``mask_sky_y`` + explicit windows),
- provenance of those windows (user ROI vs auto detector),
- diagnostics and flags.

The primary entry point is :func:`compute_sky_geometry`.

Notes on conventions
--------------------
All Y ranges are **inclusive** endpoints, matching the pipeline GUI ROI dialog.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

# SciPy is a core dependency (v5.40.2+). It provides robust, well-tested
# morphology operations used by ROI guardrails.
from scipy import ndimage as _ndi


@dataclass(frozen=True)
class ROISelection:
    """User ROI description.

    All tuples are inclusive y0..y1. Any field may be None (missing).
    """

    obj_band: tuple[int, int] | None
    sky_band_low: tuple[int, int] | None
    sky_band_high: tuple[int, int] | None

    roi_version: str | None = None
    roi_hash: str | None = None


@dataclass(frozen=True)
class SkyGeometry:
    """Computed slit geometry masks and diagnostics."""

    mask_obj_y: np.ndarray  # (ny,) bool
    mask_sky_y: np.ndarray  # (ny,) bool
    sky_windows: list[tuple[int, int]]  # inclusive windows
    object_spans: list[tuple[int, int]]  # inclusive spans
    roi_used: dict[str, Any]
    metrics: dict[str, Any]


def _clip_pair(pair: Any, ny: int) -> tuple[int, int] | None:
    if pair is None:
        return None
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        try:
            y0, y1 = int(pair[0]), int(pair[1])
        except Exception:
            return None
        if y0 > y1:
            y0, y1 = y1, y0
        y0 = max(0, min(ny - 1, y0))
        y1 = max(0, min(ny - 1, y1))
        return (y0, y1)
    return None


def roi_from_cfg(cfg: dict[str, Any]) -> ROISelection | None:
    """Extract ROI from cfg.

    The current GUI stores ROI under ``cfg['sky']['roi']`` with keys:
    ``obj_y0,obj_y1, sky_bot_y0,sky_bot_y1, sky_top_y0,sky_top_y1``.

    This function also understands legacy keys ``sky_top``/``sky_bot`` as
    2-element lists.
    """

    # New GUI schema: cfg['sky']['roi']
    roi: dict[str, Any] = {}
    if isinstance(cfg.get("sky"), dict) and isinstance(cfg["sky"].get("roi"), dict):
        roi = cfg["sky"]["roi"]
    # Legacy schema (tests/older configs): cfg['roi']
    elif isinstance(cfg.get("roi"), dict):
        roi = cfg["roi"]

    if not roi:
        return None

    obj = None
    # preferred: obj_y0..obj_y1
    if roi.get("obj_y0") is not None and roi.get("obj_y1") is not None:
        obj = (int(roi.get("obj_y0")), int(roi.get("obj_y1")))
    # legacy: obj_y1..obj_y2
    elif roi.get("obj_y1") is not None and roi.get("obj_y2") is not None:
        obj = (int(roi.get("obj_y1")), int(roi.get("obj_y2")))

    top = None
    bot = None

    # preferred schema: sky_top_y0..y1 and sky_bot_y0..y1
    if roi.get("sky_top_y0") is not None and roi.get("sky_top_y1") is not None:
        top = (int(roi.get("sky_top_y0")), int(roi.get("sky_top_y1")))
    if roi.get("sky_bot_y0") is not None and roi.get("sky_bot_y1") is not None:
        bot = (int(roi.get("sky_bot_y0")), int(roi.get("sky_bot_y1")))

    # legacy schema (tests): sky_y1..y2 and sky2_y1..y2
    if bot is None and roi.get("sky_y1") is not None and roi.get("sky_y2") is not None:
        bot = (int(roi.get("sky_y1")), int(roi.get("sky_y2")))
    if top is None and roi.get("sky2_y1") is not None and roi.get("sky2_y2") is not None:
        top = (int(roi.get("sky2_y1")), int(roi.get("sky2_y2")))

    # legacy alternate keys: sky_top / sky_bot as 2-element lists
    if top is None:
        t = roi.get("sky_top")
        if isinstance(t, (list, tuple)) and len(t) == 2:
            top = (int(t[0]), int(t[1]))
    if bot is None:
        b = roi.get("sky_bot")
        if isinstance(b, (list, tuple)) and len(b) == 2:
            bot = (int(b[0]), int(b[1]))
    return ROISelection(
        obj_band=obj,
        sky_band_low=bot,
        sky_band_high=top,
        roi_version=str(roi.get("roi_version")) if roi.get("roi_version") is not None else None,
        roi_hash=str(roi.get("roi_hash")) if roi.get("roi_hash") is not None else None,
    )


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    r = int(max(0, radius))
    if r <= 0:
        return mask
    # 1D binary dilation with a flat structuring element.
    # Using SciPy avoids subtle edge-case bugs and is faster for large ny.
    structure = np.ones((2 * r + 1,), dtype=bool)
    return _ndi.binary_dilation(mask, structure=structure).astype(bool)


def _mask_from_windows(ny: int, windows: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros(ny, dtype=bool)
    for y0, y1 in windows:
        y0 = max(0, min(ny - 1, int(y0)))
        y1 = max(0, min(ny - 1, int(y1)))
        if y0 > y1:
            y0, y1 = y1, y0
        m[y0 : y1 + 1] = True
    return m


def _windows_from_mask(mask: np.ndarray, *, min_width_px: int) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.size)
    out: list[tuple[int, int]] = []
    if n == 0:
        return out
    i = 0
    mw = int(max(1, min_width_px))
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        if (j - i + 1) >= mw:
            out.append((int(i), int(j)))
        i = j + 1
    return out


def _robust_median_and_mad(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med, 1.4826 * mad


def compute_sky_geometry(
    D: np.ndarray,
    V: np.ndarray | None,
    M: np.ndarray | None,
    *,
    roi: ROISelection | None,
    roi_policy: Literal["user_only", "prefer_user", "auto"] = "prefer_user",
    fatal_bits: int = 0,
    edge_margin_px: int = 16,
    profile_x_percentile: float = 50.0,
    thresh_sigma: float = 3.0,
    dilation_px: int = 3,
    min_obj_width_px: int = 6,
    min_sky_width_px: int = 12,
    contamination_sigma: float = 3.0,
    contamination_frac_warn: float = 0.15,
) -> SkyGeometry:
    """Compute slit geometry masks and windows.

    Parameters
    ----------
    D
        Science image (ny, nx).
    V
        Variance (ny, nx) or None.
    M
        Mask (ny, nx) uint16 or None.
    roi
        User ROI selection (may be None).
    roi_policy
        - ``user_only``: require a valid ROI (otherwise roi_valid=False and empty windows).
        - ``prefer_user``: use user ROI when valid, else fall back to auto.
        - ``auto``: ignore ROI and always auto-detect.
    fatal_bits
        Mask bits that exclude a pixel from geometry estimation.
    """

    D = np.asarray(D, dtype=float)
    ny, nx = D.shape
    if ny <= 0 or nx <= 0:
        raise ValueError("D must be a non-empty 2D array")

    if V is not None:
        V = np.asarray(V, dtype=float)
    if M is not None:
        M = np.asarray(M, dtype=np.uint16)
    else:
        M = np.zeros((ny, nx), dtype=np.uint16)

    x_edge = int(max(0, edge_margin_px))
    # Clamp edge so that the valid X window is not empty (important for small nx in synthetic tests).
    x_edge = min(x_edge, max(0, (nx - 1) // 2))

    # 'core' is a Y-mask used for robust profile statistics. Do not apply the X-edge margin to Y.
    core = np.ones(ny, dtype=bool)


    flags: list[dict[str, Any]] = []

    # --- choose calm X window based on robust medians across y ---
    good = np.isfinite(D) & ((M & np.uint16(fatal_bits)) == 0)
    # Use NaNs to exclude masked pixels for robust reductions.
    Dg = np.where(good, D, np.nan)
    try:
        S_med = np.nanmedian(Dg, axis=0)
    except Exception:
        S_med = np.full(nx, np.nan, dtype=float)

    valid_x = np.arange(nx, dtype=int)
    if (nx - 2 * x_edge) >= 4:
        valid_x = np.arange(x_edge, nx - x_edge, dtype=int)

    if np.isfinite(S_med[valid_x]).sum() >= 4:
        p = float(np.nanpercentile(S_med[valid_x], float(profile_x_percentile)))
        X_win = valid_x[np.where(S_med[valid_x] <= p)[0]]
    else:
        X_win = valid_x

    if X_win.size < 4:
        X_win = valid_x
    if X_win.size < 4:
        X_win = np.arange(nx, dtype=int)

    # --- profile along y ---
    try:
        P = np.nanmedian(Dg[:, X_win], axis=1)
    except Exception:
        P = np.full(ny, np.nan, dtype=float)

    baseline, sigma = _robust_median_and_mad(P[core])
    if not np.isfinite(sigma) or sigma <= 0:
        # fall back to a small positive sigma to avoid division-by-zero
        sigma = float(np.nanstd(P[core])) if np.isfinite(np.nanstd(P[core])) else 1.0
        sigma = max(float(sigma), 1e-6)

    # --- ROI validation (if present) ---
    roi_valid = False
    roi_source = "none"
    obj_band: tuple[int, int] | None = None
    sky_low: tuple[int, int] | None = None
    sky_high: tuple[int, int] | None = None

    def _validate_user_roi(r: ROISelection) -> tuple[bool, str]:
        """Validate and normalize a user ROI.

        The ROI is treated as authoritative as long as the bands exist and do not overlap.
        Width/edge heuristics are reported as WARNs (not hard failures) because
        narrow ROIs are common in synthetic QA and can be useful in real reductions.
        """
        nonlocal obj_band, sky_low, sky_high
        # clip
        obj = _clip_pair(r.obj_band, ny)
        lo = _clip_pair(r.sky_band_low, ny)
        hi = _clip_pair(r.sky_band_high, ny)
        if obj is None or lo is None or hi is None:
            return False, "missing bands"
        # sort pairs (defensive; _clip_pair already does it)
        if obj[0] > obj[1] or lo[0] > lo[1] or hi[0] > hi[1]:
            return False, "invalid endpoints"

        # non-overlap is a hard requirement
        def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
            return not (a[1] < b[0] or b[1] < a[0])

        if _overlap(obj, lo) or _overlap(obj, hi) or _overlap(lo, hi):
            return False, "bands overlap"

        # heuristic warnings (do not invalidate the ROI)
        obj_w = int(obj[1] - obj[0] + 1)
        if obj_w < int(min_obj_width_px):
            flags.append({
                "code": "ROI_OBJECT_THIN",
                "severity": "WARN",
                "message": "Object band is thinner than the recommended minimum",
                "value": obj_w,
                "recommended_min": int(min_obj_width_px),
            })

        sky_w = int((lo[1] - lo[0] + 1) + (hi[1] - hi[0] + 1))
        if sky_w < int(min_sky_width_px):
            flags.append({
                "code": "ROI_SKY_THIN",
                "severity": "WARN",
                "message": "Sky bands are thinner than the recommended minimum",
                "value": sky_w,
                "recommended_min": int(min_sky_width_px),
            })

        for name, (y0, y1) in ("obj", obj), ("sky_low", lo), ("sky_high", hi):
            if y0 == 0 or y1 == (ny - 1):
                flags.append({
                    "code": "ROI_TOUCH_EDGE",
                    "severity": "WARN",
                    "message": f"{name} band touches the detector edge",
                    "band": name,
                })

        obj_band, sky_low, sky_high = obj, lo, hi
        return True, "ok"

    if roi is not None and roi_policy in {"user_only", "prefer_user"}:
        ok, why = _validate_user_roi(roi)
        if ok:
            roi_valid = True
            roi_source = "user"
        else:
            flags.append(
                {
                    "code": "ROI_INVALID",
                    # Per P1-C STOP-THE-LINE, ROI_INVALID is always an ERROR.
                    "severity": "ERROR",
                    "message": "ROI is present but invalid" if roi_policy == "user_only" else "ROI is present but invalid; falling back to auto ROI",
                    "hint": why,
                }
            )

    if roi_policy == "auto":
        roi_valid = False
        roi_source = "auto"

    # --- build masks/windows ---
    mask_obj_y = np.zeros(ny, dtype=bool)
    mask_sky_y = np.zeros(ny, dtype=bool)
    sky_windows: list[tuple[int, int]] = []
    object_spans: list[tuple[int, int]] = []

    if roi_valid and obj_band and sky_low and sky_high:
        mask_obj_y[obj_band[0] : obj_band[1] + 1] = True
        mask_obj_y = _dilate(mask_obj_y, int(dilation_px))

        # preserve two windows (low/high) and also their union mask
        sky_windows = [
            (int(sky_low[0]), int(sky_low[1])),
            (int(sky_high[0]), int(sky_high[1])),
        ]
        mask_sky_y = _mask_from_windows(ny, sky_windows)
        mask_sky_y &= core
        object_spans = [(int(obj_band[0]), int(obj_band[1]))]
    else:
        if roi_policy == "user_only":
            # empty, but still return diagnostics
            sky_windows = []
            object_spans = []
        else:
            # auto detector: object = profile excursion above baseline
            if np.isfinite(P).sum() < max(10, int(0.1 * ny)):
                flags.append(
                    {
                        "code": "PROFILE_TOO_SPARSE",
                        "severity": "ERROR",
                        "message": "Not enough valid pixels to build slit profile",
                    }
                )
            obj = (P > (baseline + float(thresh_sigma) * sigma)) & core & np.isfinite(P)
            obj = _dilate(obj, int(dilation_px))
            mask_obj_y = obj
            mask_sky_y = core & (~mask_obj_y)
            sky_windows = _windows_from_mask(mask_sky_y, min_width_px=int(min_sky_width_px))
            object_spans = _windows_from_mask(mask_obj_y & core, min_width_px=int(min_obj_width_px))

            if not sky_windows:
                flags.append(
                    {
                        "code": "NO_SKY_WINDOWS",
                        "severity": "ERROR",
                        "message": "Auto ROI failed to find usable sky windows",
                        "hint": "Provide explicit ROI (sky_top/sky_bot) or check that the slit is not filled",
                    }
                )

            # suspect a filled slit
            obj_frac = float(np.mean(mask_obj_y[core])) if core.any() else 0.0
            if obj_frac > 0.80:
                flags.append(
                    {
                        "code": "SLIT_FILLED_SUSPECTED",
                        "severity": "WARN",
                        "message": "A very large fraction of the slit is above the object threshold",
                        "value": obj_frac,
                    }
                )

    # --- metrics ---
    rows_sky = np.where(mask_sky_y)[0]
    rows_obj = np.where(mask_obj_y)[0]

    sky_good_frac = 0.0
    if rows_sky.size:
        sky_good_frac = float(np.mean(good[rows_sky, :]))

    sky_rows_frac = float(rows_sky.size / float(ny)) if ny > 0 else 0.0
    obj_frac = float(rows_obj.size / float(ny)) if ny > 0 else 0.0

    # contamination check for ROI-based sky bands
    sky_contam_frac = float("nan")
    if rows_sky.size and np.isfinite(P).any():
        thr = baseline + float(contamination_sigma) * sigma
        s = P[rows_sky]
        s = s[np.isfinite(s)]
        if s.size:
            sky_contam_frac = float(np.mean(s > thr))
            if np.isfinite(sky_contam_frac) and sky_contam_frac > float(contamination_frac_warn):
                flags.append(
                    {
                        "code": "SKY_BANDS_CONTAMINATED",
                        "severity": "WARN",
                        "message": "Sky windows appear contaminated by object light",
                        "value": sky_contam_frac,
                        "hint": "Move sky windows farther from the object or tighten the object ROI",
                    }
                )

    metrics: dict[str, Any] = {
        "ny": int(ny),
        "nx": int(nx),
        "edge_margin_px": int(x_edge),
        "x_edge_margin_px": int(x_edge),
        "profile_x_percentile": float(profile_x_percentile),
        "baseline": float(baseline) if np.isfinite(baseline) else None,
        "sigma": float(sigma) if np.isfinite(sigma) else None,
        "sky_rows_frac": float(sky_rows_frac),
        "sky_good_frac": float(sky_good_frac),
        "object_frac": float(obj_frac),
        "sky_contamination_metric": float(sky_contam_frac) if np.isfinite(sky_contam_frac) else None,
        "x_win_n": int(X_win.size),
        "x_win": [int(X_win.min()), int(X_win.max())] if X_win.size else None,
        "flags": flags,
    }

    roi_used: dict[str, Any] = {
        "roi_source": roi_source,
        "roi_valid": bool(roi_valid),
        "obj_band": list(obj_band) if (roi_valid and obj_band) else None,
        "sky_band_low": list(sky_low) if (roi_valid and sky_low) else None,
        "sky_band_high": list(sky_high) if (roi_valid and sky_high) else None,
        "roi_version": roi.roi_version if roi is not None else None,
        "roi_hash": roi.roi_hash if roi is not None else None,
    }

    return SkyGeometry(
        mask_obj_y=mask_obj_y,
        mask_sky_y=mask_sky_y,
        sky_windows=sky_windows,
        object_spans=object_spans,
        roi_used=roi_used,
        metrics=metrics,
    )
