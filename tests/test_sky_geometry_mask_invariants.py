"""P0 synthetic checks: the SKY mask used by SkySub must match valid windows.

These are geometry/engineering invariants, NOT certification of source-free sky.
"""
from __future__ import annotations

import numpy as np

from scorpio_pipe.sky_geometry import ROISelection, compute_sky_geometry


def _window_mask(ny: int, windows: list[tuple[int, int]]) -> np.ndarray:
    selected = np.zeros(ny, dtype=bool)
    for start, stop in windows:  # Geometry uses inclusive y endpoints.
        selected[start : stop + 1] = True
    return selected


def test_user_roi_abutting_object_cannot_cross_dilated_protection() -> None:
    raw = np.zeros((64, 80), dtype=float)
    roi = ROISelection(
        obj_band=(30, 34),
        sky_band_low=(12, 29),
        sky_band_high=(35, 52),
    )
    geom = compute_sky_geometry(
        raw, None, None, roi=roi, roi_policy="user_only", dilation_px=3
    )

    assert geom.sky_windows == []
    assert not np.any(geom.mask_sky_y)
    assert not np.any(geom.mask_sky_y & geom.mask_obj_y)
    assert geom.roi_used["roi_valid"] is False
    assert any(
        flag["code"] == "ROI_PROTECTED_MARGIN_OVERLAP"
        and flag["severity"] == "ERROR"
        for flag in geom.metrics["flags"]
    )


def test_user_roi_outside_protection_remains_exact() -> None:
    raw = np.zeros((64, 80), dtype=float)
    roi = ROISelection(
        obj_band=(30, 34),
        sky_band_low=(10, 24),
        sky_band_high=(40, 55),
    )
    geom = compute_sky_geometry(
        raw, None, None, roi=roi, roi_policy="user_only", dilation_px=3
    )

    assert geom.roi_used["roi_valid"] is True
    assert geom.sky_windows == [(10, 24), (40, 55)]
    assert np.array_equal(geom.mask_sky_y, _window_mask(64, geom.sky_windows))
    assert not np.any(geom.mask_sky_y & geom.mask_obj_y)


def test_auto_roi_drops_narrow_islands_from_actual_sky_fit_mask() -> None:
    raw = np.zeros((80, 64), dtype=float)
    raw[10:20, :] = 100.0
    raw[22:35, :] = 100.0
    geom = compute_sky_geometry(
        raw,
        None,
        None,
        roi=None,
        roi_policy="auto",
        thresh_sigma=0,
        dilation_px=0,
        min_sky_width_px=5,
        edge_margin_px=0,
    )

    # The 2-row gap y=20,21 is not a valid >=5-row sky window.
    assert geom.sky_windows == [(0, 9), (35, 79)]
    assert not np.any(geom.mask_sky_y[20:22])
    assert np.array_equal(geom.mask_sky_y, _window_mask(80, geom.sky_windows))
    assert not np.any(geom.mask_sky_y & geom.mask_obj_y)


def test_auto_roi_without_wide_windows_has_no_sky_fit_pixels() -> None:
    raw = np.zeros((40, 64), dtype=float)
    raw[1::3, :] = 100.0
    geom = compute_sky_geometry(
        raw,
        None,
        None,
        roi=None,
        roi_policy="auto",
        thresh_sigma=0,
        dilation_px=0,
        min_sky_width_px=5,
        edge_margin_px=0,
    )

    assert geom.sky_windows == []
    assert not np.any(geom.mask_sky_y)
    assert geom.metrics["sky_rows_frac"] == 0.0
    assert any(flag["code"] == "NO_SKY_WINDOWS" for flag in geom.metrics["flags"])
