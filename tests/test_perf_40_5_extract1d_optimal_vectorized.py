from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("astropy")

from scipy.ndimage import gaussian_filter1d

from scorpio_pipe.stages.extract1d import _optimal_extract


def _optimal_extract_reference(
    sci: np.ndarray,
    var: np.ndarray | None,
    mask: np.ndarray | None,
    y_center: np.ndarray,
    *,
    ap_hw: int,
    fatal_bits: int,
    min_good_frac: float,
    sky_resid: np.ndarray | None,
    apply_local_sky: bool,
    smooth_lambda_sigma: float = 3.0,
    clip_negative_profile: bool = True,
    min_profile_sum: float = 1e-6,
):
    ny, nlam = sci.shape
    ap_hw = int(max(1, ap_hw))
    nwin = int(2 * ap_hw + 1)
    min_good_frac = float(np.clip(min_good_frac, 0.0, 1.0))

    have_var = var is not None and var.shape == sci.shape
    have_mask = mask is not None and mask.shape == sci.shape
    if not have_var:
        var = np.ones_like(sci, dtype=float)

    prof = np.zeros((nwin, nlam), dtype=float)
    good_prof = np.zeros((nwin, nlam), dtype=bool)

    for j in range(nlam):
        yc = float(y_center[j]) if j < len(y_center) else float(np.nanmedian(y_center))
        y0 = int(np.clip(int(round(yc - ap_hw)), 0, ny - 1))
        y1 = int(np.clip(int(round(yc + ap_hw)), 0, ny - 1))
        sl = sci[y0 : y1 + 1, j].astype(float, copy=False)
        vv = var[y0 : y1 + 1, j].astype(float, copy=False)
        good = np.isfinite(sl) & np.isfinite(vv) & (vv > 0)
        if have_mask:
            mm = mask[y0 : y1 + 1, j]
            good &= (mm & fatal_bits) == 0
        if sl.size == 0:
            continue

        p = sl.copy()
        if clip_negative_profile:
            p = np.where(np.isfinite(p), np.maximum(p, 0.0), 0.0)
        else:
            p = np.where(np.isfinite(p), p, 0.0)

        k0 = int(y0 - (int(round(yc)) - ap_hw))
        k0 = int(np.clip(k0, 0, nwin - 1))
        k1 = min(nwin, k0 + sl.size)
        prof[k0:k1, j] = p[: (k1 - k0)]
        good_prof[k0:k1, j] = good[: (k1 - k0)]

    sig = float(max(0.0, smooth_lambda_sigma))
    if sig > 0:
        for k in range(nwin):
            prof[k, :] = gaussian_filter1d(prof[k, :], sigma=sig, mode="nearest")

    P = np.zeros_like(prof, dtype=float)
    for j in range(nlam):
        g = good_prof[:, j]
        s = float(np.sum(prof[g, j])) if np.any(g) else 0.0
        if not (s > min_profile_sum):
            if np.any(g):
                P[g, j] = 1.0 / float(np.sum(g))
            else:
                P[:, j] = 1.0 / float(nwin)
        else:
            P[g, j] = prof[g, j] / s

    flux = np.full(nlam, np.nan, dtype=float)
    out_var = np.full(nlam, np.nan, dtype=float)
    out_mask = np.zeros(nlam, dtype=np.uint16)
    out_npix = np.zeros(nlam, dtype=np.int16)
    no_coverage = np.uint16(16)

    for j in range(nlam):
        yc = float(y_center[j]) if j < len(y_center) else float(np.nanmedian(y_center))
        y0 = int(np.clip(int(round(yc - ap_hw)), 0, ny - 1))
        y1 = int(np.clip(int(round(yc + ap_hw)), 0, ny - 1))
        sl = sci[y0 : y1 + 1, j].astype(float, copy=False)
        vv = var[y0 : y1 + 1, j].astype(float, copy=False)
        good = np.isfinite(sl) & np.isfinite(vv) & (vv > 0)
        if have_mask:
            mm = mask[y0 : y1 + 1, j]
            good &= (mm & fatal_bits) == 0

        n_tot = int(y1 - y0 + 1)
        n_good = int(np.sum(good))
        out_npix[j] = np.int16(n_good)
        if n_tot <= 0 or n_good <= 0:
            out_mask[j] |= no_coverage
            continue
        if n_good < int(math.ceil(min_good_frac * n_tot)):
            out_mask[j] |= no_coverage

        k0 = int(y0 - (int(round(yc)) - ap_hw))
        k0 = int(np.clip(k0, 0, nwin - 1))
        k1 = min(nwin, k0 + sl.size)
        Pj = np.zeros_like(sl, dtype=float)
        Pj[: (k1 - k0)] = P[k0:k1, j][: (k1 - k0)]

        denom = float(np.sum((Pj[good] ** 2) / vv[good])) if np.any(good) else 0.0
        if not (denom > 0):
            out_mask[j] |= no_coverage
            continue
        num = float(np.sum(Pj[good] * sl[good] / vv[good]))
        f = num / denom
        if apply_local_sky and sky_resid is not None and np.isfinite(sky_resid[j]):
            corr = float(np.sum(Pj[good] / vv[good])) / denom
            f = f - float(sky_resid[j]) * corr

        flux[j] = f
        out_var[j] = 1.0 / denom

    sky_out = np.asarray(sky_resid, dtype=float) if sky_resid is not None else np.full(nlam, np.nan, dtype=float)
    return flux, out_var, out_mask, sky_out, out_npix


def test_perf_40_5_optimal_vectorized_matches_reference() -> None:
    rng = np.random.default_rng(1234)
    ny, nlam = 33, 257
    sci = rng.normal(loc=10.0, scale=3.0, size=(ny, nlam))
    var = rng.uniform(0.8, 2.2, size=(ny, nlam))
    mask = np.zeros((ny, nlam), dtype=np.uint16)
    mask[rng.random((ny, nlam)) < 0.08] = 1
    y_center = rng.uniform(-1.2, ny - 0.1, size=nlam)
    sky = rng.normal(scale=0.05, size=nlam)

    ref = _optimal_extract_reference(
        sci,
        var,
        mask,
        y_center,
        ap_hw=4,
        fatal_bits=1,
        min_good_frac=0.55,
        sky_resid=sky,
        apply_local_sky=True,
        smooth_lambda_sigma=1.25,
        clip_negative_profile=True,
        min_profile_sum=1e-6,
    )
    got = _optimal_extract(
        sci,
        var,
        mask,
        y_center,
        ap_hw=4,
        fatal_bits=1,
        min_good_frac=0.55,
        sky_resid=sky,
        apply_local_sky=True,
        smooth_lambda_sigma=1.25,
        clip_negative_profile=True,
        min_profile_sum=1e-6,
        backend="numpy",
    )

    for exp, obs in zip(ref, got[:5]):
        assert np.allclose(exp, obs, equal_nan=True)
    assert got[5]["vectorized"] is True
    assert got[5]["backend_used"] == "numpy"
