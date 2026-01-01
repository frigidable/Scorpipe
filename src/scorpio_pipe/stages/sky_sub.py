"""Sky subtraction in RAW geometry.

Stage 09_sky.

Inputs
------
- Cleaned RAW science frames (prefer cosmics-clean output) and their masks.
- 08_wavesol/lambda_map.fits (λ(x,y) in the *raw* detector geometry).

Outputs
-------
- 09_sky/<stem>_skymodel_raw.fits  (MEF: SCI [+VAR,+MASK])
- 09_sky/<stem>_skysub_raw.fits    (MEF: SCI [+VAR,+MASK])
- 09_sky/preview.fits (+ preview.png)
- 09_sky/qc_sky.json
- 09_sky/sky_done.json (+ done.json alias)

Notes
-----
This stage implements the P1-C "Kelson RAW" approach in a practical, *accurate*
form:

- A *ROI-aware* sky mask is computed by :func:`scorpio_pipe.sky_geometry.compute_sky_geometry`.
- The sky model is fit using a **cubic B-spline** basis in λ with optional
  linear dependence on y ("tilt"), solved as weighted least squares with robust
  sigma-clipping.
- Flexure is estimated by scanning a small δλ and refining with a parabola.

Since v5.40.2 SciPy is a core dependency: we use its well-tested spline design
matrix and sparse least-squares solvers rather than bespoke linear algebra.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from astropy.io import fits

from scipy.interpolate import BSpline
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

from scorpio_pipe.io.mef import write_sci_var_mask
from scorpio_pipe.maskbits import BADPIX, COSMIC, NO_COVERAGE, REJECTED, SATURATED, USER
from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.provenance import add_provenance
from scorpio_pipe.qc.flags import make_flag, max_severity
from scorpio_pipe.qc.lambda_map import validate_lambda_map
from scorpio_pipe.qc_thresholds import compute_thresholds
from scorpio_pipe.sky_geometry import compute_sky_geometry, roi_from_cfg
from scorpio_pipe.version import PIPELINE_VERSION
from scorpio_pipe.workspace_paths import stage_dir

from scorpio_pipe.compare_cache import build_stage_diff, snapshot_stage
from scorpio_pipe.io.quicklook import quicklook_from_mef
from scorpio_pipe.io.atomic import atomic_write_json

log = logging.getLogger(__name__)


FATAL_BITS = int(NO_COVERAGE | BADPIX | COSMIC | SATURATED | USER | REJECTED)


def _read_lambda_map(path: Path) -> Tuple[np.ndarray, fits.Header, dict[str, Any]]:
    """Load and validate lambda_map.fits."""
    diag = validate_lambda_map(path)
    with fits.open(path, memmap=False) as hdul:
        lam = np.asarray(hdul[0].data, dtype=float)
        hdr = hdul[0].header.copy()
    return lam, hdr, diag.as_dict()


def _raw_stem_from_path(p: Path) -> str:
    """Match naming rules used by Linearize: strip common suffixes."""
    s = p.stem
    for suf in ("_clean", "_cosmics", "_ff", "_flat", "_bias"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def _try_load_mask_for_clean(path: Path) -> Optional[np.ndarray]:
    """Try to load per-frame mask written by Cosmics stage.

    Cosmics stage stores masks alongside cleaned frames as ``*_mask.fits``.
    """
    cand = path.with_name(path.stem + "_mask.fits")
    if not cand.exists():
        return None
    try:
        with fits.open(cand, memmap=False) as hdul:
            m = np.asarray(hdul[0].data, dtype=np.uint16)
        return m
    except Exception:
        return None


def _read_image(path: Path) -> Tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header.copy()
    if data is None:
        raise ValueError(f"Empty image: {path}")
    return np.asarray(data, dtype=float), hdr


def _estimate_var_adu2(sci: np.ndarray, *, read_noise_e: float, gain_e_per_adu: float) -> np.ndarray:
    """Poisson + read-noise variance in ADU^2 (very basic)."""
    g = float(gain_e_per_adu) if gain_e_per_adu else 1.0
    rn = float(read_noise_e) if read_noise_e else 5.0
    # electrons: e = sci * g ; var_e = e + rn^2
    var_e = np.maximum(sci * g, 0.0) + rn * rn
    # var_adu = var_e / g^2
    return (var_e / (g * g)).astype(np.float32)


def _robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad


def _solve_tridiag(diag: np.ndarray, off: np.ndarray, rhs: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Thomas algorithm for a symmetric tridiagonal system."""
    n = int(diag.size)
    if n == 0:
        return np.zeros(0, dtype=float)
    a = diag.astype(np.float64).copy()
    b = off.astype(np.float64).copy()
    d = rhs.astype(np.float64).copy()

    # Regularize tiny diagonal entries to avoid division by zero.
    a = np.where(np.abs(a) < eps, eps, a)

    # Forward sweep
    for i in range(1, n):
        w = b[i - 1] / a[i - 1]
        a[i] -= w * b[i - 1]
        d[i] -= w * d[i - 1]
        if abs(a[i]) < eps:
            a[i] = eps

    # Back substitution
    x = np.zeros(n, dtype=np.float64)
    x[-1] = d[-1] / a[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - b[i] * x[i + 1]) / a[i]
    return x.astype(float)


def _inv2x2(M: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Inverse of a 2x2 matrix with tiny regularization."""
    a, b = float(M[0, 0]), float(M[0, 1])
    c, d = float(M[1, 0]), float(M[1, 1])
    det = a * d - b * c
    if abs(det) < eps:
        # ridge
        a += eps
        d += eps
        det = a * d - b * c
        if abs(det) < eps:
            det = eps
    inv = np.array([[d, -b], [-c, a]], dtype=float) / det
    return inv


def _solve_block_tridiag(D: np.ndarray, U: np.ndarray, B: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Solve block tridiagonal with 2x2 blocks.

    Parameters
    ----------
    D : (n,2,2) diagonal blocks
    U : (n-1,2,2) upper blocks (between i and i+1)
    B : (n,2) rhs blocks
    """
    n = int(D.shape[0])
    if n == 0:
        return np.zeros((0, 2), dtype=float)
    D = D.astype(float).copy()
    U = U.astype(float).copy() if U.size else U
    B = B.astype(float).copy()

    # Forward elimination
    for i in range(1, n):
        inv_prev = _inv2x2(D[i - 1], eps=eps)
        m = U[i - 1].T @ inv_prev  # lower block times inv(prev)
        D[i] = D[i] - m @ U[i - 1]
        B[i] = B[i] - m @ B[i - 1]

    # Back substitution
    X = np.zeros((n, 2), dtype=float)
    X[-1] = _inv2x2(D[-1], eps=eps) @ B[-1]
    for i in range(n - 2, -1, -1):
        X[i] = _inv2x2(D[i], eps=eps) @ (B[i] - U[i] @ X[i + 1])
    return X


def _prepare_lambda_nodes(lam: np.ndarray, *, knot_step_A: float) -> Tuple[float, float, int, np.ndarray]:
    lam = np.asarray(lam, dtype=float)
    good = np.isfinite(lam)
    if not good.any():
        raise ValueError("No finite wavelength values")
    lam_min = float(np.nanmin(lam[good]))
    lam_max = float(np.nanmax(lam[good]))
    step = float(max(knot_step_A, 1e-6))
    nseg = int(math.ceil((lam_max - lam_min) / step))
    n_nodes = max(2, nseg + 1)
    nodes = lam_min + step * np.arange(n_nodes, dtype=float)
    return lam_min, step, n_nodes, nodes


def _lambda_basis_indices(lam: np.ndarray, lam0: float, step: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (i, f, a0, a1) for linear B-spline basis."""
    t = (np.asarray(lam, dtype=float) - float(lam0)) / float(step)
    i = np.floor(t).astype(np.int64)
    # clamp into [0, n_nodes-2] later by caller
    f = t - i
    a1 = np.clip(f, 0.0, 1.0)
    a0 = 1.0 - a1
    return i, a1, a0, a1


def _fit_kelson_linear_spline(
    lam: np.ndarray,
    y_norm: np.ndarray,
    d: np.ndarray,
    var: np.ndarray,
    *,
    knot_step_A: float,
    y_order: int,
    clip_sigma: float,
    max_iter: int,
    min_pixels: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Fit sky model coefficients for S(λ,y) with a linear B-spline in λ."""

    lam = np.asarray(lam, dtype=float)
    y_norm = np.asarray(y_norm, dtype=float)
    d = np.asarray(d, dtype=float)
    var = np.asarray(var, dtype=float)

    ok = np.isfinite(lam) & np.isfinite(y_norm) & np.isfinite(d) & np.isfinite(var) & (var > 0)
    if int(ok.sum()) < int(min_pixels):
        raise RuntimeError(f"Too few sky pixels for fit: {int(ok.sum())} < {int(min_pixels)}")

    lam0, step, n_nodes, _nodes = _prepare_lambda_nodes(lam[ok], knot_step_A=knot_step_A)
    # basis indices / weights
    t = (lam - lam0) / step
    idx = np.floor(t).astype(np.int64)
    frac = t - idx
    idx = np.clip(idx, 0, n_nodes - 2)
    a1 = np.clip(frac, 0.0, 1.0)
    a0 = 1.0 - a1
    idx1 = idx + 1

    # initial keep mask
    keep = ok.copy()
    sig_before = None
    clipped_frac = 0.0

    # small diagonal ridge for numerical stability
    ridge = 1e-8

    def eval_model(c0: np.ndarray, c1: Optional[np.ndarray], delta_A: float = 0.0) -> np.ndarray:
        # Evaluate at shifted λ by recomputing (idx, a0, a1) on the fly.
        if delta_A != 0.0:
            tt = (lam + float(delta_A) - lam0) / step
            ii = np.floor(tt).astype(np.int64)
            ff = tt - ii
            ii = np.clip(ii, 0, n_nodes - 2)
            aa1 = np.clip(ff, 0.0, 1.0)
            aa0 = 1.0 - aa1
            ii1 = ii + 1
        else:
            ii, ii1, aa0, aa1 = idx, idx1, a0, a1

        s0 = aa0 * c0[ii] + aa1 * c0[ii1]
        if c1 is None:
            return s0
        s1 = aa0 * c1[ii] + aa1 * c1[ii1]
        return s0 + s1 * y_norm

    # Iteratively reweighted clipping (hard sigma cut)
    for it in range(int(max_iter)):
        w = np.zeros_like(var, dtype=float)
        w[keep] = 1.0 / var[keep]

        if int(y_order) <= 0:
            # scalar tridiagonal for c0
            diag = np.zeros(n_nodes, dtype=float)
            off = np.zeros(n_nodes - 1, dtype=float)
            rhs = np.zeros(n_nodes, dtype=float)

            # diag/rhs contributions for idx and idx+1
            diag += np.bincount(idx[keep], weights=w[keep] * a0[keep] * a0[keep], minlength=n_nodes)
            diag += np.bincount(idx1[keep], weights=w[keep] * a1[keep] * a1[keep], minlength=n_nodes)
            rhs += np.bincount(idx[keep], weights=w[keep] * a0[keep] * d[keep], minlength=n_nodes)
            rhs += np.bincount(idx1[keep], weights=w[keep] * a1[keep] * d[keep], minlength=n_nodes)
            off += np.bincount(idx[keep], weights=w[keep] * a0[keep] * a1[keep], minlength=n_nodes - 1)

            # ridge
            diag += ridge

            c0 = _solve_tridiag(diag, off, rhs)
            c1 = None
        else:
            # block (2x2) tridiagonal for [c0, c1]
            y = y_norm
            y2 = y * y

            A00 = np.zeros(n_nodes, dtype=float)
            A01 = np.zeros(n_nodes, dtype=float)
            A11 = np.zeros(n_nodes, dtype=float)
            b0 = np.zeros(n_nodes, dtype=float)
            b1 = np.zeros(n_nodes, dtype=float)

            # contributions for node idx
            a02 = a0 * a0
            a12 = a1 * a1
            A00 += np.bincount(idx[keep], weights=w[keep] * a02[keep], minlength=n_nodes)
            A01 += np.bincount(idx[keep], weights=w[keep] * a02[keep] * y[keep], minlength=n_nodes)
            A11 += np.bincount(idx[keep], weights=w[keep] * a02[keep] * y2[keep], minlength=n_nodes)
            b0 += np.bincount(idx[keep], weights=w[keep] * a0[keep] * d[keep], minlength=n_nodes)
            b1 += np.bincount(idx[keep], weights=w[keep] * a0[keep] * y[keep] * d[keep], minlength=n_nodes)

            # contributions for node idx+1
            A00 += np.bincount(idx1[keep], weights=w[keep] * a12[keep], minlength=n_nodes)
            A01 += np.bincount(idx1[keep], weights=w[keep] * a12[keep] * y[keep], minlength=n_nodes)
            A11 += np.bincount(idx1[keep], weights=w[keep] * a12[keep] * y2[keep], minlength=n_nodes)
            b0 += np.bincount(idx1[keep], weights=w[keep] * a1[keep] * d[keep], minlength=n_nodes)
            b1 += np.bincount(idx1[keep], weights=w[keep] * a1[keep] * y[keep] * d[keep], minlength=n_nodes)

            # off-diagonal blocks (between k and k+1, stored at k)
            O00 = np.zeros(n_nodes - 1, dtype=float)
            O01 = np.zeros(n_nodes - 1, dtype=float)
            O11 = np.zeros(n_nodes - 1, dtype=float)
            O00 += np.bincount(idx[keep], weights=w[keep] * a0[keep] * a1[keep], minlength=n_nodes - 1)
            O01 += np.bincount(idx[keep], weights=w[keep] * a0[keep] * a1[keep] * y[keep], minlength=n_nodes - 1)
            O11 += np.bincount(idx[keep], weights=w[keep] * a0[keep] * a1[keep] * y2[keep], minlength=n_nodes - 1)

            # Assemble blocks
            Dblk = np.zeros((n_nodes, 2, 2), dtype=float)
            Dblk[:, 0, 0] = A00 + ridge
            Dblk[:, 0, 1] = A01
            Dblk[:, 1, 0] = A01
            Dblk[:, 1, 1] = A11 + ridge

            Ublk = np.zeros((n_nodes - 1, 2, 2), dtype=float)
            Ublk[:, 0, 0] = O00
            Ublk[:, 0, 1] = O01
            Ublk[:, 1, 0] = O01
            Ublk[:, 1, 1] = O11

            Bblk = np.zeros((n_nodes, 2), dtype=float)
            Bblk[:, 0] = b0
            Bblk[:, 1] = b1

            X = _solve_block_tridiag(Dblk, Ublk, Bblk)
            c0 = X[:, 0]
            c1 = X[:, 1]

        pred = eval_model(c0, c1)
        r = (d - pred) / np.sqrt(var)
        r_keep = r[keep]
        sig = _robust_sigma(r_keep)
        if sig_before is None:
            sig_before = sig

        if not np.isfinite(sig) or sig <= 0:
            break

        new_keep = keep & (np.abs(r) <= float(clip_sigma))
        if new_keep.sum() == keep.sum():
            break
        keep = new_keep

    clipped_frac = float(1.0 - (keep.sum() / max(ok.sum(), 1)))
    pred = eval_model(c0, c1)
    r = (d - pred) / np.sqrt(var)
    sig_after = _robust_sigma(r[keep])

    return {
        "lam0": float(lam0),
        "step_A": float(step),
        "n_nodes": int(n_nodes),
        "y_order": int(y_order),
        "clip_sigma": float(clip_sigma),
        "max_iter": int(max_iter),
        "keep_frac": float(keep.sum() / max(ok.sum(), 1)),
        "clipped_frac": float(clipped_frac),
        "sigma_r_before": float(sig_before) if (sig_before is not None and np.isfinite(sig_before)) else None,
        "sigma_r_after": float(sig_after) if np.isfinite(sig_after) else None,
        "coeff": {
            "c0": c0.astype(np.float32),
            "c1": (c1.astype(np.float32) if c1 is not None else None),
        },
        "_cache": {
            "lam": lam,
            "y_norm": y_norm,
            "d": d,
            "var": var,
            "ok": ok,
            "keep": keep,
        },
        "eval": eval_model,
    }


def _make_bspline_knot_vector(lam_min: float, lam_max: float, *, knot_step_A: float, degree: int = 3) -> np.ndarray:
    """Construct a clamped (open) uniform knot vector.

    The spline domain is [t[degree], t[-degree-1]]. We clamp evaluations to this
    domain to avoid pathological extrapolation when flexure shifts λ slightly.
    """

    k = int(max(0, degree))
    step = float(max(knot_step_A, 1e-6))
    lo = float(lam_min)
    hi = float(lam_max)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("Invalid wavelength range for spline")

    # interior knots exclude endpoints
    interior = np.arange(lo + step, hi - step + 0.5 * step, step, dtype=float)
    t = np.concatenate(
        [np.repeat(lo, k + 1).astype(float), interior.astype(float), np.repeat(hi, k + 1).astype(float)]
    )
    return t


def _eval_bspline(t: np.ndarray, c: np.ndarray, k: int, lam: np.ndarray) -> np.ndarray:
    """Evaluate a BSpline safely on an array of λ."""

    t = np.asarray(t, dtype=float)
    c = np.asarray(c, dtype=float)
    k = int(k)
    lam = np.asarray(lam, dtype=float)

    dom_lo = float(t[k])
    dom_hi = float(t[-k - 1])
    lam_c = np.clip(lam, dom_lo, dom_hi)
    spl = BSpline(t, c, k, extrapolate=True)
    return np.asarray(spl(lam_c), dtype=float)


def _fit_kelson_bspline(
    lam: np.ndarray,
    y_norm: np.ndarray,
    d: np.ndarray,
    var: np.ndarray,
    *,
    knot_step_A: float,
    y_order: int,
    clip_sigma: float,
    max_iter: int,
    min_pixels: int,
    degree: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Kelson RAW: robust WLS fit of S(λ,y) using cubic B-splines.

    Model:
      S(λ,y) = Σ_j c0_j B_j(λ) + (y_norm) Σ_j c1_j B_j(λ)   (if y_order=1)
    """

    from scipy.sparse import hstack

    lam = np.asarray(lam, dtype=float)
    y_norm = np.asarray(y_norm, dtype=float)
    d = np.asarray(d, dtype=float)
    var = np.asarray(var, dtype=float)

    ok = np.isfinite(lam) & np.isfinite(y_norm) & np.isfinite(d) & np.isfinite(var) & (var > 0)
    if int(ok.sum()) < int(min_pixels):
        raise RuntimeError(f"Too few sky pixels for fit: {int(ok.sum())} < {int(min_pixels)}")

    lam_ok = lam[ok]
    y_ok = y_norm[ok]
    d_ok = d[ok]
    var_ok = var[ok]

    lam_min = float(np.nanmin(lam_ok))
    lam_max = float(np.nanmax(lam_ok))
    k = int(max(1, degree))
    t = _make_bspline_knot_vector(lam_min, lam_max, knot_step_A=float(knot_step_A), degree=k)
    n_basis = int(len(t) - k - 1)

    if n_basis <= 4:
        raise RuntimeError("Not enough spline basis functions; increase wavelength span or reduce knot_step_A")

    keep = np.ones_like(lam_ok, dtype=bool)
    sig_before = None

    # Robust sigma clipping loop
    for it in range(int(max_iter)):
        idx = np.where(keep)[0]
        if idx.size < int(min_pixels):
            break

        lam_k = lam_ok[idx]
        y_k = y_ok[idx]
        d_k = d_ok[idx]
        var_k = var_ok[idx]

        # Design matrix for B-spline basis
        B = BSpline.design_matrix(lam_k, t, k)  # sparse
        if int(y_order) <= 0:
            A = B
        else:
            By = B.multiply(y_k[:, None])
            A = hstack([B, By], format="csr")

        # Weighted LS via lsqr on row-scaled system
        w_sqrt = (1.0 / np.sqrt(np.maximum(var_k, 1e-20))).astype(float)
        Aw = A.multiply(w_sqrt[:, None])
        dw = (d_k * w_sqrt).astype(float)

        sol = lsqr(Aw, dw, atol=1e-10, btol=1e-10, iter_lim=2000)
        c = np.asarray(sol[0], dtype=float)

        c0 = c[:n_basis]
        c1 = c[n_basis:] if int(y_order) > 0 else None

        # Residuals for clipping on the full ok-set
        pred0 = _eval_bspline(t, c0, k, lam_ok)
        if c1 is not None:
            pred1 = _eval_bspline(t, c1, k, lam_ok)
            pred = pred0 + pred1 * y_ok
        else:
            pred = pred0

        r = (d_ok - pred) / np.sqrt(np.maximum(var_ok, 1e-20))
        sig = _robust_sigma(r[keep])
        if sig_before is None:
            sig_before = sig

        new_keep = keep & (np.abs(r) <= float(clip_sigma))
        if new_keep.sum() == keep.sum():
            keep = new_keep
            break
        keep = new_keep

    # Final coefficients using last keep-set
    idx = np.where(keep)[0]
    if idx.size < int(min_pixels):
        raise RuntimeError("Too few pixels left after sigma-clipping")

    lam_k = lam_ok[idx]
    y_k = y_ok[idx]
    d_k = d_ok[idx]
    var_k = var_ok[idx]
    B = BSpline.design_matrix(lam_k, t, k)
    if int(y_order) <= 0:
        A = B
    else:
        By = B.multiply(y_k[:, None])
        A = hstack([B, By], format="csr")
    w_sqrt = (1.0 / np.sqrt(np.maximum(var_k, 1e-20))).astype(float)
    Aw = A.multiply(w_sqrt[:, None])
    dw = (d_k * w_sqrt).astype(float)
    sol = lsqr(Aw, dw, atol=1e-10, btol=1e-10, iter_lim=2000)
    c = np.asarray(sol[0], dtype=float)
    c0 = c[:n_basis]
    c1 = c[n_basis:] if int(y_order) > 0 else None

    pred0 = _eval_bspline(t, c0, k, lam_ok)
    if c1 is not None:
        pred1 = _eval_bspline(t, c1, k, lam_ok)
        pred = pred0 + pred1 * y_ok
    else:
        pred = pred0
    r = (d_ok - pred) / np.sqrt(np.maximum(var_ok, 1e-20))
    sig_after = _robust_sigma(r[keep])
    clipped_frac = float(1.0 - (keep.sum() / max(lam_ok.size, 1)))

    return {
        "basis": {
            "type": "bspline",
            "degree": int(k),
            "knot_step_A": float(knot_step_A),
            "n_basis": int(n_basis),
            "t": t.astype(np.float64),
            "domain_A": [float(t[k]), float(t[-k - 1])],
        },
        "y_order": int(y_order),
        "clip_sigma": float(clip_sigma),
        "max_iter": int(max_iter),
        "keep_frac": float(keep.sum() / max(lam_ok.size, 1)),
        "clipped_frac": float(clipped_frac),
        "sigma_r_before": float(sig_before) if (sig_before is not None and np.isfinite(sig_before)) else None,
        "sigma_r_after": float(sig_after) if np.isfinite(sig_after) else None,
        "coeff": {
            "c0": c0.astype(np.float32),
            "c1": (c1.astype(np.float32) if c1 is not None else None),
        },
        "_cache": {
            "lam": lam_ok.astype(np.float32),
            "y_norm": y_ok.astype(np.float32),
            "d": d_ok.astype(np.float32),
            "var": var_ok.astype(np.float32),
            "keep": keep,
        },
    }


def _estimate_flexure_delta(fit: dict[str, Any], *, delta_max_A: float, delta_step_A: float, n_samples: int, rng: np.random.Generator) -> dict[str, Any]:
    """Estimate δλ by scanning χ²(δ) on a small grid (Kelson RAW)."""

    cache = fit.get("_cache") or {}
    lam = np.asarray(cache.get("lam"), dtype=float)
    y = np.asarray(cache.get("y_norm"), dtype=float)
    d = np.asarray(cache.get("d"), dtype=float)
    var = np.asarray(cache.get("var"), dtype=float)
    keep = np.asarray(cache.get("keep"), dtype=bool)

    idx_all = np.where(keep)[0]
    if idx_all.size < 50:
        return {
            "delta_A": 0.0,
            "sigma_delta_A": None,
            "flexure_score": None,
            "grid": {"delta_A": [], "chi2": []},
            "flag": "FLEXURE_UNCERTAIN",
        }

    if idx_all.size > int(n_samples):
        idx = rng.choice(idx_all, size=int(n_samples), replace=False)
    else:
        idx = idx_all

    lam_s = lam[idx]
    y_s = y[idx]
    d_s = d[idx]
    var_s = var[idx]

    basis = fit.get("basis") or {}
    t = np.asarray(basis.get("t"), dtype=float)
    k = int(basis.get("degree") or 3)
    c0 = np.asarray(fit.get("coeff", {}).get("c0"), dtype=float)
    c1 = fit.get("coeff", {}).get("c1")
    c1 = (np.asarray(c1, dtype=float) if c1 is not None else None)

    step = float(max(delta_step_A, 1e-6))
    dmax = float(max(delta_max_A, 0.0))
    grid = np.arange(-dmax, dmax + 0.5 * step, step, dtype=float)
    chi2 = np.zeros_like(grid)

    for ii, delta in enumerate(grid):
        lam_e = lam_s + float(delta)
        s0 = _eval_bspline(t, c0, k, lam_e)
        if c1 is not None:
            s1 = _eval_bspline(t, c1, k, lam_e)
            pred = s0 + s1 * y_s
        else:
            pred = s0
        r = d_s - pred
        chi2[ii] = float(np.sum((r * r) / np.maximum(var_s, 1e-20)))

    k0 = int(np.argmin(chi2))
    delta_best = float(grid[k0])
    sigma_delta = None

    # Parabolic refinement around k0
    if 0 < k0 < (grid.size - 1):
        y1, y2, y3 = float(chi2[k0 - 1]), float(chi2[k0]), float(chi2[k0 + 1])
        denom = (y1 - 2.0 * y2 + y3)
        if abs(denom) > 0:
            delta_best = float(grid[k0]) + 0.5 * (y1 - y3) * step / denom
            # curvature ≈ d2chi2/dδ2
            d2 = denom / (step * step)
            if d2 > 0:
                sigma_delta = float(math.sqrt(2.0 / d2))

    flexure_score = None
    if sigma_delta is not None and sigma_delta > 0:
        flexure_score = float(abs(delta_best) / sigma_delta)

    return {
        "delta_A": float(delta_best),
        "sigma_delta_A": float(sigma_delta) if sigma_delta is not None else None,
        "flexure_score": float(flexure_score) if flexure_score is not None else None,
        "grid": {"delta_A": [float(x) for x in grid.tolist()], "chi2": [float(v) for v in chi2.tolist()]},
    }


def _eval_model_map(
    lambda_map: np.ndarray,
    *,
    t: np.ndarray,
    degree: int,
    c0: np.ndarray,
    c1: Optional[np.ndarray],
    y_norm_rows: np.ndarray,
    delta_A: float = 0.0,
) -> np.ndarray:
    """Evaluate S(λ,y) for the full 2D frame (B-spline basis)."""

    lam = np.asarray(lambda_map, dtype=float)
    ny, nx = lam.shape
    y_norm = np.asarray(y_norm_rows, dtype=float).reshape(ny, 1)
    lam_e = lam + float(delta_A)

    s0 = _eval_bspline(t, c0, int(degree), lam_e)
    if c1 is None:
        return s0.astype(np.float32)
    s1 = _eval_bspline(t, c1, int(degree), lam_e)
    return (s0 + s1 * y_norm).astype(np.float32)


def _eval_linear_spline_map(
    lambda_map: np.ndarray,
    *,
    lam0: float,
    step_A: float,
    a_nodes: np.ndarray,
    delta_A: float = 0.0,
) -> np.ndarray:
    """Evaluate a linear B-spline (piecewise-linear) a(λ) on the full 2D frame."""

    lam = np.asarray(lambda_map, dtype=float)
    tt = (lam + float(delta_A) - float(lam0)) / float(step_A)
    ii = np.floor(tt).astype(np.int64)
    ff = tt - ii
    ii = np.clip(ii, 0, int(a_nodes.size) - 2)
    aa1 = np.clip(ff, 0.0, 1.0)
    aa0 = 1.0 - aa1
    ii1 = ii + 1
    out = aa0 * a_nodes[ii] + aa1 * a_nodes[ii1]
    return out.astype(np.float32)


def _fit_sky_scale(
    lam: np.ndarray,
    T: np.ndarray,
    D: np.ndarray,
    var: np.ndarray,
    *,
    knot_step_A: float,
    clip_sigma: float,
    max_iter: int,
    min_pixels: int,
) -> dict[str, Any]:
    """Fit a(λ) such that a(λ) * T ≈ D, using a linear-spline a(λ)."""
    lam = np.asarray(lam, dtype=float)
    T = np.asarray(T, dtype=float)
    D = np.asarray(D, dtype=float)
    var = np.asarray(var, dtype=float)
    ok = np.isfinite(lam) & np.isfinite(T) & np.isfinite(D) & np.isfinite(var) & (var > 0) & (np.abs(T) > 0)
    if int(ok.sum()) < int(min_pixels):
        raise RuntimeError(f"Too few pixels for sky_scale_raw fit: {int(ok.sum())} < {int(min_pixels)}")

    lam0, step, n_nodes, _nodes = _prepare_lambda_nodes(lam[ok], knot_step_A=knot_step_A)
    tt = (lam - lam0) / step
    idx = np.floor(tt).astype(np.int64)
    frac = tt - idx
    idx = np.clip(idx, 0, n_nodes - 2)
    a1 = np.clip(frac, 0.0, 1.0)
    a0 = 1.0 - a1
    idx1 = idx + 1

    keep = ok.copy()
    sig_before = None
    ridge = 1e-8

    for _it in range(int(max_iter)):
        w = np.zeros_like(var, dtype=float)
        w[keep] = 1.0 / var[keep]

        diag = np.zeros(n_nodes, dtype=float)
        off = np.zeros(n_nodes - 1, dtype=float)
        rhs = np.zeros(n_nodes, dtype=float)

        # design is basis * T
        t0 = a0 * T
        t1 = a1 * T
        diag += np.bincount(idx[keep], weights=w[keep] * t0[keep] * t0[keep], minlength=n_nodes)
        diag += np.bincount(idx1[keep], weights=w[keep] * t1[keep] * t1[keep], minlength=n_nodes)
        rhs += np.bincount(idx[keep], weights=w[keep] * t0[keep] * D[keep], minlength=n_nodes)
        rhs += np.bincount(idx1[keep], weights=w[keep] * t1[keep] * D[keep], minlength=n_nodes)
        off += np.bincount(idx[keep], weights=w[keep] * t0[keep] * t1[keep], minlength=n_nodes - 1)

        diag += ridge
        a_nodes = _solve_tridiag(diag, off, rhs)

        # predict
        pred = (a0 * a_nodes[idx] + a1 * a_nodes[idx1]) * T
        r = (D - pred) / np.sqrt(var)
        sig = _robust_sigma(r[keep])
        if sig_before is None:
            sig_before = sig
        if not np.isfinite(sig) or sig <= 0:
            break
        new_keep = keep & (np.abs(r) <= float(clip_sigma))
        if new_keep.sum() == keep.sum():
            break
        keep = new_keep

    sig_after = _robust_sigma(((D - (a0 * a_nodes[idx] + a1 * a_nodes[idx1]) * T) / np.sqrt(var))[keep])
    clipped_frac = float(1.0 - (keep.sum() / max(ok.sum(), 1)))
    return {
        "lam0": float(lam0),
        "step_A": float(step),
        "n_nodes": int(n_nodes),
        "clip_sigma": float(clip_sigma),
        "max_iter": int(max_iter),
        "keep_frac": float(keep.sum() / max(ok.sum(), 1)),
        "clipped_frac": float(clipped_frac),
        "sigma_r_before": float(sig_before) if (sig_before is not None and np.isfinite(sig_before)) else None,
        "sigma_r_after": float(sig_after) if np.isfinite(sig_after) else None,
        "a_nodes": a_nodes.astype(np.float32),
    }


def _median_template(frames: Iterable[Path]) -> Tuple[np.ndarray, fits.Header]:
    imgs = []
    hdr0 = None
    for p in frames:
        try:
            im, hdr = _read_image(p)
            if hdr0 is None:
                hdr0 = hdr
            imgs.append(im.astype(np.float32))
        except Exception as e:
            log.warning("Skip sky frame %s: %s", p, e)
    if not imgs:
        raise RuntimeError("No readable sky frames for template")
    stack = np.stack(imgs, axis=0)
    return np.nanmedian(stack, axis=0).astype(np.float32), (hdr0 or fits.Header())


def run_sky_sub(cfg: dict[str, Any]) -> dict[str, Path]:
    """Entry point used by the pipeline runner."""

    work_dir = resolve_work_dir(cfg)
    out_dir = stage_dir(work_dir, "sky")
    out_dir.mkdir(parents=True, exist_ok=True)

    done_path = out_dir / "done.json"
    sky_done_path = out_dir / "sky_done.json"
    qc_path = out_dir / "qc_sky.json"

    sky_cfg = cfg.get("sky") if isinstance(cfg.get("sky"), dict) else {}
    geom_cfg = sky_cfg.get("geometry") if isinstance(sky_cfg.get("geometry"), dict) else {}
    kel_cfg = sky_cfg.get("kelson_raw") if isinstance(sky_cfg.get("kelson_raw"), dict) else {}
    scl_cfg = sky_cfg.get("sky_scale_raw") if isinstance(sky_cfg.get("sky_scale_raw"), dict) else {}

    primary = str(sky_cfg.get("primary_method") or sky_cfg.get("method") or "kelson_raw").lower()

    # RNG for subsampling
    seed = int(sky_cfg.get("rng_seed") or 12345)
    rng = np.random.default_rng(seed)

    stage_flags: list[dict[str, Any]] = []
    per_exp: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []

    outs: dict[str, Path] = {
        "done_json": done_path,
        "sky_done": sky_done_path,
        "qc_sky_json": qc_path,
        "sky_preview_fits": out_dir / "preview.fits",
        "sky_preview_png": out_dir / "preview.png",
    }

    # --- Compare A/B cache (before overwrite) ---
    compare_stamp: str | None = None
    compare_a: Path | None = None
    if done_path.exists():
        try:
            compare_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            compare_a = snapshot_stage(
                stage_key="sky",
                stage_dir=out_dir,
                label="A",
                stamp=compare_stamp,
                patterns=(
                    "done.json",
                    "sky_done.json",
                    "qc_sky.json",
                    "*_skysub_raw.fits",
                    "*_skymodel_raw.fits",
                    "*_skysub_raw.png*",
                    "*_skymodel_raw.png*",
                    "*_skysub_raw_skywin.png*",
                ),
            )
        except Exception:
            compare_stamp = None
            compare_a = None

    # --- inputs ---
    frames = cfg.get("frames") if isinstance(cfg.get("frames"), dict) else {}
    obj_frames = frames.get("obj") if isinstance(frames.get("obj"), list) else []
    sky_frames = frames.get("sky") if isinstance(frames.get("sky"), list) else []

    # Prefer cosmics-cleaned frames if present
    cos_clean_dir = stage_dir(work_dir, "cosmics") / "clean"
    clean_inputs = sorted(p for p in cos_clean_dir.glob("*_clean.fits") if p.is_file()) if cos_clean_dir.exists() else []
    raw_inputs = [Path(str(p)) for p in clean_inputs] if clean_inputs else [Path(str(p)) for p in obj_frames]

    lam_map_path = stage_dir(work_dir, "wavesol") / "lambda_map.fits"
    if not lam_map_path.exists():
        # Older layout: wavesol_dir may be elsewhere
        lam_map_path = (work_dir / "wavesol" / "lambda_map.fits")

    lam_map = None
    lam_hdr = None
    lam_diag: dict[str, Any] = {}
    try:
        lam_map, lam_hdr, lam_diag = _read_lambda_map(lam_map_path)
    except Exception as e:
        stage_flags.append(make_flag("LAMBDA_MAP_INVALID", "FATAL", str(e), path=str(lam_map_path)))

    # thresholds
    thr, thr_meta = compute_thresholds(cfg)

    preview_stack = []

    try:
        if lam_map is None:
            raise RuntimeError("lambda_map is missing or invalid")

        # optional sky template
        sky_template = None
        if primary == "sky_scale_raw":
            if sky_frames:
                sky_template, _hdr_t = _median_template([Path(str(p)) for p in sky_frames])
            else:
                stage_flags.append(
                    make_flag(
                        "NO_SKY_FRAMES",
                        "ERROR",
                        "primary_method=sky_scale_raw but cfg.frames.sky is empty",
                        hint="Provide dedicated sky frames or switch to kelson_raw",
                    )
                )
                primary = "kelson_raw"

        # process exposures
        for p in raw_inputs:
            stem = _raw_stem_from_path(p)
            try:
                sci, hdr = _read_image(p)
                mask = _try_load_mask_for_clean(p)
                if mask is None:
                    mask = np.zeros_like(sci, dtype=np.uint16)

                # basic variance estimate (can be replaced later by a more faithful model)
                read_noise_e = float(sky_cfg.get("read_noise_e", 5.0) or 5.0)
                gain = float(sky_cfg.get("gain_e_per_adu", 1.0) or 1.0)
                var = _estimate_var_adu2(sci, read_noise_e=read_noise_e, gain_e_per_adu=gain)

                # geometry
                roi = roi_from_cfg(cfg)
                geom = compute_sky_geometry(
                    sci,
                    var,
                    mask,
                    roi=roi,
                    roi_policy=str(geom_cfg.get("roi_policy") or "prefer_user"),
                    fatal_bits=FATAL_BITS,
                    edge_margin_px=int(geom_cfg.get("edge_margin_px", 16) or 16),
                    profile_x_percentile=float(geom_cfg.get("profile_x_percentile", 50.0) or 50.0),
                    thresh_sigma=float(geom_cfg.get("thresh_sigma", 3.0) or 3.0),
                    dilation_px=int(geom_cfg.get("dilation_px", 3) or 3),
                    min_obj_width_px=int(geom_cfg.get("min_obj_width_px", 6) or 6),
                    min_sky_width_px=int(geom_cfg.get("min_sky_width_px", 12) or 12),
                    contamination_sigma=float(geom_cfg.get("contamination_sigma", 3.0) or 3.0),
                    contamination_frac_warn=float(geom_cfg.get("contamination_frac_warn", 0.15) or 0.15),
                )

                # propagate geom flags
                for f in geom.metrics.get("flags") or []:
                    stage_flags.append(make_flag(str(f.get("code")), str(f.get("severity") or "INFO"), str(f.get("message") or ""), **{k: v for k, v in f.items() if k not in {"code", "severity", "message"}}))

                # build sky pixel list
                ny, nx = sci.shape
                y_idx, x_idx = np.where(geom.mask_sky_y[:, None])
                if y_idx.size == 0:
                    stage_flags.append(
                        make_flag(
                            "NO_SKY_WINDOWS",
                            "ERROR",
                            "No valid sky windows could be defined",
                            hint="Provide ROI (sky_top/sky_bot) or ensure the slit is not filled",
                            stem=stem,
                        )
                    )
                    raise RuntimeError("No sky pixels")

                # Keep only good pixels
                good_pix = (mask[y_idx, x_idx] & FATAL_BITS) == 0
                good_pix &= np.isfinite(sci[y_idx, x_idx]) & np.isfinite(var[y_idx, x_idx])
                good_pix &= np.isfinite(lam_map[y_idx, x_idx])
                y_idx = y_idx[good_pix]
                x_idx = x_idx[good_pix]
                if y_idx.size < 100:
                    raise RuntimeError("Too few good sky pixels after masking")

                lam_s = lam_map[y_idx, x_idx].astype(float)
                d_s = sci[y_idx, x_idx].astype(float)
                var_s = var[y_idx, x_idx].astype(float)
                y_rows = np.arange(ny, dtype=float)
                y_norm_rows = (2.0 * (y_rows - 0.5 * (ny - 1)) / max((ny - 1), 1.0)).astype(float)
                y_s_norm = y_norm_rows[y_idx]

                # choose method
                method_used = primary
                flex = None
                fit_meta: dict[str, Any] = {}
                if method_used == "sky_scale_raw" and sky_template is not None:
                    T_s = sky_template[y_idx, x_idx].astype(float)
                    fit = _fit_sky_scale(
                        lam_s,
                        T_s,
                        d_s,
                        var_s,
                        knot_step_A=float(scl_cfg.get("knot_step_A", 1.0) or 1.0),
                        clip_sigma=float(scl_cfg.get("robust_clip_sigma", 4.5) or 4.5),
                        max_iter=int(scl_cfg.get("max_iter", 4) or 4),
                        min_pixels=int(scl_cfg.get("min_sky_pixels", 2000) or 2000),
                    )
                    fit_meta = dict(fit)
                    a_nodes = np.asarray(fit["a_nodes"], dtype=float)
                    amp = _eval_linear_spline_map(
                        lam_map,
                        lam0=fit["lam0"],
                        step_A=fit["step_A"],
                        a_nodes=a_nodes,
                        delta_A=0.0,
                    )
                    model = (amp * sky_template.astype(np.float32)).astype(np.float32)
                else:
                    # kelson_raw
                    fit = _fit_kelson_bspline(
                        lam_s,
                        y_s_norm,
                        d_s,
                        var_s,
                        knot_step_A=float(kel_cfg.get("knot_step_A", 1.0) or 1.0),
                        y_order=int(kel_cfg.get("y_order", 0) or 0),
                        clip_sigma=float(kel_cfg.get("robust_clip_sigma", 4.5) or 4.5),
                        max_iter=int(kel_cfg.get("max_iter", 4) or 4),
                        min_pixels=int(kel_cfg.get("min_sky_pixels", 2000) or 2000),
                        degree=int(kel_cfg.get("bspline_degree", 3) or 3),
                        rng=rng,
                    )
                    # flexure scan
                    flex = _estimate_flexure_delta(
                        fit,
                        delta_max_A=float(kel_cfg.get("delta_max_A", 10.0) or 10.0),
                        delta_step_A=float(kel_cfg.get("delta_step_A", 0.2) or 0.2),
                        n_samples=int(kel_cfg.get("flexure_n_samples", 200000) or 200000),
                        rng=rng,
                    )
                    delta_A = float(flex.get("delta_A") or 0.0)
                    fit_meta = {k: v for k, v in fit.items() if k not in {"_cache", "coeff"}}
                    fit_meta["coeff"] = {
                        "basis_lambda": str((kel_cfg.get("basis_lambda") or "bspline")).lower(),
                        "knot_step_A": float(kel_cfg.get("knot_step_A", 1.0) or 1.0),
                        "bspline_degree": int(fit.get("basis", {}).get("degree") or 3),
                        "y_order": int(fit.get("y_order") or 0),
                    }
                    model = _eval_model_map(
                        lam_map,
                        t=np.asarray(fit["basis"]["t"], dtype=float),
                        degree=int(fit["basis"]["degree"]),
                        c0=np.asarray(fit["coeff"]["c0"], dtype=float),
                        c1=(np.asarray(fit["coeff"]["c1"], dtype=float) if fit["coeff"]["c1"] is not None else None),
                        y_norm_rows=y_norm_rows,
                        delta_A=delta_A,
                    )

                    # FLEXURE_UNCERTAIN heuristic
                    sigd = flex.get("sigma_delta_A")
                    score = flex.get("flexure_score")
                    if sigd is None or (isinstance(sigd, (int, float)) and float(sigd) > float(kel_cfg.get("delta_uncertain_A", 2.0) or 2.0)):
                        stage_flags.append(make_flag("FLEXURE_UNCERTAIN", "WARN", "Flexure estimate is uncertain", stem=stem, sigma_delta_A=sigd, flexure_score=score))
                    if score is not None and float(score) < float(kel_cfg.get("delta_score_warn", 2.5) or 2.5):
                        stage_flags.append(make_flag("FLEXURE_UNCERTAIN", "WARN", "Flexure significance is low", stem=stem, sigma_delta_A=sigd, flexure_score=score))

                skysub = (sci.astype(np.float32) - model.astype(np.float32)).astype(np.float32)

                # QC on sky residuals
                res = skysub[y_idx, x_idx]
                rms_sky = float(np.sqrt(np.nanmean((res[np.isfinite(res)]) ** 2))) if np.isfinite(res).any() else None
                r_snr = (res / np.sqrt(np.maximum(var_s, 1e-20)))
                sigma_r_after = _robust_sigma(r_snr)

                # object-eating metric: median residual inside object spans on calm X
                obj_eat = None
                try:
                    # calm X from geometry metrics
                    xwin = geom.metrics.get("x_win")
                    if isinstance(xwin, list) and len(xwin) == 2:
                        x0, x1 = int(xwin[0]), int(xwin[1])
                        x0 = max(0, min(nx - 1, x0))
                        x1 = max(0, min(nx - 1, x1))
                        if x1 > x0:
                            yy = geom.object_spans[0] if geom.object_spans else None
                            if yy is not None:
                                oy0, oy1 = int(yy[0]), int(yy[1])
                                oy0 = max(0, min(ny - 1, oy0))
                                oy1 = max(0, min(ny - 1, oy1))
                                sl = skysub[oy0 : oy1 + 1, x0 : x1 + 1]
                                if sl.size:
                                    med = float(np.nanmedian(sl))
                                    if sigma_r_after and np.isfinite(sigma_r_after) and sigma_r_after > 0:
                                        # normalize by sky noise in ADU using var_s
                                        med_sig = float(np.nanmedian(np.sqrt(var_s)))
                                        if med_sig > 0:
                                            obj_eat = float(med / med_sig)
                except Exception:
                    obj_eat = None

                if obj_eat is not None and obj_eat < float(kel_cfg.get("object_eating_warn", -1.0) or -1.0):
                    stage_flags.append(
                        make_flag(
                            "OBJECT_EATING_RISK",
                            "WARN",
                            "Sky subtraction may be eating the object flux",
                            stem=stem,
                            metric=obj_eat,
                        )
                    )

                # write products
                hdr0 = hdr.copy()
                hdr0["SKYMD"] = (str(method_used), "Sky model method")
                hdr0["SKYKS"] = (float(kel_cfg.get("knot_step_A", 1.0) or 1.0), "λ knot step (Angstrom)")
                if flex is not None:
                    hdr0["SKYDLAM"] = (float(flex.get("delta_A") or 0.0), "Flexure δλ applied (Angstrom)")
                hdr0 = add_provenance(hdr0, cfg, stage="sky")

                skymodel_path = out_dir / f"{stem}_skymodel_raw.fits"
                skysub_path = out_dir / f"{stem}_skysub_raw.fits"
                write_sci_var_mask(skymodel_path, model, var, mask, header=hdr0)
                write_sci_var_mask(skysub_path, skysub, var, mask, header=hdr0)

                # Quicklook PNGs (robust stretch, fatal-mask aware)
                skysub_png = out_dir / f"{stem}_skysub_raw.png"
                skymodel_png = out_dir / f"{stem}_skymodel_raw.png"
                skysub_skywin_png = out_dir / f"{stem}_skysub_raw_skywin.png"
                try:
                    quicklook_from_mef(skysub_path, skysub_png, k=4.0, method="asinh")
                    quicklook_from_mef(skymodel_path, skymodel_png, k=4.0, method="linear")
                    if hasattr(geom, "mask_sky_y"):
                        quicklook_from_mef(
                            skysub_path,
                            skysub_skywin_png,
                            k=4.0,
                            method="asinh",
                            row_mask=np.asarray(geom.mask_sky_y, dtype=bool),
                            meta={"region": "sky_windows"},
                        )
                except Exception:
                    pass

                preview_stack.append(skysub)

                # metrics for qc_report compatibility
                # pixel scale: convert δλ to pixels using median dλ/dx at center row
                flex_pix = None
                flex_A = None
                if flex is not None:
                    flex_A = float(flex.get("delta_A") or 0.0)
                    try:
                        midy = int(ny // 2)
                        dldx = np.nanmedian(np.abs(np.diff(lam_map[midy, :]))).item()
                        if np.isfinite(dldx) and dldx > 0:
                            flex_pix = float(flex_A / dldx)
                    except Exception:
                        flex_pix = None

                metrics = {
                    "rms_sky": float(rms_sky) if rms_sky is not None and np.isfinite(rms_sky) else None,
                    "robust_sigma_r_after": float(sigma_r_after) if np.isfinite(sigma_r_after) else None,
                    "robust_sigma_r_before": fit_meta.get("sigma_r_before"),
                    "clipped_frac": fit_meta.get("clipped_frac"),
                    "sky_rows_frac": geom.metrics.get("sky_rows_frac"),
                    "sky_good_frac": geom.metrics.get("sky_good_frac"),
                    "sky_contamination_metric": geom.metrics.get("sky_contamination_metric"),
                    "object_eating_metric": obj_eat,
                    "flexure_shift_A": flex_A,
                    "flexure_shift_pix": flex_pix,
                    "flexure_sigma_A": flex.get("sigma_delta_A") if flex is not None else None,
                    "flexure_score": flex.get("flexure_score") if flex is not None else None,
                }

                per_exp.append(
                    {
                        "stem": stem,
                        "input": str(p),
                        "outputs": {
                            "skymodel_raw": str(skymodel_path),
                            "skysub_raw": str(skysub_path),
                            "skymodel_raw_png": str(skymodel_png),
                            "skysub_raw_png": str(skysub_png),
                            "skysub_raw_skywin_png": str(skysub_skywin_png),
                        },
                        "method": {"name": method_used, "params": {"kelson_raw": kel_cfg, "sky_scale_raw": scl_cfg}},
                        "roi_used": geom.roi_used,
                        "geometry_metrics": geom.metrics,
                        "fit": fit_meta,
                        "flexure": flex,
                        "metrics": metrics,
                    }
                )

                qc_rows.append({"stem": stem, "metrics": metrics})

                # thresholds-based flags (coverage)
                try:
                    sgf = float(geom.metrics.get("sky_good_frac") or 0.0)
                    srf = float(geom.metrics.get("sky_rows_frac") or 0.0)

                    if srf < float(getattr(thr, "sky_rows_frac_bad", 0.10)):
                        stage_flags.append(make_flag("NO_SKY_WINDOWS", "ERROR", "Sky window rows fraction is too low", stem=stem, sky_rows_frac=srf))
                    elif srf < float(getattr(thr, "sky_rows_frac_warn", 0.20)):
                        stage_flags.append(make_flag("SKY_ROWS_LOW", "WARN", "Sky window rows fraction is low", stem=stem, sky_rows_frac=srf))

                    if sgf < float(getattr(thr, "sky_good_frac_bad", 0.60)):
                        stage_flags.append(make_flag("SKY_COVERAGE_TOO_LOW", "ERROR", "Sky good-pixel coverage is too low", stem=stem, sky_good_frac=sgf))
                    elif sgf < float(getattr(thr, "sky_good_frac_warn", 0.80)):
                        stage_flags.append(make_flag("SKY_COVERAGE_LOW", "WARN", "Sky good-pixel coverage is low", stem=stem, sky_good_frac=sgf))
                except Exception:
                    pass

            except Exception as e:
                stage_flags.append(make_flag("SKY_SUB_FAILED", "FATAL", str(e), stem=stem, input=str(p)))

        # preview
        if preview_stack:
            stack = np.nanmedian(np.stack(preview_stack, axis=0), axis=0).astype(np.float32)
            hdr0 = fits.Header()
            hdr0["BUNIT"] = (str(lam_hdr.get("BUNIT", "ADU")), "Data unit")
            hdr0["NEXP"] = (int(len(preview_stack)), "Number of exposures (median)")
            hdr0 = add_provenance(hdr0, cfg, stage="sky")
            write_sci_var_mask(outs["sky_preview_fits"], stack, None, None, header=hdr0)

            if bool(sky_cfg.get("save_png", True)):
                try:
                    import matplotlib.pyplot as plt

                    from scorpio_pipe.plot_style import mpl_style

                    with mpl_style():
                        finite = stack[np.isfinite(stack)]
                        vmin, vmax = (np.nanpercentile(finite, [5, 99]) if finite.size else (None, None))
                        fig = plt.figure(figsize=(10, 4))
                        ax = fig.add_subplot(111)
                        ax.imshow(stack, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
                        ax.set_title("Sky-subtracted (RAW) preview")
                        ax.set_xlabel("x (pix)")
                        ax.set_ylabel("y (pix)")
                        fig.tight_layout()
                        fig.savefig(outs["sky_preview_png"], dpi=150)
                        plt.close(fig)
                except Exception as e:
                    stage_flags.append(make_flag("SKY_PREVIEW_PNG_FAILED", "WARN", str(e)))

    finally:
        created_utc = datetime.now(timezone.utc).isoformat()

        qc_max = max_severity(stage_flags)
        if qc_max in {"FATAL", "ERROR"}:
            status = "fail"
        elif qc_max == "WARN":
            status = "warn"
        else:
            status = "ok"

        # Aggregate outputs (per exposure)
        skymodel_map: dict[str, str] = {}
        skysub_map: dict[str, str] = {}
        for e in per_exp:
            stem = str(e.get("stem") or "")
            outs_e = e.get("outputs") or {}
            if stem:
                if outs_e.get("skymodel_raw"):
                    skymodel_map[stem] = str(outs_e.get("skymodel_raw"))
                if outs_e.get("skysub_raw"):
                    skysub_map[stem] = str(outs_e.get("skysub_raw"))

        # Aggregate metrics
        def _med(key: str) -> float | None:
            vals: list[float] = []
            for e in per_exp:
                m = (e.get("metrics") or {}).get(key)
                if m is None:
                    continue
                try:
                    v = float(m)
                except Exception:
                    continue
                if np.isfinite(v):
                    vals.append(v)
            if not vals:
                return None
            return float(np.nanmedian(np.asarray(vals, dtype=float)))

        sky_cov_vals: list[float] = []
        for e in per_exp:
            mm = e.get("metrics") or {}
            try:
                sgf = float(mm.get("sky_good_frac"))
                srf = float(mm.get("sky_rows_frac"))
                if np.isfinite(sgf) and np.isfinite(srf):
                    sky_cov_vals.append(float(sgf * srf))
            except Exception:
                continue
        sky_coverage_frac = float(np.nanmedian(sky_cov_vals)) if sky_cov_vals else None

        # ROI summary
        roi_requested = roi_from_cfg(cfg)
        roi_used = (per_exp[0].get("roi_used") if per_exp else None)
        roi_summary = {
            "requested": roi_requested,
            "roi_source": (roi_used or {}).get("roi_source"),
            "roi_valid": (roi_used or {}).get("roi_valid"),
            "obj_band": (roi_used or {}).get("obj_band"),
            "sky_band_low": (roi_used or {}).get("sky_band_low"),
            "sky_band_high": (roi_used or {}).get("sky_band_high"),
            "sky_contamination_metric": _med("sky_contamination_metric"),
        }

        # Flexure summary (Kelson RAW only)
        flex_deltas: list[float] = []
        flex_sigmas: list[float] = []
        flex_scores: list[float] = []
        for e in per_exp:
            fx = e.get("flexure")
            if not isinstance(fx, dict):
                continue
            if fx.get("delta_A") is not None:
                try:
                    flex_deltas.append(float(fx.get("delta_A")))
                except Exception:
                    pass
            if fx.get("sigma_delta_A") is not None:
                try:
                    flex_sigmas.append(float(fx.get("sigma_delta_A")))
                except Exception:
                    pass
            if fx.get("flexure_score") is not None:
                try:
                    flex_scores.append(float(fx.get("flexure_score")))
                except Exception:
                    pass
        flexure_summary = {
            "delta_A_median": float(np.nanmedian(flex_deltas)) if flex_deltas else None,
            "sigma_delta_A_median": float(np.nanmedian(flex_sigmas)) if flex_sigmas else None,
            "flexure_score_median": float(np.nanmedian(flex_scores)) if flex_scores else None,
        }

        # User intent: did we request residual cleanup in Linearize?
        lin_cfg = cfg.get("linearize") if isinstance(cfg.get("linearize"), dict) else {}
        residual_cleanup = str((lin_cfg or {}).get("cleanup") or (lin_cfg or {}).get("residual_cleanup") or "auto").lower()

        done = {
            "stage": "sky",
            "status": status,
            "version": PIPELINE_VERSION,
            "created_utc": created_utc,
            "method": {
                "primary": str(primary),
                "secondary": None,
                "params": {
                    "geometry": geom_cfg,
                    "kelson_raw": kel_cfg,
                    "sky_scale_raw": scl_cfg,
                },
                "residual_cleanup_requested": residual_cleanup,
            },
            "roi": roi_summary,
            "inputs": {
                "science_files": [str(p) for p in raw_inputs],
                "sky_files": [str(p) for p in sky_frames],
                "lambda_map_path": str(lam_map_path),
                "lambda_map_diagnostics": lam_diag,
            },
            "flexure": flexure_summary,
            "metrics": {
                "robust_sigma_r_before_median": _med("robust_sigma_r_before"),
                "robust_sigma_r_after_median": _med("robust_sigma_r_after"),
                "clipped_frac_median": _med("clipped_frac"),
                "rms_sky_median": _med("rms_sky"),
                "sky_coverage_frac_median": sky_coverage_frac,
                "object_eating_metric_median": _med("object_eating_metric"),
            },
            "outputs": {
                "skymodel_raw_path": skymodel_map,
                "skysub_raw_path": skysub_map,
                "preview_fits": str(outs["sky_preview_fits"]) if outs.get("sky_preview_fits") and outs["sky_preview_fits"].exists() else None,
                "preview_png": str(outs["sky_preview_png"]) if outs.get("sky_preview_png") and outs["sky_preview_png"].exists() else None,
                "qc_sky_json": str(qc_path),
            },
            "qc": {
                "flags": stage_flags,
                "max_severity": qc_max,
                "thresholds": thr.to_dict() if hasattr(thr, "to_dict") else {},
                "thresholds_meta": thr_meta,
                "per_exposure": qc_rows,
            },
            "per_exposure": per_exp,
            # compatibility alias
            "per_exp": per_exp,
        }

        atomic_write_json(sky_done_path, done, indent=2, ensure_ascii=False)
        atomic_write_json(done_path, done, indent=2, ensure_ascii=False)

        qc = {
            "stage": "sky",
            "status": status,
            "version": PIPELINE_VERSION,
            "created_utc": created_utc,
            "per_exposure": qc_rows,
            "flags": stage_flags,
            "max_severity": qc_max,
            "thresholds": thr.to_dict() if hasattr(thr, "to_dict") else {},
            "thresholds_meta": thr_meta,
        }
        atomic_write_json(qc_path, qc, indent=2, ensure_ascii=False)

        # --- Compare A/B cache (after run) ---
        if compare_stamp and compare_a is not None:
            try:
                compare_b = snapshot_stage(
                    stage_key="sky",
                    stage_dir=out_dir,
                    label="B",
                    patterns=(
                        "done.json",
                        "sky_done.json",
                        "qc_sky.json",
                        "*_skysub_raw.fits",
                        "*_skymodel_raw.fits",
                        "*_skysub_raw.png*",
                        "*_skymodel_raw.png*",
                        "*_skysub_raw_skywin.png*",
                    ),
                    stamp=compare_stamp,
                )
                stems = [row.get("stem") for row in per_exp if row.get("stem")]
                build_stage_diff(
                    stage_key="sky",
                    stamp=compare_stamp,
                    run_root=work_dir,
                    a_dir=compare_a,
                    b_dir=compare_b,
                    stems=stems,
                    a_suffix="_skysub_raw.fits",
                    b_suffix="_skysub_raw.fits",
                )
            except Exception:
                pass

    return outs
