"""Small alignment helpers (subpixel cross-correlation and safe shifts).

The long-slit pipeline needs a couple of pragmatic *subpixel* alignment steps:

- global Δλ (flexure) correction by sky lines (1D xcorr on a sky spectrum)
- y-alignment of the series before stacking (1D xcorr on spatial profiles)

We intentionally keep this module NumPy-only (no SciPy dependency) because the
project aims to stay lightweight and PyInstaller-friendly.

Implementation notes
--------------------
We estimate a subpixel shift as:
  1) brute-force normalized dot product for integer shifts in [-max,+max]
  2) parabola refinement around the best integer peak (3-point fit)

This is robust enough for typical long-slit drift values (a few pixels).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class XCorrShift:
    """Subpixel shift estimate."""

    shift_pix: float
    score: float


def _norm1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x.copy()
    m = np.nanmedian(x)
    if np.isfinite(m):
        x -= m
    s = np.nanstd(x)
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    x /= s
    return x


def xcorr_shift_subpix(ref: np.ndarray, cur: np.ndarray, max_shift: int) -> XCorrShift:
    """Estimate subpixel shift to apply to ``cur`` to best match ``ref``.

    The returned shift has the same *sign convention* as the pipeline's existing
    integer shifters:
      out[i + shift] <- in[i]

    Parameters
    ----------
    ref, cur:
        1D arrays of equal length.
    max_shift:
        Search window in pixels.
    """

    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    if ref.size != cur.size or ref.size < 16:
        return XCorrShift(0.0, float("-inf"))

    max_shift = int(max(0, max_shift))
    if max_shift == 0:
        return XCorrShift(0.0, 0.0)

    a = _norm1d(ref)
    b = _norm1d(cur)
    n = a.size

    shifts = np.arange(-max_shift, max_shift + 1, dtype=int)
    scores = np.full(shifts.size, -np.inf, dtype=float)

    for i, s in enumerate(shifts):
        if s == 0:
            aa = a
            bb = b
        elif s > 0:
            aa = a[s:]
            bb = b[: n - s]
        else:
            ss = -s
            aa = a[: n - ss]
            bb = b[ss:]
        good = np.isfinite(aa) & np.isfinite(bb)
        if good.sum() < 16:
            continue
        scores[i] = float(np.nansum(aa[good] * bb[good]) / good.sum())

    imax = int(np.nanargmax(scores))
    s0 = float(shifts[imax])
    best = float(scores[imax])

    # Parabolic refinement using neighbors.
    if 0 < imax < (scores.size - 1):
        y1 = float(scores[imax - 1])
        y2 = float(scores[imax])
        y3 = float(scores[imax + 1])
        denom = (y1 - 2.0 * y2 + y3)
        if np.isfinite(denom) and abs(denom) > 1e-12:
            delta = 0.5 * (y1 - y3) / denom
            if np.isfinite(delta):
                # clamp to a reasonable subpixel range
                delta = float(np.clip(delta, -1.0, 1.0))
                s0 += delta

    return XCorrShift(float(s0), best)


# ------------------------- 2D subpixel shifters -------------------------

def _coerce_shift_per_row(shift_pix_y: np.ndarray | float, ny: int) -> np.ndarray:
    s = np.asarray(shift_pix_y, dtype=float).reshape(-1)
    if s.size == 1:
        return np.full(ny, float(s[0]), dtype=float)
    if s.size != ny:
        raise ValueError(f"shift_pix_y must have length {ny} (got {s.size})")
    return s


def shift2d_subpix_x(arr: np.ndarray, shift_pix_y: np.ndarray | float, *, fill: float = float('nan')) -> tuple[np.ndarray, np.ndarray]:
    """Shift a 2D array along X by a *per-row* subpixel shift.

    Sign convention is consistent with :func:`xcorr_shift_subpix`:
      out[y, i + shift] <- in[y, i]

    Parameters
    ----------
    arr:
        2D array (ny, nx)
    shift_pix_y:
        Either a scalar shift applied to all rows, or an array of length ny.
    fill:
        Value used for pixels that fall outside the input domain.

    Returns
    -------
    out, filled:
        ``out`` is the shifted array. ``filled`` is a boolean mask of pixels
        where ``fill`` was applied (outside coverage).

    Notes
    -----
    This uses simple linear interpolation. It is intended for small flexure
    corrections on an already-linearized wavelength grid.
    """

    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError('arr must be 2D')
    ny, nx = a.shape
    s = _coerce_shift_per_row(shift_pix_y, ny)

    x = np.arange(nx, dtype=float)
    xin = x[None, :] - s[:, None]

    i0 = np.floor(xin).astype(np.int64)
    i1 = i0 + 1
    frac = xin - i0

    valid = (i0 >= 0) & (i1 < nx) & np.isfinite(frac)
    i0c = np.clip(i0, 0, nx - 1)
    i1c = np.clip(i1, 0, nx - 1)

    a0 = np.take_along_axis(a, i0c, axis=1)
    a1 = np.take_along_axis(a, i1c, axis=1)

    out = (1.0 - frac) * a0 + frac * a1
    filled = ~valid
    if np.any(filled):
        out = out.astype(float, copy=False)
        out[filled] = float(fill)
    return out, filled


def shift2d_subpix_x_var(var: np.ndarray, shift_pix_y: np.ndarray | float, *, fill: float = float('inf')) -> tuple[np.ndarray, np.ndarray]:
    """Shift a variance image along X with kernel propagation.

    For linear interpolation ``out = w0*a0 + w1*a1``, this uses:
      var_out = w0^2*var0 + w1^2*var1

    Returns ``(out, filled)`` similarly to :func:`shift2d_subpix_x`.
    """

    v = np.asarray(var)
    if v.ndim != 2:
        raise ValueError('var must be 2D')
    ny, nx = v.shape
    s = _coerce_shift_per_row(shift_pix_y, ny)

    x = np.arange(nx, dtype=float)
    xin = x[None, :] - s[:, None]

    i0 = np.floor(xin).astype(np.int64)
    i1 = i0 + 1
    frac = xin - i0

    valid = (i0 >= 0) & (i1 < nx) & np.isfinite(frac)
    i0c = np.clip(i0, 0, nx - 1)
    i1c = np.clip(i1, 0, nx - 1)

    v0 = np.take_along_axis(v, i0c, axis=1)
    v1 = np.take_along_axis(v, i1c, axis=1)

    w1 = frac
    w0 = 1.0 - w1
    out = (w0 * w0) * v0 + (w1 * w1) * v1

    filled = ~valid
    if np.any(filled):
        out = out.astype(float, copy=False)
        out[filled] = float(fill)
    return out, filled


def shift2d_subpix_x_mask(mask: np.ndarray, shift_pix_y: np.ndarray | float, *, no_cov: int = 1, no_coverage_bit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Shift a uint16 mask along X; filled pixels get ``no_cov`` bit.

    Compatibility: older callers may pass ``no_coverage_bit=...``.
    """

    if no_coverage_bit is not None:
        no_cov = int(no_coverage_bit)

    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError('mask must be 2D')
    ny, nx = m.shape
    s = _coerce_shift_per_row(shift_pix_y, ny)

    x = np.arange(nx, dtype=float)
    xin = x[None, :] - s[:, None]

    i0 = np.floor(xin).astype(np.int64)
    i1 = i0 + 1
    frac = xin - i0

    valid = (i0 >= 0) & (i1 < nx) & np.isfinite(frac)
    i0c = np.clip(i0, 0, nx - 1)
    i1c = np.clip(i1, 0, nx - 1)

    m0 = np.take_along_axis(m, i0c, axis=1).astype(np.uint16, copy=False)
    m1 = np.take_along_axis(m, i1c, axis=1).astype(np.uint16, copy=False)

    out = (m0 | m1).astype(np.uint16, copy=False)
    filled = ~valid
    if np.any(filled):
        out[filled] |= np.uint16(no_cov)
    return out, filled
