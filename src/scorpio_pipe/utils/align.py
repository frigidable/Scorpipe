from __future__ import annotations

import numpy as np

# Keep consistent with stages that use uint16 masks.
MASK_NO_COVERAGE = np.uint16(1 << 0)


def xcorr_integer_shift_1d(ref: np.ndarray, cur: np.ndarray, max_shift: int) -> int:
    """Return integer shift to apply to `cur` to best match `ref`.

    Searches shifts in [-max_shift, +max_shift]. Uses a normalized dot product
    on the overlapping range, ignoring non-finite values.

    Sign convention: returned `shift` is the translation applied to the *input*
    array: `out[i + shift] <- in[i]`.
    """
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    if ref.size != cur.size or ref.size < 16:
        return 0
    max_shift = int(max(0, max_shift))
    if max_shift == 0:
        return 0

    def _norm(x: np.ndarray) -> np.ndarray:
        x = x.copy()
        m = np.nanmedian(x)
        x -= m
        s = np.nanstd(x)
        if not np.isfinite(s) or s <= 0:
            s = 1.0
        x /= s
        return x

    a = _norm(ref)
    b = _norm(cur)

    best_s = 0
    best_score = -np.inf
    n = a.size
    for s in range(-max_shift, max_shift + 1):
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
        score = float(np.nansum(aa[good] * bb[good]) / good.sum())
        if score > best_score:
            best_score = score
            best_s = s
    return int(best_s)


def shift_int_fill_float(
    arr: np.ndarray, shift: int, *, axis: int, fill: float
) -> tuple[np.ndarray, np.ndarray]:
    """Shift array by integer `shift` along `axis`, fill empty pixels with `fill`.

    Returns (shifted, filled_mask), where filled_mask marks pixels that were filled.
    """
    arr = np.asarray(arr)
    shift = int(shift)
    if shift == 0:
        return arr.copy(), np.zeros(arr.shape, dtype=bool)

    out = np.full(
        arr.shape, fill, dtype=np.float32 if arr.dtype.kind == "f" else arr.dtype
    )
    filled = np.ones(arr.shape, dtype=bool)
    n = arr.shape[axis]
    s = abs(shift)
    if s >= n:
        return out, filled

    sl_src = [slice(None)] * arr.ndim
    sl_dst = [slice(None)] * arr.ndim
    if shift > 0:
        sl_src[axis] = slice(0, n - s)
        sl_dst[axis] = slice(s, n)
    else:
        sl_src[axis] = slice(s, n)
        sl_dst[axis] = slice(0, n - s)
    out[tuple(sl_dst)] = arr[tuple(sl_src)]
    filled[tuple(sl_dst)] = False
    return out, filled


def shift_int_fill_mask(
    mask: np.ndarray, shift: int, *, axis: int, no_cov: np.uint16 = MASK_NO_COVERAGE
) -> np.ndarray:
    """Shift uint16 mask; introduced pixels get `no_cov`."""
    mask = np.asarray(mask, dtype=np.uint16)
    shift = int(shift)
    if shift == 0:
        return mask.copy()
    out = np.full(mask.shape, no_cov, dtype=np.uint16)
    n = mask.shape[axis]
    s = abs(shift)
    if s >= n:
        return out
    sl_src = [slice(None)] * mask.ndim
    sl_dst = [slice(None)] * mask.ndim
    if shift > 0:
        sl_src[axis] = slice(0, n - s)
        sl_dst[axis] = slice(s, n)
    else:
        sl_src[axis] = slice(s, n)
        sl_dst[axis] = slice(0, n - s)
    out[tuple(sl_dst)] = mask[tuple(sl_src)]
    return out


def take_block_yshift(
    arr: np.ndarray, y0: int, y1: int, shift: int, *, fill: float
) -> tuple[np.ndarray, np.ndarray]:
    """Take arr block [y0:y1, :] from a frame shifted by `shift` in y.

    Interprets `shift` as: out[y + shift] <- in[y].
    """
    arr = np.asarray(arr)
    ny, nx = arr.shape
    shift = int(shift)
    out = np.full(
        (y1 - y0, nx), fill, dtype=np.float32 if arr.dtype.kind == "f" else arr.dtype
    )
    filled = np.ones((y1 - y0, nx), dtype=bool)

    src0 = y0 - shift
    src1 = y1 - shift
    v0 = max(0, src0)
    v1 = min(ny, src1)
    if v1 <= v0:
        return out, filled
    dst0 = (v0 + shift) - y0
    dst1 = dst0 + (v1 - v0)
    out[dst0:dst1, :] = arr[v0:v1, :]
    filled[dst0:dst1, :] = False
    return out, filled


def take_block_yshift_mask(
    mask: np.ndarray,
    y0: int,
    y1: int,
    shift: int,
    *,
    no_cov: np.uint16 = MASK_NO_COVERAGE,
) -> np.ndarray:
    """Take uint16 mask block with y-shift; filled pixels get `no_cov`."""
    mask = np.asarray(mask, dtype=np.uint16)
    ny, nx = mask.shape
    shift = int(shift)
    out = np.full((y1 - y0, nx), no_cov, dtype=np.uint16)
    src0 = y0 - shift
    src1 = y1 - shift
    v0 = max(0, src0)
    v1 = min(ny, src1)
    if v1 <= v0:
        return out
    dst0 = (v0 + shift) - y0
    dst1 = dst0 + (v1 - v0)
    out[dst0:dst1, :] = mask[v0:v1, :]
    return out
