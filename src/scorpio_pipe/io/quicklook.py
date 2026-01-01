from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from scorpio_pipe.io.mef import DEFAULT_FATAL_BITS, read_sci_var_mask
from scorpio_pipe.io.atomic import atomic_write_json


def robust_center_scale(
    arr: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    fatal_bits: int = DEFAULT_FATAL_BITS,
) -> tuple[float, float, int]:
    """Return (median, sigma_MAD, n_good) using finite, non-fatal pixels."""

    a = np.asarray(arr, dtype=np.float64)
    if mask is None:
        good = np.isfinite(a)
    else:
        m = np.asarray(mask)
        good = np.isfinite(a) & ((m & fatal_bits) == 0)
    vals = a[good]
    n = int(vals.size)
    if n <= 0:
        return float("nan"), float("nan"), 0
    med = float(np.nanmedian(vals))
    mad = float(np.nanmedian(np.abs(vals - med)))
    sig = 1.4826 * mad
    # fallback if MAD collapses
    if not np.isfinite(sig) or sig <= 0:
        try:
            p16, p84 = np.nanpercentile(vals, [16, 84])
            sig = float(0.5 * (p84 - p16))
        except Exception:
            sig = float("nan")
    return med, sig, n


def _normalize_linear(a: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(a, dtype=np.float32)
    x = (a - vmin) / (vmax - vmin)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _normalize_asinh(a: np.ndarray, m: float, s: float, k: float) -> np.ndarray:
    # I_norm = asinh((I-m)/(k*s))/asinh(1)  -> [-1,1] when I in [m-k*s, m+k*s]
    denom = float(np.arcsinh(1.0))
    if not np.isfinite(m) or not np.isfinite(s) or s <= 0 or not np.isfinite(k) or k <= 0:
        return np.zeros_like(a, dtype=np.float32)
    u = np.arcsinh((a - m) / (k * s)) / denom
    u = np.clip(u, -1.0, 1.0)
    return (0.5 * (u + 1.0)).astype(np.float32)


def write_quicklook_png(
    arr: np.ndarray,
    out_png: Path,
    *,
    mask: np.ndarray | None = None,
    fatal_bits: int = DEFAULT_FATAL_BITS,
    k: float = 4.0,
    method: Literal["linear", "asinh"] = "linear",
    meta: dict[str, Any] | None = None,
    write_sidecar: bool = True,
) -> dict[str, Any]:
    """Write a robust-stretched PNG quicklook (and optional sidecar JSON)."""

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    m, s, n = robust_center_scale(arr, mask=mask, fatal_bits=fatal_bits)
    vmin = float(m - k * s) if np.isfinite(m) and np.isfinite(s) else float("nan")
    vmax = float(m + k * s) if np.isfinite(m) and np.isfinite(s) else float("nan")

    a = np.asarray(arr, dtype=np.float64)
    if method == "asinh":
        img = _normalize_asinh(a, m, s, float(k))
    else:
        img = _normalize_linear(a, vmin, vmax)

    # NaNs -> 0 for PNG
    img = np.where(np.isfinite(img), img, 0.0).astype(np.float32)

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt

        plt.imsave(str(out_png), img, cmap="gray", vmin=0.0, vmax=1.0, origin="lower")
    except Exception:
        # As a last resort, try pillow (if installed)
        try:
            from PIL import Image

            im8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            Image.fromarray(im8).save(str(out_png))
        except Exception:
            pass

    meta_out: dict[str, Any] = {
        "method": str(method),
        "k": float(k),
        "median": m,
        "sigma_mad": s,
        "n_good": int(n),
        "vmin": vmin,
        "vmax": vmax,
        "fatal_bits": int(fatal_bits),
    }
    if meta:
        meta_out.update(meta)

    if write_sidecar:
        try:
            atomic_write_json(out_png.with_suffix(out_png.suffix + ".json"), meta_out, indent=2)
        except Exception:
            pass

    return meta_out


def quicklook_from_mef(
    fits_path: Path,
    out_png: Path,
    *,
    k: float = 4.0,
    method: Literal["linear", "asinh"] = "linear",
    fatal_bits: int = DEFAULT_FATAL_BITS,
    row_mask: np.ndarray | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a quicklook from a MEF FITS (SCI/VAR/MASK).

    If ``row_mask`` is provided (shape ``(ny,)``), rows where it is False are
    set to NaN (so the statistics are driven only by the selected rows).
    """

    sci, _var, msk, _hdr = read_sci_var_mask(Path(fits_path))
    a = np.asarray(sci, dtype=np.float64)
    m = np.asarray(msk) if msk is not None else None

    if row_mask is not None:
        rm = np.asarray(row_mask, dtype=bool)
        if rm.ndim == 1 and a.ndim == 2 and rm.shape[0] == a.shape[0]:
            keep = rm[:, None]
            a = np.where(keep, a, np.nan)

    base_meta = {"fits": str(Path(fits_path).name)}
    if meta:
        base_meta.update(meta)
    return write_quicklook_png(a, out_png, mask=m, fatal_bits=fatal_bits, k=k, method=method, meta=base_meta)
