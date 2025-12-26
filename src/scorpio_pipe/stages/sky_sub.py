from __future__ import annotations

"""Night-sky subtraction (Kelson-like baseline implementation).

v5.0 implements a pragmatic, fast variant suitable for interactive use:

1) build a robust sky spectrum S(λ) from user-selected sky rows
   (top + bottom regions, excluding the object rows)
2) smooth S(λ) with a cubic B-spline fit with iterative sigma clipping
3) model spatial variation with a low-order polynomial in y:
      sky(y,λ) ≈ a(y) * S(λ) + b(y)
   where a(y), b(y) are fitted using sky rows only
4) subtract the model and write:
   - sky_model.fits
   - obj_sky_sub.fits

This is *not* a full re-implementation of Kelson (2003) on unrectified data,
but provides the same user-facing semantics and can be swapped out by a more
advanced method later.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import csv
import shutil
import numpy as np
from astropy.io import fits

from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.plot_style import mpl_style
from scorpio_pipe.shift_utils import xcorr_shift_subpix, shift2d_subpix_x, shift2d_subpix_x_var, shift2d_subpix_x_mask
from scorpio_pipe.io.mef import read_sci_var_mask, write_sci_var_mask, try_read_grid
from scorpio_pipe.version import PIPELINE_VERSION
from scorpio_pipe.maskbits import (
    NO_COVERAGE,
    EDGE,
    BADPIX,
    COSMIC,
    SATURATED,
    USER,
    REJECTED,
    header_cards as mask_header_cards,
)


_FATAL_BITS = np.uint16(NO_COVERAGE | BADPIX | COSMIC | SATURATED | USER | REJECTED)




def _has_linear_wcs_x(hdr: fits.Header) -> bool:
    """Return True if header contains a simple linear WCS along X."""

    crval1 = hdr.get("CRVAL1")
    cdelt1 = hdr.get("CDELT1", hdr.get("CD1_1"))
    return crval1 is not None and cdelt1 is not None


def _apply_xcorr_windows_any(
    spec: np.ndarray,
    hdr: fits.Header,
    windows_A: Any | None,
    windows_pix: Any | None,
    *,
    unit: str = "auto",
) -> np.ndarray:
    """Apply a wavelength/pixel window selection to a 1D spectrum.

    Parameters
    ----------
    spec:
        1D spectrum to be windowed.
    hdr:
        FITS header used to map Angstrom windows to pixel indices (linear WCS).
    windows_A:
        Iterable of [l0, l1] (Angstrom) pairs.
    windows_pix:
        Iterable of [x0, x1] (pixel) pairs.
    unit:
        "auto" (default): use Angstrom windows if WCS is present, else pixel windows.
        "A": force Angstrom windows.
        "pix": force pixel windows.

    Returns
    -------
    spec2:
        Same shape, but values outside the windows are set to NaN.
    """

    spec = np.asarray(spec)
    nx = int(spec.size)

    u = str(unit or "auto").strip().lower()
    if u not in ("auto", "a", "pix"):
        u = "auto"

    has_wcs = _has_linear_wcs_x(hdr)

    # Decide which window set to use.
    use_pix = False
    if u == "pix":
        use_pix = True
    elif u == "a":
        use_pix = False
    else:  # auto
        use_pix = not has_wcs

    # Pixel windows
    if use_pix:
        if windows_pix is None:
            return spec
        try:
            sel = np.zeros(nx, dtype=bool)
            for w in list(windows_pix):
                if isinstance(w, (list, tuple)) and len(w) >= 2:
                    a, b = int(w[0]), int(w[1])
                elif isinstance(w, dict):
                    a, b = int(w.get("x0")), int(w.get("x1"))
                else:
                    continue
                if a > b:
                    a, b = b, a
                a = max(0, min(nx - 1, a))
                b = max(0, min(nx - 1, b))
                sel[a : b + 1] = True
            if sel.sum() < 16:
                return spec
            out = np.asarray(spec, dtype=float).copy()
            out[~sel] = np.nan
            return out
        except Exception:
            return spec

    # Angstrom windows
    if windows_A is None or not has_wcs:
        return spec

    crval1 = hdr.get("CRVAL1")
    cdelt1 = hdr.get("CDELT1", hdr.get("CD1_1"))
    crpix1 = hdr.get("CRPIX1", 1.0)
    try:
        crval1 = float(crval1)
        cdelt1 = float(cdelt1)
        crpix1 = float(crpix1)
    except Exception:
        return spec
    if not np.isfinite(cdelt1) or cdelt1 == 0.0:
        return spec

    # λ(x) = crval1 + (x+1 - crpix1)*cdelt1
    # x(λ) = ( (λ - crval1)/cdelt1 + crpix1 ) - 1
    sel = np.zeros(nx, dtype=bool)
    try:
        for w in list(windows_A):
            if isinstance(w, (list, tuple)) and len(w) >= 2:
                l0, l1 = float(w[0]), float(w[1])
            elif isinstance(w, dict):
                l0, l1 = float(w.get("l0")), float(w.get("l1"))
            else:
                continue
            if l0 > l1:
                l0, l1 = l1, l0
            x0 = int(np.floor(((l0 - crval1) / cdelt1 + crpix1) - 1.0))
            x1 = int(np.ceil(((l1 - crval1) / cdelt1 + crpix1) - 1.0))
            x0 = max(0, min(nx - 1, x0))
            x1 = max(0, min(nx - 1, x1))
            sel[x0 : x1 + 1] = True
    except Exception:
        return spec

    if sel.sum() < 16:
        return spec

    out = np.asarray(spec, dtype=float).copy()
    out[~sel] = np.nan
    return out

def _xcorr_subpix_shift(ref: np.ndarray, cur: np.ndarray, max_shift: int) -> tuple[float, float]:
    """Return (shift_pix, score) to apply to `cur` to best match `ref`.

    Uses NumPy-only normalized dot-product xcorr with a parabola refinement
    around the best integer shift (see :mod:`scorpio_pipe.shift_utils`).
    """

    est = xcorr_shift_subpix(ref, cur, max_shift=max_shift)
    return float(est.shift_pix), float(est.score)


def _shift_subpix_fill_float(arr: np.ndarray, shift: float, *, axis: int, fill: float) -> tuple[np.ndarray, np.ndarray]:
    """Subpixel shift with linear interpolation.

    Sign convention matches the integer shifter:
      out[i + shift] <- in[i]

    Returns (shifted, filled_mask) where filled_mask marks pixels that were
    filled with `fill` (i.e. fell outside the input domain).
    """

    arr = np.asarray(arr)
    s = float(shift)
    if not np.isfinite(s) or abs(s) < 1e-9:
        return arr.copy(), np.zeros(arr.shape, dtype=bool)

    if axis not in (0, 1):
        raise ValueError("sky_sub subpixel shifter supports axis 0/1 only")

    if arr.ndim != 2:
        raise ValueError("expected 2D array")

    ny, nx = arr.shape
    if axis == 1:
        n = nx
        x = np.arange(n, dtype=float)
        xin = x - s
        i0 = np.floor(xin).astype(int)
        i1 = i0 + 1
        frac = (xin - i0).astype(float)
        valid = (i0 >= 0) & (i1 < n)
        i0c = np.clip(i0, 0, n - 1)
        i1c = np.clip(i1, 0, n - 1)
        out = np.full((ny, nx), fill, dtype=np.float32 if arr.dtype.kind == "f" else arr.dtype)
        filled = np.ones((ny, nx), dtype=bool)
        if np.any(valid):
            a0 = arr[:, i0c]
            a1 = arr[:, i1c]
            w1 = frac[None, :]
            w0 = 1.0 - w1
            out[:, valid] = (w0[:, valid] * a0[:, valid] + w1[:, valid] * a1[:, valid]).astype(out.dtype, copy=False)
            filled[:, valid] = False
        return out, filled
    else:
        n = ny
        y = np.arange(n, dtype=float)
        yin = y - s
        i0 = np.floor(yin).astype(int)
        i1 = i0 + 1
        frac = (yin - i0).astype(float)
        valid = (i0 >= 0) & (i1 < n)
        i0c = np.clip(i0, 0, n - 1)
        i1c = np.clip(i1, 0, n - 1)
        out = np.full((ny, nx), fill, dtype=np.float32 if arr.dtype.kind == "f" else arr.dtype)
        filled = np.ones((ny, nx), dtype=bool)
        if np.any(valid):
            a0 = arr[i0c, :]
            a1 = arr[i1c, :]
            w1 = frac[:, None]
            w0 = 1.0 - w1
            out[valid, :] = (w0[valid, :] * a0[valid, :] + w1[valid, :] * a1[valid, :]).astype(out.dtype, copy=False)
            filled[valid, :] = False
        return out, filled


def _shift_subpix_fill_var(var: np.ndarray, shift: float, *, axis: int, fill: float) -> tuple[np.ndarray, np.ndarray]:
    """Subpixel shift for variance, using (w0^2, w1^2) propagation."""

    var = np.asarray(var, dtype=float)
    s = float(shift)
    if not np.isfinite(s) or abs(s) < 1e-9:
        return var.copy(), np.zeros(var.shape, dtype=bool)

    if axis not in (0, 1) or var.ndim != 2:
        raise ValueError("expected 2D var array and axis 0/1")

    ny, nx = var.shape
    if axis == 1:
        n = nx
        x = np.arange(n, dtype=float)
        xin = x - s
        i0 = np.floor(xin).astype(int)
        i1 = i0 + 1
        frac = (xin - i0).astype(float)
        valid = (i0 >= 0) & (i1 < n)
        i0c = np.clip(i0, 0, n - 1)
        i1c = np.clip(i1, 0, n - 1)
        out = np.full((ny, nx), fill, dtype=np.float32)
        filled = np.ones((ny, nx), dtype=bool)
        if np.any(valid):
            v0 = var[:, i0c]
            v1 = var[:, i1c]
            w1 = frac[None, :]
            w0 = 1.0 - w1
            out[:, valid] = (w0[:, valid] ** 2) * v0[:, valid] + (w1[:, valid] ** 2) * v1[:, valid]
            filled[:, valid] = False
        return out, filled
    else:
        n = ny
        y = np.arange(n, dtype=float)
        yin = y - s
        i0 = np.floor(yin).astype(int)
        i1 = i0 + 1
        frac = (yin - i0).astype(float)
        valid = (i0 >= 0) & (i1 < n)
        i0c = np.clip(i0, 0, n - 1)
        i1c = np.clip(i1, 0, n - 1)
        out = np.full((ny, nx), fill, dtype=np.float32)
        filled = np.ones((ny, nx), dtype=bool)
        if np.any(valid):
            v0 = var[i0c, :]
            v1 = var[i1c, :]
            w1 = frac[:, None]
            w0 = 1.0 - w1
            out[valid, :] = (w0[valid, :] ** 2) * v0[valid, :] + (w1[valid, :] ** 2) * v1[valid, :]
            filled[valid, :] = False
        return out, filled


def _shift_subpix_mask(mask: np.ndarray, shift: float, *, axis: int) -> np.ndarray:
    """Subpixel shift for uint16 mask.

    Conservative rule: output mask is OR of contributing input pixels.
    Pixels outside the input domain are marked as NO_COVERAGE.
    """

    mask = np.asarray(mask, dtype=np.uint16)
    s = float(shift)
    if not np.isfinite(s) or abs(s) < 1e-9:
        return mask.copy()

    if axis not in (0, 1) or mask.ndim != 2:
        raise ValueError("expected 2D mask and axis 0/1")

    ny, nx = mask.shape
    out = np.full((ny, nx), NO_COVERAGE, dtype=np.uint16)
    if axis == 1:
        n = nx
        x = np.arange(n, dtype=float)
        xin = x - s
        i0 = np.floor(xin).astype(int)
        i1 = i0 + 1
        valid = (i0 >= 0) & (i1 < n)
        i0c = np.clip(i0, 0, n - 1)
        i1c = np.clip(i1, 0, n - 1)
        if np.any(valid):
            m0 = mask[:, i0c]
            m1 = mask[:, i1c]
            out[:, valid] = (m0[:, valid] | m1[:, valid]).astype(np.uint16, copy=False)
        return out
    else:
        n = ny
        y = np.arange(n, dtype=float)
        yin = y - s
        i0 = np.floor(yin).astype(int)
        i1 = i0 + 1
        valid = (i0 >= 0) & (i1 < n)
        i0c = np.clip(i0, 0, n - 1)
        i1c = np.clip(i1, 0, n - 1)
        if np.any(valid):
            m0 = mask[i0c, :]
            m1 = mask[i1c, :]
            out[valid, :] = (m0[valid, :] | m1[valid, :]).astype(np.uint16, copy=False)
        return out


def _shift_int_fill_float(arr: np.ndarray, shift: int, *, axis: int, fill: float) -> tuple[np.ndarray, np.ndarray]:
    """Shift array by integer `shift` along `axis`, fill empty pixels with `fill`.

    Returns (shifted, filled_mask), where filled_mask marks pixels that were filled.
    """
    arr = np.asarray(arr)
    shift = int(shift)
    if shift == 0:
        return arr.copy(), np.zeros(arr.shape, dtype=bool)

    out = np.full(arr.shape, fill, dtype=np.float32 if arr.dtype.kind == "f" else arr.dtype)
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


def _shift_int_fill_mask(mask: np.ndarray, shift: int, *, axis: int) -> np.ndarray:
    """Shift uint16 mask; introduced pixels get NO_COVERAGE."""
    mask = np.asarray(mask, dtype=np.uint16)
    shift = int(shift)
    if shift == 0:
        return mask.copy()
    out = np.full(mask.shape, NO_COVERAGE, dtype=np.uint16)
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


def _write_mef(
    path: Path,
    sci: np.ndarray,
    hdr: fits.Header,
    *,
    var: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> None:
    """Write MEF product with canonical SCI/VAR/MASK.

    Primary stores SCI for legacy; SCI extension is canonical.
    """
    hdr2 = hdr.copy()
    if mask is not None:
        try:
            for k, v, c in mask_header_cards():
                hdr2[k] = (v, c)
        except Exception:
            pass
    grid = try_read_grid(hdr2)
    write_sci_var_mask(path, sci, var=var, mask=mask, header=hdr2, grid=grid, primary_data=sci)

def _roi_from_cfg(cfg: dict[str, Any]) -> ROI:
    # Canonical v5.x location: cfg['sky']['roi'].
    # Backward compatibility:
    #  - cfg['roi'] (older smoke tests)
    #  - alternative key spellings (obj_y1/obj_y2, etc.)
    sky = (cfg.get("sky") or {}) if isinstance(cfg.get("sky"), dict) else {}
    roi = (sky.get("roi") or {}) if isinstance(sky.get("roi"), dict) else {}
    if not roi and isinstance(cfg.get("roi"), dict):
        roi = cfg.get("roi") or {}

    def _g(*keys: str, default: int | None = None) -> int:
        for k in keys:
            if k in roi and roi[k] is not None:
                return int(roi[k])
        if default is None:
            raise KeyError(f"Missing ROI key(s): {keys}")
        return int(default)

    return ROI(
        obj_y0=_g("obj_y0", "obj_ymin"),
        obj_y1=_g("obj_y1", "obj_y2", "obj_ymax"),
        sky_top_y0=_g("sky_top_y0", "sky_up_y0", "sky1_y0"),
        sky_top_y1=_g("sky_top_y1", "sky_up_y1", "sky1_y1"),
        sky_bot_y0=_g("sky_bot_y0", "sky_down_y0", "sky2_y0"),
        sky_bot_y1=_g("sky_bot_y1", "sky_down_y1", "sky2_y1"),
    )


def _wave_from_header(hdr: fits.Header, n: int) -> np.ndarray:
    crval = float(hdr.get("CRVAL1", 0.0))
    cdelt = float(hdr.get("CDELT1", 1.0))
    crpix = float(hdr.get("CRPIX1", 1.0))
    # FITS WCS: world = CRVAL + (pix-CRPIX)*CDELT, pix is 1-based
    pix = np.arange(n, dtype=float) + 1.0
    return crval + (pix - crpix) * cdelt


def _bspline_basis(x: np.ndarray, t: np.ndarray, deg: int) -> np.ndarray:
    """Evaluate B-spline basis matrix B(x) for knot vector t and degree deg.

    Returns B of shape (len(x), n_basis).
    Pure NumPy Cox–de Boor recursion. Sufficient for moderate sizes (N~2000).
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    if deg < 0:
        raise ValueError("deg must be >=0")
    n_basis = len(t) - deg - 1
    if n_basis <= 0:
        raise ValueError("Invalid knot vector length")

    # k=0
    B = np.zeros((x.size, n_basis), dtype=float)
    for i in range(n_basis):
        left = t[i]
        right = t[i + 1]
        sel = (x >= left) & (x < right)
        B[sel, i] = 1.0
    # include last point exactly at end
    B[x == t[-1], -1] = 1.0

    # recursion
    for k in range(1, deg + 1):
        Bk = np.zeros_like(B)
        for i in range(n_basis):
            d1 = t[i + k] - t[i]
            d2 = t[i + k + 1] - t[i + 1]
            term1 = 0.0
            term2 = 0.0
            if d1 > 0:
                term1 = (x - t[i]) / d1 * B[:, i]
            if d2 > 0 and i + 1 < n_basis:
                term2 = (t[i + k + 1] - x) / d2 * B[:, i + 1]
            Bk[:, i] = term1 + term2
        B = Bk
    return B


def _fit_bspline_1d(x: np.ndarray, y: np.ndarray, *, step: float, deg: int = 3, sigma_clip: float = 3.0, maxiter: int = 6) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < max(20, deg + 5):
        return y

    x0 = float(np.nanmin(x[m]))
    x1 = float(np.nanmax(x[m]))
    if step <= 0:
        step = (x1 - x0) / 200.0

    # internal knots (excluding endpoints)
    internal = np.arange(x0, x1 + step, step, dtype=float)
    if internal.size < 4:
        return y

    # open knot vector
    t = np.concatenate([
        np.full(deg + 1, x0, dtype=float),
        internal[1:-1],
        np.full(deg + 1, x1, dtype=float),
    ])

    B = _bspline_basis(x, t, deg)

    mask = m.copy()
    for _ in range(int(maxiter)):
        xm = x[mask]
        ym = y[mask]
        Bm = B[mask, :]
        if xm.size < max(20, deg + 5):
            break
        # least squares
        try:
            c, *_ = np.linalg.lstsq(Bm, ym, rcond=None)
        except Exception:
            break
        yfit = B @ c
        resid = y - yfit
        r = resid[mask]
        if r.size < 20:
            break
        med = np.nanmedian(r)
        mad = np.nanmedian(np.abs(r - med))
        sigma = 1.4826 * mad if mad > 0 else np.nanstd(r)
        if not np.isfinite(sigma) or sigma <= 0:
            break
        new_mask = m & (np.abs(resid - med) <= sigma_clip * sigma)
        if new_mask.sum() == mask.sum():
            mask = new_mask
            break
        mask = new_mask

    # final
    try:
        c, *_ = np.linalg.lstsq(B[mask, :], y[mask], rcond=None)
        return B @ c
    except Exception:
        return y


def run_sky_sub(cfg: dict[str, Any], *, lin_fits: Path | None = None, out_dir: Path | None = None) -> dict[str, Any]:
    """Run sky subtraction.

    Parameters
    ----------
    cfg
        Resolved config dict.
    lin_fits
        Optional path to the linearized stacked frame. If None,
        uses work_dir/lin/obj_sum_lin.fits.
    out_dir
        Output directory. Defaults to work_dir/sky.
    """

    sky_cfg = (cfg.get("sky") or {}) if isinstance(cfg.get("sky"), dict) else {}
    # v5.12 defaults (best-practice): products/ as canonical outputs.
    wd = resolve_work_dir(cfg)
    products_root = wd / "products"
    legacy_root = wd / "sky"
    if not bool(sky_cfg.get("enabled", True)):
        # Write a marker anyway to keep resume/QC stable.
        wd = resolve_work_dir(cfg)
        out_dir = Path(out_dir) if out_dir is not None else (wd / "sky")
        out_dir.mkdir(parents=True, exist_ok=True)
        done = out_dir / "sky_sub_done.json"
        done.write_text(json.dumps({"skipped": True, "reason": "sky.enabled=false"}, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"skipped": True, "reason": "sky.enabled=false", "out_dir": str(out_dir)}

    # ROI selection: headless from config, or interactive (Qt) if requested and available.
    def _try_pick_roi_interactive(preview_fits: Path) -> dict[str, int] | None:
        try:
            sky = (cfg.get("sky") or {}) if isinstance(cfg.get("sky"), dict) else {}
            if not bool(sky.get("roi_interactive", False)):
                return None
            from PySide6 import QtWidgets
            from scorpio_pipe.ui.sky_roi_dialog import SkyRoiDialog

            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            dlg = SkyRoiDialog(str(preview_fits))
            if dlg.exec() == QtWidgets.QDialog.Accepted:
                return dlg.get_roi_dict()
        except Exception:
            return None
        return None

    # Resolve a preview frame for ROI selection (stacked linearize preview preferred).
    roi: ROI | None = None
    roi_dict: dict[str, int] | None = None
    try:
        roi = _roi_from_cfg(cfg)
    except Exception:
        # Try interactive mode if enabled.
        wd = resolve_work_dir(cfg)
        cand = [
            wd / "products" / "lin" / "lin_preview.fits",
            wd / "lin" / "obj_sum_lin.fits",
        ]
        preview = next((p for p in cand if p.exists()), None)
        if preview is not None:
            roi_dict = _try_pick_roi_interactive(preview)
        if roi_dict is None:
            raise
        # Inject into cfg in-memory so downstream uses it.
        cfg.setdefault("sky", {})
        if isinstance(cfg["sky"], dict):
            cfg["sky"].setdefault("roi", {})
            if isinstance(cfg["sky"]["roi"], dict):
                cfg["sky"]["roi"].update(roi_dict)
        roi = _roi_from_cfg(cfg)

    out_dir = Path(out_dir) if out_dir is not None else (products_root / "sky")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_exposure = bool(sky_cfg.get("per_exposure", True))
    # NOTE (v5.15): stacking is a dedicated stage (stack2d). We keep the
    # ``stack_after`` flag for UI/backward compatibility, but this stage does
    # not perform stacking anymore.
    stack_after = bool(sky_cfg.get("stack_after", True))
    save_per_exp_model = bool(sky_cfg.get("save_per_exp_model", False))
    save_spectrum_1d = bool(sky_cfg.get("save_spectrum_1d", False))

    # Helper: mirror a product into legacy folder for backward compatibility.
    def _mirror_legacy(src: Path, rel: str) -> None:
        try:
            dst = legacy_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        except Exception:
            pass

    # Flexure (Δλ) reference sky spectrum for cross-correlation (set on first processed frame).
    flex_ref_spec: np.ndarray | None = None

    # QC: user-defined "critical" wavelength zones (e.g. strong OH residual region).
    crit_windows_A = sky_cfg.get("critical_windows_A") or sky_cfg.get("critical_windows") or [[6800, 6900]]
    try:
        crit_windows_A = [[float(a), float(b)] for a, b in crit_windows_A]
    except Exception:
        crit_windows_A = [[6800.0, 6900.0]]

    def _read_lin_frame(p: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, fits.Header]:
        """Read a rectified/linearized frame.

        Prefer SCI/VAR/MASK extensions; fallback to primary HDU for legacy.
        """
        try:
            sci, var, mask, hdr = read_sci_var_mask(p)
            return np.asarray(sci, dtype=float), (None if var is None else np.asarray(var, dtype=float)), (
                None if mask is None else np.asarray(mask, dtype=np.uint16)
            ), fits.Header(hdr)
        except Exception:
            with fits.open(p, memmap=False) as hdul:
                hdr = fits.Header(hdul[0].header)
                sci = np.asarray(hdul[0].data, dtype=float)
                var = None
                mask = None
                if "VAR" in hdul:
                    try:
                        var = np.asarray(hdul["VAR"].data, dtype=float)
                    except Exception:
                        var = None
                if "MASK" in hdul:
                    try:
                        mask = np.asarray(hdul["MASK"].data, dtype=np.uint16)
                    except Exception:
                        mask = None
                return sci, var, mask, hdr

    def _qc_residual_metrics(resid: np.ndarray, wave: np.ndarray, sky_rows: np.ndarray) -> dict[str, float]:
        rr = resid[sky_rows, :]
        rms = float(np.sqrt(np.nanmean(rr**2))) if np.isfinite(rr).any() else float("nan")
        mae = float(np.nanmean(np.abs(rr))) if np.isfinite(rr).any() else float("nan")
        out: dict[str, float] = {"rms_sky": rms, "mae_sky": mae}
        # Critical windows
        for i, (a0, a1) in enumerate(crit_windows_A):
            sel = (wave >= min(a0, a1)) & (wave <= max(a0, a1))
            if not np.any(sel):
                continue
            r2 = rr[:, sel]
            out[f"rms_crit_{i}"] = float(np.sqrt(np.nanmean(r2**2))) if np.isfinite(r2).any() else float("nan")
            out[f"mae_crit_{i}"] = float(np.nanmean(np.abs(r2))) if np.isfinite(r2).any() else float("nan")
        return out

    def _process_one(lin_path: Path, *, tag: str, base_dir: Path, write_model: bool) -> dict[str, Any]:
        nonlocal flex_ref_spec
        data, var, mask, hdr = _read_lin_frame(lin_path)

        ny, nx = data.shape
        wave = _wave_from_header(hdr, nx)

        def _clip(a: int) -> int:
            return int(np.clip(a, 0, ny - 1))

        obj_y0, obj_y1 = sorted((_clip(roi.obj_y0), _clip(roi.obj_y1)))
        st0, st1 = sorted((_clip(roi.sky_top_y0), _clip(roi.sky_top_y1)))
        sb0, sb1 = sorted((_clip(roi.sky_bot_y0), _clip(roi.sky_bot_y1)))

        sky_rows = np.zeros(ny, dtype=bool)
        sky_rows[st0 : st1 + 1] = True
        sky_rows[sb0 : sb1 + 1] = True
        sky_rows[obj_y0 : obj_y1 + 1] = False
        if sky_rows.sum() < 3:
            raise ValueError("Sky ROI is too small (need at least a few rows)")

        sky_pix = data[sky_rows, :]
        if mask is not None:
            bad = (mask[sky_rows, :] & _FATAL_BITS) != 0
            sky_pix = np.where(bad, np.nan, sky_pix)
        if var is not None:
            sky_pix = np.where(~np.isfinite(var[sky_rows, :]) | (var[sky_rows, :] <= 0), np.nan, sky_pix)
        sky_spec_raw = np.nanmedian(sky_pix, axis=0)

        # Optional: per-exposure flexure correction (Delta-lambda) using sky spectrum cross-correlation.
        # Two modes are supported:
        #   - global shift (single Delta-lambda per exposure)
        #   - y-dependent shift model Delta-lambda(y) (low-order polynomial), measured from sky rows
        flex = sky_cfg.get("flexure") if isinstance(sky_cfg.get("flexure"), dict) else {}
        flex_enabled = bool(flex.get("enabled", sky_cfg.get("flexure_enabled", False)))
        flex_max = int(flex.get("max_shift_pix", sky_cfg.get("flexure_max_shift_pix", 5)))
        flex_mode = str(flex.get("mode", "full")).lower()
        flex_windows_A = flex.get("windows_A") or flex.get("windows") or flex.get("windows_angstrom")
        flex_windows_pix = flex.get("windows_pix") or flex.get("windows_pixels")
        flex_windows_unit = str(flex.get("windows_unit", "auto") or "auto")
        flex_y_dependent = bool(flex.get("y_dependent", False))
        flex_y_step = int(flex.get("y_step", 32))
        flex_y_bin = int(flex.get("y_bin", 24))
        flex_y_poly_deg = int(flex.get("y_poly_deg", 1))
        flex_y_smooth_bins = int(flex.get("y_smooth_bins", 5) or 5)
        flex_y_sigma_clip = float(flex.get("y_sigma_clip", 3.5) or 3.5)
        flex_y_fit_maxiter = int(flex.get("y_fit_maxiter", 3) or 3)
        flex_min_score = float(flex.get("min_score", 0.05))
        flex_save_curve = bool(flex.get("save_curve", True))
        flex_save_curve_png = bool(flex.get("save_curve_png", True))

        flex_shift_pix: float = 0.0
        flex_shift_A = None
        flex_score: float | None = None
        flex_poly: list[float] | None = None
        flex_poly_meta: dict[str, Any] | None = None

        if flex_enabled:
            # Prepare spectra for xcorr (optionally masked to wavelength windows).
            cur_for_xcorr = sky_spec_raw
            if flex_mode == "windows":
                cur_for_xcorr = _apply_xcorr_windows_any(sky_spec_raw, hdr, flex_windows_A, flex_windows_pix, unit=flex_windows_unit)

            if flex_ref_spec is None:
                # First frame becomes the reference.
                flex_ref_spec = np.array(cur_for_xcorr, dtype=float, copy=True)
                flex_shift_pix = 0.0
                flex_score = None
            else:
                if not flex_y_dependent:
                    flex_shift_pix, _score = _xcorr_subpix_shift(flex_ref_spec, cur_for_xcorr, flex_max)
                    try:
                        flex_score = float(_score)
                    except Exception:
                        flex_score = None

                    if abs(float(flex_shift_pix)) > 1e-6:
                        if mask is None:
                            mask = np.zeros_like(data, dtype=np.uint16)
                        data, filled = _shift_subpix_fill_float(data, float(flex_shift_pix), axis=1, fill=float("nan"))
                        mask = _shift_subpix_mask(mask, float(flex_shift_pix), axis=1)
                        mask[filled] |= NO_COVERAGE
                        if var is not None:
                            var, _filled_v = _shift_subpix_fill_var(var, float(flex_shift_pix), axis=1, fill=float("inf"))
                        sky_pix = data[sky_rows, :]
                        sky_spec_raw = np.nanmedian(sky_pix, axis=0)

                else:
                    # Measure Delta-lambda(y) from binned sky rows and fit a smooth polynomial.
                    ys = np.arange(ny, dtype=int)
                    y_cent: list[int] = []
                    shs: list[float] = []
                    scs: list[float] = []
                    if flex_y_step < 1:
                        flex_y_step = 1
                    if flex_y_bin < 1:
                        flex_y_bin = 1

                    for y0 in range(0, ny, flex_y_step):
                        y1 = min(ny, y0 + flex_y_bin)
                        rows = sky_rows[y0:y1]
                        if int(rows.sum()) < 3:
                            continue
                        spec = np.nanmedian(data[ys[y0:y1][rows], :], axis=0)
                        if flex_mode == "windows":
                            spec = _apply_xcorr_windows_any(spec, hdr, flex_windows_A, flex_windows_pix, unit=flex_windows_unit)
                        sh, sc = _xcorr_subpix_shift(flex_ref_spec, spec, flex_max)
                        try:
                            sc = float(sc)
                        except Exception:
                            sc = float('nan')
                        if not np.isfinite(sh) or not np.isfinite(sc) or sc < flex_min_score:
                            continue
                        y_cent.append(int((y0 + y1 - 1) // 2))
                        shs.append(float(sh))
                        scs.append(float(sc))

                    if len(y_cent) >= max(flex_y_poly_deg + 2, 6):
                        y_arr = np.asarray(y_cent, dtype=float)
                        sh_arr = np.asarray(shs, dtype=float)
                        w_arr = np.asarray(scs, dtype=float)
                        # Normalize y to [0,1] for numerically stable polynomial fits.
                        y0m = float(np.nanmin(y_arr))
                        y1m = float(np.nanmax(y_arr))
                        denom = (y1m - y0m) if (y1m > y0m) else 1.0
                        t = (y_arr - y0m) / denom

                        # Sort by y and (optionally) smooth the measured shifts for stability.
                        order = np.argsort(t)
                        t = t[order]
                        sh_arr = sh_arr[order]
                        w_arr = w_arr[order]

                        def _median_smooth(x: np.ndarray, k: int) -> np.ndarray:
                            k = int(k)
                            if k < 3:
                                return x
                            if k % 2 == 0:
                                k += 1
                            pad = k // 2
                            xp = np.pad(x, pad, mode='edge')
                            out = np.empty_like(x, dtype=float)
                            for i in range(x.size):
                                out[i] = float(np.nanmedian(xp[i : i + k]))
                            return out

                        try:
                            sh_s = _median_smooth(sh_arr.astype(float), int(flex_y_smooth_bins))
                        except Exception:
                            sh_s = sh_arr.astype(float)

                        deg = int(max(0, flex_y_poly_deg))
                        # Iterative sigma-clipped fit (keeps Δλ(y) smooth and robust).
                        good = np.isfinite(t) & np.isfinite(sh_s) & np.isfinite(w_arr)
                        if good.sum() < max(deg + 2, 6):
                            good = np.isfinite(t) & np.isfinite(sh_s)

                        coef = None
                        for _ in range(max(1, int(flex_y_fit_maxiter))):
                            if int(good.sum()) < max(deg + 2, 4):
                                break
                            try:
                                coef = np.polyfit(t[good], sh_s[good], deg=deg, w=np.clip(w_arr[good], 1e-6, None))
                            except Exception:
                                coef = np.polyfit(t[good], sh_s[good], deg=deg)
                            fit = np.polyval(coef, t[good])
                            resid = sh_s[good] - fit
                            med = float(np.nanmedian(resid))
                            sig = 1.4826 * float(np.nanmedian(np.abs(resid - med)))
                            if not np.isfinite(sig) or sig <= 0:
                                break
                            keep = np.abs(resid - med) <= float(flex_y_sigma_clip) * sig
                            idx = np.where(good)[0]
                            new_good = good.copy()
                            new_good[idx] = keep
                            if int(new_good.sum()) == int(good.sum()):
                                good = new_good
                                break
                            good = new_good

                        if coef is None:
                            coef = np.polyfit(t, sh_s, deg=deg)

                        # Evaluate for all rows.
                        t_all = (np.arange(ny, dtype=float) - y0m) / denom
                        shift_pix_y = np.polyval(coef, t_all).astype(float)
                        # Clip to configured bounds.
                        shift_pix_y = np.clip(shift_pix_y, -float(flex_max), float(flex_max))

                        if mask is None:
                            mask = np.zeros_like(data, dtype=np.uint16)
                        data, _filled = shift2d_subpix_x(data, shift_pix_y, fill=float("nan"))
                        mask, _filled_m = shift2d_subpix_x_mask(mask, shift_pix_y, no_coverage_bit=NO_COVERAGE)
                        if var is not None:
                            var, _filled_v = shift2d_subpix_x_var(var, shift_pix_y, fill=float("inf"))

                        # Recompute sky spectrum on corrected frame
                        sky_pix = data[sky_rows, :]
                        sky_spec_raw = np.nanmedian(sky_pix, axis=0)

                        # Summarize (median) shift for QC
                        try:
                            flex_shift_pix = float(np.nanmedian(shift_pix_y[sky_rows]))
                        except Exception:
                            flex_shift_pix = float(np.nanmedian(shift_pix_y))
                        flex_score = float(np.nanmedian(w_arr[good])) if np.any(good) else float(np.nanmedian(w_arr))

                        flex_poly = [float(x) for x in coef.tolist()]
                        flex_poly_meta = {
                            "deg": int(flex_y_poly_deg),
                            "y0": y0m,
                            "y1": y1m,
                            "coef_pix": flex_poly,
                            "samples": {"y": y_cent, "shift_pix": [float(x) for x in shs], "score": [float(x) for x in scs]},
                        }

                        if flex_save_curve:
                            try:
                                out_csv = base_dir / f"{tag}_flexure_ycurve.csv"
                                with out_csv.open("w", newline="", encoding="utf-8") as f:
                                    w = csv.writer(f)
                                    w.writerow(["y", "shift_pix"])
                                    for yy, shv in enumerate(shift_pix_y.tolist()):
                                        w.writerow([int(yy), float(shv)])
                            except Exception:
                                pass

                        if flex_save_curve_png:
                            try:
                                import matplotlib.pyplot as plt

                                out_png = base_dir / f"{tag}_flexure_ycurve.png"
                                with mpl_style():
                                    fig = plt.figure(figsize=(7.0, 3.5), dpi=140)
                                    ax = fig.add_subplot(1, 1, 1)
                                    ax.plot(y_arr, sh_arr, ".", label="measured")
                                    ax.plot(np.arange(ny), shift_pix_y, "-", label="poly")
                                    ax.axhline(0.0, lw=1)
                                    ax.set_xlabel("y (row)")
                                    ax.set_ylabel("shift (pix)")
                                    ax.set_title("Flexure correction: Delta-lambda(y)")
                                    ax.legend(loc="best", frameon=False)
                                    fig.tight_layout()
                                    fig.savefig(out_png)
                                    plt.close(fig)
                            except Exception:
                                pass

                    else:
                        # Not enough information: fall back to global shift.
                        flex_shift_pix, _score = _xcorr_subpix_shift(flex_ref_spec, cur_for_xcorr, flex_max)
                        try:
                            flex_score = float(_score)
                        except Exception:
                            flex_score = None
                        if abs(float(flex_shift_pix)) > 1e-6:
                            if mask is None:
                                mask = np.zeros_like(data, dtype=np.uint16)
                            data, filled = _shift_subpix_fill_float(data, float(flex_shift_pix), axis=1, fill=float("nan"))
                            mask = _shift_subpix_mask(mask, float(flex_shift_pix), axis=1)
                            mask[filled] |= NO_COVERAGE
                            if var is not None:
                                var, _filled_v = _shift_subpix_fill_var(var, float(flex_shift_pix), axis=1, fill=float("inf"))
                            sky_pix = data[sky_rows, :]
                            sky_spec_raw = np.nanmedian(sky_pix, axis=0)

            # Convert to Angstrom if we know the dispersion step.
            cdelt = hdr.get("CDELT1", hdr.get("CD1_1"))
            try:
                if cdelt is not None:
                    flex_shift_A = float(flex_shift_pix) * float(cdelt)
            except Exception:
                flex_shift_A = None
        
        deg = int(sky_cfg.get("bsp_degree", 3))

        step = float(sky_cfg.get("bsp_step_A", 3.0))
        sigma_clip = float(sky_cfg.get("sigma_clip", 3.0))
        maxiter = int(sky_cfg.get("maxiter", 6))
        sky_spec = _fit_bspline_1d(wave, sky_spec_raw, step=step, deg=deg, sigma_clip=sigma_clip, maxiter=maxiter)

        use_spatial = bool(sky_cfg.get("use_spatial_scale", True))
        poly_deg = int(sky_cfg.get("spatial_poly_deg", 1))
        if poly_deg < 0:
            poly_deg = 0
        a_y = np.ones(ny, dtype=float)
        b_y = np.zeros(ny, dtype=float)
        if use_spatial:
            ys = np.where(sky_rows)[0].astype(float)
            a_s = []
            b_s = []
            for y in ys.astype(int):
                row = data[y, :]
                m = np.isfinite(row) & np.isfinite(sky_spec)
                if m.sum() < 30:
                    a_s.append(np.nan)
                    b_s.append(np.nan)
                    continue
                X = np.vstack([sky_spec[m], np.ones(m.sum(), dtype=float)]).T
                try:
                    (a, b), *_ = np.linalg.lstsq(X, row[m], rcond=None)
                except Exception:
                    a, b = np.nan, np.nan
                a_s.append(a)
                b_s.append(b)
            a_s = np.asarray(a_s, dtype=float)
            b_s = np.asarray(b_s, dtype=float)
            good = np.isfinite(a_s) & np.isfinite(b_s)
            if good.sum() >= max(poly_deg + 2, 5):
                pa = np.polyfit(ys[good], a_s[good], deg=poly_deg)
                pb = np.polyfit(ys[good], b_s[good], deg=poly_deg)
                y_all = np.arange(ny, dtype=float)
                a_y = np.polyval(pa, y_all)
                b_y = np.polyval(pb, y_all)

        sky_model = a_y[:, None] * sky_spec[None, :] + b_y[:, None]
        sky_sub = data - sky_model

        hdr_out = hdr.copy()
        hdr_out["HISTORY"] = f"Scorpio Pipe {PIPELINE_VERSION}: sky subtraction (Kelson-like)"
        hdr_out["SKYMETH"] = str(sky_cfg.get("method", "kelson"))
        if flex_enabled:
            hdr_out["DLAMPIX"] = float(flex_shift_pix)
            if flex_shift_A is not None:
                hdr_out["DLAM_A"] = float(flex_shift_A)
            # If we used a y-dependent flexure model, store the polynomial for reproducibility
            if flex_poly_meta is not None:
                try:
                    hdr_out["FLEXY"] = True
                    hdr_out["DLAMDEG"] = int(flex_poly_meta.get("deg", 0))
                    hdr_out["DLAMY0"] = float(flex_poly_meta.get("y0", 0.0))
                    hdr_out["DLAMY1"] = float(flex_poly_meta.get("y1", 0.0))
                    for i, c in enumerate(flex_poly_meta.get("coef_pix", [])):
                        hdr_out[f"DLAMC{i}"] = float(c)
                except Exception:
                    pass
        hdr_out["OBJY0"] = int(obj_y0)
        hdr_out["OBJY1"] = int(obj_y1)

        # per-exposure naming if needed
        sky_model_path = base_dir / f"{tag}_sky_model.fits"
        # New canonical naming (v5.19): *_skysub.fits
        sky_sub_path = base_dir / f"{tag}_skysub.fits"
        sky_sub_legacy = base_dir / f"{tag}_sky_sub.fits"
        sky_spec_csv = base_dir / f"{tag}_sky_spectrum.csv"
        sky_spec_json = base_dir / f"{tag}_sky_spectrum.json"

        if write_model:
            _write_mef(sky_model_path, np.asarray(sky_model, dtype=np.float32), hdr_out)

        _write_mef(
            sky_sub_path,
            np.asarray(sky_sub, dtype=np.float32),
            hdr_out,
            var=None if var is None else np.asarray(var, dtype=np.float32),
            mask=None if mask is None else np.asarray(mask, dtype=np.uint16),
        )
        # Backward-compatible alias
        try:
            if sky_sub_legacy != sky_sub_path:
                shutil.copy2(sky_sub_path, sky_sub_legacy)
        except Exception:
            pass

        # 1D sky spectrum export (optional)
        if save_spectrum_1d:
            try:
                with sky_spec_csv.open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["wave_A", "sky_raw", "sky_fit"])
                    for i in range(nx):
                        w.writerow([float(wave[i]), float(sky_spec_raw[i]), float(sky_spec[i])])
                sky_spec_json.write_text(
                    json.dumps(
                        {
                            "tag": tag,
                            "flexure_shift_pix": float(flex_shift_pix) if flex_enabled else None,
                            "flexure_shift_A": float(flex_shift_A) if (flex_enabled and flex_shift_A is not None) else None,
                            "flexure_model": flex_poly_meta if flex_poly_meta is not None else None,
                            "wave_A": wave.tolist(),
                            "sky_raw": sky_spec_raw.tolist(),
                            "sky_fit": sky_spec.tolist(),
                            "bsp": {"degree": deg, "step_A": step, "sigma_clip": sigma_clip, "maxiter": maxiter},
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass

        # QC metrics in sky rows
        resid_sky = sky_sub[sky_rows, :]
        rms_sky = float(np.sqrt(np.nanmean(resid_sky**2))) if np.isfinite(resid_sky).any() else float("nan")
        mae_sky = float(np.nanmean(np.abs(resid_sky))) if np.isfinite(resid_sky).any() else float("nan")
        masked_frac_sky = None
        if mask is not None:
            m = (mask[sky_rows, :] & _FATAL_BITS) != 0
            masked_frac_sky = float(np.mean(m)) if m.size else None

        # Critical windows (default includes the most painful OH forest region).
        crit_A = sky_cfg.get("critical_windows_A") or [[6800, 6900]]
        crit_pix = sky_cfg.get("critical_windows_pix")
        crit_unit = str(sky_cfg.get("critical_windows_unit", "auto") or "auto")
        sel = np.ones(nx, dtype=bool)
        try:
            tmp = _apply_xcorr_windows_any(np.ones(nx, dtype=float), hdr, crit_A, crit_pix, unit=crit_unit)
            sel = np.isfinite(tmp)
        except Exception:
            sel = np.ones(nx, dtype=bool)

        resid_crit = resid_sky[:, sel]
        rms_crit = float(np.sqrt(np.nanmean(resid_crit**2))) if np.isfinite(resid_crit).any() else None
        mae_crit = float(np.nanmean(np.abs(resid_crit))) if np.isfinite(resid_crit).any() else None

        # Diagnostic plots
        png_spec = base_dir / f"{tag}_sky_spectrum.png"
        png_resid = base_dir / f"{tag}_sky_residuals.png"
        try:
            import matplotlib.pyplot as plt

            with mpl_style():
                # 1) sky spectrum raw vs spline
                fig = plt.figure(figsize=(10, 4))
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(wave, sky_spec_raw, lw=1, label="sky raw")
                ax.plot(wave, sky_spec, lw=1.5, label="B-spline")
                ax.set_xlabel(f"Wavelength [{hdr.get('CUNIT1','Angstrom')}]" if _has_linear_wcs_x(hdr) else "X")
                ax.set_ylabel("Sky (ADU)")
                ax.set_title(f"{tag}: sky spectrum")
                ax.legend(loc="best")
                fig.tight_layout()
                fig.savefig(png_spec, dpi=160)
                plt.close(fig)

                # 2) residual map (sky rows only)
                img = np.full_like(sky_sub, np.nan, dtype=float)
                img[sky_rows, :] = sky_sub[sky_rows, :]
                fig = plt.figure(figsize=(10, 4))
                ax = fig.add_subplot(1, 1, 1)
                finite = img[np.isfinite(img)]
                if finite.size:
                    vmin, vmax = np.nanpercentile(finite, [5, 95])
                else:
                    vmin, vmax = None, None
                ax.imshow(img, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
                ax.axhspan(obj_y0, obj_y1, color="#00aa00", alpha=0.08)
                ax.set_title(f"{tag}: residuals in sky rows")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                fig.tight_layout()
                fig.savefig(png_resid, dpi=160)
                plt.close(fig)
        except Exception:
            png_spec = None
            png_resid = None

        q = {
            "tag": tag,
            "rms_sky": rms_sky,
            "mae_sky": mae_sky,
            "rms_critical": rms_crit,
            "mae_critical": mae_crit,
            "critical_windows_A": crit_A,
            "masked_frac_sky": masked_frac_sky,
            "diag_spectrum_png": str(png_spec) if isinstance(png_spec, Path) else None,
            "diag_residuals_png": str(png_resid) if isinstance(png_resid, Path) else None,
            "n_sky_rows": int(sky_rows.sum()),
            "flexure_shift_pix": float(flex_shift_pix) if flex_enabled else None,
            "flexure_shift_A": float(flex_shift_A) if (flex_enabled and flex_shift_A is not None) else None,
            "flexure_score": float(flex_score) if (flex_enabled and flex_score is not None) else None,
            "flexure_y_dependent": bool(flex_poly_meta is not None) if flex_enabled else None,
            "flexure_poly_deg": int(flex_poly_meta.get("deg", 0)) if (flex_enabled and flex_poly_meta is not None) else None,
        }
        return {
            "ok": True,
            "lin_fits": str(lin_path),
            "sky_model": str(sky_model_path) if write_model else None,
            "sky_sub": str(sky_sub_path),
            "sky_sub_legacy": str(sky_sub_legacy) if sky_sub_legacy.exists() else None,
            "sky_spec_csv": str(sky_spec_csv) if (save_spectrum_1d and sky_spec_csv.exists()) else None,
            "sky_spec_json": str(sky_spec_json) if (save_spectrum_1d and sky_spec_json.exists()) else None,
            "qc": q,
            "metrics": q,
        }

    # Determine inputs
    if not per_exposure:
        if lin_fits is None:
            # Prefer canonical products/... then legacy.
            cand = [
                wd / "products" / "lin" / "lin_preview.fits",
                wd / "lin" / "obj_sum_lin.fits",
            ]
            lin_fits = next((p for p in cand if p.exists()), cand[-1])
        lin_fits = Path(lin_fits)
        if not lin_fits.exists():
            raise FileNotFoundError(f"Missing linearized sum: {lin_fits} (run linearize first)")
        one = _process_one(
            lin_fits,
            tag="obj",
            base_dir=out_dir,
            write_model=bool(sky_cfg.get("save_sky_model", True)),
        )
        # Mirror to legacy names
        if one.get("sky_sub"):
            _mirror_legacy(Path(one["sky_sub"]), "obj_sky_sub.fits")
        if one.get("sky_model"):
            _mirror_legacy(Path(one["sky_model"]), "sky_model.fits")
        payload = {"mode": "stack", "out_dir": str(out_dir), "result": one}
    else:
        per_dir_cand = [
            wd / "products" / "lin" / "per_exp",
            wd / "lin" / "per_exp",
        ]
        per_dir = next((p for p in per_dir_cand if p.exists()), per_dir_cand[0])
        if not per_dir.exists():
            raise FileNotFoundError(
                "Missing per-exposure linearized frames. Expected one of: "
                + ", ".join(str(p) for p in per_dir_cand)
            )
        out_per = out_dir / "per_exp"
        out_per.mkdir(parents=True, exist_ok=True)
        # Prefer canonical outputs from linearize (v5.19+): *_rectified.fits.
        # Fall back to legacy *_lin.fits if needed. Avoid processing both (duplicates).
        files = sorted(per_dir.glob("*_rectified.fits"))
        if not files:
            files = sorted(per_dir.glob("*_lin.fits"))
        if not files:
            files = sorted(per_dir.glob("*.fits"))
        if not files:
            raise FileNotFoundError(f"No per-exposure linearized FITS found in {per_dir}")
        results: list[dict[str, Any]] = []
        for f in files:
            tag = f.stem
            if tag.endswith("_lin"):
                tag = tag[:-4]
            if tag.endswith("_rectified"):
                tag = tag[:-10]
            res = _process_one(
                f,
                tag=tag,
                base_dir=out_per,
                write_model=bool(sky_cfg.get("save_sky_model", True)) and save_per_exp_model,
            )
            results.append(res)
        payload = {
            "mode": "per_exposure",
            "out_dir": str(out_dir),
            "per_exp": results,
            # preferred alias (kept in sync for downstream tools)
            "per_exposure": results,
            "stack_after_requested": stack_after,
        }

    # Persist ROI used (useful for QC / reproducibility, especially if chosen interactively).
    roi_path = out_dir / "roi.json"
    try:
        roi_path.write_text(
            json.dumps(
                {
                    "obj_y0": int(roi.obj_y0),
                    "obj_y1": int(roi.obj_y1),
                    "sky_top_y0": int(roi.sky_top_y0),
                    "sky_top_y1": int(roi.sky_top_y1),
                    "sky_bot_y0": int(roi.sky_bot_y0),
                    "sky_bot_y1": int(roi.sky_bot_y1),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        roi_path = None

    # QC summary JSON.
    qc_path = out_dir / "qc_sky.json"
    try:
        if payload.get("mode") == "stack":
            qcs = [((payload.get("result") or {}).get("qc") or {})]
        else:
            qcs = [((r.get("qc") or {})) for r in (payload.get("per_exp") or []) if isinstance(r, dict)]
        qc_doc = {
            "stage": "sky",
            "mode": payload.get("mode"),
            "out_dir": str(out_dir),
            "roi": {
                "obj_y0": int(roi.obj_y0),
                "obj_y1": int(roi.obj_y1),
                "sky_top_y0": int(roi.sky_top_y0),
                "sky_top_y1": int(roi.sky_top_y1),
                "sky_bot_y0": int(roi.sky_bot_y0),
                "sky_bot_y1": int(roi.sky_bot_y1),
            },
            "critical_windows_A": crit_windows_A,
            "exposures": qcs,
        }
        qc_path.write_text(json.dumps(qc_doc, indent=2, ensure_ascii=False), encoding="utf-8")
        _mirror_legacy(qc_path, "qc_sky.json")
        payload["qc_sky_json"] = str(qc_path)
        if isinstance(roi_path, Path):
            payload["roi_json"] = str(roi_path)
    except Exception:
        pass

    done = out_dir / "sky_sub_done.json"
    done.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _mirror_legacy(done, "sky_sub_done.json")
    return payload
