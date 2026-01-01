"""Stack rectified sky-subtracted frames in (λ, y).

P1-E contract (v5.40.4)
----------------------
* Inputs: **only** ``10_linearize/<stem>_skysub.fits`` MEF products.
* Consistency checks: shapes, wavelength grid, units.
* Normalize to per-second units (ADU/s, e-/s, ...) if needed.
* Robust inverse-variance combine (default: Huber downweighting).
* Propagate masks conservatively (OR over contributing samples).
* Write coverage map (COV) and a structured ``stack_done.json`` report.
* Optional empirical variance scaling :math:`\eta(\lambda)` from sky rows.

Implementation is chunked along y to keep memory bounded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import json
import time

import numpy as np
from astropy.io import fits
import astropy.units as u

from scorpio_pipe.fits_utils import open_fits_smart

from scorpio_pipe.io.mef import write_sci_var_mask, try_read_grid, validate_sci_var_mask
from scorpio_pipe.version import PIPELINE_VERSION
from scorpio_pipe.maskbits import BADPIX, COSMIC, NO_COVERAGE, REJECTED, SATURATED, USER, EDGE

from ..plot_style import mpl_style
from scorpio_pipe.paths import resolve_work_dir
from ..shift_utils import xcorr_shift_subpix


MASK_NO_COVERAGE = NO_COVERAGE
MASK_ROBUST_REJECTED = REJECTED

# Bits that make a pixel unusable for stacking (P1-E).
# NOTE: REJECTED is produced by robust combine; it is not treated as fatal input.
FATAL_BITS = np.uint16(NO_COVERAGE | BADPIX | COSMIC | SATURATED | USER)


def _get_primary_header(hdul: fits.HDUList) -> fits.Header:
    try:
        return hdul[0].header
    except Exception:
        return fits.Header()


def _get_sci_data(hdul: fits.HDUList) -> np.ndarray:
    sci = None
    try:
        sci = hdul[0].data
    except Exception:
        sci = None
    if sci is None and "SCI" in hdul:
        sci = hdul["SCI"].data
    if sci is None:
        raise ValueError("No SCI data")
    return np.asarray(sci)


def _get_var_data(hdul: fits.HDUList) -> np.ndarray:
    if "VAR" not in hdul:
        raise ValueError("VAR extension missing")
    v = hdul["VAR"].data
    if v is None:
        raise ValueError("VAR extension has no data")
    return np.asarray(v)


def _get_mask_data(hdul: fits.HDUList) -> np.ndarray:
    if "MASK" not in hdul:
        raise ValueError("MASK extension missing")
    m = hdul["MASK"].data
    if m is None:
        raise ValueError("MASK extension has no data")
    return np.asarray(m, dtype=np.uint16)


def _split_rate_unit(unit: str) -> tuple[str, bool]:
    """Split a unit string into (base, per_second).

    We intentionally keep this string-based because detector units like ADU
    are not understood by :mod:`astropy.units`.
    """

    u0 = str(unit or "").strip()
    if not u0:
        return "", False
    ul = u0.lower().replace(" ", "")
    per_sec = False
    if "/s" in ul or "/sec" in ul or "s^-1" in ul or "s-1" in ul:
        per_sec = True
        # Remove common rate suffixes.
        for tok in ("/sec", "/s", "s^-1", "s-1"):
            ul = ul.replace(tok, "")
    base = ul
    # Restore capitalization for readability (best-effort).
    if base == "adu":
        base = "ADU"
    elif base in {"e-", "electron", "electrons"}:
        base = "e-"
    elif base:
        base = base.upper() if base.isalpha() else base
    return base, per_sec


def _get_exptime_seconds(hdr: fits.Header) -> float | None:
    for k in ("TEXPS", "EXPTIME", "EXPOSURE"):
        v = hdr.get(k)
        if v is None:
            continue
        try:
            vv = float(v)
        except Exception:
            continue
        if np.isfinite(vv) and vv > 0:
            return vv
    return None


def _grid_to_unit(grid: Any, target_unit: str) -> tuple[float, float, int, str]:
    """Return (lambda0, dlambda, nlam, unit) converted to ``target_unit``."""
    if grid is None:
        raise ValueError("Missing wavelength grid")
    try:
        u_in = u.Unit(str(getattr(grid, "unit", "Angstrom")))
        u_out = u.Unit(str(target_unit))
        fac = (1.0 * u_in).to(u_out).value
    except Exception:
        # Fall back to common names.
        in_u = str(getattr(grid, "unit", "Angstrom")).strip().lower()
        out_u = str(target_unit).strip().lower()
        fac = 1.0
        if in_u in {"angstrom", "a", "aa"} and out_u in {"angstrom", "a", "aa"}:
            fac = 1.0
        elif in_u in {"nm", "nanometer", "nanometre"} and out_u in {"angstrom", "a", "aa"}:
            fac = 10.0
        elif in_u in {"angstrom", "a", "aa"} and out_u in {"nm", "nanometer", "nanometre"}:
            fac = 0.1
        else:
            raise
    return (
        float(getattr(grid, "lambda0")) * float(fac),
        float(getattr(grid, "dlambda")) * float(fac),
        int(getattr(grid, "nlam")),
        str(target_unit),
    )


def _xcorr_subpix_shift_1d(
    ref: np.ndarray, cur: np.ndarray, max_shift: int
) -> tuple[float, float]:
    """Return (shift_pix, score) to apply to `cur` to best match `ref`.

    Uses NumPy-only normalized dot-product xcorr with a parabola refinement
    around the best integer shift (see :mod:`scorpio_pipe.shift_utils`).
    """

    est = xcorr_shift_subpix(ref, cur, max_shift=max_shift)
    return float(est.shift_pix), float(est.score)


def _colsel_from_windows(
    hdr: fits.Header, nx: int, windows_A: Any | None, windows_pix: Any | None = None
) -> np.ndarray | None:
    """Return boolean column selector for wavelength windows.

    Supports either Angstrom windows on a linear WCS (CRVAL1/CDELT1[/CD1_1]/CRPIX1)
    or pixel windows. Returns None if no usable selector can be built.
    """

    # Pixel windows take precedence.
    if windows_pix is not None:
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
            if sel.sum() >= 16:
                return sel
        except Exception:
            pass

    if windows_A is None:
        return None

    crval1 = hdr.get("CRVAL1")
    cdelt1 = hdr.get("CDELT1", hdr.get("CD1_1"))
    crpix1 = hdr.get("CRPIX1", 1.0)
    if crval1 is None or cdelt1 is None:
        return None

    try:
        crval1 = float(crval1)
        cdelt1 = float(cdelt1)
        crpix1 = float(crpix1)
    except Exception:
        return None

    if not np.isfinite(cdelt1) or abs(cdelt1) <= 0:
        return None

    def w_to_i(w: float) -> float:
        # WCS: w = CRVAL1 + (i+1 - CRPIX1)*CDELT1
        return (w - crval1) / cdelt1 + crpix1 - 1.0

    sel = np.zeros(nx, dtype=bool)
    try:
        for win in list(windows_A):
            if isinstance(win, dict):
                w0 = float(win.get("w0", win.get("lo")))
                w1 = float(win.get("w1", win.get("hi")))
            elif isinstance(win, (list, tuple)) and len(win) >= 2:
                w0, w1 = float(win[0]), float(win[1])
            else:
                continue
            if w0 > w1:
                w0, w1 = w1, w0
            i0 = int(np.floor(w_to_i(w0)))
            i1 = int(np.ceil(w_to_i(w1)))
            i0 = max(0, min(nx - 1, i0))
            i1 = max(0, min(nx - 1, i1))
            sel[i0 : i1 + 1] = True
    except Exception:
        return None

    if sel.sum() < 16:
        return None
    return sel


def _take_block_yshift(
    arr: np.ndarray, y0: int, y1: int, shift: int, *, fill: float
) -> tuple[np.ndarray, np.ndarray]:
    """Take arr block [y0:y1, :] from a frame shifted by `shift` in y.

    We interpret `shift` as a translation applied to the *input* frame to align
    it into the *output* y-grid:
        out[y + shift] <- in[y]

    Returns (block, filled_mask) where filled_mask marks pixels that were filled.
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


def _take_block_yshift_subpix(
    arr: np.ndarray, y0: int, y1: int, shift: float, *, fill: float
) -> tuple[np.ndarray, np.ndarray]:
    """Subpixel y-shifted block (SCI-like), using linear interpolation.

    Sign convention matches the integer helper: out[y+shift] <- in[y].
    For fractional shifts we sample input rows at y_in = y_out - shift.

    Returns (block, filled_mask).
    """

    # IMPORTANT: do not cast the full array (it can be a memmap). We only
    # materialize the y-rows we need.
    arr = np.asarray(arr)
    ny, nx = arr.shape
    s = float(shift)
    if not np.isfinite(s) or abs(s) < 1e-9:
        blk = arr[y0:y1, :].copy()
        filled = ~np.isfinite(blk)
        return blk, filled

    y_out = np.arange(y0, y1, dtype=float)
    y_in = y_out - s
    i0 = np.floor(y_in).astype(int)
    i1 = i0 + 1
    frac = (y_in - i0).astype(float)
    valid = (i0 >= 0) & (i1 < ny)
    i0c = np.clip(i0, 0, ny - 1)
    i1c = np.clip(i1, 0, ny - 1)

    out = np.full((y1 - y0, nx), fill, dtype=np.float32)
    filled = np.ones((y1 - y0, nx), dtype=bool)
    if np.any(valid):
        a0 = arr[i0c, :]
        a1 = arr[i1c, :]
        w1 = frac[:, None]
        w0 = 1.0 - w1
        vv = valid[:, None]
        out[vv] = w0[vv] * a0[vv] + w1[vv] * a1[vv]
        filled[vv] = False
    return out, filled


def _take_block_yshift_subpix_var(
    var: np.ndarray, y0: int, y1: int, shift: float, *, fill: float
) -> tuple[np.ndarray, np.ndarray]:
    """Subpixel y-shifted block for VAR with (w0^2, w1^2) propagation."""

    # IMPORTANT: do not cast the full array (it can be a memmap). We only
    # materialize the y-rows we need.
    var = np.asarray(var)
    ny, nx = var.shape
    s = float(shift)
    if not np.isfinite(s) or abs(s) < 1e-9:
        blk = var[y0:y1, :].copy()
        filled = ~np.isfinite(blk)
        return np.asarray(blk, dtype=np.float32), filled

    y_out = np.arange(y0, y1, dtype=float)
    y_in = y_out - s
    i0 = np.floor(y_in).astype(int)
    i1 = i0 + 1
    frac = (y_in - i0).astype(float)
    valid = (i0 >= 0) & (i1 < ny)
    i0c = np.clip(i0, 0, ny - 1)
    i1c = np.clip(i1, 0, ny - 1)

    out = np.full((y1 - y0, nx), fill, dtype=np.float32)
    filled = np.ones((y1 - y0, nx), dtype=bool)
    if np.any(valid):
        v0 = var[i0c, :]
        v1 = var[i1c, :]
        w1 = frac[:, None]
        w0 = 1.0 - w1
        vv = valid[:, None]
        out[vv] = (w0[vv] ** 2) * v0[vv] + (w1[vv] ** 2) * v1[vv]
        filled[vv] = False
    return out, filled


def _take_block_yshift_subpix_mask(
    mask: np.ndarray, y0: int, y1: int, shift: float
) -> np.ndarray:
    """Subpixel y-shifted mask block (conservative OR)."""

    mask = np.asarray(mask, dtype=np.uint16)
    ny, nx = mask.shape
    s = float(shift)
    if not np.isfinite(s) or abs(s) < 1e-9:
        return mask[y0:y1, :].copy()

    y_out = np.arange(y0, y1, dtype=float)
    y_in = y_out - s
    i0 = np.floor(y_in).astype(int)
    i1 = i0 + 1
    valid = (i0 >= 0) & (i1 < ny)
    i0c = np.clip(i0, 0, ny - 1)
    i1c = np.clip(i1, 0, ny - 1)

    out = np.full((y1 - y0, nx), MASK_NO_COVERAGE, dtype=np.uint16)
    if np.any(valid):
        m0 = mask[i0c, :]
        m1 = mask[i1c, :]
        vv = valid[:, None]
        out[vv] = m0[vv] | m1[vv]
    return out


def _take_block_yshift_mask(
    mask: np.ndarray, y0: int, y1: int, shift: int
) -> np.ndarray:
    """Take uint16 mask block with y-shift; filled pixels get MASK_NO_COVERAGE."""
    mask = np.asarray(mask, dtype=np.uint16)
    ny, nx = mask.shape
    shift = int(shift)
    out = np.full((y1 - y0, nx), MASK_NO_COVERAGE, dtype=np.uint16)

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


def _open_mef(
    path: Path,
) -> tuple[np.ndarray, fits.Header, np.ndarray | None, np.ndarray | None]:
    """Return (sci, hdr, var, mask) from a MEF or simple FITS."""
    # Use memmap='auto' to avoid Astropy strict_memmap failures when the file
    # declares BZERO/BSCALE/BLANK (common for unsigned MASK extensions).
    with open_fits_smart(path, memmap="auto") as hdul:
        hdr = hdul[0].header.copy()
        sci = hdul[0].data
        if sci is None:
            # Try SCI ext
            if "SCI" in hdul:
                sci = hdul["SCI"].data
                hdr = hdul["SCI"].header.copy()
            else:
                raise ValueError(f"No SCI data found in {path}")
        sci = np.asarray(sci, dtype=np.float32)
        var = None
        mask = None
        if "VAR" in hdul:
            try:
                var = np.asarray(hdul["VAR"].data, dtype=np.float32)
            except Exception:
                var = None
        if "MASK" in hdul:
            try:
                mask = np.asarray(hdul["MASK"].data, dtype=np.uint16)
            except Exception:
                mask = None
        return sci, hdr, var, mask


def _write_mef(
    path: Path,
    sci: np.ndarray,
    hdr: fits.Header,
    *,
    var: np.ndarray | None,
    mask: np.ndarray | None,
    cov: np.ndarray | None,
) -> None:
    """Write stacked 2D product as MEF (SCI/VAR/MASK [+COV])."""
    grid = try_read_grid(hdr)
    extra: list[fits.ImageHDU] = []
    if cov is not None:
        extra.append(fits.ImageHDU(np.asarray(cov, dtype=np.int16), name="COV"))
    write_sci_var_mask(
        path, sci, var=var, mask=mask, header=hdr, grid=grid, extra_hdus=extra
    )


def _iter_slices(ny: int, chunk: int) -> Iterable[slice]:
    chunk = int(max(8, chunk))
    for y0 in range(0, int(ny), chunk):
        yield slice(y0, min(ny, y0 + chunk))


def run_stack2d(
    cfg: dict[str, Any], *, inputs: Iterable[Path], out_dir: Path | None = None
) -> dict[str, Any]:
    st_cfg = (cfg.get("stack2d") or {}) if isinstance(cfg.get("stack2d"), dict) else {}
    if not bool(st_cfg.get("enabled", True)):
        # Keep a stable payload shape even when skipped.
        return {
            "skipped": True,
            "reason": "stack2d.enabled=false",
            "stack2d_fits": None,
            "stacked2d_fits": None,
            "coverage_png": None,
            "qc_png": None,
        }

    t0 = time.time()

    wd = resolve_work_dir(cfg)
    if out_dir is None:
        from scorpio_pipe.workspace_paths import stage_dir

        out_dir = stage_dir(wd, "stack2d")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [Path(p) for p in inputs]
    files = [p for p in files if p.exists()]
    if not files:
        raise FileNotFoundError("stack2d: no input FITS files")

    # Enforce P1-E input contract: only 10_linearize/<stem>_skysub.fits.
    from scorpio_pipe.workspace_paths import stage_dir

    lin_dir = stage_dir(wd, "linearize").resolve()
    for p in files:
        pr = p.resolve()
        try:
            if not pr.is_relative_to(lin_dir):
                raise ValueError(
                    f"stack2d input must come from {lin_dir.name}: got {p}"
                )
        except AttributeError:
            # Python < 3.9 fallback
            if str(lin_dir) not in str(pr):
                raise ValueError(
                    f"stack2d input must come from {lin_dir.name}: got {p}"
                )

    normalize_exptime = bool(st_cfg.get("normalize_exptime", True))
    method = str(st_cfg.get("method", "invvar_huber") or "invvar_huber").strip().lower()
    robust_iter = int(st_cfg.get("robust_iter", st_cfg.get("maxiter", 3)) or 3)
    huber_c = float(st_cfg.get("huber_c", 2.0) or 2.0)
    clip_sigma = float(st_cfg.get("clip_sigma", st_cfg.get("sigma_clip", 4.0)) or 4.0)
    chunk = int(st_cfg.get("chunk_rows", 128))

    # Optional y-alignment (subpixel shifts) before stacking.
    ya_cfg = st_cfg.get("y_align")
    if isinstance(ya_cfg, dict):
        y_align_enabled = bool(ya_cfg.get("enabled", False))
        y_align_max = int(ya_cfg.get("max_shift_pix", 10))
    else:
        y_align_enabled = bool(st_cfg.get("y_align_enabled", False))
        y_align_max = int(st_cfg.get("y_align_max_shift_pix", 10))

    # Open all MEFs (memmap) and collect per-input metadata.
    hduls = [open_fits_smart(p, memmap="auto") for p in files]
    report: dict[str, Any] = {
        "ok": False,
        "status": "fail",
        "stage": "stack2d",
        "pipeline_version": PIPELINE_VERSION,
        "started_at_unix": float(t0),
        "inputs": [],
        "config": {
            "method": method,
            "robust_iter": robust_iter,
            "huber_c": huber_c,
            "clip_sigma": clip_sigma,
            "normalize_exptime": normalize_exptime,
            "chunk_rows": chunk,
            "y_align_enabled": bool(y_align_enabled),
            "y_align_max_shift_pix": int(y_align_max),
        },
    }

    # Products (filled later)
    out_fits = out_dir / "stack2d.fits"
    out_fits_legacy = out_dir / "stacked2d.fits"
    eta_fits = out_dir / "eta_lambda.fits"
    # Navigator-friendly quicklooks
    out_stack_png = out_dir / "stack2d.png"
    out_cov_png = out_dir / "coverage.png"
    out_cov_hist_png = out_dir / "coverage_hist.png"
    out_eta_png = out_dir / "eta_lambda.png"
    stack_done = out_dir / "stack_done.json"
    stack2d_done = out_dir / "stack2d_done.json"
    done_json = out_dir / "done.json"

    # We keep the geometry object (for var_floor/eta) and a JSON-friendly summary.
    sky_geom = None
    sky_geometry_meta: dict[str, Any] | None = None

    try:
        # First exposure defines reference shape, WCS/grid, and base unit.
        hdr0 = _get_primary_header(hduls[0]).copy()
        sci0 = np.asarray(_get_sci_data(hduls[0]), dtype=np.float32)
        var0 = np.asarray(_get_var_data(hduls[0]), dtype=np.float32)
        mask0 = np.asarray(_get_mask_data(hduls[0]), dtype=np.uint16)

        validate_sci_var_mask(sci0, var0, mask0, fatal_bits=int(FATAL_BITS))
        ny, nx = sci0.shape

        # Wavelength grid consistency (P1-E).
        grid0 = try_read_grid(hdr0)
        if grid0 is None:
            raise ValueError(f"stack2d: missing wavelength grid in {files[0].name}")
        lam0_aa, dlam_aa, nlam, grid_unit = _grid_to_unit(grid0, "Angstrom")
        if int(nlam) != int(nx):
            raise ValueError(
                f"stack2d: wavelength grid nlam={nlam} does not match image nx={nx}"
            )

        # Optional: unified slit geometry (used for var_floor and eta(λ)).
        try:
            from scorpio_pipe.sky_geometry import compute_sky_geometry, roi_from_cfg

            roi = roi_from_cfg(cfg)
            sky = cfg.get("sky") if isinstance(cfg.get("sky"), dict) else {}
            gcfg = sky.get("geometry") if isinstance(sky.get("geometry"), dict) else {}
            sky_geom = compute_sky_geometry(
                sci0,
                var0,
                mask0,
                roi=roi,
                roi_policy=str(gcfg.get("roi_policy", "prefer_user")),
                fatal_bits=int(FATAL_BITS),
                edge_margin_px=int(gcfg.get("edge_margin_px", 16) or 16),
                profile_x_percentile=float(gcfg.get("profile_x_percentile", 50.0) or 50.0),
                thresh_sigma=float(gcfg.get("thresh_sigma", 3.0) or 3.0),
                dilation_px=int(gcfg.get("dilation_px", 3) or 3),
                min_obj_width_px=int(gcfg.get("min_obj_width_px", 6) or 6),
                min_sky_width_px=int(gcfg.get("min_sky_width_px", 12) or 12),
            )
            sky_geometry_meta = {
                "roi_used": sky_geom.roi_used,
                "metrics": sky_geom.metrics,
                "sky_windows": [list(x) for x in sky_geom.sky_windows],
                "object_spans": [list(x) for x in sky_geom.object_spans],
            }
        except Exception:
            sky_geom = None
            sky_geometry_meta = None

        # Units & exposure-time normalization (P1-E).
        unit0_raw = hdr0.get("BUNIT", hdr0.get("SCIUNIT", ""))
        base0, per_sec0 = _split_rate_unit(str(unit0_raw))
        norm0 = bool(hdr0.get("NORMEXP", False))
        per_sec0 = bool(per_sec0 or norm0)
        if not base0:
            raise ValueError("stack2d: missing BUNIT/SCIUNIT in first input")

        base_ref = base0
        # Per-frame scaling factors applied to SCI and VAR.
        sci_scales: list[float] = []
        var_scales: list[float] = []
        exptimes: list[float | None] = []

        # Per-input metadata + strict validation.
        for p, h in zip(files, hduls):
            hdr = _get_primary_header(h)
            sci = _get_sci_data(h)
            var = _get_var_data(h)
            mask = _get_mask_data(h)

            if sci.shape != (ny, nx):
                raise ValueError(
                    f"stack2d: shape mismatch in {p.name}: {sci.shape} != {(ny, nx)}"
                )
            if var.shape != (ny, nx):
                raise ValueError(
                    f"stack2d: VAR shape mismatch in {p.name}: {var.shape} != {(ny, nx)}"
                )
            if mask.shape != (ny, nx):
                raise ValueError(
                    f"stack2d: MASK shape mismatch in {p.name}: {mask.shape} != {(ny, nx)}"
                )

            validate_sci_var_mask(
                np.asarray(sci, dtype=np.float32),
                np.asarray(var, dtype=np.float32),
                np.asarray(mask, dtype=np.uint16),
                fatal_bits=int(FATAL_BITS),
            )

            g = try_read_grid(hdr)
            if g is None:
                raise ValueError(f"stack2d: missing wavelength grid in {p.name}")
            l0, dl, nn, _ = _grid_to_unit(g, "Angstrom")
            if int(nn) != int(nx):
                raise ValueError(
                    f"stack2d: wavelength grid nlam={nn} does not match nx={nx} in {p.name}"
                )
            tol = float(st_cfg.get("grid_tol_angstrom", 1e-6) or 1e-6)
            if (abs(l0 - lam0_aa) > tol) or (abs(dl - dlam_aa) > tol):
                raise ValueError(
                    f"stack2d: wavelength grid mismatch vs first frame (tol={tol} Å) in {p.name}"
                )

            unit_raw = hdr.get("BUNIT", hdr.get("SCIUNIT", ""))
            base, per_sec = _split_rate_unit(str(unit_raw))
            norm = bool(hdr.get("NORMEXP", False))
            per_sec = bool(per_sec or norm)
            if base != base_ref:
                raise ValueError(
                    f"stack2d: incompatible SCI unit base in {p.name}: {base!r} != {base_ref!r}"
                )

            exptime = _get_exptime_seconds(hdr)
            exptimes.append(exptime)

            s_scale = 1.0
            applied_norm = False
            if normalize_exptime and (not per_sec):
                if exptime is None:
                    raise ValueError(
                        f"stack2d: need EXPTIME to normalize units for {p.name}"
                    )
                s_scale = 1.0 / float(exptime)
                applied_norm = True
                per_sec = True

            sci_scales.append(float(s_scale))
            var_scales.append(float(s_scale * s_scale))

            report["inputs"].append(
                {
                    "file": p.name,
                    "path": str(p),
                    "shape": [int(ny), int(nx)],
                    "bunit": str(unit_raw),
                    "base_unit": base,
                    "per_second": bool(per_sec),
                    "exptime_s": float(exptime) if exptime is not None else None,
                    "normalized_by_stack": bool(applied_norm),
                    "grid": {
                        "lambda0_A": float(l0),
                        "dlambda_A": float(dl),
                        "nlam": int(nn),
                    },
                }
            )

        # Decide output unit string.
        out_bunit = f"{base_ref}/s" if normalize_exptime else str(unit0_raw)

        # Var-floor (P1-E): percentile in sky rows (subsampled).
        var_floor_enabled = bool(st_cfg.get("var_floor_enabled", True))
        var_floor = None
        var_floor_meta: dict[str, Any] = {"enabled": var_floor_enabled, "value": None}
        if var_floor_enabled:
            pctl = float(st_cfg.get("var_floor_percentile", 1.0) or 1.0)
            scale = float(st_cfg.get("var_floor_scale", 1.0) or 1.0)
            max_rows = int(st_cfg.get("var_floor_sample_rows", 96) or 96)
            max_x = int(st_cfg.get("var_floor_sample_cols", 256) or 256)

            if sky_geom is not None:
                sky_rows = np.where(np.asarray(sky_geom.mask_sky_y, dtype=bool))[0]
            else:
                sky_rows = np.arange(ny)
            if sky_rows.size > 0:
                # Evenly spaced rows.
                if sky_rows.size > max_rows:
                    idx = np.linspace(0, sky_rows.size - 1, max_rows).astype(int)
                    sky_rows = sky_rows[idx]

            cols = np.arange(nx)
            if cols.size > max_x:
                cols = np.linspace(0, nx - 1, max_x).astype(int)

            samples: list[np.ndarray] = []
            for i, h in enumerate(hduls):
                v = np.asarray(_get_var_data(h)[np.ix_(sky_rows, cols)], dtype=np.float32)
                m = np.asarray(_get_mask_data(h)[np.ix_(sky_rows, cols)], dtype=np.uint16)
                v = v * float(var_scales[i])
                good = np.isfinite(v) & (v > 0) & ((m & FATAL_BITS) == 0)
                if np.any(good):
                    samples.append(v[good])

            if samples:
                vv = np.concatenate(samples, axis=0)
                if vv.size >= 128:
                    try:
                        vf = float(np.percentile(vv, pctl)) * float(scale)
                        if np.isfinite(vf) and vf > 0:
                            var_floor = float(vf)
                    except Exception:
                        var_floor = None

            var_floor_meta.update(
                {
                    "percentile": float(pctl),
                    "scale": float(scale),
                    "sample_rows": int(sky_rows.size),
                    "sample_cols": int(cols.size),
                    "value": float(var_floor) if var_floor is not None else None,
                }
            )
        report["var_floor"] = var_floor_meta

        # Output arrays.
        out_sci = np.full((ny, nx), np.nan, dtype=np.float32)
        out_var = np.full((ny, nx), np.nan, dtype=np.float32)
        out_mask = np.zeros((ny, nx), dtype=np.uint16)
        out_cov = np.zeros((ny, nx), dtype=np.int16)

        # Precompute per-exposure y offsets (subpixel) if requested.
        y_shifts = [0.0 for _ in files]
        y_scores: list[float | None] = [None for _ in files]
        y_offsets: list[dict[str, Any]] = []
        if y_align_enabled and len(files) > 1:
            y_align_mode = "full"
            y_align_windows_A = None
            y_align_windows_pix = None
            y_align_windows_unit = "auto"
            y_align_use_positive = True
            if isinstance(ya_cfg, dict):
                y_align_mode = str(ya_cfg.get("mode", "full") or "full").strip().lower()
                y_align_windows_A = (
                    ya_cfg.get("windows_A")
                    or ya_cfg.get("windows")
                    or ya_cfg.get("windows_angstrom")
                )
                y_align_windows_pix = ya_cfg.get("windows_pix") or ya_cfg.get(
                    "windows_pixels"
                )
                y_align_windows_unit = (
                    str(ya_cfg.get("windows_unit", "auto") or "auto").strip().lower()
                )
                y_align_use_positive = bool(ya_cfg.get("use_positive_flux", True))

            profiles = []
            for h in hduls:
                s = np.asarray(_get_sci_data(h), dtype=np.float32)
                m = np.asarray(_get_mask_data(h), dtype=np.uint16)
                good = np.isfinite(s) & ((m & FATAL_BITS) == 0)

                sel = None
                if y_align_mode == "windows":
                    try:
                        has_wcs = (hdr0.get("CRVAL1") is not None) and (
                            hdr0.get("CDELT1") is not None or hdr0.get("CD1_1") is not None
                        )
                        unit = str(y_align_windows_unit or "auto").lower()
                        wA = y_align_windows_A
                        wp = y_align_windows_pix
                        if unit in ("a", "angstrom"):
                            wp = None
                        elif unit in ("pix", "pixel", "pixels"):
                            wA = None
                        else:
                            if has_wcs and wA is not None:
                                wp = None
                            elif (not has_wcs) and wp is not None:
                                wA = None
                        sel = _colsel_from_windows(hdr0, nx, wA, windows_pix=wp)
                    except Exception:
                        sel = None

                if sel is not None:
                    s2 = s[:, sel]
                    good2 = good[:, sel]
                else:
                    s2 = s
                    good2 = good

                if y_align_use_positive:
                    s2 = np.maximum(s2, 0.0)
                prof = np.nansum(np.where(good2, s2, 0.0), axis=1)
                profiles.append(prof)

            ref = profiles[0]
            for i in range(1, len(profiles)):
                sh, sc = _xcorr_subpix_shift_1d(ref, profiles[i], y_align_max)
                y_shifts[i] = float(sh)
                y_scores[i] = float(sc)

            y_offsets = [
                {"file": p.name, "y_shift_pix": float(sh), "score": y_scores[i]}
                for i, (p, sh) in enumerate(zip(files, y_shifts))
            ]
        else:
            y_offsets = [{"file": p.name, "y_shift_pix": 0.0, "score": None} for p in files]

        report["y_offsets"] = y_offsets
        report["sky_geometry"] = sky_geometry_meta

        # Accumulators for diagnostics.
        suppressed_any = 0
        suppressed_total = 0

        # Main stacking loop (chunked).
        for ys in _iter_slices(ny, chunk):
            y0 = int(ys.start or 0)
            y1 = int(ys.stop or ny)

            sci_stack: list[np.ndarray] = []
            var_stack: list[np.ndarray] = []
            mask_stack: list[np.ndarray] = []

            for i, h in enumerate(hduls):
                sh = float(y_shifts[i]) if (y_align_enabled and i < len(y_shifts)) else 0.0

                sci_src = _get_sci_data(h)
                var_src = _get_var_data(h)
                mask_src = _get_mask_data(h)

                if y_align_enabled and abs(sh) > 1e-6:
                    block_s, _ = _take_block_yshift_subpix(sci_src, y0, y1, sh, fill=float("nan"))
                    block_v, _ = _take_block_yshift_subpix_var(var_src, y0, y1, sh, fill=float("nan"))
                    block_m = _take_block_yshift_subpix_mask(mask_src, y0, y1, sh)
                else:
                    block_s, _ = _take_block_yshift(sci_src, y0, y1, int(round(sh)), fill=float("nan"))
                    block_v, _ = _take_block_yshift(var_src, y0, y1, int(round(sh)), fill=float("nan"))
                    block_m = _take_block_yshift_mask(mask_src, y0, y1, int(round(sh)))

                # Apply exposure normalization if requested.
                s = np.asarray(block_s, dtype=np.float32) * float(sci_scales[i])
                v = np.asarray(block_v, dtype=np.float32) * float(var_scales[i])
                m = np.asarray(block_m, dtype=np.uint16)

                sci_stack.append(s)
                var_stack.append(v)
                mask_stack.append(m)

            S = np.stack(sci_stack, axis=0)  # (nexp, y, x)
            V = np.stack(var_stack, axis=0)
            M = np.stack(mask_stack, axis=0)

            # Valid samples: finite + positive variance + no fatal mask bits.
            valid = np.isfinite(S) & np.isfinite(V) & (V > 0) & ((M & FATAL_BITS) == 0)

            # Variance floor.
            if var_floor is not None and var_floor > 0:
                Veff = np.maximum(V, float(var_floor)).astype(np.float32)
            else:
                Veff = V

            W0 = np.where(valid, 1.0 / Veff, 0.0).astype(np.float32)
            W = W0.copy()

            # Robust iterations.
            keep = valid.copy()
            hub = np.ones_like(W, dtype=np.float32)
            for _ in range(max(0, robust_iter)):
                wsum = np.sum(W, axis=0)
                mu = np.where(wsum > 0, np.sum(W * S, axis=0) / wsum, np.nan)

                # Normalized residuals.
                r = (S - mu[None, :, :]) / np.sqrt(np.maximum(Veff, 1e-20))
                if method in {"invvar_huber", "huber", "huber_invvar"}:
                    ar = np.abs(r)
                    hub = np.where(ar <= huber_c, 1.0, huber_c / np.maximum(ar, 1e-20)).astype(
                        np.float32
                    )
                    W = W0 * hub
                elif method in {"invvar_clip", "sigma_clip", "clip"}:
                    keep &= (np.abs(r) <= clip_sigma)
                    W = np.where(keep, W0, 0.0)
                else:
                    # Fallback: plain inverse-variance mean.
                    W = W0

            wsum = np.sum(W, axis=0)
            cov = np.sum(W > 0, axis=0).astype(np.int16)
            mu = np.where(wsum > 0, np.sum(W * S, axis=0) / wsum, np.nan)
            var_out = np.where(wsum > 0, 1.0 / wsum, np.nan).astype(np.float32)

            # Mask propagation: OR over contributing samples.
            m_out = np.zeros(mu.shape, dtype=np.uint16)
            m_out = np.where(wsum <= 0, m_out | MASK_NO_COVERAGE, m_out)
            m_or = np.bitwise_or.reduce(np.where(W > 0, M, 0), axis=0).astype(np.uint16)
            m_out |= m_or

            # Mark robustly suppressed/rejected pixels.
            if method in {"invvar_clip", "sigma_clip", "clip"}:
                suppressed = np.any(valid & (W == 0) & (W0 > 0), axis=0)
            elif method in {"invvar_huber", "huber", "huber_invvar"}:
                suppressed = np.any(valid & (hub < 1.0), axis=0)
            else:
                suppressed = np.zeros(mu.shape, dtype=bool)
            m_out = np.where(suppressed, m_out | MASK_ROBUST_REJECTED, m_out)

            suppressed_any += int(np.count_nonzero(suppressed))
            suppressed_total += int(suppressed.size)

            out_sci[ys, :] = mu.astype(np.float32)
            out_var[ys, :] = var_out
            out_mask[ys, :] = m_out
            out_cov[ys, :] = cov

        # Empirical variance scaling eta(lambda) (optional).
        eta_cfg = st_cfg.get("eta") if isinstance(st_cfg.get("eta"), dict) else {}
        eta_enabled = bool(eta_cfg.get("enabled", True))
        eta_meta: dict[str, Any] = {"enabled": eta_enabled, "applied": False}
        if eta_enabled and sky_geom is not None:
            try:
                from scipy.ndimage import median_filter

                sky_rows_mask = np.asarray(sky_geom.mask_sky_y, dtype=bool)
                good = (
                    sky_rows_mask[:, None]
                    & np.isfinite(out_sci)
                    & np.isfinite(out_var)
                    & (out_cov > 0)
                    & ((out_mask & FATAL_BITS) == 0)
                )
                if np.count_nonzero(sky_rows_mask) >= 8 and np.count_nonzero(good) >= 256:
                    Z = np.where(good, out_sci, np.nan)
                    med = np.nanmedian(Z, axis=0)
                    mad = np.nanmedian(np.abs(Z - med[None, :]), axis=0)
                    sigma = 1.4826 * mad
                    var_emp = sigma * sigma
                    var_pred = np.nanmedian(np.where(good, out_var, np.nan), axis=0)
                    eta_raw = np.where(
                        np.isfinite(var_emp) & np.isfinite(var_pred) & (var_pred > 0),
                        var_emp / var_pred,
                        np.nan,
                    )

                    eta_min = float(eta_cfg.get("eta_min", 1.0) or 1.0)
                    eta_max = float(eta_cfg.get("eta_max", 5.0) or 5.0)
                    win = int(eta_cfg.get("smooth_window", 31) or 31)
                    win = int(max(3, win // 2 * 2 + 1))  # odd >=3

                    eta0 = np.where(np.isfinite(eta_raw), eta_raw, 1.0).astype(np.float32)
                    eta_s = median_filter(eta0, size=win, mode="nearest").astype(np.float32)
                    eta_s = np.clip(eta_s, eta_min, eta_max)

                    # Apply in-place.
                    out_var *= eta_s[None, :]

                    # Write eta(lambda) to a sidecar FITS.
                    eh = fits.Header()
                    eh["BUNIT"] = "dimensionless"
                    eh["ETAMETH"] = ("sky_MAD/var_pred", "eta(lambda) definition")
                    eh["ETASMO"] = (win, "median filter window")
                    eh["ETAMIN"] = (eta_min, "eta lower bound")
                    eh["ETAMAX"] = (eta_max, "eta upper bound")
                    eh["HISTORY"] = f"Scorpio Pipe {PIPELINE_VERSION}: eta(lambda)"
                    fits.PrimaryHDU(data=eta_s.astype(np.float32), header=eh).writeto(
                        eta_fits, overwrite=True
                    )

                    eta_meta.update(
                        {
                            "applied": True,
                            "eta_min": float(eta_min),
                            "eta_max": float(eta_max),
                            "smooth_window": int(win),
                            "eta_median": float(np.nanmedian(eta_raw))
                            if np.any(np.isfinite(eta_raw))
                            else None,
                            "frac_eta_lt_1": (
                                float(np.mean(eta_raw[np.isfinite(eta_raw)] < 1.0))
                                if np.any(np.isfinite(eta_raw))
                                else None
                            ),
                            "eta_fits": str(eta_fits),
                        }
                    )
            except Exception as e:
                eta_meta.update({"applied": False, "error": str(e)})

        report["eta"] = eta_meta

        # Write outputs.
        hdr = hdr0.copy()
        hdr["HISTORY"] = f"Scorpio Pipe {PIPELINE_VERSION}: stack2d"
        hdr["BUNIT"] = out_bunit
        _out_per_sec = "/s" in str(out_bunit).replace(" ", "").lower() or "s-1" in str(
            out_bunit
        ).lower()
        hdr["NORMEXP"] = (
            bool(_out_per_sec),
            "SCI/VAR in per-second units" if _out_per_sec else "SCI/VAR not normalized per second",
        )
        hdr["STKMETH"] = (method, "stack2d combine method")
        if method in {"invvar_huber", "huber", "huber_invvar"}:
            hdr["STKHC"] = (float(huber_c), "Huber c")
        if method in {"invvar_clip", "sigma_clip", "clip"}:
            hdr["STKCLIP"] = (float(clip_sigma), "clip sigma")
        if var_floor is not None:
            hdr["STKVFLO"] = (float(var_floor), "variance floor")
        # Always stamp ETAAPPL so downstream stages can interpret VAR correctly.
        # If eta is not applied, ETAAPPL=False is still an important piece of provenance.
        if report.get("eta", {}).get("applied"):
            hdr["ETAAPPL"] = (True, "eta(lambda) applied to VAR")
            hdr["ETAPATH"] = (eta_fits.name, "eta(lambda) sidecar FITS")
        else:
            hdr["ETAAPPL"] = (False, "eta(lambda) applied to VAR")

        _write_mef(out_fits, out_sci, hdr, var=out_var, mask=out_mask, cov=out_cov)
        # Legacy output for older runs/UI.
        try:
            _write_mef(out_fits_legacy, out_sci, hdr, var=out_var, mask=out_mask, cov=out_cov)
        except Exception:
            pass

        # Navigator-friendly quicklooks.
        if bool(st_cfg.get("save_png", True)):
            # Stacked SCI image
            try:
                from scorpio_pipe.io.quicklook import quicklook_from_mef

                quicklook_from_mef(out_fits, out_stack_png, method="linear", k=4.0)
            except Exception:
                pass

            # Coverage map (COV extension, scaled 0..max)
            try:
                import numpy as _np
                from PIL import Image

                mx = int(_np.nanmax(out_cov)) if out_cov.size else 0
                mx = mx if mx > 0 else 1
                img = (_np.clip(out_cov, 0, mx) / float(mx) * 255.0).astype("uint8")
                Image.fromarray(img).save(out_cov_png)
            except Exception:
                pass

            # Coverage histogram (useful for diagnosing drizzle gaps)
            try:
                import matplotlib.pyplot as plt

                with mpl_style():
                    fig = plt.figure(figsize=(6.0, 3.6))
                    ax = fig.add_subplot(111)
                    ax.hist(out_cov.ravel(), bins=range(int(out_cov.max()) + 2))
                    ax.set_xlabel("Coverage (N exposures)")
                    ax.set_ylabel("Pixels")
                    fig.tight_layout()
                    fig.savefig(out_cov_hist_png)
                    plt.close(fig)
            except Exception:
                pass

            # eta(lambda) plot
            if eta_fits.exists():
                try:
                    from astropy.io import fits
                    import matplotlib.pyplot as plt
                    import numpy as _np

                    with fits.open(eta_fits, memmap=False) as _hdul:
                        _eta = _np.asarray(_hdul[0].data, dtype=float)
                    with mpl_style():
                        fig = plt.figure(figsize=(6.0, 3.6))
                        ax = fig.add_subplot(111)
                        ax.plot(_eta)
                        ax.set_xlabel("Lambda index")
                        ax.set_ylabel("eta")
                        fig.tight_layout()
                        fig.savefig(out_eta_png)
                        plt.close(fig)
                except Exception:
                    pass

        report.update(
            {
                "ok": True,
                "status": "ok",
                "shape": [int(ny), int(nx)],
                "n_inputs": len(files),
                "outputs": {
                    "stack2d_fits": str(out_fits),
                    "stacked2d_fits": str(out_fits_legacy) if out_fits_legacy.exists() else None,
                    "stack2d_png": str(out_stack_png) if out_stack_png.exists() else None,
                    "coverage_png": str(out_cov_png) if out_cov_png.exists() else None,
                    "coverage_hist_png": str(out_cov_hist_png) if out_cov_hist_png.exists() else None,
                    "eta_lambda_fits": str(eta_fits) if eta_fits.exists() else None,
                    "eta_lambda_png": str(out_eta_png) if out_eta_png.exists() else None,
                },
                "metrics": {
                    "cov_nonzero": float(np.count_nonzero(out_cov > 0) / float(out_cov.size))
                    if out_cov.size
                    else 0.0,
                    "suppressed_pixels_fraction": float(suppressed_any / suppressed_total)
                    if suppressed_total
                    else 0.0,
                },
            }
        )

        # Backward-friendly top-level shortcuts (used by pipeline_runner / UI).
        report["stack2d"] = str(out_fits)
        report["stacked2d"] = (
            str(out_fits_legacy) if out_fits_legacy.exists() else str(out_fits)
        )
        if out_cov_png.exists():
            report["coverage_png"] = str(out_cov_png)
        if out_cov_hist_png.exists():
            report["coverage_hist_png"] = str(out_cov_hist_png)
        if out_stack_png.exists():
            report["stack2d_png"] = str(out_stack_png)
        if eta_fits.exists():
            report["eta_lambda_fits"] = str(eta_fits)
        if out_eta_png.exists():
            report["eta_lambda_png"] = str(out_eta_png)

        # Stage-level QC flags (used by the QC gate)
        try:
            from scorpio_pipe.qc.flags import make_flag, max_severity
            from scorpio_pipe.qc_thresholds import compute_thresholds

            thr, thr_meta = compute_thresholds(cfg)
            stage_flags: list[dict[str, Any]] = []

            cov_nonzero = float(report.get("metrics", {}).get("cov_nonzero", 0.0) or 0.0)
            rejected_pix_frac = float(np.mean((out_mask & MASK_ROBUST_REJECTED) != 0)) if out_mask.size else 0.0

            if cov_nonzero <= float(thr.linearize_cov_nonzero_bad):
                stage_flags.append(
                    make_flag(
                        "COVERAGE_LOW",
                        "ERROR",
                        "Stack2D coverage nonzero fraction is critically low",
                        value=cov_nonzero,
                        warn_le=float(thr.linearize_cov_nonzero_warn),
                        bad_le=float(thr.linearize_cov_nonzero_bad),
                    )
                )
            elif cov_nonzero <= float(thr.linearize_cov_nonzero_warn):
                stage_flags.append(
                    make_flag(
                        "COVERAGE_LOW",
                        "WARN",
                        "Stack2D coverage nonzero fraction is low",
                        value=cov_nonzero,
                        warn_le=float(thr.linearize_cov_nonzero_warn),
                        bad_le=float(thr.linearize_cov_nonzero_bad),
                    )
                )

            if rejected_pix_frac >= float(thr.linearize_rejected_frac_bad):
                stage_flags.append(
                    make_flag(
                        "REJECTED_FRAC_HIGH",
                        "ERROR",
                        "Stack2D rejected/suppressed pixel fraction is too high",
                        value=rejected_pix_frac,
                        warn_ge=float(thr.linearize_rejected_frac_warn),
                        bad_ge=float(thr.linearize_rejected_frac_bad),
                    )
                )
            elif rejected_pix_frac >= float(thr.linearize_rejected_frac_warn):
                stage_flags.append(
                    make_flag(
                        "REJECTED_FRAC_HIGH",
                        "WARN",
                        "Stack2D rejected/suppressed pixel fraction is high",
                        value=rejected_pix_frac,
                        warn_ge=float(thr.linearize_rejected_frac_warn),
                        bad_ge=float(thr.linearize_rejected_frac_bad),
                    )
                )

            # Coverage gaps: columns with zero coverage (useful to diagnose poor dither/alignment)
            try:
                if out_cov.size:
                    col_max = np.max(out_cov, axis=0)
                    gap_frac = float(np.mean(col_max <= 0))
                else:
                    gap_frac = 0.0
                if gap_frac >= 0.05:
                    stage_flags.append(
                        make_flag(
                            "COVERAGE_GAPS",
                            "ERROR",
                            "Stack2D has substantial zero-coverage wavelength gaps",
                            value=gap_frac,
                            warn_ge=0.01,
                            bad_ge=0.05,
                        )
                    )
                elif gap_frac >= 0.01:
                    stage_flags.append(
                        make_flag(
                            "COVERAGE_GAPS",
                            "WARN",
                            "Stack2D has wavelength gaps with zero coverage",
                            value=gap_frac,
                            warn_ge=0.01,
                            bad_ge=0.05,
                        )
                    )
            except Exception:
                gap_frac = None

            # ETA sanity check
            try:
                eta = report.get("eta") if isinstance(report.get("eta"), dict) else {}
                if eta.get("applied"):
                    mn = eta.get("eta_min")
                    mx = eta.get("eta_max")
                    med = eta.get("eta_median")
                    if mn is not None and mx is not None:
                        mnf = float(mn)
                        mxf = float(mx)
                        medf = float(med) if med is not None else None
                        if mnf < 0.2 or mxf > 5.0 or (medf is not None and (medf < 0.3 or medf > 3.0)):
                            stage_flags.append(
                                make_flag(
                                    "ETA_ANOMALY",
                                    "WARN",
                                    "eta(lambda) looks suspicious; check VAR model or sky noise estimates",
                                    value=float(medf) if medf is not None else None,
                                )
                            )
            except Exception:
                pass

            report["qc"] = {
                "flags": stage_flags,
                "max_severity": max_severity(stage_flags),
                "thresholds": thr.to_dict(),
                "thresholds_meta": thr_meta,
                "metrics": {
                    "cov_nonzero": cov_nonzero,
                    "rejected_pixels_fraction": rejected_pix_frac,
                    "gap_fraction": gap_frac,
                    "eta_median": (eta.get("eta_median") if isinstance(eta, dict) else None),
                },
            }

            _sev = str(report.get("qc", {}).get("max_severity", "OK") or "OK").upper()
            report["status"] = (
                "ok" if _sev in {"OK", "INFO"} else ("warn" if _sev == "WARN" else "fail")
            )
            report["ok"] = bool(report["status"] != "fail")
        except Exception:
            pass

    except Exception as e:
        report["ok"] = False
        report["status"] = "fail"
        report["error"] = str(e)
        raise

    finally:
        # Always write a done report (even on failure), then close HDUs.
        report["finished_at_unix"] = float(time.time())
        report["runtime_s"] = float(report["finished_at_unix"] - t0)
        try:
            stack_done.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass
        try:
            stack2d_done.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass
        try:
            done_json.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass
        for h in hduls:
            try:
                h.close()
            except Exception:
                pass

    return report
