from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from scorpio_pipe.fits_utils import read_image_smart

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class CosmicsSummary:
    kind: str
    n_frames: int
    k: float
    replaced_pixels: int
    replaced_fraction: float
    per_frame_fraction: list[float]
    outputs: dict[str, str]


def _as_path(x: Any) -> Path:
    return x if isinstance(x, Path) else Path(str(x))


def _load_cfg_any(cfg: Any) -> dict[str, Any]:
    """Normalize config input (path/dict/RunContext) into a config dict."""
    from scorpio_pipe.config import load_config_any

    return load_config_any(cfg)


def _resolve_path(p: Path, *, data_dir: Path, work_dir: Path, base_dir: Path) -> Path:
    """Resolve a possibly-relative path.

    Preference order is chosen to match user expectation:
    1) data_dir (raw frames)
    2) work_dir (derived products)
    3) base_dir (config dir)
    """
    if p.is_absolute():
        return p
    for root in (data_dir, work_dir, base_dir):
        cand = (root / p).resolve()
        if cand.exists():
            return cand
    # last resort: resolve against work_dir even if it doesn't exist yet
    return (work_dir / p).resolve()


def _load_superbias(work_dir: Path) -> np.ndarray | None:
    # New layout (v5+): work_dir/calibs/*.fits, but keep legacy fallback too.
    for rel in (Path("calibs") / "superbias.fits", Path("calib") / "superbias.fits"):
        p = (work_dir / rel)
        if not p.is_file():
            continue
        try:
            return fits.getdata(p, memmap=False).astype(np.float32)
        except Exception:
            continue
    return None


def _robust_mad(x: np.ndarray, axis: int = 0) -> np.ndarray:
    med = np.median(x, axis=axis)
    mad = np.median(np.abs(x - np.expand_dims(med, axis=axis)), axis=axis)
    return mad


def _robust_sigma_mad_1d(x: np.ndarray) -> float:
    """Robust sigma estimate from MAD (1D flatten)."""

    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def _boxcar_mean2d(img: np.ndarray, r: int) -> np.ndarray:
    """Fast boxcar mean using an integral image (no SciPy).

    Important: must preserve the input shape.

    The integral-image inclusion/exclusion formula requires a leading
    row/column of zeros. Without it, the output becomes (ny-1, nx-1) and
    downstream broadcasting fails.
    """

    if r <= 0:
        return np.asarray(img, dtype=np.float32)

    a = np.asarray(img, dtype=np.float32)
    pad = int(r)
    ap = np.pad(a, ((pad, pad), (pad, pad)), mode="reflect")

    # Integral image with a leading 0 row/col.
    s = np.pad(ap, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
    s = s.cumsum(axis=0).cumsum(axis=1)

    k = 2 * pad + 1
    total = (
        s[k:, k:]
        - s[:-k, k:]
        - s[k:, :-k]
        + s[:-k, :-k]
    )
    # `total` now has the same shape as the original `a`.
    return (total / float(k * k)).astype(np.float32)


def _boxcar_mean2d_masked(img: np.ndarray, mask_bad: np.ndarray, r: int) -> np.ndarray:
    """Fast boxcar mean excluding masked pixels.

    Parameters
    ----------
    img
        Image.
    mask_bad
        Boolean mask where True means "exclude".
    r
        Radius of the square window.

    Notes
    -----
    - Uses integral images, no SciPy.
    - Returns an array with the same shape as `img`.
    """

    if r <= 0:
        return np.asarray(img, dtype=np.float32)
    a = np.asarray(img, dtype=np.float32)
    bad = np.asarray(mask_bad, dtype=bool)
    good = (~bad).astype(np.float32)

    pad = int(r)
    ap = np.pad(a * good, ((pad, pad), (pad, pad)), mode="reflect")
    gp = np.pad(good, ((pad, pad), (pad, pad)), mode="reflect")

    # Integral images with leading 0 row/col.
    s = np.pad(ap, ((1, 0), (1, 0)), mode="constant", constant_values=0.0).cumsum(axis=0).cumsum(axis=1)
    c = np.pad(gp, ((1, 0), (1, 0)), mode="constant", constant_values=0.0).cumsum(axis=0).cumsum(axis=1)

    k = 2 * pad + 1
    total = s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]
    count = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    mean = total / np.maximum(count, 1.0)
    return mean.astype(np.float32)


def _laplacian4(img: np.ndarray) -> np.ndarray:
    """4-neighbor Laplacian with reflect padding."""
    a = np.asarray(img, dtype=np.float32)
    ip = np.pad(a, ((1, 1), (1, 1)), mode="reflect")
    return (
        -4.0 * ip[1:-1, 1:-1]
        + ip[:-2, 1:-1]
        + ip[2:, 1:-1]
        + ip[1:-1, :-2]
        + ip[1:-1, 2:]
    ).astype(np.float32)


def _dilate_mask(mask: np.ndarray, r: int = 1) -> np.ndarray:
    """Binary dilation with a (2r+1)x(2r+1) square structuring element."""

    if r <= 0:
        return mask
    m = mask.astype(bool)
    h, w = m.shape
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            y0_src = max(0, -dy)
            y1_src = min(h, h - dy)
            x0_src = max(0, -dx)
            x1_src = min(w, w - dx)
            y0_dst = y0_src + dy
            y1_dst = y1_src + dy
            x0_dst = x0_src + dx
            x1_dst = x1_src + dx
            out[y0_dst:y1_dst, x0_dst:x1_dst] |= m[y0_src:y1_src, x0_src:x1_src]
    return out


def _two_frame_diff_clean(
    paths: list[Path],
    *,
    out_dir: Path,
    superbias: np.ndarray | None,
    k: float,
    bias_subtract: bool,
    save_png: bool,
    save_mask_fits: bool,
    dilate: int = 1,
    local_r: int = 2,
    k2_scale: float = 0.8,
    k2_min: float = 5.0,
    thr_local_a: float = 4.0,
    thr_local_b: float = 2.5,
) -> CosmicsSummary:
    """Cosmic cleaning specialized for N=2 exposures.

    With only two frames, MAD-around-median is mathematically unable to flag
    outliers (the MAD scales with the difference itself). Instead we build a
    per-pixel difference frame and flag spikes that are both globally and
    locally deviant; then we replace them by the value from the other exposure.
    """

    if len(paths) != 2:
        raise ValueError("_two_frame_diff_clean expects exactly 2 frames")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clean").mkdir(parents=True, exist_ok=True)
    if save_png:
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    if save_mask_fits:
        (out_dir / "masks_fits").mkdir(parents=True, exist_ok=True)

    datas: list[np.ndarray] = []
    headers: list[fits.Header] = []
    names: list[str] = []

    for p in paths:
        data, hdr, _info = read_image_smart(p, memmap="auto", dtype=np.float32)
        datas.append(data)
        headers.append(hdr)
        names.append(Path(p).stem)

    # Some nights contain frames with a 1-pixel geometry mismatch (e.g. last
    # row/column missing in one exposure). Downstream stages (stack/linearize)
    # require consistent geometry, so we crop to the common overlap here.
    shapes = [d.shape for d in datas]
    ny = min(s[0] for s in shapes)
    nx = min(s[1] for s in shapes)
    if any(s != (ny, nx) for s in shapes):
        log.warning("Cosmics(two_frame_diff): input shapes differ %s; cropping to (%d, %d)", shapes, ny, nx)
        datas = [d[:ny, :nx] for d in datas]
        for hdr, s in zip(headers, shapes):
            if s != (ny, nx):
                hdr["CROPY"] = (ny, "Cropped to common height")
                hdr["CROPX"] = (nx, "Cropped to common width")
                hdr["HISTORY"] = f"scorpio_pipe cosmics: cropped from {s} to {(ny, nx)} for geometry match"

    # Apply superbias after cropping if requested.
    if bias_subtract and superbias is not None:
        if superbias.shape[0] >= ny and superbias.shape[1] >= nx:
            sb = superbias[:ny, :nx]
            datas = [d - sb for d in datas]
            for hdr in headers:
                hdr["BIASSUB"] = (True, "Superbias subtracted")
                hdr["HISTORY"] = "scorpio_pipe cosmics: bias subtracted using superbias.fits"
        else:
            log.warning(
                "Cosmics(two_frame_diff): superbias shape %s is smaller than cropped science (%d, %d); skipping bias subtraction",
                getattr(superbias, "shape", None),
                ny,
                nx,
            )

    a, b = datas
    diff = (a - b).astype(np.float32)
    absdiff = np.abs(diff)

    sigma = _robust_sigma_mad_1d(diff)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(diff)) if diff.size else 0.0

    # local scale proxy: (2*local_r+1)^2 mean of |diff|
    loc = _boxcar_mean2d(absdiff, r=int(local_r))

    k2 = max(float(k2_min), float(k2_scale) * float(k))
    thr_global = float(k2 * sigma)
    # require also locally deviant (spike-like)
    thr_local = (float(thr_local_a) * loc + float(thr_local_b) * sigma).astype(np.float32)

    cand = absdiff > np.maximum(thr_global, thr_local)

    # Assign to the higher frame (sign of diff)
    m0 = cand & (diff > 0)
    m1 = cand & (diff < 0)

    # Expand masks slightly to cover halo pixels
    m0 = _dilate_mask(m0, r=int(dilate))
    m1 = _dilate_mask(m1, r=int(dilate))

    cleaned0 = a.copy()
    cleaned1 = b.copy()
    cleaned0[m0] = b[m0]
    cleaned1[m1] = a[m1]

    # Summaries
    replaced_pixels = int(m0.sum() + m1.sum())
    total_pix = int(a.size + b.size)
    replaced_fraction = float(replaced_pixels) / float(total_pix) if total_pix else 0.0
    per_frame_fraction = [float(m0.mean()), float(m1.mean())]

    out_files: dict[str, str] = {}
    for i, (name, hdr, img, m) in enumerate(
        [(names[0], headers[0], cleaned0, m0), (names[1], headers[1], cleaned1, m1)]
    ):
        out_f = out_dir / "clean" / f"{name}_clean.fits"
        h = hdr.copy()
        h["COSMCLEA"] = (True, "Cosmics cleaned")
        h["COSM_K"] = (float(k), "Threshold control")
        h["COSM_MD"] = ("two_frame_diff", "Cosmics method")
        h["HISTORY"] = "scorpio_pipe cosmics: replaced spikes using the other exposure"
        fits.writeto(out_f, img.astype(np.float32), header=h, overwrite=True)
        out_files[name] = str(out_f)

        if save_png:
            _save_png(out_dir / "masks" / f"{name}_mask.png", m.astype(np.uint8), title=f"Cosmic mask: {name}")

        if save_mask_fits:
            mf = out_dir / "masks_fits" / f"{name}_mask.fits"
            # Store as uint8 to avoid FITS unsigned-int scaling keywords (BZERO/BSCALE)
            # that may break strict memmap readers.
            fits.writeto(mf, m.astype(np.uint8), overwrite=True)

    sum_excl = (a * (~m0) + b * (~m1)).astype(np.float32)
    cov = ((~m0).astype(np.int16) + (~m1).astype(np.int16)).astype(np.int16)
    sum_f = out_dir / "sum_excl_cosmics.fits"
    cov_f = out_dir / "coverage.fits"
    fits.writeto(sum_f, sum_excl, overwrite=True)
    fits.writeto(cov_f, cov, overwrite=True)
    if save_png:
        _save_png(out_dir / "sum_excl_cosmics.png", sum_excl, title="Sum (cosmics excluded)")
        _save_png(out_dir / "coverage.png", cov, title="Coverage (non-cosmic count)")

    outputs = {
        "clean_dir": str((out_dir / "clean").resolve()),
        "sum_excl_fits": str(sum_f.resolve()),
        "coverage_fits": str(cov_f.resolve()),
        "sigma_diff": float(sigma),
    }
    if save_png:
        outputs.update(
            {
                "sum_excl_png": str((out_dir / "sum_excl_cosmics.png").resolve()),
                "coverage_png": str((out_dir / "coverage.png").resolve()),
                "masks_dir": str((out_dir / "masks").resolve()),
            }
        )
    if save_mask_fits:
        outputs["masks_fits_dir"] = str((out_dir / "masks_fits").resolve())

    return CosmicsSummary(
        kind="",
        n_frames=2,
        k=float(k),
        replaced_pixels=replaced_pixels,
        replaced_fraction=replaced_fraction,
        per_frame_fraction=per_frame_fraction,
        outputs=outputs,
    )


def _single_frame_laplacian_clean(
    path: Path,
    *,
    out_dir: Path,
    superbias: np.ndarray | None,
    k: float,
    bias_subtract: bool,
    save_png: bool,
    save_mask_fits: bool,
    dilate: int = 1,
    local_r: int = 2,
    k_scale: float = 0.8,
    k_min: float = 5.0,
) -> CosmicsSummary:
    """Single-frame fallback using a Laplacian high-pass detector."""

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clean").mkdir(parents=True, exist_ok=True)
    if save_png:
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    if save_mask_fits:
        (out_dir / "masks_fits").mkdir(parents=True, exist_ok=True)

    img, hdr, _info = read_image_smart(path, memmap="auto", dtype=np.float32)

    if bias_subtract and superbias is not None and superbias.shape == img.shape:
        img = img - superbias
        hdr["BIASSUB"] = (True, "Superbias subtracted")

    # Laplacian (4-neighbor) with reflect padding
    lap = _laplacian4(img)

    sigma = _robust_sigma_mad_1d(lap)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(lap)) if lap.size else 0.0

    # Local baseline for replacement: (2*local_r+1)^2 mean
    loc_mean = _boxcar_mean2d(img, r=int(local_r))

    # Candidate pixels: strong high-frequency outliers.
    thr = max(float(k_min), float(k_scale) * float(k)) * sigma
    m = np.abs(lap) > float(thr)
    m = _dilate_mask(m, r=int(dilate))

    cleaned = img.copy()
    cleaned[m] = loc_mean[m]

    replaced_pixels = int(m.sum())
    replaced_fraction = float(m.mean()) if m.size else 0.0

    name = Path(path).stem
    out_f = out_dir / "clean" / f"{name}_clean.fits"
    h = hdr.copy()
    h["COSMCLEA"] = (True, "Cosmics cleaned")
    h["COSM_K"] = (float(k), "Threshold control")
    h["COSM_MD"] = ("laplacian", "Cosmics method")
    h["HISTORY"] = "scorpio_pipe cosmics: laplacian detector + local mean replacement"
    fits.writeto(out_f, cleaned.astype(np.float32), header=h, overwrite=True)

    if save_png:
        _save_png(out_dir / "masks" / f"{name}_mask.png", m.astype(np.uint8), title=f"Cosmic mask: {name}")
    if save_mask_fits:
        # Store as uint8 to avoid FITS unsigned-int scaling keywords (BZERO/BSCALE)
        # that may break strict memmap readers.
        fits.writeto(out_dir / "masks_fits" / f"{name}_mask.fits", m.astype(np.uint8), overwrite=True)

    # Reference QC products for single frame
    sum_excl = cleaned * (~m)
    cov = (~m).astype(np.int16)
    fits.writeto(out_dir / "sum_excl_cosmics.fits", sum_excl.astype(np.float32), header=h, overwrite=True)
    fits.writeto(out_dir / "coverage.fits", cov, overwrite=True)
    if save_png:
        _save_png(out_dir / "sum_excl_cosmics.png", sum_excl, title="Sum excl. cosmics")
        _save_png(out_dir / "coverage.png", cov, title="Coverage")

    outputs = {
        "clean_dir": str((out_dir / "clean").resolve()),
        "masks_dir": str((out_dir / "masks").resolve()),
        "masks_fits_dir": str((out_dir / "masks_fits").resolve()),
        "sum_excl_fits": str((out_dir / "sum_excl_cosmics.fits").resolve()),
        "coverage_fits": str((out_dir / "coverage.fits").resolve()),
        "sum_excl_png": str((out_dir / "sum_excl_cosmics.png").resolve()) if save_png else None,
        "coverage_png": str((out_dir / "coverage.png").resolve()) if save_png else None,
    }
    return CosmicsSummary(
        kind="",
        n_frames=1,
        k=float(k),
        replaced_pixels=replaced_pixels,
        replaced_fraction=replaced_fraction,
        per_frame_fraction=[replaced_fraction],
        outputs=outputs,
    )



def _single_frame_lacosmic_clean(
    path: Path,
    *,
    out_dir: Path,
    superbias: np.ndarray | None,
    bias_subtract: bool,
    save_png: bool,
    save_mask_fits: bool,
    niter: int = 4,
    sigclip: float = 4.5,
    sigfrac: float = 0.3,
    objlim: float = 5.0,
    replace_r: int = 2,
    dilate: int = 1,
    protect_lines: bool = True,
    protect_half_x: int = 6,
    protect_half_y: int = 1,
    protect_k: float = 5.0,
) -> CosmicsSummary:
    """Single-frame L.A.Cosmic-like detector (van Dokkum 2001 style).

    This is a lightweight, SciPy-free implementation meant to be conservative
    on long-slit data: prefer missing a few cosmics to eating real narrow
    emission/sky features.

    If `astroscrappy` is available, it will be used as the reference engine.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clean").mkdir(parents=True, exist_ok=True)
    if save_png:
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    if save_mask_fits:
        (out_dir / "masks_fits").mkdir(parents=True, exist_ok=True)

    img, hdr, _info = read_image_smart(path, memmap="auto", dtype=np.float32)
    if bias_subtract and superbias is not None and superbias.shape == img.shape:
        img = img - superbias
        hdr["BIASSUB"] = (True, "Superbias subtracted")

    # Optional safeguard for long-slit: don't classify strong vertical sky/arc
    # features as cosmics. This is intentionally conservative: it protects
    # line cores, but leaves most of the continuum available for cleaning.
    line_protect: np.ndarray | None = None
    if bool(protect_lines):
        rx = img - _boxcar_mean2d(img, r=int(max(1, protect_half_x)))
        ry = img - _boxcar_mean2d(img, r=int(max(1, protect_half_y)))
        sx = _robust_sigma_mad_1d(rx)
        sy = _robust_sigma_mad_1d(ry)
        if np.isfinite(sx) and np.isfinite(sy) and sx > 0 and sy > 0:
            # Sky/arc lines: sharp in X, but extended/consistent in Y.
            line_protect = (np.abs(rx) > float(protect_k) * sx) & (np.abs(ry) < 0.8 * float(protect_k) * sy)

    # Try astroscrappy first (closest to van Dokkum L.A.Cosmic).
    try:
        import astroscrappy  # type: ignore

        crmask, cleaned = astroscrappy.detect_cosmics(
            img,
            sigclip=float(sigclip),
            sigfrac=float(sigfrac),
            objlim=float(objlim),
            niter=int(max(1, niter)),
            verbose=False,
        )
        m = np.asarray(crmask, dtype=bool)
        if line_protect is not None:
            m = m & (~line_protect)
        cleaned = np.asarray(cleaned, dtype=np.float32)
        engine = "astroscrappy"
        sigma_lap = float("nan")
    except Exception:
        # Fallback: iterative Laplacian threshold + object-likeness discriminator.
        a = img.astype(np.float32)
        m = np.zeros_like(a, dtype=bool)
        sigma_lap = float("nan")

        # Use two-scale "fine structure" proxy to reject object/line-like features.
        # (In original L.A.Cosmic this is based on median filters.)
        def _fine_structure(x: np.ndarray) -> np.ndarray:
            x1 = _boxcar_mean2d(x, r=1)
            x3 = _boxcar_mean2d(x, r=3)
            return (x1 - x3).astype(np.float32)

        thr_seed = float(sigclip)
        thr_grow = float(sigclip) * float(sigfrac)
        for _ in range(int(max(1, niter))):
            # High-frequency component
            smooth = _boxcar_mean2d(a, r=2)
            resid = (a - smooth).astype(np.float32)
            lap = _laplacian4(resid)

            sigma = _robust_sigma_mad_1d(lap)
            if not np.isfinite(sigma) or sigma <= 0:
                sigma = float(np.std(lap)) if lap.size else 0.0
            sigma_lap = float(sigma)
            if sigma <= 0:
                break

            fine = _fine_structure(a)
            # Object-likeness: cosmics are sharp in Laplacian but not supported
            # by broader fine-structure.
            denom = np.maximum(np.abs(fine), 1e-3 * sigma).astype(np.float32)
            ratio = np.abs(lap) / denom

            seed = (np.abs(lap) > (thr_seed * sigma)) & (ratio > float(objlim))
            if line_protect is not None:
                seed = seed & (~line_protect)
            grow = (np.abs(lap) > (thr_grow * sigma)) & (ratio > float(objlim))
            if line_protect is not None:
                grow = grow & (~line_protect)

            # Grow around existing mask and new seed.
            m_new = m | seed
            if np.any(m_new):
                m_new = m_new | (grow & _dilate_mask(m_new, r=1))
            m = m_new

            # Replace masked pixels conservatively using local mean on unmasked.
            repl = _boxcar_mean2d_masked(a, m, r=int(replace_r))
            a = a.copy()
            a[m] = repl[m]

        cleaned = a
        engine = "fallback"

    # Final small dilation to cover halos.
    m = _dilate_mask(m, r=int(max(0, dilate)))

    replaced_pixels = int(m.sum())
    replaced_fraction = float(m.mean()) if m.size else 0.0

    name = Path(path).stem
    out_f = out_dir / "clean" / f"{name}_clean.fits"
    h = hdr.copy()
    h["COSMCLEA"] = (True, "Cosmics cleaned")
    h["COSM_MD"] = ("la_cosmic", "Cosmics method")
    h["COSM_ENG"] = (str(engine), "L.A.Cosmic engine")
    h["COSM_IT"] = (int(max(1, niter)), "L.A.Cosmic iterations")
    h["COSM_SC"] = (float(sigclip), "L.A.Cosmic sigclip")
    h["COSM_SF"] = (float(sigfrac), "L.A.Cosmic sigfrac")
    h["COSM_OL"] = (float(objlim), "L.A.Cosmic objlim")
    h["HISTORY"] = "scorpio_pipe cosmics: la_cosmic (van Dokkum-like)"
    fits.writeto(out_f, cleaned.astype(np.float32), header=h, overwrite=True)

    if save_png:
        _save_png(out_dir / "masks" / f"{name}_mask.png", m.astype(np.uint8), title=f"Cosmic mask: {name}")
    if save_mask_fits:
        # Store as uint8 to avoid BZERO/BSCALE scaling keywords.
        fits.writeto(out_dir / "masks_fits" / f"{name}_mask.fits", m.astype(np.uint8), overwrite=True)

    sum_f = out_dir / "sum_excl_cosmics.fits"
    cov_f = out_dir / "coverage.fits"
    fits.writeto(sum_f, (img * (~m)).astype(np.float32), overwrite=True)
    fits.writeto(cov_f, (~m).astype(np.int16), overwrite=True)

    outputs = {
        "clean_dir": str((out_dir / "clean").resolve()),
        "sum_excl_fits": str(sum_f.resolve()),
        "coverage_fits": str(cov_f.resolve()),
        "engine": str(engine),
        "sigma_lap": float(sigma_lap) if np.isfinite(sigma_lap) else None,
    }
    if save_png:
        outputs["masks_dir"] = str((out_dir / "masks").resolve())
    if save_mask_fits:
        outputs["masks_fits_dir"] = str((out_dir / "masks_fits").resolve())

    return CosmicsSummary(
        kind="",
        n_frames=1,
        k=float(sigclip),
        replaced_pixels=replaced_pixels,
        replaced_fraction=float(replaced_fraction),
        per_frame_fraction=[float(replaced_fraction)],
        outputs=outputs,
    )


def _save_png(path: Path, arr: np.ndarray, title: str | None = None) -> None:
    # Optional visualization; avoid heavy dependencies.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 5), dpi=160)
    ax = fig.add_subplot(111)
    im = ax.imshow(arr, origin="lower", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)



def _mask_u8_to_bool(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.uint8) > 0


def _pad_mask_to_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Pad/crop mask to `shape` placing existing pixels at (0,0)."""
    m = np.asarray(mask, dtype=bool)
    ny, nx = shape
    out = np.zeros((ny, nx), dtype=bool)
    sy = min(ny, m.shape[0])
    sx = min(nx, m.shape[1])
    out[:sy, :sx] = m[:sy, :sx]
    return out


def _apply_manual_masks(
    kind_out: Path,
    *,
    replace_r: int,
    save_png: bool,
) -> tuple[int, float, list[float]]:
    """Merge manual masks into the final mask products and re-apply inpainting.

    This is a post-processing step that runs after any automatic method.

    It ensures that:
      - manual_masks_fits/<base>_manual_mask.fits are persisted
      - auto masks are snapshotted to auto_masks_fits/<base>_auto_mask.fits
      - masks_fits/<base>_mask.fits becomes AUTO|MANUAL
      - clean/<base>_clean.fits gets manual pixels replaced by local mean
      - sum_excl_cosmics / coverage products match the final masks

    Returns
    -------
    replaced_pixels_total, replaced_fraction_total, per_frame_fraction
    """

    clean_dir = kind_out / 'clean'
    masks_fits = kind_out / 'masks_fits'
    masks_png = kind_out / 'masks'
    auto_masks = kind_out / 'auto_masks_fits'
    manual_masks = kind_out / 'manual_masks_fits'

    auto_masks.mkdir(parents=True, exist_ok=True)
    manual_masks.mkdir(parents=True, exist_ok=True)
    masks_fits.mkdir(parents=True, exist_ok=True)
    if save_png:
        masks_png.mkdir(parents=True, exist_ok=True)

    clean_files = sorted(clean_dir.glob('*_clean.fits'))
    if not clean_files:
        return 0, 0.0, []

    final_masks: list[np.ndarray] = []
    clean_datas: list[np.ndarray] = []

    for cf in clean_files:
        base = cf.stem
        if base.endswith('_clean'):
            base = base[:-6]

        with fits.open(cf, memmap=False) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float32)
            hdr = hdul[0].header.copy()

        auto_path = masks_fits / f'{base}_mask.fits'
        if auto_path.exists():
            with fits.open(auto_path, memmap=False) as h:
                auto_m = _mask_u8_to_bool(h[0].data)
        else:
            auto_m = np.zeros(data.shape, dtype=bool)

        # snapshot current auto mask for GUI baselining
        fits.writeto(auto_masks / f'{base}_auto_mask.fits', np.asarray(auto_m, dtype=np.uint8), overwrite=True)

        man_path = manual_masks / f'{base}_manual_mask.fits'
        if man_path.exists():
            with fits.open(man_path, memmap=False) as h:
                man_m = _mask_u8_to_bool(h[0].data)
        else:
            man_m = np.zeros(data.shape, dtype=bool)

        auto_m = _pad_mask_to_shape(auto_m, data.shape)
        man_m = _pad_mask_to_shape(man_m, data.shape)

        final_m = auto_m | man_m

        # Apply manual inpainting on top of the auto-cleaned image.
        if np.any(man_m):
            repl = _boxcar_mean2d_masked(data, final_m, r=int(max(0, replace_r)))
            data = data.copy()
            data[man_m] = repl[man_m]
            hdr['MANCR'] = (True, 'Manual cosmics edits applied')
            hdr['MANCRN'] = (int(man_m.sum()), 'Manual masked pixels')
            hdr['HISTORY'] = 'scorpio_pipe cosmics: manual inpainting applied'

        fits.writeto(cf, data.astype(np.float32), header=hdr, overwrite=True)
        fits.writeto(masks_fits / f'{base}_mask.fits', final_m.astype(np.uint8), overwrite=True)
        if save_png:
            _save_png(masks_png / f'{base}_mask.png', final_m.astype(np.uint8), title=f'Cosmic mask: {base}')

        final_masks.append(final_m)
        clean_datas.append(data)

    # Reference products based on final masks and current clean frames.
    stack = np.stack(clean_datas, axis=0)
    mstack = np.stack(final_masks, axis=0)
    sum_excl = np.sum(stack * (~mstack), axis=0)
    cov = np.sum(~mstack, axis=0).astype(np.int16)

    sum_f = kind_out / 'sum_excl_cosmics.fits'
    cov_f = kind_out / 'coverage.fits'
    fits.writeto(sum_f, sum_excl.astype(np.float32), overwrite=True)
    fits.writeto(cov_f, cov, overwrite=True)
    if save_png:
        _save_png(kind_out / 'sum_excl_cosmics.png', sum_excl, title='Sum (cosmics excluded)')
        _save_png(kind_out / 'coverage.png', cov, title='Coverage (non-cosmic count)')

    replaced_pixels = int(mstack.sum())
    n_pix_total = int(mstack.size)
    replaced_fraction = float(replaced_pixels) / float(n_pix_total) if n_pix_total else 0.0
    per_frame_fraction = [float(m.mean()) for m in final_masks]

    return replaced_pixels, replaced_fraction, per_frame_fraction

def _stack_mad_clean(
    paths: list[Path],
    *,
    out_dir: Path,
    superbias: np.ndarray | None,
    k: float,
    mad_scale: float = 1.0,
    min_mad: float = 0.0,
    max_frac_per_frame: float | None = None,
    dilate: int = 0,
    bias_subtract: bool,
    save_png: bool,
    save_mask_fits: bool,
) -> CosmicsSummary:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clean").mkdir(parents=True, exist_ok=True)
    if save_png:
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    if save_mask_fits:
        (out_dir / "masks_fits").mkdir(parents=True, exist_ok=True)

    datas: list[np.ndarray] = []
    headers: list[fits.Header] = []
    names: list[str] = []

    for p in paths:
        data, hdr, _info = read_image_smart(p, memmap="auto", dtype=np.float32)
        datas.append(data)
        headers.append(hdr)
        names.append(p.stem)

    # Harmonize geometry (see comment in _two_frame_diff_clean).
    shapes = [d.shape for d in datas]
    ny = min(s[0] for s in shapes)
    nx = min(s[1] for s in shapes)
    if any(s != (ny, nx) for s in shapes):
        log.warning("Cosmics(stack_mad): input shapes differ %s; cropping to (%d, %d)", shapes, ny, nx)
        datas = [d[:ny, :nx] for d in datas]
        for hdr, s in zip(headers, shapes):
            if s != (ny, nx):
                hdr["CROPY"] = (ny, "Cropped to common height")
                hdr["CROPX"] = (nx, "Cropped to common width")
                hdr["HISTORY"] = f"scorpio_pipe cosmics: cropped from {s} to {(ny, nx)} for geometry match"

    if bias_subtract and superbias is not None:
        if superbias.shape[0] >= ny and superbias.shape[1] >= nx:
            sb = superbias[:ny, :nx]
            datas = [d - sb for d in datas]
            for hdr in headers:
                hdr["BIASSUB"] = (True, "Superbias subtracted")
                hdr["HISTORY"] = "scorpio_pipe cosmics: bias subtracted using superbias.fits"
        else:
            log.warning(
                "Cosmics(stack_mad): superbias shape %s is smaller than cropped science (%d, %d); skipping bias subtraction",
                getattr(superbias, "shape", None),
                ny,
                nx,
            )

    stack = np.stack(datas, axis=0)  # (N, H, W)
    med = np.median(stack, axis=0)
    mad = _robust_mad(stack, axis=0)

    # Protect against zero/too-small MAD (flat pixels) + allow user floor.
    eps = float(np.finfo(np.float32).eps)
    floor = max(eps, float(min_mad))
    mad = np.maximum(mad, floor)

    # Threshold score (dimensionless): |x-med| / (mad_scale*mad)
    ms = max(eps, float(mad_scale))
    score = np.abs(stack - med[None, :, :]) / (ms * mad[None, :, :])
    mask = score > float(k)

    # Optional safety: cap per-frame masked fraction by raising the effective threshold.
    if max_frac_per_frame is not None:
        mf = float(max_frac_per_frame)
        if 0.0 < mf < 1.0:
            for i in range(mask.shape[0]):
                frac = float(mask[i].mean()) if mask[i].size else 0.0
                if frac <= mf:
                    continue
                # Pick a threshold that yields ~mf masked fraction.
                # (This protects against pathological MAD underestimation.)
                thr_i = float(np.quantile(score[i].ravel(), 1.0 - mf))
                thr_i = max(float(k), thr_i)
                mask[i] = score[i] > thr_i

    # Optional dilation to catch halo pixels around a hit.
    if int(dilate) > 0:
        for i in range(mask.shape[0]):
            mask[i] = _dilate_mask(mask[i], r=int(dilate))

    # Replace cosmics by per-pixel median of the stack
    cleaned = np.where(mask, med[None, :, :], stack)

    replaced_pixels = int(mask.sum())
    replaced_fraction = float(replaced_pixels) / float(stack.size)
    per_frame_fraction = [float(mask[i].mean()) for i in range(mask.shape[0])]

    # Write per-frame cleaned files
    out_files: dict[str, str] = {}
    for i, (name, hdr) in enumerate(zip(names, headers)):
        out_f = out_dir / "clean" / f"{name}_clean.fits"
        h = hdr.copy()
        h["COSMCLEA"] = (True, "Cosmics cleaned")
        h["COSM_K"] = (float(k), "MAD threshold multiplier")
        h["COSM_MD"] = ("stack_mad", "Cosmics method")
        h["HISTORY"] = "scorpio_pipe cosmics: replaced cosmic pixels with stack median"
        fits.writeto(out_f, cleaned[i].astype(np.float32), header=h, overwrite=True)
        out_files[name] = str(out_f)

        if save_png:
            # quicklook mask
            mpath = out_dir / "masks" / f"{name}_mask.png"
            _save_png(mpath, mask[i].astype(np.uint8), title=f"Cosmic mask: {name}")

        if save_mask_fits:
            mf = out_dir / "masks_fits" / f"{name}_mask.fits"
            # Store as uint8 to avoid FITS unsigned-int scaling keywords (BZERO/BSCALE)
            # that may break strict memmap readers.
            fits.writeto(mf, mask[i].astype(np.uint8), overwrite=True)

    # Reference products: sum excluding masked pixels + coverage map
    sum_excl = np.sum(stack * (~mask), axis=0)
    cov = np.sum(~mask, axis=0).astype(np.int16)

    sum_f = out_dir / "sum_excl_cosmics.fits"
    cov_f = out_dir / "coverage.fits"
    fits.writeto(sum_f, sum_excl.astype(np.float32), overwrite=True)
    fits.writeto(cov_f, cov, overwrite=True)

    if save_png:
        _save_png(out_dir / "sum_excl_cosmics.png", sum_excl, title="Sum (cosmics excluded)")
        _save_png(out_dir / "coverage.png", cov, title="Coverage (non-cosmic count)")

    outputs = {
        "clean_dir": str((out_dir / "clean").resolve()),
        "sum_excl_fits": str(sum_f.resolve()),
        "coverage_fits": str(cov_f.resolve()),
    }
    if save_png:
        outputs.update(
            {
                "sum_excl_png": str((out_dir / "sum_excl_cosmics.png").resolve()),
                "coverage_png": str((out_dir / "coverage.png").resolve()),
                "masks_dir": str((out_dir / "masks").resolve()),
            }
        )
    if save_mask_fits:
        outputs["masks_fits_dir"] = str((out_dir / "masks_fits").resolve())

    return CosmicsSummary(
        kind="",
        n_frames=len(paths),
        k=float(k),
        replaced_pixels=replaced_pixels,
        replaced_fraction=replaced_fraction,
        per_frame_fraction=per_frame_fraction,
        outputs=outputs,
    )


def clean_cosmics(cfg: Any, *, out_dir: str | Path | None = None) -> Path:
    """Clean cosmics and write a report.

    Default method is `auto`:
      - 1 frame  -> `la_cosmic` (van Dokkum L.A.Cosmic-like)
      - 2 frames -> `two_frame_diff`
      - ≥3       -> `stack_mad`

    `laplacian` remains available as a simple SciPy-free single-frame fallback.

    Outputs:
      work_dir/cosmics/<kind>/clean/*.fits
      work_dir/cosmics/<kind>/summary.json
    """
    cfg = _load_cfg_any(cfg)
    base_dir = Path(str(cfg.get("config_dir", "."))).resolve()
    data_dir = Path(str(cfg.get("data_dir", "."))).expanduser().resolve()

    work_dir = _as_path(cfg.get("work_dir", "work"))
    if not work_dir.is_absolute():
        work_dir = (base_dir / work_dir).resolve()

    ccfg = cfg.get("cosmics", {}) or {}
    if not bool(ccfg.get("enabled", True)):
        out_root = Path(out_dir) if out_dir is not None else (work_dir / "cosmics")
        if not out_root.is_absolute():
            out_root = (work_dir / out_root).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        out_path = out_root / "summary.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"disabled": True}, f, indent=2, ensure_ascii=False)
        return out_path.resolve()

    method = str(ccfg.get("method", "auto")).strip().lower()
    apply_to = ccfg.get("apply_to", ["obj"]) or ["obj"]
    if isinstance(apply_to, str):
        apply_to = [apply_to]

    # Threshold: prefer explicit k; keep backward compat with older sigma_clip.
    k = ccfg.get("k", None)
    if k is None:
        k = ccfg.get("sigma_clip", 9.0)
    try:
        k = float(k)
    except Exception:
        k = 9.0

    bias_subtract = bool(ccfg.get("bias_subtract", True))
    save_png = bool(ccfg.get("save_png", True))
    save_mask_fits = bool(ccfg.get("save_mask_fits", True))

    # ---- optional tuning knobs (safe defaults keep legacy behavior) ----
    def _as_int(x: Any, default: int) -> int:
        try:
            return int(x)
        except Exception:
            return int(default)

    def _as_float(x: Any, default: float) -> float:
        try:
            v = float(x)
            return v
        except Exception:
            return float(default)

    dilate = _as_int(ccfg.get("dilate", 1), 1)

    mad_scale = _as_float(ccfg.get("mad_scale", 1.0), 1.0)
    min_mad = _as_float(ccfg.get("min_mad", 0.0), 0.0)
    max_frac_per_frame = ccfg.get("max_frac_per_frame", None)
    if max_frac_per_frame is not None:
        try:
            max_frac_per_frame = float(max_frac_per_frame)
        except Exception:
            max_frac_per_frame = None

    stack_dilate = ccfg.get("stack_dilate", None)
    if stack_dilate is not None:
        stack_dilate = _as_int(stack_dilate, dilate)

    local_r = _as_int(ccfg.get("local_r", 2), 2)
    two_diff_local_r = ccfg.get("two_diff_local_r", None)
    if two_diff_local_r is not None:
        two_diff_local_r = _as_int(two_diff_local_r, local_r)

    two_diff_k2_scale = _as_float(ccfg.get("two_diff_k2_scale", 0.8), 0.8)
    two_diff_k2_min = _as_float(ccfg.get("two_diff_k2_min", 5.0), 5.0)
    two_diff_thr_local_a = _as_float(ccfg.get("two_diff_thr_local_a", 4.0), 4.0)
    two_diff_thr_local_b = _as_float(ccfg.get("two_diff_thr_local_b", 2.5), 2.5)
    two_diff_dilate = ccfg.get("two_diff_dilate", None)
    if two_diff_dilate is not None:
        two_diff_dilate = _as_int(two_diff_dilate, dilate)

    lap_local_r = ccfg.get("lap_local_r", None)
    if lap_local_r is not None:
        lap_local_r = _as_int(lap_local_r, local_r)
    lap_k_scale = _as_float(ccfg.get("lap_k_scale", 0.8), 0.8)
    lap_k_min = _as_float(ccfg.get("lap_k_min", 5.0), 5.0)
    lap_dilate = ccfg.get("lap_dilate", None)
    if lap_dilate is not None:
        lap_dilate = _as_int(lap_dilate, dilate)

    # L.A.Cosmic knobs (single-frame). Conservative defaults.
    la_niter = _as_int(ccfg.get("la_niter", 4), 4)
    la_sigclip = _as_float(ccfg.get("la_sigclip", 4.5), 4.5)
    la_sigfrac = _as_float(ccfg.get("la_sigfrac", 0.3), 0.3)
    la_objlim = _as_float(ccfg.get("la_objlim", 5.0), 5.0)
    la_replace_r = _as_int(ccfg.get("la_replace_r", max(1, local_r)), max(1, local_r))
    la_dilate = _as_int(ccfg.get("la_dilate", dilate), dilate)
    la_protect_lines = bool(ccfg.get("la_protect_lines", True))
    la_protect_half_x = _as_int(ccfg.get("la_protect_half_x", 6), 6)
    la_protect_half_y = _as_int(ccfg.get("la_protect_half_y", 1), 1)
    la_protect_k = _as_float(ccfg.get("la_protect_k", 5.0), 5.0)

    superbias = _load_superbias(work_dir) if bias_subtract else None

    out_root = Path(out_dir) if out_dir is not None else (work_dir / "cosmics")
    if not out_root.is_absolute():
        out_root = (work_dir / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    frames = cfg.get("frames", {}) or {}

    all_summaries: list[dict[str, Any]] = []

    for kind in apply_to:
        kind = str(kind)
        rel_paths = frames.get(kind, []) or []
        if not isinstance(rel_paths, (list, tuple)):
            continue

        paths = [_resolve_path(_as_path(pp), data_dir=data_dir, work_dir=work_dir, base_dir=base_dir) for pp in rel_paths]
        paths = [p for p in paths if p.is_file()]

        if len(paths) == 0:
            continue

        kind_out = out_root / kind
        kind_out.mkdir(parents=True, exist_ok=True)

        summary: CosmicsSummary
        if method in (
            "auto",
            "la_cosmic",
            "lacosmic",
            "stack_mad",
            "mad_stack",
            "stack",
            "two_frame_diff",
            "diff",
            "laplacian",
            "lap",
        ):
            # Choose best available method for the frame count.
            if method in ("laplacian", "lap") and len(paths) != 1:
                log.warning("method=laplacian requires exactly 1 frame; falling back to auto")
            if method in ("two_frame_diff", "diff") and len(paths) != 2:
                log.warning("method=two_frame_diff requires exactly 2 frames; falling back to auto")
            if method in ("la_cosmic", "lacosmic") and len(paths) != 1:
                log.warning("method=la_cosmic requires exactly 1 frame; falling back to auto")

            if len(paths) >= 3 and method in ("auto", "stack_mad", "mad_stack", "stack"):
                summary = _stack_mad_clean(
                    paths,
                    out_dir=kind_out,
                    superbias=superbias,
                    k=k,
                    mad_scale=mad_scale,
                    min_mad=min_mad,
                    max_frac_per_frame=max_frac_per_frame,
                    dilate=int(stack_dilate if stack_dilate is not None else dilate),
                    bias_subtract=bias_subtract,
                    save_png=save_png,
                    save_mask_fits=save_mask_fits,
                )
            elif len(paths) == 2 and method in ("auto", "stack_mad", "mad_stack", "stack", "two_frame_diff", "diff"):
                summary = _two_frame_diff_clean(
                    paths,
                    out_dir=kind_out,
                    superbias=superbias,
                    k=k,
                    dilate=int(two_diff_dilate if two_diff_dilate is not None else dilate),
                    local_r=int(two_diff_local_r if two_diff_local_r is not None else local_r),
                    k2_scale=two_diff_k2_scale,
                    k2_min=two_diff_k2_min,
                    thr_local_a=two_diff_thr_local_a,
                    thr_local_b=two_diff_thr_local_b,
                    bias_subtract=bias_subtract,
                    save_png=save_png,
                    save_mask_fits=save_mask_fits,
                )
            elif len(paths) == 1 and method in ("auto", "la_cosmic", "lacosmic"):
                # Single-frame preferred route: LA Cosmic-like detector.
                summary = _single_frame_lacosmic_clean(
                    paths[0],
                    out_dir=kind_out,
                    superbias=superbias,
                    bias_subtract=bias_subtract,
                    save_png=save_png,
                    save_mask_fits=save_mask_fits,
                    niter=la_niter,
                    sigclip=la_sigclip,
                    sigfrac=la_sigfrac,
                    objlim=la_objlim,
                    replace_r=la_replace_r,
                    dilate=la_dilate,
                    protect_lines=la_protect_lines,
                    protect_half_x=la_protect_half_x,
                    protect_half_y=la_protect_half_y,
                    protect_k=la_protect_k,
                )
            elif len(paths) == 1 and method in ("laplacian", "lap"):
                summary = _single_frame_laplacian_clean(
                    paths[0],
                    out_dir=kind_out,
                    superbias=superbias,
                    k=k,
                    dilate=int(lap_dilate if lap_dilate is not None else dilate),
                    local_r=int(lap_local_r if lap_local_r is not None else local_r),
                    k_scale=lap_k_scale,
                    k_min=lap_k_min,
                    bias_subtract=bias_subtract,
                    save_png=save_png,
                    save_mask_fits=save_mask_fits,
                )
            else:
                # Nothing to do
                (kind_out / "clean").mkdir(parents=True, exist_ok=True)
                outputs = {"note": f"not enough frames for cosmics cleaning (n={len(paths)})"}
                summary = CosmicsSummary(
                    kind=kind,
                    n_frames=len(paths),
                    k=float(k),
                    replaced_pixels=0,
                    replaced_fraction=0.0,
                    per_frame_fraction=[0.0] * len(paths),
                    outputs=outputs,
                )

            # Ensure 'kind' is set in summary
            if summary.kind != kind:
                summary = CosmicsSummary(
                    kind=kind,
                    n_frames=summary.n_frames,
                    k=summary.k,
                    replaced_pixels=summary.replaced_pixels,
                    replaced_fraction=summary.replaced_fraction,
                    per_frame_fraction=summary.per_frame_fraction,
                    outputs=summary.outputs,
                )

            # --- manual mask post-processing (GUI-driven) ---
            preserve_manual = bool(ccfg.get("preserve_manual", True))
            manual_replace_r = _as_int(ccfg.get("manual_replace_r", la_replace_r), la_replace_r)
            if preserve_manual:
                try:
                    repix, refrac, per = _apply_manual_masks(kind_out, replace_r=int(manual_replace_r), save_png=save_png)
                    # Update summary statistics to reflect final masks (AUTO|MANUAL).
                    summary = CosmicsSummary(
                        kind=summary.kind,
                        n_frames=summary.n_frames,
                        k=summary.k,
                        replaced_pixels=int(repix),
                        replaced_fraction=float(refrac),
                        per_frame_fraction=list(per),
                        outputs={
                            **(summary.outputs or {}),
                            "auto_masks_fits_dir": str((kind_out / "auto_masks_fits").resolve()),
                            "manual_masks_fits_dir": str((kind_out / "manual_masks_fits").resolve()),
                            "sum_excl_fits": str((kind_out / "sum_excl_cosmics.fits").resolve()),
                            "coverage_fits": str((kind_out / "coverage.fits").resolve()),
                        },
                    )
                except Exception as e:
                    log.warning("Failed to apply manual cosmics masks for %s: %s", kind, e, exc_info=True)
        else:
            outputs = {"note": f"method={method} not supported"}
            summary = CosmicsSummary(
                kind=kind,
                n_frames=len(paths),
                k=float(k),
                replaced_pixels=0,
                replaced_fraction=0.0,
                per_frame_fraction=[0.0] * len(paths),
                outputs=outputs,
            )

        all_summaries.append(
            {
                "kind": summary.kind,
                "n_frames": summary.n_frames,
                "method": method,
                "k": summary.k,
                "replaced_pixels": summary.replaced_pixels,
                "replaced_fraction": summary.replaced_fraction,
                "per_frame_fraction": summary.per_frame_fraction,
                "outputs": summary.outputs,
            }
        )

        # write per-kind summary too
        with (kind_out / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(all_summaries[-1], f, indent=2, ensure_ascii=False)

    out = {
        "method": method,
        "k": float(k),
        "bias_subtract": bias_subtract,
        "save_png": save_png,
        "save_mask_fits": save_mask_fits,
        "dilate": int(dilate),
        "stack": {
            "mad_scale": float(mad_scale),
            "min_mad": float(min_mad),
            "max_frac_per_frame": (float(max_frac_per_frame) if max_frac_per_frame is not None else None),
            "dilate": int(stack_dilate if stack_dilate is not None else dilate),
        },
        "two_frame_diff": {
            "local_r": int(two_diff_local_r if two_diff_local_r is not None else local_r),
            "k2_scale": float(two_diff_k2_scale),
            "k2_min": float(two_diff_k2_min),
            "thr_local_a": float(two_diff_thr_local_a),
            "thr_local_b": float(two_diff_thr_local_b),
            "dilate": int(two_diff_dilate if two_diff_dilate is not None else dilate),
        },
        "laplacian": {
            "local_r": int(lap_local_r if lap_local_r is not None else local_r),
            "k_scale": float(lap_k_scale),
            "k_min": float(lap_k_min),
            "dilate": int(lap_dilate if lap_dilate is not None else dilate),
        },
        "items": all_summaries,
    }

    out_path = out_root / "summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return out_path.resolve()
