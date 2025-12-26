from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

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
        with fits.open(p) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float32)
            hdr = hdul[0].header.copy()
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
            fits.writeto(mf, (m.astype(np.uint16)) * 1, overwrite=True)

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

    with fits.open(path) as hdul:
        img = np.asarray(hdul[0].data, dtype=np.float32)
        hdr = hdul[0].header.copy()

    if bias_subtract and superbias is not None and superbias.shape == img.shape:
        img = img - superbias
        hdr["BIASSUB"] = (True, "Superbias subtracted")

    # Laplacian (4-neighbor) with reflect padding
    ip = np.pad(img, ((1, 1), (1, 1)), mode="reflect")
    lap = (
        -4.0 * ip[1:-1, 1:-1]
        + ip[:-2, 1:-1]
        + ip[2:, 1:-1]
        + ip[1:-1, :-2]
        + ip[1:-1, 2:]
    ).astype(np.float32)

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
        fits.writeto(out_dir / "masks_fits" / f"{name}_mask.fits", (m.astype(np.uint16)) * 1, overwrite=True)

    sum_f = out_dir / "sum_excl_cosmics.fits"
    cov_f = out_dir / "coverage.fits"
    fits.writeto(sum_f, (img * (~m)).astype(np.float32), overwrite=True)
    fits.writeto(cov_f, (~m).astype(np.int16), overwrite=True)

    outputs = {
        "clean_dir": str((out_dir / "clean").resolve()),
        "sum_excl_fits": str(sum_f.resolve()),
        "coverage_fits": str(cov_f.resolve()),
        "sigma_lap": float(sigma),
    }
    if save_png:
        outputs["masks_dir"] = str((out_dir / "masks").resolve())
    if save_mask_fits:
        outputs["masks_fits_dir"] = str((out_dir / "masks_fits").resolve())

    return CosmicsSummary(
        kind="",
        n_frames=1,
        k=float(k),
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
        with fits.open(p) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float32)
            hdr = hdul[0].header.copy()
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
            # uint16 mask: 1 = cosmic (first reserved bit)
            fits.writeto(mf, (mask[i].astype(np.uint16)) * 1, overwrite=True)

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

    Default method is the robust stack-based MAD detection inspired by the user's
    `SKY_MODEL Object Cosmic.py` workflow: build per-pixel median/MAD across a
    stack of exposures, mask outliers, and replace them by the median.

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

    method = str(ccfg.get("method", "stack_mad")).strip().lower()
    apply_to = ccfg.get("apply_to", ["obj"]) or ["obj"]
    if isinstance(apply_to, str):
        apply_to = [apply_to]

    # Threshold: prefer explicit k; keep backward compat with older sigma_clip
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
        if method in ("auto", "stack_mad", "mad_stack", "stack", "two_frame_diff", "laplacian"):
            # Choose best available method for the frame count.
            if method in ("laplacian",) and len(paths) != 1:
                log.warning("method=laplacian requires exactly 1 frame; falling back to auto")
            if method in ("two_frame_diff",) and len(paths) != 2:
                log.warning("method=two_frame_diff requires exactly 2 frames; falling back to auto")

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
            elif len(paths) == 2 and method in ("auto", "stack_mad", "mad_stack", "stack", "two_frame_diff"):
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
            elif len(paths) == 1 and method in ("auto", "laplacian"):
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
