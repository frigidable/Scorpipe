from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from scorpio_pipe.io.atomic_fits import atomic_write_fits

from scorpio_pipe.fits_utils import read_image_smart
from scorpio_pipe.workspace_paths import stage_dir
from scorpio_pipe import maskbits
from scorpio_pipe.detector_saturation import carry_detector_saturation
from scorpio_pipe.cosmics_dq import read_clean_dq_sidecar
from scorpio_pipe.adapters.parallel_map import parallel_map

log = logging.getLogger(__name__)


def _default_out_root(work_dir: Path) -> Path:
    """Choose the default Cosmics output root.

    Canonical layout is the stage directory (``run_root/05_cosmics``). For
    backward compatibility, if an existing legacy directory ``run_root/cosmics``
    already contains products for this run and the canonical dir does not exist
    yet, we keep writing into the legacy dir to avoid splitting outputs.
    """

    canonical = stage_dir(work_dir, "cosmics")
    legacy = work_dir / "cosmics"

    # If a run was already started with legacy layout, keep it consistent.
    try:
        legacy_has_products = legacy.exists() and any(
            (legacy / k / "clean").exists() for k in ("obj", "sky")
        )
    except Exception:
        legacy_has_products = False

    if canonical.exists():
        return canonical
    if legacy_has_products and not canonical.exists():
        return legacy
    return canonical


def _write_clean_mask_bitmask(
    clean_dir: Path, base: str, mask_bool: np.ndarray, *,
    detector_dq: np.ndarray | None = None,
    provenance_header: fits.Header | None = None,
) -> Path:
    """Persist the complete science DQ, not only cosmic-replacement locations.

    The adjacent primary-HDU clean image stays compatible with legacy GUIs.
    Its mandatory, typed sidecar carries COSMIC plus original detector bits.
    """
    clean_dir.mkdir(parents=True, exist_ok=True)
    mask_bool = np.asarray(mask_bool, dtype=bool)
    m = np.where(mask_bool, maskbits.COSMIC, 0).astype(np.uint16)
    if detector_dq is not None:
        dq = np.asarray(detector_dq)
        if dq.shape != m.shape or not np.issubdtype(dq.dtype, np.integer):
            raise ValueError("Cosmics detector DQ must be an integer image matching SCI")
        if np.any(dq < 0) or np.any(dq > np.iinfo(np.uint16).max):
            raise ValueError("Cosmics detector DQ outside uint16 range")
        m |= np.asarray(dq, dtype=np.uint16)
    h = fits.Header()
    for k, v in maskbits.header_cards(prefix="SCORP").items():
        h[k] = (str(v), "Mask schema")
    h["BUNIT"] = ("bitmask", "MASK unit (uint16 bitfield)")
    h["SATMSK"] = ("DQ_V1", "Typed cosmics DQ sidecar contract")
    if provenance_header is not None:
        for key in ("SATQ", "SATADU", "SATMARG", "SATSRC"):
            if key in provenance_header:
                h[key] = provenance_header[key]
    out = clean_dir / f"{base}_clean_mask.fits"
    atomic_write_fits(out, m, header=h, overwrite=True)
    return out


def _write_cosmics_frame_output(task: dict[str, Any]) -> dict[str, str]:
    name = str(task["name"])
    out_dir = Path(str(task["out_dir"]))
    hdr = task["header"]
    img = np.asarray(task["image"], dtype=np.float32)
    mask = np.asarray(task["mask"], dtype=bool)
    save_png = bool(task.get("save_png", False))
    save_mask_fits = bool(task.get("save_mask_fits", False))
    roi = _coerce_roi_spec(task.get("roi"), img.shape)
    history = str(task.get("history", "scorpio_pipe cosmics"))
    cosm_k = float(task.get("cosm_k", 0.0))
    cosm_md = str(task.get("cosm_md", "auto"))

    out_f = out_dir / "clean" / f"{name}_clean.fits"
    h = hdr.copy()
    h["COSMCLEA"] = (True, "Cosmics cleaned")
    h["SATMSK"] = ("DQ_V1", "Typed cosmics DQ sidecar contract")
    h["COSM_K"] = (float(cosm_k), "Threshold control")
    h["COSM_MD"] = (str(cosm_md), "Cosmics method")
    _stamp_roi_header(h, roi)
    h["HISTORY"] = history
    atomic_write_fits(out_f, img, header=h, overwrite=True)

    if save_png:
        _save_png(out_dir / "masks" / f"{name}_mask.png", mask.astype(np.uint8), title=f"Cosmic mask: {name}")

    if save_mask_fits:
        mf = out_dir / "masks_fits" / f"{name}_mask.fits"
        atomic_write_fits(mf, mask.astype(np.uint8), overwrite=True)

    # DQ persistence is required for qualified SATQ. Never silently drop it.
    _write_clean_mask_bitmask(
        out_dir / "clean", name, mask,
        detector_dq=task.get("detector_dq"), provenance_header=h,
    )

    return {"name": name, "clean_path": str(out_f)}

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
    """Load superbias (if available) for optional bias subtraction.

    Canonical location (v5.38.5):
      run_root/03_biascorr/superbias.fits

    We also keep legacy fallbacks under calibs/ and calib/.
    """

    from scorpio_pipe.workspace_paths import stage_dir

    candidates = [
        stage_dir(work_dir, "superbias") / "superbias.fits",
        work_dir / "calibs" / "superbias.fits",
        work_dir / "calib" / "superbias.fits",
    ]

    for p in candidates:
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
    total = s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]
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
    s = (
        np.pad(ap, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
        .cumsum(axis=0)
        .cumsum(axis=1)
    )
    c = (
        np.pad(gp, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
        .cumsum(axis=0)
        .cumsum(axis=1)
    )

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
    """Binary dilation with a (2r+1)x(2r+1) square structuring element.

    Uses SciPy's optimized kernel when available, otherwise falls back to the
    legacy pure-NumPy implementation.
    """

    if r <= 0:
        return np.asarray(mask, dtype=bool)
    m = np.asarray(mask, dtype=bool)
    try:  # fast kernel if SciPy is available
        from scipy import ndimage as ndi  # type: ignore

        size = int(2 * int(r) + 1)
        return np.asarray(
            ndi.binary_dilation(m, structure=np.ones((size, size), dtype=bool)),
            dtype=bool,
        )
    except Exception:
        pass

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


@dataclass(frozen=True)
class _RoiSpec:
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.y1 - self.y0), int(self.x1 - self.x0))

    def slices(self) -> tuple[slice, slice]:
        return slice(int(self.y0), int(self.y1)), slice(int(self.x0), int(self.x1))

    def to_dict(self) -> dict[str, int]:
        return {
            "y0": int(self.y0),
            "y1": int(self.y1),
            "x0": int(self.x0),
            "x1": int(self.x1),
            "height": int(self.y1 - self.y0),
            "width": int(self.x1 - self.x0),
        }


def _parse_roi_spec(roi_cfg: Any, shape: tuple[int, int]) -> _RoiSpec | None:
    """Parse an optional ROI config into a clipped image ROI.

    Accepted forms:
      - {x0,x1,y0,y1}
      - {xmin,xmax,ymin,ymax}
      - {x:[x0,x1], y:[y0,y1]}
      - [y0, y1, x0, x1]

    Missing bounds default to the full frame. If the ROI covers the full frame,
    returns None to keep the hot path branch-free.
    """

    ny, nx = int(shape[0]), int(shape[1])
    if ny <= 0 or nx <= 0 or roi_cfg in (None, False, {}, []):
        return None

    if isinstance(roi_cfg, (list, tuple)) and len(roi_cfg) >= 4:
        y0, y1, x0, x1 = roi_cfg[:4]
    elif isinstance(roi_cfg, dict):
        if roi_cfg.get("enabled", True) is False:
            return None
        xr = roi_cfg.get("x") or roi_cfg.get("cols") or roi_cfg.get("col_range")
        yr = roi_cfg.get("y") or roi_cfg.get("rows") or roi_cfg.get("row_range")
        y0 = roi_cfg.get("y0", roi_cfg.get("ymin", roi_cfg.get("y_min", None)))
        y1 = roi_cfg.get("y1", roi_cfg.get("ymax", roi_cfg.get("y_max", None)))
        x0 = roi_cfg.get("x0", roi_cfg.get("xmin", roi_cfg.get("x_min", None)))
        x1 = roi_cfg.get("x1", roi_cfg.get("xmax", roi_cfg.get("x_max", None)))
        if isinstance(yr, (list, tuple)) and len(yr) >= 2:
            y0 = yr[0] if y0 is None else y0
            y1 = yr[1] if y1 is None else y1
        if isinstance(xr, (list, tuple)) and len(xr) >= 2:
            x0 = xr[0] if x0 is None else x0
            x1 = xr[1] if x1 is None else x1
        pad = int(roi_cfg.get("pad", 0) or 0)
    else:
        return None

    def _ibound(v: Any, default: int) -> int:
        try:
            return int(v)
        except Exception:
            return int(default)

    y0 = _ibound(y0, 0)
    y1 = _ibound(y1, ny)
    x0 = _ibound(x0, 0)
    x1 = _ibound(x1, nx)
    if isinstance(roi_cfg, dict):
        y0 -= pad
        x0 -= pad
        y1 += pad
        x1 += pad

    y0 = max(0, min(ny, y0))
    y1 = max(y0, min(ny, y1))
    x0 = max(0, min(nx, x0))
    x1 = max(x0, min(nx, x1))

    if y0 == 0 and y1 == ny and x0 == 0 and x1 == nx:
        return None
    if y1 <= y0 or x1 <= x0:
        return None
    return _RoiSpec(y0=y0, y1=y1, x0=x0, x1=x1)


def _coerce_roi_spec(roi: Any, shape: tuple[int, int]) -> _RoiSpec | None:
    if isinstance(roi, _RoiSpec):
        return _parse_roi_spec(roi.to_dict(), shape)
    return _parse_roi_spec(roi, shape)


def _crop2d(img: np.ndarray, roi: _RoiSpec | None) -> np.ndarray:
    if roi is None:
        return np.asarray(img)
    ys, xs = roi.slices()
    return np.asarray(img)[ys, xs]


def _merge_roi_bool(mask_roi: np.ndarray, shape: tuple[int, int], roi: _RoiSpec | None) -> np.ndarray:
    if roi is None:
        return np.asarray(mask_roi, dtype=bool)
    out = np.zeros(shape, dtype=bool)
    ys, xs = roi.slices()
    out[ys, xs] = np.asarray(mask_roi, dtype=bool)
    return out


def _merge_roi_float(base: np.ndarray, roi_arr: np.ndarray, roi: _RoiSpec | None) -> np.ndarray:
    if roi is None:
        return np.asarray(roi_arr, dtype=np.float32)
    out = np.asarray(base, dtype=np.float32).copy()
    ys, xs = roi.slices()
    out[ys, xs] = np.asarray(roi_arr, dtype=np.float32)
    return out


def _stamp_roi_header(hdr: fits.Header, roi: _RoiSpec | None) -> None:
    if roi is None:
        return
    hdr["COSMROI"] = (True, "Cosmics detection limited to ROI")
    hdr["COSMRY0"] = (int(roi.y0), "Cosmics ROI y0 [px]")
    hdr["COSMRY1"] = (int(roi.y1), "Cosmics ROI y1 [px]")
    hdr["COSMRX0"] = (int(roi.x0), "Cosmics ROI x0 [px]")
    hdr["COSMRX1"] = (int(roi.x1), "Cosmics ROI x1 [px]")


def _two_frame_diff_clean(
    paths: list[Path],
    *,
    out_dir: Path,
    superbias: np.ndarray | None,
    k: float,
    bias_subtract: bool,
    save_png: bool,
    save_mask_fits: bool,
    cfg: dict | None = None,
    dilate: int = 1,
    local_r: int = 2,
    k2_scale: float = 0.8,
    k2_min: float = 5.0,
    thr_local_a: float = 4.0,
    thr_local_b: float = 2.5,
    roi: _RoiSpec | None = None,
    parallel_n_jobs: int | None = None,
    parallel_backend: str = "thread",
) -> CosmicsSummary:
    """Cosmic cleaning specialized for N=2 exposures."""

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
    detector_dqs: list[np.ndarray] = []
    names: list[str] = []

    for p in paths:
        data, hdr, _info = read_image_smart(p, memmap="auto", dtype=np.float32)
        detector_dq, hdr = carry_detector_saturation(data, None, hdr, cfg or {})
        detector_dqs.append(detector_dq)
        datas.append(data)
        headers.append(hdr)
        names.append(Path(p).stem)

    shapes = [d.shape for d in datas]
    ny = min(s[0] for s in shapes)
    nx = min(s[1] for s in shapes)
    if any(s != (ny, nx) for s in shapes):
        log.warning(
            "Cosmics(two_frame_diff): input shapes differ %s; cropping to (%d, %d)",
            shapes,
            ny,
            nx,
        )
        datas = [d[:ny, :nx] for d in datas]
        detector_dqs = [m[:ny, :nx] for m in detector_dqs]
        for hdr, s in zip(headers, shapes):
            if s != (ny, nx):
                hdr["CROPY"] = (ny, "Cropped to common height")
                hdr["CROPX"] = (nx, "Cropped to common width")
                hdr["HISTORY"] = (
                    f"scorpio_pipe cosmics: cropped from {s} to {(ny, nx)} for geometry match"
                )

    if bias_subtract and superbias is not None:
        if superbias.shape[0] >= ny and superbias.shape[1] >= nx:
            sb = superbias[:ny, :nx]
            datas = [d - sb for d in datas]
            for hdr in headers:
                hdr["BIASSUB"] = (True, "Superbias subtracted")
                hdr["HISTORY"] = (
                    "scorpio_pipe cosmics: bias subtracted using superbias.fits"
                )
        else:
            log.warning(
                "Cosmics(two_frame_diff): superbias shape %s is smaller than cropped science (%d, %d); skipping bias subtraction",
                getattr(superbias, "shape", None),
                ny,
                nx,
            )

    roi = _coerce_roi_spec(roi, (ny, nx))

    a, b = datas
    a_eval = _crop2d(a, roi)
    b_eval = _crop2d(b, roi)
    diff = (a_eval - b_eval).astype(np.float32)
    absdiff = np.abs(diff)

    sigma = _robust_sigma_mad_1d(diff)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    loc = _boxcar_mean2d(absdiff, r=int(local_r))
    k2 = max(float(k2_min), float(k2_scale) * float(k))
    thr_global = float(k2 * sigma)
    thr_local = (float(thr_local_a) * loc + float(thr_local_b) * sigma).astype(
        np.float32
    )

    cand = absdiff > np.maximum(thr_global, thr_local)
    m0_roi = cand & (diff > 0)
    m1_roi = cand & (diff < 0)

    # Backstop for tiny ROIs / near-empty synthetic frames: if the robust
    # detector found nothing but there is still an obvious extreme pixel, seed
    # the mask at the strongest outlier so the paired exposure can repair it.
    if not bool(np.any(m0_roi) or np.any(m1_roi)) and absdiff.size:
        try:
            ij = np.unravel_index(int(np.nanargmax(absdiff)), absdiff.shape)
            if np.isfinite(absdiff[ij]) and float(absdiff[ij]) > float(thr_global):
                if float(diff[ij]) > 0:
                    m0_roi[ij] = True
                elif float(diff[ij]) < 0:
                    m1_roi[ij] = True
        except Exception:
            pass

    m0_roi = _dilate_mask(m0_roi, r=int(dilate))
    m1_roi = _dilate_mask(m1_roi, r=int(dilate))

    m0 = _merge_roi_bool(m0_roi, a.shape, roi)
    m1 = _merge_roi_bool(m1_roi, b.shape, roi)
    cleaned0 = a.copy()
    cleaned1 = b.copy()
    if roi is None:
        cleaned0[m0] = b[m0]
        cleaned1[m1] = a[m1]
    else:
        ys, xs = roi.slices()
        c0 = a_eval.copy()
        c1 = b_eval.copy()
        c0[m0_roi] = b_eval[m0_roi]
        c1[m1_roi] = a_eval[m1_roi]
        cleaned0[ys, xs] = c0
        cleaned1[ys, xs] = c1

    replaced_pixels = int(m0.sum() + m1.sum())
    total_pix = int(a.size + b.size)
    replaced_fraction = float(replaced_pixels) / float(total_pix) if total_pix else 0.0
    per_frame_fraction = [float(m0.mean()), float(m1.mean())]

    out_files: dict[str, str] = {}
    write_tasks = [
        {
            "name": names[0],
            "out_dir": str(out_dir),
            "header": headers[0],
            "image": cleaned0,
            "mask": m0,
            "detector_dq": detector_dqs[0],
            "save_png": save_png,
            "save_mask_fits": save_mask_fits,
            "roi": (None if roi is None else roi.to_dict()),
            "history": "scorpio_pipe cosmics: replaced spikes using the other exposure",
            "cosm_k": float(k),
            "cosm_md": "two_frame_diff",
        },
        {
            "name": names[1],
            "out_dir": str(out_dir),
            "header": headers[1],
            "image": cleaned1,
            "mask": m1,
            "detector_dq": detector_dqs[1],
            "save_png": save_png,
            "save_mask_fits": save_mask_fits,
            "roi": (None if roi is None else roi.to_dict()),
            "history": "scorpio_pipe cosmics: replaced spikes using the other exposure",
            "cosm_k": float(k),
            "cosm_md": "two_frame_diff",
        },
    ]
    for res in parallel_map(write_tasks, _write_cosmics_frame_output, n_jobs=parallel_n_jobs, backend=parallel_backend):
        out_files[str(res["name"])] = str(res["clean_path"])

    sum_excl = (a * (~m0) + b * (~m1)).astype(np.float32)
    cov = ((~m0).astype(np.int16) + (~m1).astype(np.int16)).astype(np.int16)
    sum_f = out_dir / "sum_excl_cosmics.fits"
    cov_f = out_dir / "coverage.fits"
    atomic_write_fits(sum_f, sum_excl, overwrite=True)
    atomic_write_fits(cov_f, cov, overwrite=True)
    if save_png:
        _save_png(
            out_dir / "sum_excl_cosmics.png", sum_excl, title="Sum (cosmics excluded)"
        )
        _save_png(out_dir / "coverage.png", cov, title="Coverage (non-cosmic count)")

    outputs: dict[str, Any] = {
        "clean_dir": str((out_dir / "clean").resolve()),
        "sum_excl_fits": str(sum_f.resolve()),
        "coverage_fits": str(cov_f.resolve()),
        "sigma_diff": float(sigma),
    }
    if roi is not None:
        outputs["roi"] = roi.to_dict()
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
    cfg: dict | None = None,
    dilate: int = 1,
    local_r: int = 2,
    k_scale: float = 0.8,
    k_min: float = 5.0,
    niter: int = 2,
    edge_margin: int = 2,
    roi: _RoiSpec | None = None,
) -> CosmicsSummary:
    """Single-frame Laplacian cosmic-ray detector (requires SciPy)."""
    try:
        from scipy import ndimage as ndi  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Cosmics method 'laplacian' requires SciPy. Install optional extras: "
            "pip install scorpio-pipe[science]  (or install scipy>=1.10)."
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clean").mkdir(parents=True, exist_ok=True)
    if save_png:
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    if save_mask_fits:
        (out_dir / "masks_fits").mkdir(parents=True, exist_ok=True)

    img, hdr, _info = read_image_smart(path, memmap="auto", dtype=np.float32)
    # Evaluate detector ADC saturation before any bias correction/inpainting.
    detector_dq, hdr = carry_detector_saturation(img, None, hdr, cfg or {})

    if bias_subtract and superbias is not None and superbias.shape == img.shape:
        img = img - superbias
        hdr["BIASSUB"] = (True, "Superbias subtracted")

    roi = _coerce_roi_spec(roi, img.shape)
    img_eval = _crop2d(img, roi)
    line_protect: np.ndarray | None = None
    try:
        rx = img_eval - _boxcar_mean2d(img_eval, r=6)
        ry = img_eval - _boxcar_mean2d(img_eval, r=1)
        sx = _robust_sigma_mad_1d(rx)
        sy = _robust_sigma_mad_1d(ry)
        if np.isfinite(sx) and np.isfinite(sy) and sx > 0 and sy > 0:
            line_protect = (np.abs(rx) > 5.0 * sx) & (np.abs(ry) < 4.0 * sy)
    except Exception:
        line_protect = None

    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    cleaned_eval = np.asarray(img_eval, dtype=np.float32).copy()
    m_eval = np.zeros_like(cleaned_eval, dtype=bool)
    win = int(2 * max(1, local_r) + 1)

    sigma_lap = float("nan")
    thr_sig = float(max(float(k_min), float(k) * float(k_scale)))

    lap0 = ndi.convolve(cleaned_eval, kernel, mode="mirror")
    sigma0 = _robust_sigma_mad_1d(lap0)
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = float(np.std(lap0)) if lap0.size else 0.0
    sigma_lap = float(sigma0)
    thr = thr_sig * sigma_lap

    for _ in range(int(max(1, int(niter)))):
        lap = ndi.convolve(cleaned_eval, kernel, mode="mirror")
        cand = (lap > thr) & np.isfinite(lap)
        em = int(edge_margin)
        if em > 0:
            cand[:em, :] = False
            cand[-em:, :] = False
            cand[:, :em] = False
            cand[:, -em:] = False
        if line_protect is not None:
            cand = cand & (~line_protect)
        if int(dilate) > 0:
            cand = np.asarray(ndi.binary_dilation(cand, iterations=int(dilate)), dtype=bool)
        new = cand & (~m_eval)
        if not bool(new.any()):
            m_eval = cand | m_eval
            break
        m_eval = cand | m_eval
        med = ndi.median_filter(cleaned_eval, size=win, mode="mirror")
        cleaned_eval[new] = med[new]

    m = _merge_roi_bool(m_eval, img.shape, roi)
    cleaned = _merge_roi_float(img, cleaned_eval, roi)

    name = path.stem
    out_f = out_dir / "clean" / f"{name}_clean.fits"
    h = hdr.copy()
    h["COSMCLEA"] = (True, "Cosmics cleaned")
    h["SATMSK"] = ("DQ_V1", "Typed cosmics DQ sidecar contract")
    h["COSM_MD"] = ("laplacian", "Cosmics method")
    h["COSM_K"] = (float(k), "Base threshold control (sigmas)")
    h["COSM_KS"] = (float(k_scale), "Laplacian k scale")
    h["COSM_KMN"] = (float(k_min), "Laplacian k min")
    h["COSM_THS"] = (float(thr_sig), "Effective threshold (sigmas)")
    h["COSM_SIG"] = (float(sigma_lap), "Robust sigma of Laplacian response")
    h["COSM_DIL"] = (int(dilate), "Binary mask dilation iters")
    h["COSM_NIT"] = (int(niter), "Laplacian iterations")
    h["COSM_EDG"] = (int(edge_margin), "Edge margin excluded from detection [px]")
    h["COSM_WIN"] = (int(win), "Local median window size")
    _stamp_roi_header(h, roi)
    h["HISTORY"] = "scorpio_pipe cosmics: SciPy laplacian detector + local median replacement"
    atomic_write_fits(out_f, cleaned.astype(np.float32), header=h, overwrite=True)

    if save_png:
        _save_png(
            out_dir / "masks" / f"{name}_mask.png",
            m.astype(np.uint8),
            title=f"Cosmic mask: {name}",
        )
    if save_mask_fits:
        atomic_write_fits(
            out_dir / "masks_fits" / f"{name}_mask.fits",
            m.astype(np.uint8),
            overwrite=True,
        )

    _write_clean_mask_bitmask(
        out_dir / "clean", name, m,
        detector_dq=detector_dq, provenance_header=h,
    )

    replaced_pixels = int(m.sum())
    replaced_fraction = float(replaced_pixels / float(m.size)) if m.size else 0.0
    sum_excl = img * (~m)
    cov = (~m).astype(np.int16)
    atomic_write_fits(out_dir / "sum_excl_cosmics.fits", sum_excl.astype(np.float32), overwrite=True)
    atomic_write_fits(out_dir / "coverage.fits", cov, overwrite=True)
    if save_png:
        _save_png(out_dir / "sum_excl_cosmics.png", sum_excl, title="Sum (cosmics excluded)")
        _save_png(out_dir / "coverage.png", cov, title="Coverage (non-cosmic count)")

    outputs: dict[str, Any] = {
        "clean_dir": str((out_dir / "clean").resolve()),
        "sum_excl_fits": str((out_dir / "sum_excl_cosmics.fits").resolve()),
        "coverage_fits": str((out_dir / "coverage.fits").resolve()),
        "sigma_lap": float(sigma_lap),
    }
    if roi is not None:
        outputs["roi"] = roi.to_dict()
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
    cfg: dict | None = None,
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
    roi: _RoiSpec | None = None,
) -> CosmicsSummary:
    """Single-frame L.A.Cosmic-like detector (van Dokkum 2001 style)."""

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clean").mkdir(parents=True, exist_ok=True)
    if save_png:
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    if save_mask_fits:
        (out_dir / "masks_fits").mkdir(parents=True, exist_ok=True)

    img, hdr, _info = read_image_smart(path, memmap="auto", dtype=np.float32)
    # Evaluate detector ADC saturation before any bias correction/inpainting.
    detector_dq, hdr = carry_detector_saturation(img, None, hdr, cfg or {})
    if bias_subtract and superbias is not None and superbias.shape == img.shape:
        img = img - superbias
        hdr["BIASSUB"] = (True, "Superbias subtracted")

    roi = _coerce_roi_spec(roi, img.shape)
    img_eval = _crop2d(img, roi)
    line_protect: np.ndarray | None = None
    if bool(protect_lines):
        rx = img_eval - _boxcar_mean2d(img_eval, r=int(max(1, protect_half_x)))
        ry = img_eval - _boxcar_mean2d(img_eval, r=int(max(1, protect_half_y)))
        sx = _robust_sigma_mad_1d(rx)
        sy = _robust_sigma_mad_1d(ry)
        if np.isfinite(sx) and np.isfinite(sy) and sx > 0 and sy > 0:
            line_protect = (np.abs(rx) > float(protect_k) * sx) & (
                np.abs(ry) < 0.8 * float(protect_k) * sy
            )

    try:
        import astroscrappy  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "astroscrappy is required for cosmics method 'la_cosmic' (LACOSMIC). "
            "Install scorpio-pipe with the required dependencies (astroscrappy) or choose another cosmics method."
        ) from e

    crmask, cleaned_eval = astroscrappy.detect_cosmics(
        img_eval,
        sigclip=float(sigclip),
        sigfrac=float(sigfrac),
        objlim=float(objlim),
        niter=int(max(1, niter)),
        verbose=False,
    )
    m_eval = np.asarray(crmask, dtype=bool)
    if line_protect is not None:
        m_eval = m_eval & (~line_protect)
    cleaned_eval = np.asarray(cleaned_eval, dtype=np.float32)
    engine = "astroscrappy"
    sigma_lap = float("nan")

    m_eval = _dilate_mask(m_eval, r=int(max(0, dilate)))
    m = _merge_roi_bool(m_eval, img.shape, roi)
    cleaned = _merge_roi_float(img, cleaned_eval, roi)

    replaced_pixels = int(m.sum())
    replaced_fraction = float(m.mean()) if m.size else 0.0

    name = Path(path).stem
    out_f = out_dir / "clean" / f"{name}_clean.fits"
    h = hdr.copy()
    h["COSMCLEA"] = (True, "Cosmics cleaned")
    h["SATMSK"] = ("DQ_V1", "Typed cosmics DQ sidecar contract")
    h["COSM_MD"] = ("la_cosmic", "Cosmics method")
    h["COSM_ENG"] = (str(engine), "L.A.Cosmic engine")
    h["COSM_IT"] = (int(max(1, niter)), "L.A.Cosmic iterations")
    h["COSM_SC"] = (float(sigclip), "L.A.Cosmic sigclip")
    h["COSM_SF"] = (float(sigfrac), "L.A.Cosmic sigfrac")
    h["COSM_OL"] = (float(objlim), "L.A.Cosmic objlim")
    _stamp_roi_header(h, roi)
    h["HISTORY"] = "scorpio_pipe cosmics: la_cosmic (van Dokkum-like)"
    atomic_write_fits(out_f, cleaned.astype(np.float32), header=h, overwrite=True)

    if save_png:
        _save_png(
            out_dir / "masks" / f"{name}_mask.png",
            m.astype(np.uint8),
            title=f"Cosmic mask: {name}",
        )
    if save_mask_fits:
        atomic_write_fits(
            out_dir / "masks_fits" / f"{name}_mask.fits",
            m.astype(np.uint8),
            overwrite=True,
        )

    _write_clean_mask_bitmask(
        out_dir / "clean", name, m,
        detector_dq=detector_dq, provenance_header=h,
    )

    sum_f = out_dir / "sum_excl_cosmics.fits"
    cov_f = out_dir / "coverage.fits"
    atomic_write_fits(sum_f, (img * (~m)).astype(np.float32), overwrite=True)
    atomic_write_fits(cov_f, (~m).astype(np.int16), overwrite=True)

    outputs: dict[str, Any] = {
        "clean_dir": str((out_dir / "clean").resolve()),
        "sum_excl_fits": str(sum_f.resolve()),
        "coverage_fits": str(cov_f.resolve()),
        "engine": str(engine),
        "sigma_lap": float(sigma_lap) if np.isfinite(sigma_lap) else None,
    }
    if roi is not None:
        outputs["roi"] = roi.to_dict()
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

    clean_dir = kind_out / "clean"
    masks_fits = kind_out / "masks_fits"
    masks_png = kind_out / "masks"
    auto_masks = kind_out / "auto_masks_fits"
    manual_masks = kind_out / "manual_masks_fits"

    auto_masks.mkdir(parents=True, exist_ok=True)
    manual_masks.mkdir(parents=True, exist_ok=True)
    masks_fits.mkdir(parents=True, exist_ok=True)
    if save_png:
        masks_png.mkdir(parents=True, exist_ok=True)

    clean_files = sorted(clean_dir.glob("*_clean.fits"))
    if not clean_files:
        return 0, 0.0, []

    final_masks: list[np.ndarray] = []
    clean_datas: list[np.ndarray] = []

    for cf in clean_files:
        base = cf.stem
        if base.endswith("_clean"):
            base = base[:-6]

        with fits.open(cf, memmap=False) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float32)
            hdr = hdul[0].header.copy()

        # Load authoritative DQ before even considering optional legacy masks.
        # They may be disabled without losing the automatic COSMIC bit.
        detector_dq = read_clean_dq_sidecar(cf, hdr, data.shape)
        if detector_dq is None:
            detector_dq = np.zeros(data.shape, dtype=np.uint16)
            if str(hdr.get("SATQ") or "").strip().upper() == "RAW_EVAL":
                hdr["SATQ"] = ("UNKNOWN", "Original detector DQ unavailable")

        auto_path = masks_fits / f"{base}_mask.fits"
        if auto_path.exists():
            with fits.open(auto_path, memmap=False) as h:
                auto_m = _mask_u8_to_bool(h[0].data)
        else:
            # Optional legacy masks may be disabled. The typed DQ stores
            # automatic cosmic detections independently of the GUI bitmap.
            auto_m = (detector_dq & maskbits.COSMIC) != 0

        # snapshot current auto mask for GUI baselining
        atomic_write_fits(
            auto_masks / f"{base}_auto_mask.fits",
            np.asarray(auto_m, dtype=np.uint8),
            overwrite=True,
        )

        man_path = manual_masks / f"{base}_manual_mask.fits"
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
            hdr["MANCR"] = (True, "Manual cosmics edits applied")
            hdr["MANCRN"] = (int(man_m.sum()), "Manual masked pixels")
            hdr["HISTORY"] = "scorpio_pipe cosmics: manual inpainting applied"

        hdr["SATMSK"] = ("DQ_V1", "Typed cosmics DQ sidecar contract")
        atomic_write_fits(cf, data.astype(np.float32), header=hdr, overwrite=True)
        atomic_write_fits(
            masks_fits / f"{base}_mask.fits", final_m.astype(np.uint8), overwrite=True
        )
        # Replace only cosmic bits, ORing the preexisting non-cosmic DQ.
        # Missing/corrupt qualified sidecars fail above instead of being
        # silently replaced with an all-zero detector mask.
        _write_clean_mask_bitmask(
            clean_dir, base, final_m,
            detector_dq=(detector_dq & np.uint16(~int(maskbits.COSMIC) & 0xFFFF)),
            provenance_header=hdr,
        )
        if save_png:
            _save_png(
                masks_png / f"{base}_mask.png",
                final_m.astype(np.uint8),
                title=f"Cosmic mask: {base}",
            )

        final_masks.append(final_m)
        clean_datas.append(data)

    # Reference products based on final masks and current clean frames.
    stack = np.stack(clean_datas, axis=0)
    mstack = np.stack(final_masks, axis=0)
    sum_excl = np.sum(stack * (~mstack), axis=0)
    cov = np.sum(~mstack, axis=0).astype(np.int16)

    sum_f = kind_out / "sum_excl_cosmics.fits"
    cov_f = kind_out / "coverage.fits"
    atomic_write_fits(sum_f, sum_excl.astype(np.float32), overwrite=True)
    atomic_write_fits(cov_f, cov, overwrite=True)
    if save_png:
        _save_png(
            kind_out / "sum_excl_cosmics.png", sum_excl, title="Sum (cosmics excluded)"
        )
        _save_png(kind_out / "coverage.png", cov, title="Coverage (non-cosmic count)")

    replaced_pixels = int(mstack.sum())
    n_pix_total = int(mstack.size)
    replaced_fraction = (
        float(replaced_pixels) / float(n_pix_total) if n_pix_total else 0.0
    )
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
    cfg: dict | None = None,
    roi: _RoiSpec | None = None,
    parallel_n_jobs: int | None = None,
    parallel_backend: str = "thread",
) -> CosmicsSummary:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clean").mkdir(parents=True, exist_ok=True)
    if save_png:
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    if save_mask_fits:
        (out_dir / "masks_fits").mkdir(parents=True, exist_ok=True)

    datas: list[np.ndarray] = []
    headers: list[fits.Header] = []
    detector_dqs: list[np.ndarray] = []
    names: list[str] = []

    for p in paths:
        data, hdr, _info = read_image_smart(p, memmap="auto", dtype=np.float32)
        detector_dq, hdr = carry_detector_saturation(data, None, hdr, cfg or {})
        detector_dqs.append(detector_dq)
        datas.append(data)
        headers.append(hdr)
        names.append(p.stem)

    shapes = [d.shape for d in datas]
    ny = min(s[0] for s in shapes)
    nx = min(s[1] for s in shapes)
    if any(s != (ny, nx) for s in shapes):
        log.warning(
            "Cosmics(stack_mad): input shapes differ %s; cropping to (%d, %d)",
            shapes,
            ny,
            nx,
        )
        datas = [d[:ny, :nx] for d in datas]
        detector_dqs = [m[:ny, :nx] for m in detector_dqs]
        for hdr, s in zip(headers, shapes):
            if s != (ny, nx):
                hdr["CROPY"] = (ny, "Cropped to common height")
                hdr["CROPX"] = (nx, "Cropped to common width")
                hdr["HISTORY"] = (
                    f"scorpio_pipe cosmics: cropped from {s} to {(ny, nx)} for geometry match"
                )

    if bias_subtract and superbias is not None:
        if superbias.shape[0] >= ny and superbias.shape[1] >= nx:
            sb = superbias[:ny, :nx]
            datas = [d - sb for d in datas]
            for hdr in headers:
                hdr["BIASSUB"] = (True, "Superbias subtracted")
                hdr["HISTORY"] = (
                    "scorpio_pipe cosmics: bias subtracted using superbias.fits"
                )
        else:
            log.warning(
                "Cosmics(stack_mad): superbias shape %s is smaller than cropped science (%d, %d); skipping bias subtraction",
                getattr(superbias, "shape", None),
                ny,
                nx,
            )

    stack = np.stack(datas, axis=0)
    roi = _coerce_roi_spec(roi, (ny, nx))
    stack_eval = stack if roi is None else stack[:, roi.y0:roi.y1, roi.x0:roi.x1]
    med = np.median(stack_eval, axis=0)
    mad = _robust_mad(stack_eval, axis=0)

    eps = float(np.finfo(np.float32).eps)
    floor = max(eps, float(min_mad))
    mad = np.maximum(mad, floor)

    ms = max(eps, float(mad_scale))
    score = np.abs(stack_eval - med[None, :, :]) / (ms * mad[None, :, :])
    mask_eval = score > float(k)

    if max_frac_per_frame is not None:
        mf = float(max_frac_per_frame)
        if 0.0 < mf < 1.0:
            for i in range(mask_eval.shape[0]):
                frac = float(mask_eval[i].mean()) if mask_eval[i].size else 0.0
                if frac <= mf:
                    continue
                thr_i = float(np.quantile(score[i].ravel(), 1.0 - mf))
                thr_i = max(float(k), thr_i)
                mask_eval[i] = score[i] > thr_i

    if int(dilate) > 0:
        for i in range(mask_eval.shape[0]):
            mask_eval[i] = _dilate_mask(mask_eval[i], r=int(dilate))

    mask = np.zeros_like(stack, dtype=bool)
    cleaned = stack.copy()
    if roi is None:
        mask = mask_eval
        cleaned = np.where(mask_eval, med[None, :, :], stack_eval)
    else:
        ys, xs = roi.slices()
        mask[:, ys, xs] = mask_eval
        cleaned[:, ys, xs] = np.where(mask_eval, med[None, :, :], stack_eval)

    replaced_pixels = int(mask.sum())
    replaced_fraction = float(replaced_pixels) / float(stack.size)
    per_frame_fraction = [float(mask[i].mean()) for i in range(mask.shape[0])]

    out_files: dict[str, str] = {}
    write_tasks = [
        {
            "name": name,
            "out_dir": str(out_dir),
            "header": hdr,
            "image": cleaned[i],
            "mask": mask[i],
            "detector_dq": detector_dqs[i],
            "save_png": save_png,
            "save_mask_fits": save_mask_fits,
            "roi": (None if roi is None else roi.to_dict()),
            "history": "scorpio_pipe cosmics: replaced cosmic pixels with stack median",
            "cosm_k": float(k),
            "cosm_md": "stack_mad",
        }
        for i, (name, hdr) in enumerate(zip(names, headers))
    ]
    for res in parallel_map(write_tasks, _write_cosmics_frame_output, n_jobs=parallel_n_jobs, backend=parallel_backend):
        out_files[str(res["name"])] = str(res["clean_path"])

    sum_excl = np.sum(stack * (~mask), axis=0)
    cov = np.sum(~mask, axis=0).astype(np.int16)

    sum_f = out_dir / "sum_excl_cosmics.fits"
    cov_f = out_dir / "coverage.fits"
    atomic_write_fits(sum_f, sum_excl.astype(np.float32), overwrite=True)
    atomic_write_fits(cov_f, cov, overwrite=True)

    if save_png:
        _save_png(
            out_dir / "sum_excl_cosmics.png", sum_excl, title="Sum (cosmics excluded)"
        )
        _save_png(out_dir / "coverage.png", cov, title="Coverage (non-cosmic count)")

    outputs: dict[str, Any] = {
        "clean_dir": str((out_dir / "clean").resolve()),
        "sum_excl_fits": str(sum_f.resolve()),
        "coverage_fits": str(cov_f.resolve()),
    }
    if roi is not None:
        outputs["roi"] = roi.to_dict()
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

    Config:
      cosmics.method = "current"|"laplacian"
        - "current" (default): keep legacy behavior (auto-select by N frames)
        - "laplacian": single-frame Laplacian detector (requires SciPy)

    Legacy method names are still accepted for backward compatibility:
      "auto", "la_cosmic", "two_frame_diff", "stack_mad", "lap"

    Default method is `auto`:
      - 1 frame  -> `la_cosmic` (van Dokkum L.A.Cosmic-like; uses astroscrappy if available)
      - 2 frames -> `two_frame_diff`
      - ≥3       -> `stack_mad`

    Outputs:
      run_root/05_cosmics/<kind>/clean/*.fits
      run_root/05_cosmics/<kind>/summary.json

    Legacy runs may still write into ``run_root/cosmics`` if such a directory
    already exists for the current workspace.
    """
    cfg = _load_cfg_any(cfg)
    base_dir = Path(str(cfg.get("config_dir", "."))).resolve()
    data_dir = Path(str(cfg.get("data_dir", "."))).expanduser().resolve()

    work_dir = _as_path(cfg.get("work_dir", "work"))
    if not work_dir.is_absolute():
        work_dir = (base_dir / work_dir).resolve()

    # Engine code should pass ``out_dir`` explicitly. Keep a compatibility
    # fallback for tests/helper scripts that still call the stage directly.
    if out_dir is None:
        out_dir = stage_dir(work_dir, "cosmics")
    out_root = Path(out_dir)
    if not out_root.is_absolute():
        out_root = (work_dir / out_root).resolve()

    ccfg = cfg.get("cosmics", {}) or {}
    if not bool(ccfg.get("enabled", True)):
        out_root.mkdir(parents=True, exist_ok=True)
        out_path = out_root / "summary.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"disabled": True}, f, indent=2, ensure_ascii=False)
        return out_path.resolve()

    method = str(ccfg.get("method", "auto")).strip().lower()
    # Backward/forward compatible aliases
    if method in ("current", "default"):
        method = "auto"
    mode_profile = str(ccfg.get("mode_profile", "current") or "current").strip().lower()
    roi_cfg = ccfg.get("roi", None)
    perf_cfg = ccfg.get("perf") if isinstance(ccfg.get("perf"), dict) else {}
    parallel_n_jobs = perf_cfg.get("n_jobs")
    parallel_backend = str(perf_cfg.get("backend", "thread") or "thread").strip().lower()

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

    lap_niter = _as_int(ccfg.get("lap_niter", 2), 2)
    lap_edge_margin = _as_int(ccfg.get("lap_edge_margin", 2), 2)

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

    out_root = Path(out_root)
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

        paths = [
            _resolve_path(
                _as_path(pp), data_dir=data_dir, work_dir=work_dir, base_dir=base_dir
            )
            for pp in rel_paths
        ]
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
                log.warning(
                    "method=laplacian requires exactly 1 frame; falling back to auto"
                )
            if method in ("two_frame_diff", "diff") and len(paths) != 2:
                log.warning(
                    "method=two_frame_diff requires exactly 2 frames; falling back to auto"
                )
            if method in ("la_cosmic", "lacosmic") and len(paths) != 1:
                log.warning(
                    "method=la_cosmic requires exactly 1 frame; falling back to auto"
                )

            if len(paths) >= 3 and method in (
                "auto",
                "stack_mad",
                "mad_stack",
                "stack",
            ):
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
                    cfg=cfg,
                    roi=roi_cfg,
                    parallel_n_jobs=parallel_n_jobs,
                    parallel_backend=parallel_backend,
                )
            elif len(paths) == 2 and method in (
                "auto",
                "stack_mad",
                "mad_stack",
                "stack",
                "two_frame_diff",
                "diff",
            ):
                summary = _two_frame_diff_clean(
                    paths,
                    out_dir=kind_out,
                    superbias=superbias,
                    k=k,
                    dilate=int(
                        two_diff_dilate if two_diff_dilate is not None else dilate
                    ),
                    local_r=int(
                        two_diff_local_r if two_diff_local_r is not None else local_r
                    ),
                    k2_scale=two_diff_k2_scale,
                    k2_min=two_diff_k2_min,
                    thr_local_a=two_diff_thr_local_a,
                    thr_local_b=two_diff_thr_local_b,
                    bias_subtract=bias_subtract,
                    save_png=save_png,
                    save_mask_fits=save_mask_fits,
                    cfg=cfg,
                    roi=roi_cfg,
                    parallel_n_jobs=parallel_n_jobs,
                    parallel_backend=parallel_backend,
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
                    cfg=cfg,
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
                    roi=roi_cfg,
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
                    niter=lap_niter,
                    edge_margin=lap_edge_margin,
                    bias_subtract=bias_subtract,
                    save_png=save_png,
                    save_mask_fits=save_mask_fits,
                    cfg=cfg,
                    roi=roi_cfg,
                )
            else:
                # Nothing to do
                (kind_out / "clean").mkdir(parents=True, exist_ok=True)
                outputs = {
                    "note": f"not enough frames for cosmics cleaning (n={len(paths)})"
                }
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
            manual_replace_r = _as_int(
                ccfg.get("manual_replace_r", la_replace_r), la_replace_r
            )
            if preserve_manual:
                try:
                    repix, refrac, per = _apply_manual_masks(
                        kind_out, replace_r=int(manual_replace_r), save_png=save_png
                    )
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
                            "auto_masks_fits_dir": str(
                                (kind_out / "auto_masks_fits").resolve()
                            ),
                            "manual_masks_fits_dir": str(
                                (kind_out / "manual_masks_fits").resolve()
                            ),
                            "sum_excl_fits": str(
                                (kind_out / "sum_excl_cosmics.fits").resolve()
                            ),
                            "coverage_fits": str(
                                (kind_out / "coverage.fits").resolve()
                            ),
                        },
                    )
                except Exception as e:
                    log.warning(
                        "Failed to apply manual cosmics masks for %s: %s",
                        kind,
                        e,
                        exc_info=True,
                    )
                    # A qualified DQ_V1 product cannot be reported complete
                    # after a partial or failed manual rewrite. Legacy runs
                    # retain their original nonfatal behavior.
                    if any(
                        str(fits.getheader(cf).get("SATMSK") or "").strip().upper()
                        == "DQ_V1"
                        for cf in (kind_out / "clean").glob("*_clean.fits")
                    ):
                        raise RuntimeError(
                            f"Qualified cosmics DQ manual processing failed for {kind}"
                        ) from e
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

        # Write per-kind QC snapshot (machine-readable) for UI/diagnostics.
        try:
            qc_obj = {
                "kind": summary.kind,
                "method": method,
                "mode_profile": mode_profile,
                "n_frames": summary.n_frames,
                "k": float(summary.k),
                "replaced_pixels": int(summary.replaced_pixels),
                "replaced_fraction": float(summary.replaced_fraction),
                "per_frame_fraction": [float(x) for x in summary.per_frame_fraction],
                "outputs": dict(summary.outputs),
            }
            qc_path = kind_out / "qc.json"
            with qc_path.open("w", encoding="utf-8") as f:
                json.dump(qc_obj, f, indent=2, ensure_ascii=False)
            # Attach to outputs for discoverability
            outs2 = dict(summary.outputs)
            outs2["qc_json"] = str(qc_path.resolve())
            summary = CosmicsSummary(
                kind=summary.kind,
                n_frames=summary.n_frames,
                k=summary.k,
                replaced_pixels=summary.replaced_pixels,
                replaced_fraction=summary.replaced_fraction,
                per_frame_fraction=summary.per_frame_fraction,
                outputs=outs2,
            )
        except Exception as e:
            log.warning("Failed to write cosmics qc.json for %s: %s", kind, e, exc_info=True)


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
                "mode_profile": mode_profile,
                "roi": ({k: int(v) for k, v in dict(summary.outputs.get("roi") or {}).items() if k in {"y0", "y1", "x0", "x1"}} if isinstance(summary.outputs, dict) and summary.outputs.get("roi") else None),
            }
        )

        # write per-kind summary too
        with (kind_out / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(all_summaries[-1], f, indent=2, ensure_ascii=False)

    out = {
        "method": method,
        "mode_profile": mode_profile,
        "k": float(k),
        "bias_subtract": bias_subtract,
        "save_png": save_png,
        "save_mask_fits": save_mask_fits,
        "roi": roi_cfg,
        "dilate": int(dilate),
        "stack": {
            "mad_scale": float(mad_scale),
            "min_mad": float(min_mad),
            "max_frac_per_frame": (
                float(max_frac_per_frame) if max_frac_per_frame is not None else None
            ),
            "dilate": int(stack_dilate if stack_dilate is not None else dilate),
        },
        "two_frame_diff": {
            "local_r": int(
                two_diff_local_r if two_diff_local_r is not None else local_r
            ),
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
            "niter": int(lap_niter),
            "edge_margin": int(lap_edge_margin),
        },
        "items": all_summaries,
    }

    out_path = out_root / "summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return out_path.resolve()
