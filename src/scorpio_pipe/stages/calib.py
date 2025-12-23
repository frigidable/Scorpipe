from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from astropy.io import fits

import logging

log = logging.getLogger(__name__)


def _load_cfg_any(cfg: Any) -> Dict:
    """Normalize config input (path/dict/RunContext) into a config dict."""
    from scorpio_pipe.config import load_config_any
    return load_config_any(cfg)


def _read_fits_data(path: Path) -> Tuple[np.ndarray, fits.Header]:
    # максимально живучее открытие
    with fits.open(path, memmap=False, ignore_missing_end=True, ignore_missing_simple=True) as hdul:
        return hdul[0].data, hdul[0].header


def build_superbias(cfg: Any, out_path: str | Path | None = None) -> Path:
    """
    Строит superbias (по умолчанию — медианой) по bias-кадрам.
    Пишет FITS в work_dir/calib/superbias.fits (или out_path).
    """
    c = _load_cfg_any(cfg)

    work_dir = Path(c["work_dir"])
    bias_paths = [Path(p) for p in c["frames"]["bias"]]

    if not bias_paths:
        raise ValueError("No bias frames in config.frames.bias")

    log.info("Superbias: %d input bias frames", len(bias_paths))

    setup = c.get("frames", {}).get("__setup__", {}) or {}
    target_shape_str = str(setup.get("shape", "") or "")


    # читаем первый кадр, чтобы зафиксировать форму
    # если в конфиге указан target shape (из науки объекта) — используем его
    target_shape = None
    if "x" in target_shape_str:
        try:
            ny, nx = target_shape_str.split("x")
            target_shape = (int(ny), int(nx))
        except Exception:
            target_shape = None

    # выберем первый bias, который подходит по форме (или просто первый, если target_shape неизвестен)
    h0 = None
    shape = None
    filtered = []
    bad_open = 0

    for p in bias_paths:
        try:
            d, h = _read_fits_data(p)
        except Exception:
            bad_open += 1
            continue
        if d is None:
            bad_open += 1
            continue
        if target_shape is not None and d.shape != target_shape:
            continue
        if shape is None:
            shape = d.shape
            h0 = h
        filtered.append(p)

    if not filtered:
        raise RuntimeError(f"No bias frames match target shape={target_shape_str!r}. "
                           f"Check config.frames.__setup__.shape and your bias shapes.")

    bias_paths = filtered


    calib_cfg = c.get("calib", {}) or {}
    combine = str(calib_cfg.get("bias_combine", "median")).lower()
    # Project requirement: superbias is built with the most robust median combine.
    # Keep backward compatibility for old configs but force the method.
    if combine != "median":
        log.warning("Superbias: forcing combine=median (was %s)", combine)
        combine = "median"
    sigma_clip = float(calib_cfg.get("bias_sigma_clip", 0.0) or 0.0)

    log.info("Superbias settings: combine=%s, sigma_clip=%g", combine, sigma_clip)

    bad = 0

    # Median: always stack frames (bias sets are typically small: ~10–50).
    # This gives the most robust superbias and matches the "IDL-style" workflow.
    if combine == "median":
        frames = []
        for p in bias_paths:
            try:
                d, _ = _read_fits_data(p)
                if d is None or d.shape != shape:
                    bad += 1
                    continue
                frames.append(d.astype(np.float32))
            except Exception:
                bad += 1

        if not frames:
            raise RuntimeError("All bias frames failed to read or had wrong shape")

        arr = np.stack(frames, axis=0)
        superbias = np.median(arr, axis=0).astype(np.float32)
        n_used = int(arr.shape[0])

    # If requested, do a simple sigma-clipped mean in a single stack
    # (rare cosmic hits / bad pixels).
    elif sigma_clip > 0:
        frames = []
        for p in bias_paths:
            try:
                d, _ = _read_fits_data(p)
                if d is None or d.shape != shape:
                    bad += 1
                    continue
                frames.append(d.astype(np.float32))
            except Exception:
                bad += 1

        if not frames:
            raise RuntimeError("All bias frames failed to read or had wrong shape")

        arr = np.stack(frames, axis=0)  # (N, H, W)
        med = np.median(arr, axis=0)
        mad = np.median(np.abs(arr - med), axis=0)
        sig = 1.4826 * mad
        sig[sig <= 0] = 1.0
        mask = np.abs(arr - med) <= (sigma_clip * sig)
        w = mask.astype(np.float32)
        num = (arr * w).sum(axis=0)
        den = w.sum(axis=0)
        den[den == 0] = 1.0
        superbias = (num / den).astype(np.float32)
        n_used = int(arr.shape[0])

    else:
        # Streaming mean: minimal memory
        acc = np.zeros(shape, dtype=np.float64)
        n_used = 0

        for p in bias_paths:
            try:
                d, _ = _read_fits_data(p)
                if d is None or d.shape != shape:
                    bad += 1
                    continue
                acc += d.astype(np.float64)
                n_used += 1
            except Exception:
                bad += 1

        if n_used == 0:
            raise RuntimeError("All bias frames failed to read or had wrong shape")

        superbias = (acc / n_used).astype(np.float32)

    if out_path is None:
        out_dir = work_dir / "calib"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "superbias.fits"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # header
    hdr = fits.Header()
    hdr.extend(h0, update=True)
    hdr["NBIAS"] = (int(n_used), "Number of bias frames used")
    hdr["SB_BAD"] = (int(bad), "Number of bias frames skipped")
    hdr["SB_METH"] = (combine, "Combine method for superbias")
    hdr["SB_CLIP"] = (float(sigma_clip), "Sigma clip (0=disabled)")
    hdr.add_history("Built by scorpio_pipe.stages.calib.build_superbias")

    fits.writeto(out_path, superbias, hdr, overwrite=True)
    log.info("Wrote superbias: %s", out_path)
    return out_path


def build_superflat(cfg: Any, out_path: str | Path | None = None) -> Path:
    """
    Строит superflat как среднее/медиану по flat-кадрам с последующей нормировкой.
    Пишет FITS в work_dir/calib/superflat.fits (или out_path).
    """
    c = _load_cfg_any(cfg)

    work_dir = Path(c["work_dir"])
    flat_paths = [Path(p) for p in c["frames"]["flat"]]

    if not flat_paths:
        raise ValueError("No flat frames in config.frames.flat")

    log.info("Superflat: %d input flat frames", len(flat_paths))

    setup = c.get("frames", {}).get("__setup__", {}) or {}
    target_shape_str = str(setup.get("shape", "") or "")

    # читаем первый кадр, чтобы зафиксировать форму
    # если в конфиге указан target shape (из науки объекта) — используем его
    target_shape = None
    if "x" in target_shape_str:
        try:
            ny, nx = target_shape_str.split("x")
            target_shape = (int(ny), int(nx))
        except Exception:
            target_shape = None

    # выберем первый flat, который подходит по форме (или просто первый, если target_shape неизвестен)
    h0 = None
    shape = None
    filtered = []
    bad_open = 0

    for p in flat_paths:
        try:
            d, h = _read_fits_data(p)
        except Exception:
            bad_open += 1
            continue
        if d is None:
            bad_open += 1
            continue
        if target_shape is not None and d.shape != target_shape:
            continue
        if shape is None:
            shape = d.shape
            h0 = h
        filtered.append(p)

    if not filtered:
        raise RuntimeError(f"No flat frames match target shape={target_shape_str!r}. "
                           f"Check config.frames.__setup__.shape and your flat shapes.")

    flat_paths = filtered

    calib_cfg = c.get("calib", {}) or {}
    combine = str(calib_cfg.get("flat_combine", "mean")).lower()
    sigma_clip = float(calib_cfg.get("flat_sigma_clip", 0.0) or 0.0)

    log.info("Superflat settings: combine=%s, sigma_clip=%g", combine, sigma_clip)

    bad = 0

    # If requested, do a simple sigma-clipped mean in a single stack
    # (flat sets are usually manageable and sigma clipping helps with bad pixels).
    if sigma_clip > 0:
        frames = []
        for p in flat_paths:
            try:
                d, _ = _read_fits_data(p)
                if d is None or d.shape != shape:
                    bad += 1
                    continue
                frames.append(d.astype(np.float32))
            except Exception:
                bad += 1

        if not frames:
            raise RuntimeError("All flat frames failed to read or had wrong shape")

        arr = np.stack(frames, axis=0)  # (N, H, W)
        if combine == "median":
            superflat = np.median(arr, axis=0).astype(np.float32)
        else:
            med = np.median(arr, axis=0)
            mad = np.median(np.abs(arr - med), axis=0)
            sig = 1.4826 * mad
            sig[sig <= 0] = 1.0
            mask = np.abs(arr - med) <= (sigma_clip * sig)
            w = mask.astype(np.float32)
            num = (arr * w).sum(axis=0)
            den = w.sum(axis=0)
            den[den == 0] = 1.0
            superflat = (num / den).astype(np.float32)
        n_used = int(arr.shape[0])

    else:
        # Streaming mean: minimal memory
        acc = np.zeros(shape, dtype=np.float64)
        n_used = 0

        for p in flat_paths:
            try:
                d, _ = _read_fits_data(p)
                if d is None or d.shape != shape:
                    bad += 1
                    continue
                acc += d.astype(np.float64)
                n_used += 1
            except Exception:
                bad += 1

        if n_used == 0:
            raise RuntimeError("All flat frames failed to read or had wrong shape")

        if combine == "median":
            # median requested but sigma_clip==0: fall back to mean to avoid storing all frames
            superflat = (acc / n_used).astype(np.float32)
            combine = "mean"
        else:
            superflat = (acc / n_used).astype(np.float32)

    finite = np.isfinite(superflat)
    if not np.any(finite):
        raise RuntimeError("Superflat normalization failed: no finite pixels")
    norm = float(np.median(superflat[finite]))
    if norm == 0.0:
        norm = float(np.mean(superflat[finite]))
    if norm == 0.0:
        raise RuntimeError("Superflat normalization failed: zero median/mean")

    superflat = (superflat / norm).astype(np.float32)

    if out_path is None:
        out_dir = work_dir / "calib"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "superflat.fits"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # header
    hdr = fits.Header()
    hdr.extend(h0, update=True)
    hdr["NFLAT"] = (int(n_used), "Number of flat frames used")
    hdr["SF_BAD"] = (int(bad), "Number of flat frames skipped")
    hdr["SF_METH"] = (combine, "Combine method for superflat")
    hdr["SF_CLIP"] = (float(sigma_clip), "Sigma clip (0=disabled)")
    hdr["SF_NORM"] = (float(norm), "Normalization factor (median)")
    hdr.add_history("Built by scorpio_pipe.stages.calib.build_superflat")

    fits.writeto(out_path, superflat, hdr, overwrite=True)
    log.info("Wrote superflat: %s", out_path)
    return out_path
