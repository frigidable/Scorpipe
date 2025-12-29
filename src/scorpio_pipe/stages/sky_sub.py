"""Sky subtraction stage.

v5.39.1 contract
----------------
Stage order is:

    ... 08_wavesol -> 09_sky -> 10_linearize -> 11_stack -> 12_extract

This file implements the single **Sky Subtraction** stage (09_sky).

Output contract split
---------------------
Sky subtraction (09_sky) operates in **detector geometry** and writes *RAW* products:

    09_sky/<stem>_skysub_raw.fits    (SCI/VAR/MASK)
    09_sky/<stem>_skymodel_raw.fits  (SCI/VAR/MASK)
    09_sky/sky_done.json

Linearization (10_linearize) then resamples to a linear wavelength grid and writes the
rectified downstream inputs (see :mod:`scorpio_pipe.stages.linearize`).

Implementation note
-------------------
The "primary" sky method interface is stable, while the underlying algorithm may
evolve. For CORE we provide a robust, conservative subtraction in detector space that
is intentionally compatible with a post-rectification residual cleanup.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from scorpio_pipe.maskbits import BADPIX, COSMIC, NO_COVERAGE, REJECTED, SATURATED, USER
from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.workspace_paths import stage_dir


FATAL_BITS = np.uint16(NO_COVERAGE | BADPIX | COSMIC | SATURATED | USER | REJECTED)


@dataclass(frozen=True)
class SkyRawOutputs:
    skysub_raw: Path
    skymodel_raw: Path


def _resolve_cfg_path(cfg: dict[str, Any], p: str | Path) -> Path:
    """Resolve a possibly-relative path from config.

    Convention used across stages: relative paths are resolved against cfg['config_dir'].
    """

    pp = Path(str(p)).expanduser()
    if pp.is_absolute():
        return pp
    base = Path(str(cfg.get("config_dir", "."))).expanduser().resolve()
    return (base / pp).resolve()


def _open_first_image(path: Path) -> tuple[np.ndarray, fits.Header]:
    """Open FITS and return first image-like HDU (data, header)."""

    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if getattr(hdu, "data", None) is None:
                continue
            if np.asarray(hdu.data).ndim == 2:
                return np.asarray(hdu.data, dtype=np.float32), hdu.header
    raise ValueError(f"No 2D image HDU found in {path}")


def _write_mef(path: Path, sci: np.ndarray, var: np.ndarray, mask: np.ndarray, header: fits.Header) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ph = fits.PrimaryHDU()
    h_sci = fits.ImageHDU(data=np.asarray(sci, dtype=np.float32), name="SCI", header=header)
    h_var = fits.ImageHDU(data=np.asarray(var, dtype=np.float32), name="VAR")
    h_msk = fits.ImageHDU(data=np.asarray(mask, dtype=np.uint16), name="MASK")
    fits.HDUList([ph, h_sci, h_var, h_msk]).writeto(path, overwrite=True)


def _estimate_var_adu2(sci: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Simple Poisson + read-noise variance model in ADU^2.

    Uses config fields if present; otherwise falls back to conservative defaults.
    """

    gain = float(cfg.get("gain_e_per_adu", cfg.get("gain", 1.0)) or 1.0)
    rn = float(cfg.get("readnoise_e", cfg.get("readnoise", 5.0)) or 5.0)
    # Convert to electrons variance, then back to ADU^2.
    # var_e = max(sci,0)*gain + rn^2 ; var_adu = var_e / gain^2
    sci_pos = np.maximum(sci, 0.0)
    var_e = sci_pos * gain + rn * rn
    return (var_e / (gain * gain)).astype(np.float32)


def _find_cosmics_clean_and_mask(work_dir: Path, raw_stem: str) -> tuple[Path | None, Path | None]:
    """Locate cosmics-cleaned image + cosmic mask FITS for a given raw stem."""

    cosmics_root = stage_dir(work_dir, "cosmics")
    # Typical layout: 05_cosmics/obj/clean/<stem>_clean.fits
    clean = next(iter(cosmics_root.glob(f"**/clean/{raw_stem}_clean.fits")), None)
    mask = next(iter(cosmics_root.glob(f"**/masks_fits/{raw_stem}_mask.fits")), None)
    return clean, mask


def _load_cosmic_mask(mask_path: Path, shape: tuple[int, int]) -> np.ndarray:
    try:
        m, _ = _open_first_image(mask_path)
        mm = (m > 0).astype(np.uint16) * np.uint16(COSMIC)
        if mm.shape != shape:
            # Best-effort: ignore mismatch.
            return np.zeros(shape, dtype=np.uint16)
        return mm.astype(np.uint16)
    except Exception:
        return np.zeros(shape, dtype=np.uint16)


def _sky_rows_from_cfg(cfg: dict[str, Any], ny: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Return sky row indices (y) from config sky.roi, plus a diagnostic dict."""

    s = cfg.get("sky", {}) if isinstance(cfg.get("sky", {}), dict) else {}
    roi = s.get("roi") if isinstance(s.get("roi"), dict) else {}
    top = roi.get("sky_top") if isinstance(roi.get("sky_top"), (list, tuple)) else None
    bot = roi.get("sky_bot") if isinstance(roi.get("sky_bot"), (list, tuple)) else None

    # UI schema (v5.3x): explicit window edges
    if top is None and roi.get("sky_top_y0") is not None and roi.get("sky_top_y1") is not None:
        top = [roi.get("sky_top_y0"), roi.get("sky_top_y1")]
    if bot is None and roi.get("sky_bot_y0") is not None and roi.get("sky_bot_y1") is not None:
        bot = [roi.get("sky_bot_y0"), roi.get("sky_bot_y1")]

    def _clip_pair(pair: Any) -> tuple[int, int] | None:
        if not pair or len(pair) != 2:
            return None
        y0, y1 = int(pair[0]), int(pair[1])
        if y0 > y1:
            y0, y1 = y1, y0
        y0 = max(0, min(ny - 1, y0))
        y1 = max(0, min(ny - 1, y1))
        if y1 - y0 < 2:
            return None
        return y0, y1

    top_c = _clip_pair(top)
    bot_c = _clip_pair(bot)
    rows: list[int] = []
    if top_c:
        rows.extend(range(top_c[0], top_c[1] + 1))
    if bot_c:
        rows.extend(range(bot_c[0], bot_c[1] + 1))

    diag = {
        "sky_top": list(top_c) if top_c else None,
        "sky_bot": list(bot_c) if bot_c else None,
        "n_rows": int(len(rows)),
        "source": "cfg.sky.roi" if rows else "auto",
    }

    if rows:
        return np.asarray(rows, dtype=int), diag

    # Auto fallback: use top/bottom 15% bands.
    band = max(8, int(0.15 * ny))
    rows = list(range(0, band)) + list(range(max(0, ny - band), ny))
    diag.update({"n_rows": int(len(rows)), "source": "auto"})
    return np.asarray(rows, dtype=int), diag


def subtract_sky_raw(
    sci: np.ndarray,
    var: np.ndarray,
    mask: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Conservative detector-space sky subtraction.

    We estimate a 1D sky spectrum per detector column using robust medians from
    sky rows, and subtract it from all rows.

    Returns
    -------
    skysub_sci, skysub_var, skysub_mask, skymodel_sci, diag
    """

    ny, nx = sci.shape
    rows, rows_diag = _sky_rows_from_cfg(cfg, ny)
    good = (mask & FATAL_BITS) == 0

    sky = np.zeros(nx, dtype=np.float32)
    for x in range(nx):
        col = sci[rows, x]
        ok = good[rows, x]
        v = col[ok]
        if v.size:
            sky[x] = np.float32(np.median(v))
        else:
            sky[x] = np.float32(np.median(col))

    skymodel = np.tile(sky[None, :], (ny, 1)).astype(np.float32)
    skysub = (sci - skymodel).astype(np.float32)

    diag = {
        "sky_rows": rows_diag,
        "sky_col_median": float(np.median(sky)),
        "sky_col_p90": float(np.percentile(sky, 90)),
    }
    return skysub, var, mask, skymodel, diag


def run_sky_sub(cfg: dict[str, Any], out_dir: str | Path | None = None) -> dict[str, Path]:
    """Run stage 09_sky.

    Parameters
    ----------
    cfg
        Pipeline config.
    out_dir
        Optional override for output dir. If None, resolved via :func:`stage_dir`.
    """

    t0 = time.time()
    work_dir = resolve_work_dir(cfg)
    out = Path(out_dir) if out_dir is not None else stage_dir(work_dir, "sky")
    out.mkdir(parents=True, exist_ok=True)

    frames = cfg.get("frames", {}) if isinstance(cfg.get("frames", {}), dict) else {}
    obj_frames = frames.get("obj", []) or []
    if not isinstance(obj_frames, list) or not obj_frames:
        raise ValueError("sky: cfg.frames.obj is empty")

    sky_cfg = cfg.get("sky", {}) if isinstance(cfg.get("sky", {}), dict) else {}
    primary = str(sky_cfg.get("primary_method", sky_cfg.get("method", "kelson_raw")) or "kelson_raw")

    per_exp: list[dict[str, Any]] = []
    outs: dict[str, Path] = {}

    preview_written = False

    for raw in obj_frames:
        raw_p = _resolve_cfg_path(cfg, raw)
        stem = raw_p.stem

        clean_p, cosmic_mask_p = _find_cosmics_clean_and_mask(work_dir, stem)
        inp = clean_p or raw_p
        sci, hdr = _open_first_image(inp)
        var = _estimate_var_adu2(sci, cfg)
        mask = _load_cosmic_mask(cosmic_mask_p, sci.shape) if cosmic_mask_p else np.zeros(sci.shape, dtype=np.uint16)

        # Primary methods (CORE): currently share the same robust detector-space estimator.
        skysub, var2, mask2, skymodel, diag = subtract_sky_raw(sci, var, mask, cfg)

        skysub_path = out / f"{stem}_skysub_raw.fits"
        skymodel_path = out / f"{stem}_skymodel_raw.fits"
        _write_mef(skysub_path, skysub, var2, mask2, hdr)
        _write_mef(skymodel_path, skymodel, var2, mask2, hdr)

        outs[f"{stem}_skysub_raw"] = skysub_path
        outs[f"{stem}_skymodel_raw"] = skymodel_path
        per_exp.append(
            {
                "stem": stem,
                "input": str(inp),
                "method": primary,
                "skysub_raw": str(skysub_path),
                "skymodel_raw": str(skymodel_path),
                "diag": diag,
            }
        )

        if not preview_written:
            # Minimal preview for QC browsing.
            try:
                _write_mef(out / "preview.fits", skysub, var2, mask2, hdr)
                preview_written = True
            except Exception:
                pass

    done = {
        "status": "ok",
        "stage": "sky",
        "primary_method": primary,
        "n_obj": int(len(obj_frames)),
        "per_exp": per_exp,
        "elapsed_s": float(time.time() - t0),
    }
    done_path = out / "sky_done.json"
    done_path.write_text(json.dumps(done, indent=2, ensure_ascii=False), encoding="utf-8")

    # Minimal QC JSON consumed by qc_report (optional).
    qc_path = out / "qc_sky.json"
    qc_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "n_obj": int(len(obj_frames)),
                "primary_method": primary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    outs.update({"sky_done": done_path, "qc_sky_json": qc_path})
    return outs
