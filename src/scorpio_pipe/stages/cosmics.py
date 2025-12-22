from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from astropy.io import fits

import logging

log = logging.getLogger(__name__)


def _load_cfg_any(cfg: Any) -> Dict:
    """
    Принимает:
      - путь к yaml (str/Path)
      - dict
      - dataclass-подобный объект (с полями work_dir/frames)
    Возвращает dict-структуру конфигурации.
    """
    if isinstance(cfg, (str, Path)):
        from scorpio_pipe.config import load_config

        return load_config(cfg)
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "__dict__"):
        d = dict(cfg.__dict__)
        if "frames" in d and not isinstance(d["frames"], dict) and hasattr(d["frames"], "__dict__"):
            d["frames"] = dict(d["frames"].__dict__)
        return d
    raise TypeError(f"Unsupported cfg type: {type(cfg)}")


def _median_filter_3x3(img: np.ndarray) -> np.ndarray:
    pad = np.pad(img, 1, mode="edge")
    h, w = img.shape
    windows = []
    for dy in range(3):
        for dx in range(3):
            windows.append(pad[dy : dy + h, dx : dx + w])
    stack = np.stack(windows, axis=0)
    return np.median(stack, axis=0)


def _clean_cosmics_single(img: np.ndarray, sigma_clip: float) -> tuple[np.ndarray, int]:
    med = _median_filter_3x3(img)
    resid = img - med
    resid_med = np.median(resid)
    mad = np.median(np.abs(resid - resid_med))
    sigma = float(1.4826 * mad)
    if sigma <= 0:
        sigma = float(np.std(resid)) or 1.0
    mask = resid > (sigma_clip * sigma)
    cleaned = img.copy()
    cleaned[mask] = med[mask]
    return cleaned, int(mask.sum())


def clean_cosmics(cfg: Any, out_dir: str | Path | None = None) -> Path:
    """
    Простая очистка космиков на 2D кадрах:
    - медианный фильтр 3x3
    - выбросы выше sigma_clip заменяются медианой

    Пишет очищенные FITS в work_dir/cosmics/<kind>/... и summary.json.
    """
    c = _load_cfg_any(cfg)
    work_dir = Path(c["work_dir"])

    cosm_cfg = c.get("cosmics") or {}
    enabled = bool(cosm_cfg.get("enabled", True))
    if not enabled:
        log.info("Cosmics cleaning disabled by config (cosmics.enabled=false)")
        out_dir = Path(out_dir) if out_dir is not None else (work_dir / "cosmics")
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {"enabled": False, "frames": {}}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return out_dir / "summary.json"

    sigma_clip = float(cosm_cfg.get("sigma_clip", 5.0) or 5.0)
    apply_to = cosm_cfg.get("apply_to", ["obj", "sky"])
    if not isinstance(apply_to, list):
        apply_to = ["obj", "sky"]

    frames = c.get("frames") or {}
    if not isinstance(frames, dict):
        raise ValueError("Config.frames must be a dict")

    out_dir = Path(out_dir) if out_dir is not None else (work_dir / "cosmics")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, int]] = {}
    total = 0

    for kind in apply_to:
        paths = [Path(p) for p in frames.get(kind, []) if isinstance(p, (str, Path))]
        if not paths:
            continue
        kind_dir = out_dir / str(kind)
        kind_dir.mkdir(parents=True, exist_ok=True)
        summary[kind] = {"frames": 0, "pixels_replaced": 0}
        for p in paths:
            try:
                data, hdr = fits.getdata(p, header=True, memmap=False)
            except Exception:
                log.warning("Failed to read FITS: %s", p)
                continue
            if data is None or data.ndim != 2:
                log.warning("Skip non-2D FITS: %s", p)
                continue
            cleaned, n_pix = _clean_cosmics_single(data.astype(np.float32), sigma_clip)
            out_path = kind_dir / f"{p.stem}_cr.fits"
            hdr = hdr or fits.Header()
            hdr.add_history("Cosmics cleaned by scorpio_pipe.stages.cosmics.clean_cosmics")
            hdr["CR_SIGMA"] = (float(sigma_clip), "Sigma threshold for cosmic cleaning")
            hdr["CR_PIX"] = (int(n_pix), "Pixels replaced by median filter")
            fits.writeto(out_path, cleaned.astype(np.float32), hdr, overwrite=True)
            summary[kind]["frames"] += 1
            summary[kind]["pixels_replaced"] += int(n_pix)
            total += 1

    if total == 0:
        raise ValueError("No frames found for cosmic cleaning (check cosmics.apply_to and frames)")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Cosmics cleaning done: %s", summary_path)
    return summary_path
