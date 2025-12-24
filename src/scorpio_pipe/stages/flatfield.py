from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io import fits

from scorpio_pipe.app_paths import ensure_dir
from scorpio_pipe.wavesol_paths import resolve_work_dir


def _read_fits(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header.copy()
    if data is None:
        raise ValueError(f"Empty FITS data: {path}")
    return np.asarray(data), hdr


def _write_fits(path: Path, data: np.ndarray, hdr: fits.Header) -> None:
    ensure_dir(path.parent)
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


def _robust_median(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.nanmedian(x))


def build_superflat(flat_paths: Iterable[Path], superbias_path: Path, out_path: Path) -> Path:
    """Build a normalized superflat (median ~ 1) from flat frames.

    Steps:
      1) subtract superbias
      2) normalize each flat by its median
      3) combine via median
      4) normalize final superflat to median=1
    """

    flat_paths = list(flat_paths)
    if not flat_paths:
        raise ValueError("No flat frames provided")

    superbias, _ = _read_fits(superbias_path)

    stack: list[np.ndarray] = []
    first_hdr: fits.Header | None = None

    for p in flat_paths:
        data, hdr = _read_fits(p)
        if first_hdr is None:
            first_hdr = hdr

        if superbias.shape != data.shape:
            raise ValueError(
                f"Shape mismatch: flat {p.name} {data.shape} vs superbias {superbias.shape}"
            )

        data = data.astype(np.float32) - superbias.astype(np.float32)

        med = _robust_median(data)
        if not np.isfinite(med) or med == 0:
            continue
        stack.append((data / med).astype(np.float32))

    if not stack:
        raise ValueError("All flats became invalid after normalization (median=0 or NaN)")

    superflat = np.nanmedian(np.stack(stack, axis=0), axis=0).astype(np.float32)

    # Final normalize
    med_sf = _robust_median(superflat)
    if np.isfinite(med_sf) and med_sf != 0:
        superflat = (superflat / med_sf).astype(np.float32)

    hdr = (first_hdr or fits.Header()).copy()
    hdr["HISTORY"] = "scorpio_pipe flatfield: superbias-subtracted flats, median-combined"
    hdr["BIASSUB"] = (True, "Superbias subtracted")
    hdr["SFLAT"] = (True, "Superflat created")
    hdr["SFLATMED"] = (float(_robust_median(superflat)), "Superflat median (after norm)")

    _write_fits(out_path, superflat, hdr)
    return out_path


def apply_flat(
    data_path: Path,
    superflat_path: Path,
    superbias_path: Path,
    out_path: Path,
    *,
    do_bias_subtract: bool = True,
) -> Path:
    """Apply flatfield correction to a single frame.

    - subtract superbias (if requested and not already marked as BIASSUB)
    - divide by superflat
    """

    data, hdr = _read_fits(data_path)
    superflat, _ = _read_fits(superflat_path)

    data = data.astype(np.float32)

    if do_bias_subtract and not bool(hdr.get("BIASSUB", False)):
        superbias, _ = _read_fits(superbias_path)
        if superbias.shape == data.shape:
            data = data - superbias.astype(np.float32)
            hdr["BIASSUB"] = (True, "Superbias subtracted")
            hdr["HISTORY"] = "scorpio_pipe flatfield: superbias subtracted"

    # Avoid division by zero
    sf = superflat.astype(np.float32)
    sf = np.where(np.isfinite(sf) & (sf != 0), sf, np.nan)

    corr = (data / sf).astype(np.float32)
    hdr["FLATCOR"] = (True, "Flat-fielding applied")
    hdr["HISTORY"] = "scorpio_pipe flatfield: divided by superflat"

    _write_fits(out_path, corr, hdr)
    return out_path


def _resolve_path(p: str | Path, base: Path) -> Path:
    pp = Path(str(p)).expanduser()
    if not pp.is_absolute():
        pp = (base / pp).resolve()
    return pp


def run_flatfield(cfg: dict, *, out_dir: Path | None = None) -> Path:
    """Run the Flat-fielding stage.

    Layout
    ------
    - calib/superflat.fits  (created if absent)
    - flatfield/<kind>/*_flat.fits
    - flatfield/flatfield_done.json

    Notes
    -----
    - If cosmics stage was run with bias_subtract=True, then the cosmics-cleaned
      frames are already superbias-subtracted (BIASSUB=True). In that case, we
      avoid subtracting the superbias again.
    """

    work_dir = resolve_work_dir(cfg)
    out_dir = out_dir or (work_dir / "flatfield")
    ensure_dir(out_dir)

    block = cfg.get("flatfield", {}) or {}
    enabled = bool(block.get("enabled", False))

    done_path = out_dir / "flatfield_done.json"

    if not enabled:
        done_path.write_text(
            json.dumps({"enabled": False, "status": "skipped"}, indent=2),
            encoding="utf-8",
        )
        return done_path.resolve()

    frames = cfg.get("frames", {}) or {}

    superbias_path = _resolve_path(
        (cfg.get("calib", {}) or {}).get("superbias_path", work_dir / "calib" / "superbias.fits"),
        work_dir,
    )
    superflat_path = _resolve_path(
        (cfg.get("calib", {}) or {}).get("superflat_path", work_dir / "calib" / "superflat.fits"),
        work_dir,
    )

    flat_paths = [_resolve_path(p, work_dir) for p in (frames.get("flat") or [])]
    if not flat_paths:
        raise ValueError("No flat frames selected (frames.flat is empty)")

    # Always build / refresh superflat for the current object.
    ensure_dir(superflat_path.parent)
    build_superflat(flat_paths, superbias_path, superflat_path)

    apply_to = list(block.get("apply_to") or ["obj", "sky", "sunsky"])  # + optional 'neon'

    cosmics_bias_sub = bool((cfg.get("cosmics", {}) or {}).get("bias_subtract", True))
    do_bias_sub_flat = bool(block.get("bias_subtract", True))

    outputs: list[str] = []

    for kind in apply_to:
        kind_frames = frames.get(kind) or []
        if not isinstance(kind_frames, list) or not kind_frames:
            continue

        kind_out = out_dir / kind
        ensure_dir(kind_out)

        kind_clean_dir = work_dir / "cosmics" / kind / "clean"

        for fp in kind_frames:
            src0 = _resolve_path(fp, work_dir)
            if not src0.exists():
                continue

            clean = kind_clean_dir / f"{src0.stem}_clean.fits"
            src = clean if clean.exists() else src0

            # If we use a cosmics-cleaned product AND cosmics already subtracted bias,
            # then avoid doing it again.
            do_bias_subtract = do_bias_sub_flat
            if clean.exists() and cosmics_bias_sub:
                do_bias_subtract = False

            dst = kind_out / f"{src0.stem}_flat.fits"
            apply_flat(src, superflat_path, superbias_path, dst, do_bias_subtract=do_bias_subtract)
            outputs.append(str(dst))

    done = {
        "enabled": True,
        "status": "ok",
        "superbias": str(superbias_path),
        "superflat": str(superflat_path),
        "apply_to": apply_to,
        "outputs": outputs,
        "n_outputs": len(outputs),
    }

    done_path.write_text(json.dumps(done, indent=2, ensure_ascii=False), encoding="utf-8")
    return done_path.resolve()
