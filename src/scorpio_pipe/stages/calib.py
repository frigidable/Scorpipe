from __future__ import annotations

from pathlib import Path
import json
import shutil
from typing import Any, Dict, Tuple

import numpy as np
from astropy.io import fits

from scorpio_pipe.frame_signature import FrameSignature, format_signature_mismatch

import logging


def _write_calib_done(work_dir: Path, name: str, payload: dict) -> Path:
    """Write a small JSON marker for calibration builders.

    GUI stages historically use per-stage *_done.json files. Superbias/superflat
    builders did not have them, but we add them for transparency/debugging.
    """
    from scorpio_pipe.work_layout import ensure_work_layout

    layout = ensure_work_layout(work_dir)
    p = layout.calibs / f"{name}_done.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def _expected_signature_from_cfg(cfg: Dict, *, fallback_path: Path | None = None) -> FrameSignature | None:
    setup = cfg.get("setup") if isinstance(cfg.get("setup"), dict) else (cfg.get("frames", {}) or {}).get("__setup__")
    if isinstance(setup, dict):
        s = FrameSignature.from_setup(setup)
        # If setup does not define shape, try to fill from fallback path.
        if (s.ny <= 0 or s.nx <= 0) and fallback_path is not None:
            try:
                fp = FrameSignature.from_path(fallback_path)
                s = FrameSignature(fp.ny, fp.nx, s.bx, s.by, window=s.window, readout=s.readout)
            except Exception:
                pass
        # If everything is unknown, ignore.
        if s.ny <= 0 and s.nx <= 0 and s.bx is None and not s.window and not s.readout:
            return None
        return s
    if fallback_path is not None:
        try:
            return FrameSignature.from_path(fallback_path)
        except Exception:
            return None
    return None


def _validate_signatures_strict(paths: list[Path], *, expected: FrameSignature | None, what: str) -> FrameSignature:
    """Ensure all frames match expected signature (or match each other if expected is None)."""
    if not paths:
        raise ValueError(f"No input frames for {what}")
    ref = expected
    if ref is None:
        ref = FrameSignature.from_path(paths[0])

    bad: list[str] = []
    for p in paths:
        try:
            sig = FrameSignature.from_path(p)
        except Exception as e:
            bad.append(f"{p.name}: read_failed: {e}")
            continue
        if not sig.is_compatible_with(ref):
            bad.append(format_signature_mismatch(expected=ref, got=sig, path=p))

    if bad:
        msg = (
            f"Incompatible {what} frame(s): expected {ref.describe()}\n"
            + "\n".join(bad[:12])
            + ("\n..." if len(bad) > 12 else "")
            + "\n\nFix: select frames with exactly the same ROI/window, binning and readout mode as your science setup."
        )
        raise RuntimeError(msg)

    return ref

log = logging.getLogger(__name__)


def _resolve_work_dir(c: Dict) -> Path:
    """Resolve work_dir robustly and ensure canonical layout exists."""
    from scorpio_pipe.paths import resolve_work_dir
    from scorpio_pipe.work_layout import ensure_work_layout

    wd = resolve_work_dir(c)
    ensure_work_layout(wd)
    return wd


def _load_cfg_any(cfg: Any) -> Dict:
    """Normalize config input (path/dict/RunContext) into a config dict."""
    from scorpio_pipe.config import load_config_any

    return load_config_any(cfg)


def _read_fits_data(path: Path) -> Tuple[np.ndarray, fits.Header]:
    # максимально живучее открытие
    with fits.open(
        path, memmap=False, ignore_missing_end=True, ignore_missing_simple=True
    ) as hdul:
        return hdul[0].data, hdul[0].header


def _resolve_superbias_path(c: Dict, work_dir: Path) -> Path:
    """Resolve superbias path with backward-compatible fallbacks."""

    from scorpio_pipe.work_layout import ensure_work_layout

    calib_cfg = c.get("calib", {}) or {}
    p = calib_cfg.get("superbias_path")
    if p:
        pp = Path(str(p)).expanduser()
        if not pp.is_absolute():
            pp = (work_dir / pp).resolve()
        return pp

    from scorpio_pipe.work_layout import ensure_work_layout

    layout = ensure_work_layout(work_dir)
    cand = [layout.calibs / "superbias.fits", layout.calib_legacy / "superbias.fits"]
    for cp in cand:
        if cp.exists():
            return cp
    # default (canonical) location even if it does not exist yet
    return cand[0]


def _robust_median_finite(a: np.ndarray) -> float:
    aa = a[np.isfinite(a)]
    if aa.size == 0:
        return float("nan")
    return float(np.median(aa))


def _build_superflat_core(
    *,
    flat_paths: list[Path],
    superbias_path: Path,
    combine: str,
    sigma_clip: float,
) -> tuple[np.ndarray, fits.Header, dict[str, int | float | str]]:
    """Canonical superflat builder used by *all* call paths.

    This is the *single source of truth* for superflat construction.
    """

    try:
        superbias, sb_hdr = _read_fits_data(superbias_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read superbias: {superbias_path}") from e
    if superbias is None:
        raise RuntimeError(f"Empty superbias FITS data: {superbias_path}")

    superbias = superbias.astype(np.float32)

    sb_sig = FrameSignature.from_header(sb_hdr, fallback_shape=superbias.shape)

    stack: list[np.ndarray] = []
    first_hdr: fits.Header | None = None
    bad_read = 0
    bad_shape = 0
    bad_norm = 0

    for p in flat_paths:
        try:
            d, h = _read_fits_data(p)
        except Exception:
            bad_read += 1
            continue
        if d is None:
            bad_read += 1
            continue
        if first_hdr is None:
            first_hdr = h
        sig = FrameSignature.from_header(h, fallback_shape=d.shape)
        if not sig.is_compatible_with(sb_sig):
            raise RuntimeError(
                "Flat frame is incompatible with superbias (ROI/binning/readout mismatch):\n"
                + format_signature_mismatch(expected=sb_sig, got=sig, path=p)
                + "\n\nFix: rebuild/choose flats and superbias with identical setup (binning + window + readout)."
            )

        data = d.astype(np.float32) - superbias
        med = _robust_median_finite(data)
        if not np.isfinite(med) or med == 0.0:
            bad_norm += 1
            continue

        stack.append((data / med).astype(np.float32))

    if not stack:
        raise RuntimeError(
            "All flats failed to read, mismatched shape, or became invalid after normalization"
        )

    arr = np.stack(stack, axis=0)  # (N, H, W)
    n_used = int(arr.shape[0])

    combine = (combine or "mean").strip().lower()
    if combine not in {"mean", "median"}:
        combine = "mean"

    if combine == "median":
        superflat = np.median(arr, axis=0).astype(np.float32)
    else:
        if sigma_clip > 0:
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
        else:
            superflat = np.mean(arr, axis=0).astype(np.float32)

    finite = np.isfinite(superflat)
    if not np.any(finite):
        raise RuntimeError("Superflat normalization failed: no finite pixels")
    norm = float(np.median(superflat[finite]))
    if norm == 0.0:
        norm = float(np.mean(superflat[finite]))
    if norm == 0.0:
        raise RuntimeError("Superflat normalization failed: zero median/mean")

    superflat = (superflat / norm).astype(np.float32)

    hdr = fits.Header()
    if first_hdr is not None:
        hdr.extend(first_hdr, update=True)

    hdr["NFLAT"] = (int(n_used), "Number of flat frames used")
    hdr["SF_BAD"] = (int(bad_read), "Number of flat frames failed to read")
    hdr["SF_SHAP"] = (int(bad_shape), "Number of flat frames with shape mismatch")
    hdr["SF_NORMB"] = (int(bad_norm), "Number of flat frames skipped (bad median)")
    hdr["SF_METH"] = (combine, "Combine method for superflat")
    hdr["SF_CLIP"] = (float(sigma_clip), "Sigma clip (0=disabled)")
    hdr["SF_NORM"] = (float(norm), "Final normalization factor (median)")
    hdr["SF_MED"] = (float(_robust_median_finite(superflat)), "Median after normalization")
    hdr["BIASSUB"] = (True, "Superbias subtracted")
    hdr["SB_REF"] = (superbias_path.name, "Superbias reference file")
    if "NBIAS" in sb_hdr:
        hdr["SB_NBIAS"] = (int(sb_hdr["NBIAS"]), "Number of bias frames used")
    hdr.add_history(
        "Built by scorpio_pipe.stages.calib.build_superflat (bias-subtracted, per-flat median norm)"
    )

    # FrameSignature for strict calibration compatibility (copied from superbias)
    hdr["FSIGSH"] = (f"{sb_sig.ny}x{sb_sig.nx}", "FrameSignature: shape")
    if sb_sig.binning():
        hdr["FSIGBIN"] = (sb_sig.binning(), "FrameSignature: binning")
    if sb_sig.window:
        hdr["FSIGROI"] = (sb_sig.window, "FrameSignature: ROI/window")
    if sb_sig.readout:
        hdr["FSIGRDO"] = (sb_sig.readout, "FrameSignature: readout")


    stats: dict[str, int | float | str] = {
        "n_used": n_used,
        "bad_read": int(bad_read),
        "bad_shape": int(bad_shape),
        "bad_norm": int(bad_norm),
        "combine": combine,
        "sigma_clip": float(sigma_clip),
        "norm": float(norm),
    }
    return superflat, hdr, stats


def build_superbias(cfg: Any, out_path: str | Path | None = None) -> Path:
    """
    Строит superbias (по умолчанию — медианой) по bias-кадрам.
    По умолчанию пишет FITS в work_dir/calibs/superbias.fits и зеркалит в legacy
    work_dir/calib/superbias.fits для обратной совместимости.
    """
    c = _load_cfg_any(cfg)

    work_dir = _resolve_work_dir(c)
    bias_paths = [Path(p) for p in c["frames"]["bias"]]

    if not bias_paths:
        raise ValueError("No bias frames in config.frames.bias")

    log.info("Superbias: %d input bias frames", len(bias_paths))

    # Strict calibration compatibility: all bias frames must share the same
    # FrameSignature and match the science setup (if known). No implicit pad/crop.
    expected_sig = _expected_signature_from_cfg(c, fallback_path=bias_paths[0])
    ref_sig = _validate_signatures_strict(bias_paths, expected=expected_sig, what="bias")

    shape = ref_sig.shape

    # Use the first bias header as a template for the master.
    d0, h0 = _read_fits_data(bias_paths[0])
    if d0 is None:
        raise RuntimeError(f"Failed to read first bias frame: {bias_paths[0]}")
    if d0.shape != shape:
        raise RuntimeError(
            f"Internal error: first bias shape {d0.shape} != validated signature {shape}"
        )

    filtered = list(bias_paths)
    bad_open = 0
    log.info("Superbias settings: combine=%s, sigma_clip=%g", combine, sigma_clip)

    bad = 0

    # Median: always stack frames (bias sets are typically small: ~10–50).
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

    # Mean (sigma-clipped) combine.
    elif combine == "mean" and sigma_clip > 0:
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

    # output paths
    if out_path is None:
        from scorpio_pipe.work_layout import ensure_work_layout

        layout = ensure_work_layout(work_dir)
        out_path = layout.calibs / "superbias.fits"
        legacy_path = layout.calib_legacy / "superbias.fits"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        legacy_path = None
        out_path = Path(out_path)
        if not out_path.is_absolute():
            out_path = (work_dir / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)



    # header
    hdr = fits.Header()
    hdr.extend(h0, update=True)
    hdr["NBIAS"] = (int(n_used), "Number of bias frames used")
    hdr["SB_BAD"] = (int(bad), "Number of bias frames skipped")
    hdr["SB_METH"] = (combine, "Combine method for superbias")
    hdr["SB_CLIP"] = (float(sigma_clip), "Sigma clip (0=disabled)")
    hdr.add_history("Built by scorpio_pipe.stages.calib.build_superbias")

    # FrameSignature for strict calibration compatibility
    hdr["FSIGSH"] = (f"{ref_sig.ny}x{ref_sig.nx}", "FrameSignature: shape")
    if ref_sig.binning():
        hdr["FSIGBIN"] = (ref_sig.binning(), "FrameSignature: binning")
    if ref_sig.window:
        hdr["FSIGROI"] = (ref_sig.window, "FrameSignature: ROI/window")
    if ref_sig.readout:
        hdr["FSIGRDO"] = (ref_sig.readout, "FrameSignature: readout")


    fits.writeto(out_path, superbias, hdr, overwrite=True)
    log.info("Wrote superbias: %s", out_path)
    if legacy_path is not None:
        try:
            if legacy_path.resolve() != out_path.resolve():
                shutil.copy2(out_path, legacy_path)
        except Exception:
            # non-fatal: legacy mirror is best-effort
            pass

    # done marker with signature used
    try:
        _write_calib_done(
            work_dir,
            "superbias",
            {
                "status": "ok",
                "frame_signature": ref_sig.to_dict(),
                "used_signature": ref_sig.to_dict(),
                "expected_signature": expected_sig.to_dict() if expected_sig is not None else None,
                "n_inputs": int(n_used),
            },
        )
    except Exception:
        # Best-effort; science output is the FITS master
        pass

    return out_path


def build_superflat(cfg: Any, out_path: str | Path | None = None) -> Path:
    """
    Строит normalized superflat (median ~ 1) из flat-кадров.

    Физически корректная (и однозначная) логика:
      1) вычесть superbias из каждого flat
      2) нормировать каждый flat по его медиане (устойчиво к разной яркости лампы/экспозициям)
      3) объединить кадры (median/mean, опционально sigma-clipped mean)
      4) нормировать итоговый superflat к медиане=1

    По умолчанию пишет FITS в work_dir/calibs/superflat.fits и зеркалит в legacy
    work_dir/calib/superflat.fits для обратной совместимости.
    """
    c = _load_cfg_any(cfg)

    work_dir = _resolve_work_dir(c)
    flat_paths = [Path(p) for p in c["frames"]["flat"]]

    if not flat_paths:
        raise ValueError("No flat frames in config.frames.flat")

    log.info("Superflat: %d input flat frames", len(flat_paths))

    # Strict calibration compatibility: all flat frames must share the same
    # FrameSignature and match the science setup (if known). No implicit pad/crop.
    expected_sig = _expected_signature_from_cfg(c, fallback_path=flat_paths[0])
    ref_sig = _validate_signatures_strict(flat_paths, expected=expected_sig, what="flat")

    shape = ref_sig.shape

    d0, h0 = _read_fits_data(flat_paths[0])
    if d0 is None:
        raise RuntimeError(f"Failed to read first flat frame: {flat_paths[0]}")
    if d0.shape != shape:
        raise RuntimeError(
            f"Internal error: first flat shape {d0.shape} != validated signature {shape}"
        )

    filtered = list(flat_paths)
    bad_open = 0
    calib_cfg = c.get("calib", {}) or {}
    combine = str(calib_cfg.get("flat_combine", "mean")).strip().lower() or "mean"
    sigma_clip = float(calib_cfg.get("flat_sigma_clip", 0.0) or 0.0)
    log.info("Superflat settings: combine=%s, sigma_clip=%g", combine, sigma_clip)

    superbias_path = _resolve_superbias_path(c, work_dir)
    if not superbias_path.exists():
        raise RuntimeError(
            f"Superflat requires superbias, but it was not found at: {superbias_path}. "
            "Run the 'superbias' stage first or set calib.superbias_path in config."
        )

    superflat, hdr, _stats = _build_superflat_core(
        flat_paths=flat_paths,
        superbias_path=superbias_path,
        combine=combine,
        sigma_clip=sigma_clip,
    )

    # output paths (canonical + legacy mirroring)
    from scorpio_pipe.work_layout import ensure_work_layout

    layout = ensure_work_layout(work_dir)
    canonical = (layout.calibs / "superflat.fits").resolve()
    legacy = (layout.calib_legacy / "superflat.fits").resolve()

    if out_path is None:
        out_path = canonical
    else:
        out_path = Path(out_path)
        if not out_path.is_absolute():
            out_path = (work_dir / out_path).resolve()

    # Decide whether we should mirror into the other location.
    legacy_path: Path | None = None
    try:
        out_res = out_path.resolve()
        if out_res == canonical:
            legacy_path = legacy
        elif out_res == legacy:
            legacy_path = canonical
    except Exception:
        legacy_path = None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if legacy_path is not None:
        legacy_path.parent.mkdir(parents=True, exist_ok=True)

    # keep a few extra bits for traceability
    hdr["SB_PATH"] = (superbias_path.name, "Superbias path (basename)")
    hdr["SB_FULL"] = (str(superbias_path), "Superbias full path")

    fits.writeto(out_path, superflat, hdr, overwrite=True)
    log.info("Wrote superflat: %s", out_path)

    if legacy_path is not None:
        try:
            if legacy_path.resolve() != out_path.resolve():
                shutil.copy2(out_path, legacy_path)
        except Exception:
            # legacy mirror is best-effort
            pass

    # done marker with signature used
    try:
        sig_out = FrameSignature.from_header(hdr, fallback_shape=superflat.shape)
        _write_calib_done(
            work_dir,
            "superflat",
            {
                "status": "ok",
                "frame_signature": sig_out.to_dict(),
                "used_signature": sig_out.to_dict(),
                "expected_signature": expected_sig.to_dict() if expected_sig is not None else None,
                "superbias": superbias_path.name,
                "n_inputs": int(stats.get("n_used", 0)),
            },
        )
    except Exception:
        pass

    return out_path
