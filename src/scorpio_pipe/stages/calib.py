from __future__ import annotations

from pathlib import Path
import json
import shutil
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Tuple, Iterable

import numpy as np
from astropy.io import fits

from scorpio_pipe.frame_signature import FrameSignature, format_signature_mismatch
from scorpio_pipe.instruments import parse_frame_meta
from scorpio_pipe.instruments.meta import ReadoutKey
from scorpio_pipe import maskbits

import logging


def _write_calib_done(work_dir: Path, name: str, payload: dict) -> Path:
    """Write a small JSON marker for calibration builders.

    GUI stages historically use per-stage *_done.json files. Superbias/superflat
    builders did not have them, but we add them for transparency/debugging.
    """
    from scorpio_pipe.work_layout import ensure_work_layout
    from scorpio_pipe.workspace_paths import stage_dir

    layout = ensure_work_layout(work_dir)
    p = stage_dir(work_dir, name) / f"{name}_done.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(payload, indent=2, ensure_ascii=False)
    p.write_text(txt, encoding="utf-8")

    # Optional legacy mirrors (do not create directories automatically).
    for legacy_root in (layout.calibs, layout.calib_legacy):
        if legacy_root.is_dir():
            try:
                lp = legacy_root / f"{name}_done.json"
                lp.write_text(txt, encoding="utf-8")
            except Exception:
                pass

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


# -----------------------------------------------------------------------------
# P0-C1: readout-aware MasterBias groups + resolver
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BiasGroupKey:
    """Grouping key for MasterBias (instrument + geometry + readout).

    Notes
    -----
    - We keep *geometry* explicit (shape, binning, ROI token) to avoid mixing
      full-frame and windowed readouts.
    - We keep FITS keyword length <= 8 in all written products.
    """

    instrument: str
    nx: int
    ny: int
    bx: int
    by: int
    roi: str
    readout: ReadoutKey

    def to_fingerprint(self) -> str:
        # Deterministic short token used for filenames.
        s = f"{self.instrument}|{self.nx}x{self.ny}|{self.bx}x{self.by}|{self.roi}|{self.readout.node}|{self.readout.rate:.3f}|{self.readout.gain:.3f}"
        h = hashlib.sha1(s.encode("utf-8")).hexdigest()
        return h[:8]

    def to_json(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "nx": int(self.nx),
            "ny": int(self.ny),
            "bx": int(self.bx),
            "by": int(self.by),
            "roi": self.roi,
            "readout": {
                "node": str(self.readout.node),
                "rate": float(self.readout.rate),
                "gain": float(self.readout.gain),
            },
        }


@dataclass(frozen=True)
class BiasSelection:
    """Resolved MasterBias selection for a given science/calib frame."""

    sci_path: Path
    var_path: Path | None
    dq_path: Path | None
    gid: str
    degraded: bool
    reason: str

    # P0-D: resolved noise parameters for this bias readout group (best-effort).
    gain_e_per_adu: float | None = None
    rdnoise_e: float | None = None
    rn_src: str = ''
    noise_src: str = ''


def _norm_readout_key(rk: ReadoutKey) -> ReadoutKey:
    """Round float fields to improve matching stability."""

    import math

    try:
        rate = float(rk.rate)
        gain = float(rk.gain)
        if not math.isfinite(rate):
            rate = -1.0
        if not math.isfinite(gain):
            gain = -1.0
        return ReadoutKey(node=str(rk.node), rate=round(rate, 3), gain=round(gain, 3))
    except Exception:
        return rk


def _roi_token_from_header(hdr: fits.Header, *, fallback_shape: tuple[int, int] | None = None) -> str:
    try:
        sig = FrameSignature.from_header(hdr, fallback_shape=fallback_shape)
        return str(sig.window or "")
    except Exception:
        return ""


def _bias_group_key_from_path(path: Path) -> BiasGroupKey:
    """Build BiasGroupKey from FITS header (strict, but with safe fallbacks)."""

    hdr = fits.getheader(path, 0)
    nx = int(hdr.get("NAXIS1") or 0)
    ny = int(hdr.get("NAXIS2") or 0)
    # If header is incomplete, fall back to data shape.
    if nx <= 0 or ny <= 0:
        data, _ = _read_fits_data(path)
        ny, nx = (int(data.shape[0]), int(data.shape[1]))

    roi = _roi_token_from_header(hdr, fallback_shape=(ny, nx))

    try:
        # Bias/flat/arc frames may have slightly different header completeness.
        meta = parse_frame_meta(hdr, strict=False)
        rk = _norm_readout_key(meta.readout_key)
        instrument = str(meta.instrument_db_key)
        bx = int(meta.binning_x)
        by = int(meta.binning_y)
    except Exception:
        # Fallback: keep geometry from header; readout is unknown.
        instrument = str(hdr.get("INSTRUME") or "")
        bx = int(hdr.get("CCDBIN1") or hdr.get("BINX") or 1)
        by = int(hdr.get("CCDBIN2") or hdr.get("BINY") or 1)
        rk = ReadoutKey(node=str(hdr.get("NODE") or ""), rate=float(hdr.get("RATE") or 0.0), gain=float(hdr.get("GAIN") or 0.0))
        rk = _norm_readout_key(rk)

    return BiasGroupKey(
        instrument=str(instrument).strip() or "UNKNOWN",
        nx=nx,
        ny=ny,
        bx=max(1, bx),
        by=max(1, by),
        roi=str(roi),
        readout=rk,
    )


def _select_default_bias_group(
    groups: dict[BiasGroupKey, list[Path]],
    *,
    expected: FrameSignature | None,
) -> BiasGroupKey:
    """Pick a 'default' group for legacy paths (superbias.fits).

    Strategy
    --------
    1) If config/setup provides an expected signature, prefer a group that matches
       it in shape/binning/window.
    2) Otherwise: pick the group with the most input frames.
    """

    if not groups:
        raise ValueError("No bias groups")

    keys = list(groups.keys())
    if expected is not None:
        # Try exact-ish match on geometry. Readout is ignored here.
        for k in keys:
            if expected.nx > 0 and expected.ny > 0:
                if int(k.nx) != int(expected.nx) or int(k.ny) != int(expected.ny):
                    continue
            if expected.bx is not None and int(k.bx) != int(expected.bx):
                continue
            if expected.by is not None and int(k.by) != int(expected.by):
                continue
            if expected.window and str(k.roi) != str(expected.window):
                continue
            return k

    # Fallback: max inputs, then stable order.
    keys.sort(key=lambda kk: (-len(groups[kk]), kk.to_fingerprint()))
    return keys[0]


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
    """Resolve superbias path with backward-compatible fallbacks.

    Priority
    --------
    1) Explicit config: calib.superbias_path
    2) Canonical stage output: <work_dir>/NN_superbias/superbias.fits
    3) Legacy/compat roots: <work_dir>/calibs/superbias.fits, <work_dir>/calib/superbias.fits

    Notes
    -----
    ``workspace_paths.resolve_input_path`` intentionally does *not* know about
    legacy ``calibs/`` and ``calib/`` roots, so we add them here.
    """

    calib_cfg = c.get("calib", {}) or {}
    p = calib_cfg.get("superbias_path")
    if p:
        pp = Path(str(p)).expanduser()
        if not pp.is_absolute():
            pp = (work_dir / pp).resolve()
        return pp

    # Prefer new P0-C1 canonical artifact if it exists.
    try:
        from scorpio_pipe.workspace_paths import stage_dir

        mb = stage_dir(work_dir, "superbias") / "master_bias.fits"
        if mb.is_file():
            return mb.resolve()
    except Exception:
        pass

    from scorpio_pipe.workspace_paths import resolve_input_path

    return resolve_input_path(
        "superbias_fits",
        work_dir,
        "superbias",
        relpath="superbias.fits",
        extra_candidates=[
            Path(work_dir) / "calibs" / "superbias.fits",
            Path(work_dir) / "calib" / "superbias.fits",
        ],
    )


def _resolve_superflat_path(c: Dict, work_dir: Path) -> Path:
    """Resolve superflat path with backward-compatible fallbacks.

    Priority
    --------
    1) Explicit config: calib.superflat_path
    2) Canonical stage output: <work_dir>/NN_superflat/superflat.fits
    3) Legacy/compat roots: <work_dir>/calibs/superflat.fits, <work_dir>/calib/superflat.fits
    """

    calib_cfg = c.get("calib", {}) or {}
    p = calib_cfg.get("superflat_path")
    if p:
        pp = Path(str(p)).expanduser()
        if not pp.is_absolute():
            pp = (work_dir / pp).resolve()
        return pp

    from scorpio_pipe.workspace_paths import resolve_input_path

    return resolve_input_path(
        "superflat_fits",
        work_dir,
        "superflat",
        relpath="superflat.fits",
        extra_candidates=[
            Path(work_dir) / "calibs" / "superflat.fits",
            Path(work_dir) / "calib" / "superflat.fits",
        ],
    )


def _master_bias_index_path(work_dir: Path) -> Path:
    from scorpio_pipe.workspace_paths import stage_dir

    return (stage_dir(work_dir, "superbias") / "master_bias_index.json").resolve()


def load_master_bias_index(work_dir: Path) -> dict[str, Any] | None:
    """Load master_bias_index.json if present."""

    ip = _master_bias_index_path(work_dir)
    if not ip.is_file():
        return None
    try:
        return json.loads(ip.read_text(encoding="utf-8"))
    except Exception:
        return None


def _bias_key_from_header(hdr: fits.Header, *, fallback_shape: tuple[int, int] | None = None) -> BiasGroupKey:
    """Build BiasGroupKey from a science/calib header."""

    nx = int(hdr.get("NAXIS1") or 0)
    ny = int(hdr.get("NAXIS2") or 0)
    if (nx <= 0 or ny <= 0) and fallback_shape is not None:
        ny, nx = int(fallback_shape[0]), int(fallback_shape[1])

    roi = _roi_token_from_header(hdr, fallback_shape=(ny, nx) if (ny and nx) else fallback_shape)
    try:
        meta = parse_frame_meta(hdr, strict=False)
        rk = _norm_readout_key(meta.readout_key)
        instrument = str(meta.instrument_db_key)
        bx = int(meta.binning_x)
        by = int(meta.binning_y)
    except Exception:
        instrument = str(hdr.get("INSTRUME") or "")
        bx = int(hdr.get("CCDBIN1") or hdr.get("BINX") or 1)
        by = int(hdr.get("CCDBIN2") or hdr.get("BINY") or 1)
        rk = ReadoutKey(node=str(hdr.get("NODE") or ""), rate=float(hdr.get("RATE") or 0.0), gain=float(hdr.get("GAIN") or 0.0))
        rk = _norm_readout_key(rk)

    return BiasGroupKey(
        instrument=str(instrument).strip() or "UNKNOWN",
        nx=int(nx),
        ny=int(ny),
        bx=max(1, int(bx)),
        by=max(1, int(by)),
        roi=str(roi),
        readout=rk,
    )


def resolve_master_bias(
    cfg: Any,
    sci_hdr: fits.Header,
    *,
    sci_shape: tuple[int, int] | None = None,
    policy_override: str | None = None,
) -> BiasSelection:
    """Resolve MasterBias artifacts for a given frame.

    Policy
    ------
    - Exact match required on (instrument + geometry + readout).
    - If no exact match exists and bias_policy=degraded, fall back to the best
      geometry-matching group and mark the selection as degraded.
    - Geometry mismatch is always fatal (shape/binning/ROI).
    """

    c = _load_cfg_any(cfg)
    work_dir = _resolve_work_dir(c)

    policy = str(
        policy_override
        or ((c.get("calib") or {}).get("bias_policy") or "degraded")
    ).strip().lower()
    if policy not in {"strict", "degraded"}:
        policy = "degraded"

    idx = load_master_bias_index(work_dir)
    if not idx or not isinstance(idx.get("groups"), list):
        # Backward-compat: resolve superbias only.
        sb = _resolve_superbias_path(c, work_dir)
        return BiasSelection(sci_path=sb, var_path=None, dq_path=None, gid="", degraded=True, reason="NOIDX", gain_e_per_adu=None, rdnoise_e=None, rn_src="", noise_src="")

    key = _bias_key_from_header(sci_hdr, fallback_shape=sci_shape)

    # Build candidates from index.
    groups = idx.get("groups") or []

    def _same_geom(gk: dict[str, Any]) -> bool:
        try:
            return (
                str(gk.get("instrument") or "").strip() == key.instrument
                and int(gk.get("nx") or 0) == int(key.nx)
                and int(gk.get("ny") or 0) == int(key.ny)
                and int(gk.get("bx") or 0) == int(key.bx)
                and int(gk.get("by") or 0) == int(key.by)
                and str(gk.get("roi") or "") == str(key.roi)
            )
        except Exception:
            return False

    def _same_readout(gk: dict[str, Any]) -> bool:
        try:
            r = gk.get("readout") or {}
            return (
                str(r.get("node") or "") == str(key.readout.node)
                and float(r.get("rate") or 0.0) == float(key.readout.rate)
                and float(r.get("gain") or 0.0) == float(key.readout.gain)
            )
        except Exception:
            return False

    def _score(g: dict[str, Any]) -> tuple[int, str]:
        # Prefer more used frames, then stable gid.
        try:
            n = int(g.get("n_used") or g.get("n_inputs") or 0)
        except Exception:
            n = 0
        return (n, str(g.get("gid") or ""))

    # Exact match.
    exact: list[dict[str, Any]] = []
    geom: list[dict[str, Any]] = []
    for g in groups:
        kjson = (g.get("key") or {}) if isinstance(g, dict) else {}
        if not _same_geom(kjson):
            continue
        geom.append(g)
        if _same_readout(kjson):
            exact.append(g)

    if exact:
        best = sorted(exact, key=_score, reverse=True)[0]
        files = best.get("files") or {}
        sdir = _resolve_work_dir(c)
        sci_p = (work_dir / str(files.get("sci"))).resolve() if files.get("sci") else _resolve_superbias_path(c, work_dir)
        var_p = (work_dir / str(files.get("var"))).resolve() if files.get("var") else None
        dq_p = (work_dir / str(files.get("dq"))).resolve() if files.get("dq") else None
        noise = (best.get('noise') or {}) if isinstance(best, dict) else {}
        return BiasSelection(
            sci_path=sci_p,
            var_path=var_p,
            dq_path=dq_p,
            gid=str(best.get('gid') or ''),
            degraded=False,
            reason='',
            gain_e_per_adu=(float(noise.get('gain_e_per_adu')) if noise.get('gain_e_per_adu') is not None else None),
            rdnoise_e=(float(noise.get('rdnoise_e')) if noise.get('rdnoise_e') is not None else None),
            rn_src=str(noise.get('rn_src') or ''),
            noise_src=str(noise.get('noisrc') or ''),
        )

    if not geom:
        raise RuntimeError(
            "MasterBias geometry mismatch: no group matches shape/binning/ROI for this frame. "
            "Rebuild superbias with the correct setup or check ROI/binning selection."
        )

    if policy == "strict":
        raise RuntimeError(
            "MasterBias readout mismatch: no exact (geometry+readout) group exists for this frame. "
            "Set calib.bias_policy=degraded to allow best-effort fallback (explicitly stamped), or rebuild bias."
        )

    # Degraded fallback: pick best geometry-matching group with highest n_used.
    best = sorted(geom, key=_score, reverse=True)[0]
    files = best.get("files") or {}
    sci_p = (work_dir / str(files.get("sci"))).resolve() if files.get("sci") else _resolve_superbias_path(c, work_dir)
    var_p = (work_dir / str(files.get("var"))).resolve() if files.get("var") else None
    dq_p = (work_dir / str(files.get("dq"))).resolve() if files.get("dq") else None

    # Reason code: try to explain which readout field mismatched.
    reason = "RDO"
    try:
        kjson = best.get("key") or {}
        r = kjson.get("readout") or {}
        if str(r.get("node") or "") != str(key.readout.node):
            reason = "NODE"
        elif float(r.get("rate") or 0.0) != float(key.readout.rate):
            reason = "RATE"
        elif float(r.get("gain") or 0.0) != float(key.readout.gain):
            reason = "GAIN"
    except Exception:
        pass

    noise = (best.get('noise') or {}) if isinstance(best, dict) else {}

    return BiasSelection(
        sci_path=sci_p,
        var_path=var_p,
        dq_path=dq_p,
        gid=str(best.get('gid') or ''),
        degraded=True,
        reason=reason,
        gain_e_per_adu=(float(noise.get('gain_e_per_adu')) if noise.get('gain_e_per_adu') is not None else None),
        rdnoise_e=(float(noise.get('rdnoise_e')) if noise.get('rdnoise_e') is not None else None),
        rn_src=str(noise.get('rn_src') or ''),
        noise_src=str(noise.get('noisrc') or ''),
    )


def stamp_bias_selection(hdr: fits.Header, sel: BiasSelection) -> fits.Header:
    """Stamp short (<=8) FITS keywords for bias selection provenance."""

    h = fits.Header(hdr)
    if sel.gid:
        h["SB_GID"] = (str(sel.gid), "MasterBias group id")
    h["DEGRAD"] = (bool(sel.degraded), "Bias selection degraded")
    if sel.degraded and sel.reason:
        h["DEGRRSN"] = (str(sel.reason), "Degraded bias reason")
    # P0-D: make bias noise metadata visible to downstream logs.
    try:
        if sel.rdnoise_e is not None and (h.get('SB_RN') is None):
            h['SB_RN'] = (float(sel.rdnoise_e), 'Bias-group read noise [e-]')
        if sel.rn_src and (h.get('SB_RNS') is None):
            h['SB_RNS'] = (str(sel.rn_src)[:8], 'Bias RN source')
    except Exception:
        pass
    return h


def _robust_median_finite(a: np.ndarray) -> float:
    aa = a[np.isfinite(a)]
    if aa.size == 0:
        return float("nan")
    return float(np.median(aa))


def _build_superflat_core(
    *,
    flat_paths: list[Path],
    superbias_path: Path,
    allow_readout_diff: bool = False,
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
        diffs = sig.diff(sb_sig)
        if diffs:
            # In degraded bias-selection mode, we allow a *readout-only* mismatch.
            # Geometry mismatches (shape/binning/ROI) are always fatal.
            if not (allow_readout_diff and all(str(dd).startswith("readout ") for dd in diffs)):
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


def _smooth_1d_boxcar(x: np.ndarray, win: int) -> np.ndarray:
    """Simple, deterministic 1D boxcar smoothing.

    We avoid SciPy here to keep this helper lightweight and easy to test.
    """

    xx = np.asarray(x, dtype=np.float64)
    w = int(win)
    if w <= 1:
        return xx.astype(np.float32)
    if w % 2 == 0:
        w += 1
    k = np.ones(w, dtype=np.float64) / float(w)
    pad = w // 2
    xp = np.pad(xx, (pad, pad), mode="reflect")
    y = np.convolve(xp, k, mode="valid")
    return y.astype(np.float32)


def _build_masterflat_core(
    *,
    cfg: Any,
    flat_paths: list[Path],
    combine: str,
    sigma_clip: float,
    dispersion_axis: str = "x",
    smooth_win: int = 101,
    coverage_min: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, fits.Header, dict[str, Any]]:
    """Build a **MasterFlat** (SCI+VAR+MASK) from multiple flat frames.

    Contract (P0-E)
    --------------
    - Each *individual* flat is bias-subtracted using a MasterBias selected by
      the flat's **own** readout signature (resolve_master_bias).
    - Per-flat VAR is estimated using the flat's GAIN/RN (P0-D), and bias VAR
      is added if available.
    - Flats are normalized per-frame (median) before combination.
    - Output is normalized to median ~ 1 and includes a pixel-to-pixel
      normalization along the dispersion axis to avoid imprinting lamp spectrum.
    - Output MASK: partial coverage is marked with EDGE; pixels with very low
      coverage are additionally marked BADPIX.
    """

    from scorpio_pipe.noise_model import estimate_variance_adu2
    from scorpio_pipe import maskbits

    if not flat_paths:
        raise ValueError("No flats provided for MasterFlat")

    stack_sci: list[np.ndarray] = []
    stack_var: list[np.ndarray] = []
    stack_mask: list[np.ndarray] = []

    first_hdr: fits.Header | None = None

    n_read_fail = 0
    n_bad_med = 0

    # Track bias-group usage for provenance.
    bias_gids: list[str] = []
    n_bias_degraded = 0
    bias_degraded_reasons: dict[str, int] = {}

    for p in flat_paths:
        try:
            d0, h0 = _read_fits_data(p)
        except Exception:
            n_read_fail += 1
            continue
        if d0 is None:
            n_read_fail += 1
            continue
        if first_hdr is None:
            first_hdr = fits.Header(h0)

        data = np.asarray(d0, dtype=np.float32)
        hdr = fits.Header(h0)

        # Readout-aware bias selection (P0-C1).
        # P0-E contract: flat must never use a MasterBias from a different
        # readout group. Use strict policy regardless of the global config.
        sel = resolve_master_bias(
            cfg,
            hdr,
            sci_shape=data.shape,
            policy_override="strict",
        )
        if not sel.sci_path.exists():
            raise RuntimeError(f"MasterBias not found for flat: {p.name} -> {sel.sci_path}")

        sb = fits.getdata(sel.sci_path).astype(np.float32)
        if sb.shape != data.shape:
            raise RuntimeError(
                f"MasterBias shape mismatch for flat: bias={sb.shape} vs flat={data.shape} ({p})"
            )

        # Optional VAR/DQ layers for the bias group.
        bvar = None
        bdq = None
        try:
            if sel.var_path and sel.var_path.exists():
                bvar = fits.getdata(sel.var_path).astype(np.float32)
        except Exception:
            bvar = None
        try:
            if sel.dq_path and sel.dq_path.exists():
                bdq = np.asarray(fits.getdata(sel.dq_path), dtype=np.uint16)
        except Exception:
            bdq = None

        data = (data - sb).astype(np.float32)

        # Per-flat variance estimate in ADU^2 (P0-D). Bias VAR is added.
        rn_hint_e = None
        try:
            if getattr(sel, "rdnoise_e", None) is not None:
                rn_hint_e = float(getattr(sel, "rdnoise_e"))
        except Exception:
            rn_hint_e = None

        var, npar = estimate_variance_adu2(
            data,
            hdr,
            cfg=_load_cfg_any(cfg),
            gain_override=None,
            rdnoise_override=None,
            bias_rn_est_e=rn_hint_e,
            instrument_hint=str((_load_cfg_any(cfg).get("instrument_hint") or "")),
            require_gain=True,
        )
        if bvar is not None and bvar.shape == var.shape:
            var = (var + bvar).astype(np.float32)

        mask = np.zeros(data.shape, dtype=np.uint16)
        if bdq is not None and bdq.shape == mask.shape:
            mask |= bdq

        # Basic data sanity -> BADPIX.
        bad = (~np.isfinite(data)) | (~np.isfinite(var)) | (var < 0)
        if np.any(bad):
            mask[bad] |= np.uint16(maskbits.BADPIX)
            data = data.copy()
            var = var.copy()
            data[bad] = 0.0
            var[bad] = 0.0

        # Per-frame median normalization (robust).
        good = (mask & np.uint16(maskbits.BADPIX)) == 0
        med = _robust_median_finite(data[good])
        if not np.isfinite(med) or med == 0.0:
            n_bad_med += 1
            continue

        data_n = (data / float(med)).astype(np.float32)
        var_n = (var / (float(med) ** 2)).astype(np.float32)

        stack_sci.append(data_n)
        stack_var.append(var_n)
        stack_mask.append(mask)

        gid = str(getattr(sel, "gid", "") or "")
        if gid:
            bias_gids.append(gid)
        if bool(getattr(sel, "degraded", False)):
            n_bias_degraded += 1
            rr = str(getattr(sel, "reason", "")) or "UNKNOWN"
            bias_degraded_reasons[rr] = int(bias_degraded_reasons.get(rr, 0)) + 1

    if not stack_sci:
        raise RuntimeError(
            "All flats failed to read or became invalid during preprocessing (bias-sub/median norm)."
        )

    arr = np.stack(stack_sci, axis=0)  # (N, H, W)
    vrr = np.stack(stack_var, axis=0)
    mrr = np.stack(stack_mask, axis=0)
    n_in = int(len(flat_paths))
    n_used = int(arr.shape[0])

    combine = (combine or "mean").strip().lower()
    if combine not in {"mean", "median"}:
        combine = "mean"
    sc = float(sigma_clip) if sigma_clip is not None else 0.0
    if not np.isfinite(sc) or sc < 0:
        sc = 0.0

    # We treat BADPIX/SATURATED/USER as invalid in the stack.
    invalid = (mrr & (np.uint16(maskbits.BADPIX) | np.uint16(maskbits.SATURATED) | np.uint16(maskbits.USER))) != 0
    data_masked = arr.astype(np.float32).copy()
    data_masked[invalid] = np.nan

    # Sigma-clip on the normalized stack (per pixel).
    rej = np.zeros_like(invalid, dtype=bool)
    if combine == "mean" and sc > 0:
        med = np.nanmedian(data_masked, axis=0)
        mad = np.nanmedian(np.abs(data_masked - med), axis=0)
        sig = (1.4826 * mad).astype(np.float32)
        sig[~np.isfinite(sig) | (sig <= 0)] = 1.0
        rej = np.abs(data_masked - med) > (sc * sig)
        data_masked[rej] = np.nan
        # Mark rejected samples in the per-frame masks (for QC provenance).
        mrr = mrr.copy()
        mrr[rej] |= np.uint16(maskbits.REJECTED)

    # Compute weights (0/1) for valid (non-nan) samples.
    w = np.isfinite(data_masked).astype(np.float32)
    wsum = w.sum(axis=0).astype(np.float32)

    sci = np.ones(arr.shape[1:], dtype=np.float32)
    var = np.zeros(arr.shape[1:], dtype=np.float32)
    mask_out = np.zeros(arr.shape[1:], dtype=np.uint16)

    covered = wsum > 0
    if combine == "median":
        sci[covered] = np.nanmedian(data_masked, axis=0)[covered].astype(np.float32)
        # Approximate variance of median: ~ (pi/2) * Var(mean).
        # Var(mean) with equal weights: sum(var_i)/N^2.
        sumv = np.nansum(vrr * (w ** 2), axis=0).astype(np.float32)
        var_mean = np.zeros_like(var)
        var_mean[covered] = sumv[covered] / (wsum[covered] ** 2)
        var[covered] = (np.pi / 2.0) * var_mean[covered]
    else:
        num = np.nansum(data_masked * w, axis=0).astype(np.float32)
        sci[covered] = (num[covered] / wsum[covered]).astype(np.float32)
        sumv = np.nansum(vrr * (w ** 2), axis=0).astype(np.float32)
        var[covered] = (sumv[covered] / (wsum[covered] ** 2)).astype(np.float32)

    # Coverage / mask summary.
    mask_out[~covered] |= np.uint16(maskbits.NO_COVERAGE)

    # Coverage fraction per pixel in the final stack (0..1).
    cov_min = float(coverage_min) if coverage_min is not None else 0.5
    if not np.isfinite(cov_min):
        cov_min = 0.5
    cov_min = max(0.0, min(1.0, cov_min))

    cov_frac = np.zeros_like(wsum, dtype=np.float32)
    if n_used > 0:
        cov_frac[covered] = (wsum[covered] / float(n_used)).astype(np.float32)

    # Any partial coverage (some frames missing/rejected) is informational.
    partial = (cov_frac > 0) & (cov_frac < (1.0 - 1e-6))
    if np.any(partial):
        mask_out[partial] |= np.uint16(maskbits.EDGE)

    # Very low coverage is treated as unreliable.
    lowcov = (cov_frac > 0) & (cov_frac < cov_min)
    if np.any(lowcov):
        mask_out[lowcov] |= np.uint16(maskbits.BADPIX)

    # Any pixel rejected in >=1 frames -> mark REJECTED (informational).
    if np.any(rej):
        rej_any = np.any(rej, axis=0)
        mask_out[rej_any] |= np.uint16(maskbits.REJECTED)

    # NOTE: We deliberately do NOT mark a pixel BADPIX merely because it was
    # masked in *some* inputs. The coverage/EDGE logic above captures that.

    # Pixel-to-pixel normalization along dispersion axis.
    ax = str(dispersion_axis or "x").strip().lower()
    if ax not in {"x", "y"}:
        ax = "x"

    if ax == "x":
        prof = np.nanmedian(np.where(covered, sci, np.nan), axis=0)
        prof_s = _smooth_1d_boxcar(prof, smooth_win)
        denom = prof_s[None, :]
    else:
        prof = np.nanmedian(np.where(covered, sci, np.nan), axis=1)
        prof_s = _smooth_1d_boxcar(prof, smooth_win)
        denom = prof_s[:, None]

    denom = np.asarray(denom, dtype=np.float32)
    bad_denom = (~np.isfinite(denom)) | (denom == 0)
    if np.any(bad_denom):
        # Avoid imprinting nonsense: mark BADPIX where denom invalid.
        mask_out[bad_denom] |= np.uint16(maskbits.BADPIX)
        denom = denom.copy()
        denom[bad_denom] = 1.0

    sci = (sci / denom).astype(np.float32)
    var = (var / (denom ** 2)).astype(np.float32)

    # Final normalization to median ~ 1.
    finite = np.isfinite(sci) & ((mask_out & np.uint16(maskbits.BADPIX)) == 0)
    if not np.any(finite):
        raise RuntimeError("MasterFlat normalization failed: no finite pixels")
    norm = float(np.median(sci[finite]))
    if norm == 0.0:
        norm = float(np.mean(sci[finite]))
    if norm == 0.0 or not np.isfinite(norm):
        raise RuntimeError("MasterFlat normalization failed: zero median/mean")
    sci = (sci / norm).astype(np.float32)
    var = (var / (norm ** 2)).astype(np.float32)

    # Header: base on first flat.
    hdr_out = fits.Header()
    if first_hdr is not None:
        hdr_out.extend(first_hdr, update=True)

    hdr_out["NCOMBINE"] = (int(n_used), "Number of flats used")
    hdr_out["NF_IN"] = (int(n_in), "Number of flat inputs")
    hdr_out["MF_METH"] = (str(combine)[:8], "MasterFlat combine method")
    hdr_out["MF_CLIP"] = (float(sc), "Sigma clip (0=disabled)")
    hdr_out["MF_AXIS"] = (str(ax).upper(), "Dispersion axis for normalization")
    hdr_out["MF_SMW"] = (int(max(1, int(smooth_win))), "Smoothing window (pix)")
    hdr_out["MF_NORM"] = (float(norm), "Final normalization factor")
    hdr_out["MF_RDF"] = (int(n_read_fail), "Input flats failed to read")
    hdr_out["MF_BMED"] = (int(n_bad_med), "Input flats skipped (bad median)")
    hdr_out["MF_BDGR"] = (int(n_bias_degraded), "# flats with degraded bias selection")

    # Mask schema header cards (compact <=8 chars).
    try:
        for k, v in maskbits.header_cards().items():
            hdr_out[k] = v
    except Exception:
        pass

    hdr_out.add_history(
        "Built by scorpio_pipe.stages.calib._build_masterflat_core (bias-sub, VAR, sigma-clip, dispersion-norm)"
    )

    # Simple QC numbers (deterministic): coverage + flat uniformity.
    cov_used = cov_frac[cov_frac > 0]
    cov_stats = {
        "coverage_min_threshold": float(cov_min),
        "coverage_median": float(np.median(cov_used)) if cov_used.size else 0.0,
        "coverage_p05": float(np.quantile(cov_used, 0.05)) if cov_used.size else 0.0,
        "coverage_p95": float(np.quantile(cov_used, 0.95)) if cov_used.size else 0.0,
    }

    vals = sci[finite].astype(np.float64)
    q01, q05, q95, q99 = [float(x) for x in np.quantile(vals, [0.01, 0.05, 0.95, 0.99])]
    # Compact histogram for QA (deterministic, bounded by robust quantiles).
    hlo, hhi = [float(x) for x in np.quantile(vals, [0.001, 0.999])]
    if not np.isfinite(hlo) or not np.isfinite(hhi) or hlo == hhi:
        hlo, hhi = float(np.min(vals)), float(np.max(vals))
        if hlo == hhi:
            hhi = hlo + 1.0
    h_counts, h_edges = np.histogram(vals, bins=50, range=(hlo, hhi))
    flat_stats = {
        "flat_median": float(_robust_median_finite(vals)),
        "flat_std": float(np.std(vals)),
        "flat_q01": q01,
        "flat_q05": q05,
        "flat_q95": q95,
        "flat_q99": q99,
        "flat_uniformity_p95_p05": float(q95 - q05),
        "flat_uniformity_q95_q05": float(q95 / q05) if q05 not in (0.0,) else float("inf"),
        "flat_hist": {
            "range": [float(hlo), float(hhi)],
            "edges": [float(x) for x in h_edges.tolist()],
            "counts": [int(x) for x in h_counts.tolist()],
        },
    }

    stats: dict[str, Any] = {
        "n_inputs": int(n_in),
        "n_used": int(n_used),
        "combine": str(combine),
        "sigma_clip": float(sc),
        "dispersion_axis": ax,
        "smooth_win": int(max(1, int(smooth_win))),
        "coverage": cov_stats,
        "norm": float(norm),
        "bias_gids": sorted(set(bias_gids)),
        "n_bias_degraded": int(n_bias_degraded),
        "bias_degraded_reasons": bias_degraded_reasons,
        "mask_summary": maskbits.summarize(mask_out),
        **flat_stats,
    }

    return sci, var, mask_out, hdr_out, stats


def build_masterflat_set(
    cfg: Any,
    *,
    flat_paths: list[Path],
    out_path: Path,
    set_id: str | None = None,
    combine: str | None = None,
    sigma_clip: float | None = None,
    dispersion_axis: str | None = None,
    smooth_win: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Build and write a MasterFlat MEF product for a single flat_set."""

    from scorpio_pipe.io.mef import write_sci_var_mask

    c = _load_cfg_any(cfg)
    cal = c.get("calib", {}) or {}

    comb = str(combine or cal.get("flat_combine") or "mean")
    clip = float(sigma_clip if sigma_clip is not None else cal.get("flat_sigma_clip", 3.0))
    ax = str(dispersion_axis or cal.get("dispersion_axis") or "x")
    smw = int(smooth_win if smooth_win is not None else cal.get("flat_smooth_win", 101))
    cov_min = float(cal.get("flat_coverage_min", 0.5) or 0.5)

    sci, var, mask, hdr, stats = _build_masterflat_core(
        cfg=c,
        flat_paths=flat_paths,
        combine=comb,
        sigma_clip=clip,
        dispersion_axis=ax,
        smooth_win=smw,
        coverage_min=cov_min,
    )

    if set_id:
        try:
            hdr["MF_SET"] = (str(set_id)[:8], "Science-set id (short)")
        except Exception:
            pass

    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # MEF with PRIMARY=SCI for backward compatibility (fits.getdata reads PRIMARY).
    write_sci_var_mask(
        out_path,
        sci,
        var=var,
        mask=mask,
        header=hdr,
        primary_data=sci,
        overwrite=True,
    )
    return out_path, stats


def build_superbias(cfg: Any, out_path: str | Path | None = None) -> Path:
    """Build MasterBias artifacts per (instrument + geometry + readout).

    P0-C1 contract
    --------------
    - Do **not** build a single master for the whole night.
    - Build one master per BiasGroupKey and write an explicit index.

    Outputs (in the canonical superbias stage dir)
    ---------------------------------------------
    - master_bias.fits        (SCI)   : default group (best match to expected signature)
    - master_bias_var.fits    (VAR)   : per-pixel variance estimate (ADU^2)
    - master_bias_dq.fits     (DQ)    : uint16 mask (MASK schema v2)
    - master_bias_index.json  : group registry + file mapping
    - superbias.fits          : legacy alias for master_bias.fits
    """

    from scorpio_pipe.work_layout import ensure_work_layout
    from scorpio_pipe.workspace_paths import stage_dir
    from scorpio_pipe import maskbits

    c = _load_cfg_any(cfg)
    work_dir = _resolve_work_dir(c)
    layout = ensure_work_layout(work_dir)

    frames = c.get("frames", {}) or {}
    bias_in = frames.get("bias") or []
    if not isinstance(bias_in, list) or not bias_in:
        raise ValueError("No bias frames in config.frames.bias")

    def _resolve(p: Any) -> Path:
        pp = Path(str(p)).expanduser()
        if not pp.is_absolute():
            pp = (work_dir / pp).resolve()
        return pp

    bias_paths = [_resolve(p) for p in bias_in]
    log.info("MasterBias: %d input bias frames", len(bias_paths))

    calib_cfg = c.get("calib", {}) or {}
    combine = str(calib_cfg.get("bias_combine", "median")).strip().lower() or "median"
    sigma_clip = float(calib_cfg.get("bias_sigma_clip", 0.0) or 0.0)

    groups: dict[BiasGroupKey, list[Path]] = {}
    bad_key = 0
    for p in bias_paths:
        try:
            k = _bias_group_key_from_path(p)
        except Exception:
            bad_key += 1
            continue
        groups.setdefault(k, []).append(p)

    if not groups:
        raise RuntimeError("All bias frames failed metadata parsing; cannot build MasterBias")

    expected_sig = _expected_signature_from_cfg(c, fallback_path=bias_paths[0])
    default_key = _select_default_bias_group(groups, expected=expected_sig)

    out_dir = stage_dir(work_dir, "superbias")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _combine_group(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, fits.Header | None, dict[str, Any]]:
        shape: tuple[int, int] | None = None
        first_hdr: fits.Header | None = None
        frames_f: list[np.ndarray] = []
        bad_read = 0
        bad_shape = 0
        for pp in paths:
            try:
                d, h = _read_fits_data(pp)
            except Exception:
                bad_read += 1
                continue
            if d is None:
                bad_read += 1
                continue
            if first_hdr is None:
                first_hdr = fits.Header(h)
            if shape is None:
                shape = (int(d.shape[0]), int(d.shape[1]))
            if d.shape != shape:
                bad_shape += 1
                continue
            frames_f.append(d.astype(np.float32))

        if not frames_f:
            raise RuntimeError("No readable bias frames in this group")

        arr = np.stack(frames_f, axis=0)
        n_used = int(arr.shape[0])

        # P0-D: Estimate read-noise from bias pairs (ADU) for this group.
        rn_est_adu = None
        rn_diag = {}
        try:
            from scorpio_pipe.noise_model import estimate_readnoise_from_bias_stack

            rn_est_adu, rn_diag = estimate_readnoise_from_bias_stack(arr)
        except Exception:
            rn_est_adu, rn_diag = None, {}

        # Combine
        if combine == "median":
            sci = np.median(arr, axis=0).astype(np.float32)
        elif combine == "mean" and sigma_clip > 0:
            med = np.median(arr, axis=0)
            mad = np.median(np.abs(arr - med), axis=0)
            sig = 1.4826 * mad
            sig[sig <= 0] = 1.0
            m = np.abs(arr - med) <= (sigma_clip * sig)
            w = m.astype(np.float32)
            num = (arr * w).sum(axis=0)
            den = w.sum(axis=0)
            den[den == 0] = 1.0
            sci = (num / den).astype(np.float32)
        else:
            sci = np.mean(arr, axis=0).astype(np.float32)

        # VAR: robust per-pixel dispersion estimator (MAD^2 in ADU^2).
        if n_used >= 2:
            med = np.median(arr, axis=0)
            mad = np.median(np.abs(arr - med), axis=0)
            sig = (1.4826 * mad).astype(np.float32)
            var = (sig ** 2).astype(np.float32)
        else:
            var = np.zeros_like(sci, dtype=np.float32)

        # DQ: start with zeros; flag non-finite pixels as BADPIX.
        dq = np.zeros(sci.shape, dtype=np.uint16)
        badpix = ~np.isfinite(sci) | ~np.isfinite(var)
        if np.any(badpix):
            dq[badpix] |= np.uint16(maskbits.BADPIX)
            sci = sci.copy(); var = var.copy()
            sci[badpix] = 0.0
            var[badpix] = 0.0

        stats = {
            "n_inputs": int(len(paths)),
            "n_used": int(n_used),
            "bad_read": int(bad_read),
            "bad_shape": int(bad_shape),
            "combine": str(combine),
            "sigma_clip": float(sigma_clip),
            "rn_est_adu": (float(rn_est_adu) if rn_est_adu is not None else None),
            "rn_pairs": int((rn_diag or {}).get('pairs_used') or 0),
            "rn_diag": (rn_diag or {}),
        }
        return sci, var, dq, first_hdr, stats

    index: dict[str, Any] = {
        "schema": "scorpio_pipe.master_bias_index",
        "schema_version": 2,
        "combine": str(combine),
        "sigma_clip": float(sigma_clip),
        "groups": [],
        "default_gid": "",
        "bad_key": int(bad_key),
    }

    # Build each group.
    for k, paths in sorted(groups.items(), key=lambda kv: kv[0].to_fingerprint()):
        gid = k.to_fingerprint()
        sci, var, dq, h0, st = _combine_group(paths)

        sci_f = out_dir / f"master_bias__{gid}.fits"
        var_f = out_dir / f"master_bias_var__{gid}.fits"
        dq_f = out_dir / f"master_bias_dq__{gid}.fits"

        # Template header: copy the first bias header in the group (best-effort).
        hdr0 = fits.Header()
        if h0 is not None:
            hdr0.extend(h0, update=True)

        sig = FrameSignature.from_header(hdr0, fallback_shape=sci.shape)

        def _stamp_common(h: fits.Header) -> None:
            # NOTE: keep keys <=8 characters (FITS standard).
            h["NBIAS"] = (int(st["n_used"]), "Number of bias frames used")
            h["SB_BAD"] = (int(st["bad_read"] + st["bad_shape"]), "Bias frames skipped")
            h["SB_METH"] = (str(combine), "Combine method")
            h["SB_CLIP"] = (float(sigma_clip), "Sigma clip (0=disabled)")
            h["SB_GID"] = (str(gid), "MasterBias group id")
            h["SBINS"] = (str(k.instrument), "Bias instrument")
            h["SBNAX1"] = (int(k.nx), "Group NAXIS1")
            h["SBNAX2"] = (int(k.ny), "Group NAXIS2")
            h["SBBINX"] = (int(k.bx), "Group binning X")
            h["SBBINY"] = (int(k.by), "Group binning Y")
            if k.roi:
                h["SBROI"] = (str(k.roi)[:68], "Group ROI token")
            h["SBNODE"] = (str(k.readout.node), "Readout node")
            h["SBRATE"] = (float(k.readout.rate), "Readout rate")
            h["SBGAIN"] = (float(k.readout.gain), "Gain")
            # P0-D: noise metadata for this readout group
            try:
                from scorpio_pipe.noise_model import resolve_noise_params, stamp_noise_keywords

                gain_hint = None
                try:
                    g0 = float(h.get('GAIN')) if h.get('GAIN') is not None else None
                    if g0 is not None and np.isfinite(g0) and g0 > 0:
                        gain_hint = None
                    else:
                        gain_hint = float(k.readout.gain) if float(k.readout.gain) > 0 else None
                except Exception:
                    gain_hint = float(k.readout.gain) if float(k.readout.gain) > 0 else None

                npar = resolve_noise_params(
                    h,
                    cfg=c,
                    gain_override=gain_hint,
                    bias_rn_est_adu=st.get('rn_est_adu'),
                    instrument_hint=str(k.instrument),
                    require_gain=False,
                )
                h = stamp_noise_keywords(h, npar, overwrite=True)
                # Extra compact diagnostics
                if st.get('rn_est_adu') is not None:
                    h['RNADU'] = (float(st.get('rn_est_adu')), 'Read noise estimate [ADU]')
                if int(st.get('rn_pairs') or 0) > 0:
                    h['RNPAIR'] = (int(st.get('rn_pairs')), 'Bias pairs used for RN')
            except Exception:
                pass
            h["DEGRAD"] = (False, "Bias selection degraded")
            # FrameSignature for compat (<=8-char keys).
            h["FSIGSH"] = (f"{sig.ny}x{sig.nx}", "FrameSignature: shape")
            if sig.binning():
                h["FSIGBIN"] = (sig.binning(), "FrameSignature: binning")
            if sig.window:
                h["FSIGROI"] = (sig.window, "FrameSignature: ROI")
            if sig.readout:
                h["FSIGRDO"] = (sig.readout, "FrameSignature: readout")

        # SCI
        hsci = fits.Header(hdr0)
        _stamp_common(hsci)
        hsci.add_history("Built by scorpio_pipe.stages.calib.build_superbias (MasterBias)")
        fits.writeto(sci_f, sci.astype(np.float32), hsci, overwrite=True)

        # VAR
        hv = fits.Header(hdr0)
        _stamp_common(hv)
        hv["BUNIT"] = ("ADU^2", "Variance unit")
        hv.add_history("MasterBias VAR (per-pixel MAD^2)")
        fits.writeto(var_f, var.astype(np.float32), hv, overwrite=True)

        # DQ
        hdq = fits.Header(hdr0)
        _stamp_common(hdq)
        for kk, vv in maskbits.header_cards(prefix="SCORP").items():
            hdq[kk] = vv
        hdq.add_history("MasterBias DQ (uint16 mask)")
        fits.writeto(dq_f, dq.astype(np.uint16), hdq, overwrite=True)

        index["groups"].append(
            {
                "gid": str(gid),
                "key": k.to_json(),
                "n_inputs": int(st["n_inputs"]),
                "n_used": int(st["n_used"]),
                "bad_read": int(st["bad_read"]),
                "bad_shape": int(st["bad_shape"]),
                "files": {
                    "sci": str(sci_f.relative_to(work_dir)),
                    "var": str(var_f.relative_to(work_dir)),
                    "dq": str(dq_f.relative_to(work_dir)),
                },
                "noise": {
                    "gain_e_per_adu": (float(hsci.get('GAIN')) if hsci.get('GAIN') is not None else None),
                    "rdnoise_e": (float(hsci.get('RDNOISE')) if hsci.get('RDNOISE') is not None else None),
                    "rn_src": str(hsci.get('RN_SRC') or ''),
                    "noisrc": str(hsci.get('NOISRC') or ''),
                    "rn_est_adu": (float(st.get('rn_est_adu')) if st.get('rn_est_adu') is not None else None),
                    "rn_pairs": int(st.get('rn_pairs') or 0),
                },
            }
        )

        if k == default_key:
            index["default_gid"] = str(gid)

    if not index.get("default_gid") and index.get("groups"):
        index["default_gid"] = str((index["groups"][0] or {}).get("gid") or "")

    # Write index.
    (out_dir / "master_bias_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Canonical aliases: default group.
    def _find_by_gid(gid: str) -> dict[str, Any]:
        for g in index.get("groups") or []:
            if str(g.get("gid") or "") == str(gid):
                return g
        raise RuntimeError("Internal error: default_gid not found in index")

    gdef = _find_by_gid(index["default_gid"])
    fdef = gdef.get("files") or {}
    sci_src = (work_dir / str(fdef.get("sci"))).resolve()
    var_src = (work_dir / str(fdef.get("var"))).resolve()
    dq_src = (work_dir / str(fdef.get("dq"))).resolve()

    master_sci = (out_dir / "master_bias.fits").resolve()
    master_var = (out_dir / "master_bias_var.fits").resolve()
    master_dq = (out_dir / "master_bias_dq.fits").resolve()
    legacy_sb = (out_dir / "superbias.fits").resolve()

    import shutil
    shutil.copy2(sci_src, master_sci)
    shutil.copy2(var_src, master_var)
    shutil.copy2(dq_src, master_dq)
    shutil.copy2(sci_src, legacy_sb)

    # Optional: mirror to legacy roots, only if dirs exist.
    for root in (layout.calibs, layout.calib_legacy):
        try:
            if root.is_dir():
                shutil.copy2(master_sci, root / "master_bias.fits")
                shutil.copy2(master_var, root / "master_bias_var.fits")
                shutil.copy2(master_dq, root / "master_bias_dq.fits")
                shutil.copy2(legacy_sb, root / "superbias.fits")
        except Exception:
            pass

    # done marker
    try:
        _write_calib_done(
            work_dir,
            "superbias",
            {
                "status": "ok",
                "n_groups": int(len(index.get("groups") or [])),
                "default_gid": str(index.get("default_gid") or ""),
                "bad_key": int(bad_key),
                "combine": str(combine),
                "sigma_clip": float(sigma_clip),
            },
        )
    except Exception:
        pass

    # If an explicit out_path was requested, also write the SCI alias there.
    if out_path is not None:
        pp = Path(str(out_path)).expanduser()
        if not pp.is_absolute():
            pp = (work_dir / pp).resolve()
        pp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(master_sci, pp)
        return pp.resolve()

    return master_sci


def build_superflat(cfg: Any, out_path: str | Path | None = None) -> Path:
    """
    Строит normalized superflat (median ~ 1) из flat-кадров.

    Физически корректная (и однозначная) логика:
      1) вычесть superbias из каждого flat
      2) нормировать каждый flat по его медиане (устойчиво к разной яркости лампы/экспозициям)
      3) объединить кадры (median/mean, опционально sigma-clipped mean)
      4) нормировать итоговый superflat к медиане=1

    По умолчанию пишет FITS в каноническую stage‑директорию
    work_dir/NN_superflat/superflat.fits и (best‑effort) зеркалит в legacy
    work_dir/calibs/superflat.fits и work_dir/calib/superflat.fits для обратной совместимости.
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

    # P0-C1: choose MasterBias matching the flat readout/geometry.
    sel = resolve_master_bias(c, fits.Header(h0), sci_shape=d0.shape)
    superbias_path = sel.sci_path
    if not superbias_path.exists():
        raise RuntimeError(
            f"Superflat requires superbias, but it was not found at: {superbias_path}. "
            "Run the 'superbias' stage first or set calib.superbias_path in config."
        )

    superflat, hdr, _stats = _build_superflat_core(
        flat_paths=flat_paths,
        superbias_path=superbias_path,
        allow_readout_diff=bool(sel.degraded),
        combine=combine,
        sigma_clip=sigma_clip,
    )

    # Propagate MasterBias selection (incl. degraded flag) into the superflat header.
    try:
        hdr = stamp_bias_selection(hdr, sel)
    except Exception:
        pass

    # output paths (canonical + legacy mirroring)
    from scorpio_pipe.work_layout import ensure_work_layout
    from scorpio_pipe.workspace_paths import stage_dir

    layout = ensure_work_layout(work_dir)
    canonical = (stage_dir(work_dir, "superflat") / "superflat.fits").resolve()
    legacy_candidates = [
        (layout.calibs / "superflat.fits").resolve(),
        (layout.calib_legacy / "superflat.fits").resolve(),
    ]

    if out_path is None:
        out_path = canonical
    else:
        out_path = Path(out_path)
        if not out_path.is_absolute():
            out_path = (work_dir / out_path).resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_res = out_path.resolve()

    # keep a few extra bits for traceability
    hdr = stamp_bias_selection(hdr, sel)
    hdr["SB_PATH"] = (superbias_path.name, "Superbias path (basename)")
    hdr["SB_FULL"] = (str(superbias_path), "Superbias full path")

    fits.writeto(out_path, superflat, hdr, overwrite=True)
    log.info("Wrote superflat: %s", out_path)

    # Mirror policy:
    #  - Always keep canonical populated.
    #  - Only mirror into legacy locations if their parent directories already exist.
    try:
        if out_res != canonical:
            canonical.parent.mkdir(parents=True, exist_ok=True)
            if canonical != out_res:
                shutil.copy2(out_path, canonical)
        else:
            for lp in legacy_candidates:
                if lp.parent.is_dir() and lp.resolve() != out_res:
                    shutil.copy2(out_path, lp)
    except Exception:
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
                "n_inputs": int((_stats or {}).get("n_used", 0)),
            },
        )
    except Exception:
        pass

    return out_path
