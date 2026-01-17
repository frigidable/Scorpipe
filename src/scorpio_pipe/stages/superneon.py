from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import os
import json
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Any

import numpy as np
from astropy.io import fits

from scorpio_pipe.frame_signature import FrameSignature

log = logging.getLogger(__name__)

# --------------------------- helpers ---------------------------


def _project_root() -> Path:
    """Project root in source layout and PyInstaller (onefile) builds."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(str(meipass)).resolve()
    return Path(__file__).resolve().parents[3]


def _resolve_from_root(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (_project_root() / p).resolve()


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_fits(path: Path) -> tuple[np.ndarray, fits.Header]:
    # Важно: ignore_missing_end спасает "подуставшие" .fts
    with fits.open(path, memmap=False, ignore_missing_end=True) as hdul:
        data = hdul[0].data.astype(np.float64)
        hdr = hdul[0].header
    return data, hdr


def _save_fits(path: Path, data: np.ndarray, header: fits.Header | None = None) -> None:
    hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=header)
    hdu.writeto(path, overwrite=True)

def _estimate_saturation_level(hdr: fits.Header) -> float | None:
    """Best-effort estimate of saturation ADU level.

    SCORPIO FITS often stores unsigned 16-bit data with BZERO=32768.
    In that case, the physical max is 65535 ADU. If signed 16-bit, max is 32767.
    Returns None for non-integer formats.
    """
    try:
        bitpix = int(hdr.get("BITPIX"))
    except Exception:
        return None
    if bitpix != 16:
        return None
    bzero = hdr.get("BZERO")
    if bzero is None:
        return 32767.0
    try:
        bz = float(bzero)
    except Exception:
        bz = 0.0
    if bz >= 32768.0:
        return 65535.0
    return 32767.0


def _saturation_fraction(img: np.ndarray, sat_level: float | None) -> float:
    if sat_level is None:
        return 0.0
    v = img[np.isfinite(img)]
    if v.size == 0:
        return 0.0
    return float(np.mean(v >= (sat_level - 0.5)))


def _write_csv(path: Path, header: str, rows: list[list[object]]) -> None:
    lines = [header]
    for r in rows:
        out: list[str] = []
        for v in r:
            s = "" if v is None else str(v)
            if "," in s or '"' in s:
                s = '"' + s.replace('"', '""') + '"'
            out.append(s)
        lines.append(",".join(out))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")



def _percentile_stretch(img: np.ndarray, p1=1.0, p2=99.7) -> tuple[float, float]:
    v = img[np.isfinite(img)]
    if v.size == 0:
        return 0.0, 1.0
    return float(np.percentile(v, p1)), float(np.percentile(v, p2))


def _save_png(path: Path, img: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt
    from scorpio_pipe.plot_style import mpl_style

    vmin, vmax = _percentile_stretch(img, 1.0, 99.7)
    with mpl_style():
        fig = plt.figure(figsize=(10, 4.6))
        ax = fig.add_subplot(111)
        im = ax.imshow(
            img, origin="lower", cmap="gray", aspect="auto", vmin=vmin, vmax=vmax
        )
        ax.set_title(title)
        ax.set_xlabel("X [px]")
        ax.set_ylabel("Y [px]")
        fig.colorbar(im, ax=ax, label="counts")
        fig.savefig(path)
        plt.close(fig)


def _median_stack(frames: list[np.ndarray]) -> np.ndarray:
    """NaN-aware median stack.

    Alignment introduces NaNs at the edges; plain `np.median` would
    propagate NaNs into the output, so we use `np.nanmedian`.
    """
    arr = np.stack(frames, axis=0)
    return np.nanmedian(arr, axis=0)


def _profile_x(img2d: np.ndarray, y0: int, y1: int) -> np.ndarray:
    y0 = max(0, int(y0))
    y1 = min(img2d.shape[0], int(y1))
    if y1 <= y0:
        y0, y1 = 0, img2d.shape[0]
    return np.nanmedian(img2d[y0:y1, :], axis=0)


def _xcorr_shift_1d(
    ref: np.ndarray, cur: np.ndarray, max_abs: int = 6
) -> tuple[int, float]:
    """Integer X-shift cur relative to ref + a quality score.

    The shift is found by maximizing FFT cross-correlation within |shift|<=max_abs.
    The score is a dimensionless peak-to-RMS ratio of the correlation curve in the search window.
    """
    ref = np.asarray(ref, float)
    cur = np.asarray(cur, float)
    ref = ref - np.nanmedian(ref)
    cur = cur - np.nanmedian(cur)
    ref = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)
    cur = np.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0)

    n = int(2 ** int(np.ceil(np.log2(ref.size + cur.size))))
    fr = np.fft.rfft(ref, n=n)
    fc = np.fft.rfft(cur, n=n)
    cc = np.fft.irfft(fr * np.conj(fc), n=n)

    lags = np.arange(n)
    signed = (lags + n // 2) % n - n // 2
    m = np.abs(signed) <= int(max_abs)
    cc_win = cc[m]
    signed_win = signed[m]
    if cc_win.size == 0:
        return 0, 0.0

    idx = int(np.argmax(cc_win))
    shift = int(signed_win[idx])

    peak = float(cc_win[idx])
    rms = float(np.sqrt(np.mean(np.square(cc_win)))) if cc_win.size else 0.0
    score = float(peak / (rms + 1e-12))
    return shift, score



def _shift_x_int(img: np.ndarray, dx: int) -> np.ndarray:
    """
    Сдвиг по X на dx (целый), с заполнением NaN по краям.
    """
    H, W = img.shape
    out = np.full_like(img, np.nan, dtype=np.float64)
    if dx == 0:
        out[:] = img
        return out
    if dx > 0:
        out[:, dx:] = img[:, : W - dx]
    else:
        out[:, : W + dx] = img[:, -dx:]
    return out


# --------------------------- peaks ---------------------------


def _find_peaks_simple(
    y: np.ndarray,
    *,
    min_height: float | None = None,
    prominence: float = 5.0,
    distance: int = 3,
) -> np.ndarray:
    """
    Поиск пиков по scipy если есть, иначе — простая эвристика.
    """
    y = np.asarray(y, float)
    y0 = y - np.nanmedian(y)
    y0[~np.isfinite(y0)] = 0.0

    try:
        from scipy.signal import find_peaks  # type: ignore

        kwargs = {"prominence": prominence, "distance": distance}
        if min_height is not None:
            kwargs["height"] = float(min_height)
        pk, _ = find_peaks(y0, **kwargs)
        return pk.astype(int)
    except Exception:
        # fallback: локальные максимумы
        pk = []
        for i in range(1, len(y0) - 1):
            if y0[i] > y0[i - 1] and y0[i] > y0[i + 1]:
                if min_height is not None and y0[i] < min_height:
                    continue
                if y0[i] < prominence:
                    continue
                pk.append(i)
        return np.array(pk, dtype=int)


def _robust_sigma(
    y: np.ndarray, n_iter: int = 4, clip: float = 3.5
) -> tuple[float, float]:
    """Robust background sigma estimate for 1D spectra.

    Iteratively excludes strong emission lines.
    Returns (sigma, median).
    """
    y = np.asarray(y, float)
    v = y[np.isfinite(y)]
    if v.size < 20:
        med = float(np.nanmedian(y)) if np.isfinite(y).any() else 0.0
        return 0.0, med

    med = float(np.median(v))
    keep = v
    for _ in range(max(1, int(n_iter))):
        mad = float(np.median(np.abs(keep - np.median(keep))))
        sigma = 1.4826 * mad
        if sigma <= 0:
            break
        m = np.abs(keep - np.median(keep)) <= clip * sigma
        if m.sum() < 20:
            break
        keep = keep[m]
    mad = float(np.median(np.abs(keep - np.median(keep))))
    sigma = 1.4826 * mad
    return float(sigma), float(np.median(keep))


def _estimate_baseline_bins(
    y: np.ndarray,
    *,
    bin_size: int = 32,
    q: float = 0.2,
    smooth_bins: int = 5,
) -> np.ndarray:
    """
    Оценка baseline(x) через низкий квантиль по бинам:
    - режем спектр на бины по X
    - в каждом бине берём квантиль q (подавляет эмиссионные линии)
    - сглаживаем baseline в бин-пространстве
    - интерполируем на полное разрешение
    """
    y = np.asarray(y, float)
    n = y.size
    if n == 0:
        return y.copy()

    bs = max(8, int(bin_size))
    nb = int(np.ceil(n / bs))
    xs = (np.arange(nb) + 0.5) * bs
    xs = np.clip(xs, 0, n - 1).astype(float)

    b = np.empty(nb, dtype=float)
    for i in range(nb):
        a = i * bs
        bb = min(n, (i + 1) * bs)
        chunk = y[a:bb]
        chunk = chunk[np.isfinite(chunk)]
        b[i] = float(np.nanquantile(chunk, q)) if chunk.size else np.nan

    if np.isnan(b).any():
        good = np.isfinite(b)
        if good.any():
            b = np.interp(np.arange(nb), np.flatnonzero(good), b[good])
        else:
            return np.zeros_like(y, dtype=float)

    sb = max(1, int(smooth_bins))
    if sb % 2 == 0:
        sb += 1
    if sb > 1 and nb >= sb:
        k = np.ones(sb, dtype=float) / sb
        b = np.convolve(b, k, mode="same")

    return np.interp(np.arange(n, dtype=float), xs, b, left=b[0], right=b[-1])


def _robust_sigma_quasi_empty(
    residual: np.ndarray,
    *,
    empty_quantile: float = 0.7,
    clip: float = 3.5,
    n_iter: int = 3,
) -> tuple[float, float]:
    """
    σ по квази-пустым участкам residual (= prof - baseline):
    - берём нижнюю долю распределения (q=empty_quantile), чтобы отсечь линии
    - затем MAD-клиппинг
    Возвращает (sigma, median).
    """
    r = np.asarray(residual, float)
    good = np.isfinite(r)
    if good.sum() < 30:
        med = float(np.nanmedian(r)) if np.isfinite(r).any() else 0.0
        return 0.0, med

    q = float(np.clip(empty_quantile, 0.3, 0.95))
    thr = float(np.nanquantile(r[good], q))
    m = good & (r <= thr)
    if m.sum() < 30:
        m = good

    keep = r[m]
    for _ in range(max(1, int(n_iter))):
        med = float(np.median(keep))
        mad = float(np.median(np.abs(keep - med)))
        sigma = 1.4826 * mad
        if not np.isfinite(sigma) or sigma <= 0:
            break
        mm = np.abs(keep - med) <= clip * sigma
        if mm.sum() < 30:
            break
        keep = keep[mm]

    med = float(np.median(keep))
    mad = float(np.median(np.abs(keep - med)))
    sigma = 1.4826 * mad
    return float(sigma), float(med)


def _refine_peak_centroid(
    x: np.ndarray, y: np.ndarray, i0: int, hw: int = 4
) -> tuple[float, float, float]:
    """
    Быстрое субпиксельное уточнение: центр масс (взвешенный) в окне.
    Возвращает (x_center, amp, fwhm_rough).
    """
    n = len(y)
    a = max(0, i0 - hw)
    b = min(n, i0 + hw + 1)
    xx = x[a:b]
    yy = y[a:b]
    base = np.nanmedian(yy)
    w = np.clip(yy - base, 0, None) + 1e-12
    xc = float(np.sum(xx * w) / np.sum(w))
    amp = float(np.nanmax(yy) - base)
    # rough fwhm: по второму моменту
    var = float(np.sum(w * (xx - xc) ** 2) / np.sum(w))
    fwhm = float(2.355 * np.sqrt(max(var, 1e-12)))
    return xc, amp, fwhm


# --------------------------- main stage ---------------------------


@dataclass
class SuperNeonResult:
    superneon_fits: Path
    peaks_csv: Path
    superneon_png: Path


def build_superneon(cfg: dict[str, Any]) -> SuperNeonResult:
    """Build SuperNeon (stacked arc/neon) as a true calibration product.

    Guarantees strict physical compatibility (FrameSignature) between:
    - all input neon frames
    - neon frames and superbias (if bias_sub=True and superbias is available)

    Produces, besides superneon.fits/png and peaks_candidates.csv:
    - superneon_shifts.csv / superneon_shifts.json: per-frame X shifts + basic QC
    - superneon_qc.json: aggregated QC summary
    """
    # work_dir: allow absolute, or relative to config_dir/cwd
    work_dir = Path(str(cfg["work_dir"]))
    if not work_dir.is_absolute():
        base = Path(str(cfg.get("config_dir", ".")))
        work_dir = (base / work_dir).resolve()
    else:
        work_dir = work_dir.resolve()

    # disperser-specific layout (so multiple gratings can live in one work_dir)
    from scorpio_pipe.wavesol_paths import wavesol_dir

    outdir = _ensure_dir(wavesol_dir(cfg))

    neon_list = [Path(p) for p in cfg["frames"].get("neon", [])]
    if not neon_list:
        raise RuntimeError(
            "Нет neon кадров в cfg['frames']['neon']. Inspect видит neon — значит autoconfig должен их записать."
        )

    # --- bias subtraction control ---
    superneon_cfg = cfg.get("superneon") or {}
    if not isinstance(superneon_cfg, dict):
        superneon_cfg = {}
    bias_sub = bool(superneon_cfg.get("bias_sub", True))

    # Resolve superbias path robustly (canonical stage dirs + legacy fallbacks).
    # Do not assume work_dir/calibs exists: GUI/CLI often use NN_* stage layout.
    from scorpio_pipe.stages.calib import _resolve_superbias_path as _resolve_sb

    superbias_path: Path | None = None
    if bias_sub:
        cand = _resolve_sb(cfg, work_dir)
        if cand is not None:
            cand = Path(str(cand)).expanduser()
            if not cand.is_absolute():
                cand = (work_dir / cand).resolve()
            if cand.is_file():
                superbias_path = cand

    superbias: np.ndarray | None = None
    superbias_hdr: fits.Header | None = None
    superbias_sig: FrameSignature | None = None
    if bias_sub:
        if superbias_path is None or not superbias_path.is_file():
            raise FileNotFoundError(
                "SuperNeon bias_sub=True requires a valid superbias master, but it was not found. "
                "Run the 'superbias' stage first or set calib.superbias_path in config.yaml."
            )
        superbias, superbias_hdr = _load_fits(superbias_path)
        superbias_sig = FrameSignature.from_fits_primary(superbias_path)

    # --- load all neon frames + strict FrameSignature checks (Block 03) ---
    frames_raw: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    headers: list[fits.Header] = []
    sigs: list[FrameSignature] = []

    n_jobs = int(superneon_cfg.get("n_jobs", 0))
    if n_jobs <= 0:
        n_jobs = max(1, min(8, os.cpu_count() or 1))

    def _read_one(p: Path) -> tuple[Path, np.ndarray, fits.Header, FrameSignature]:
        img, hdr = _load_fits(p)
        # FrameSignature.from_header() uses fallback_shape for missing NAXIS cards.
        sig = FrameSignature.from_header(hdr, fallback_shape=img.shape)
        return p, img, hdr, sig

    items: list[tuple[Path, np.ndarray, fits.Header, FrameSignature]] = []
    if len(neon_list) >= 4 and n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            for item in ex.map(_read_one, neon_list):
                items.append(item)
    else:
        for p in neon_list:
            items.append(_read_one(p))

    if not items:
        raise RuntimeError("No neon frames loaded (unexpected).")

    # strict: all neon frames must match each other
    ref_sig = items[0][3]
    for p, img, hdr, sig in items:
        diffs = ref_sig.diff(sig)
        if diffs:
            msg = (
                "SuperNeon input frames are not physically compatible (FrameSignature mismatch).\n"
                f"Reference: {ref_sig.to_dict()}\n"
                f"Current   : {sig.to_dict()}\n"
                "Differences:\n- " + "\n- ".join(diffs) + "\n"
                f"Offending frame: {p}"
            )
            raise ValueError(msg)

    # strict: superbias must match neon frames if bias_sub is enabled
    did_sub = False
    if bias_sub and superbias is not None:
        if superbias_sig is not None:
            diffs_sb = superbias_sig.diff(ref_sig)
            if diffs_sb:
                msg = (
                    "SuperNeon requires superbias to match neon frames (strict FrameSignature).\n"
                    f"Superbias : {superbias_sig.to_dict()}\n"
                    f"Neon sig  : {ref_sig.to_dict()}\n"
                    "Differences:\n- " + "\n- ".join(diffs_sb) + "\n"
                    f"Superbias : {superbias_path}"
                )
                raise ValueError(msg)

    # saturation level (best effort from first header)
    sat_level = _estimate_saturation_level(items[0][2])

    for p, img, hdr, sig in items:
        frames_raw.append(img)
        headers.append(hdr)
        sigs.append(sig)

        if bias_sub and superbias is not None:
            if superbias.shape != img.shape:
                raise ValueError(
                    "SuperNeon requires superbias to match neon frames in shape (strict). "
                    f"superbias={superbias.shape}, neon={img.shape} ({p})"
                )
            frames.append(img - superbias)
            did_sub = True
        else:
            frames.append(img)

    H, W = frames[0].shape

    # profile box for x-alignment
    wcfg = cfg.get("wavesol", {}) or {}
    prof_y = wcfg.get("profile_y", None)
    if prof_y is None:
        y_half = int(wcfg.get("y_half", 20))
        y0, y1 = (H // 2 - y_half, H // 2 + y_half)
    else:
        y0, y1 = prof_y
    y0 = int(y0)
    y1 = int(y1)
    xshift_max = int(cfg.get("wavesol", {}).get("xshift_max_abs", 6))

    ref_prof = _profile_x(frames[0], y0, y1)
    aligned = [frames[0]]

    dxs: list[int] = [0]
    scores: list[float] = [1.0]
    for i, img in enumerate(frames[1:], start=1):
        cur_prof = _profile_x(img, y0, y1)
        dx, score = _xcorr_shift_1d(ref_prof, cur_prof, max_abs=xshift_max)
        dxs.append(int(dx))
        scores.append(float(score))
        log.debug("SuperNeon shift frame #%d: dx=%d score=%.3g", i, dx, score)
        aligned.append(_shift_x_int(img, int(dx)))

    superneon = _median_stack(aligned)

    # estimated clipping fraction (diagnostic; does NOT affect the stack)
    clip_k = float(superneon_cfg.get("clip_k", 6.0))
    clip_frac = 0.0
    try:
        stack = np.stack(aligned, axis=0)
        med = np.nanmedian(stack, axis=0)
        abs_dev = np.abs(stack - med)
        mad = np.nanmedian(abs_dev, axis=0)
        sigma_pix = 1.4826 * mad
        sigma_pix = np.where(np.isfinite(sigma_pix), sigma_pix, 0.0)
        sigma_pix = np.maximum(sigma_pix, 1e-6)
        valid = np.isfinite(stack)
        out = (abs_dev > (clip_k * sigma_pix)) & valid
        n_valid = int(np.sum(valid))
        n_out = int(np.sum(out))
        clip_frac = float(n_out / n_valid) if n_valid > 0 else 0.0
    except Exception:
        clip_frac = 0.0

    # output file paths
    superneon_fits = outdir / "superneon.fits"
    superneon_png = outdir / "superneon.png"
    peaks_csv = outdir / "peaks_candidates.csv"
    shifts_csv = outdir / "superneon_shifts.csv"
    shifts_json = outdir / "superneon_shifts.json"
    qc_json = outdir / "superneon_qc.json"

    # --- peak candidates from I(x) profile ---
    prof = _profile_x(superneon, y0, y1)
    x = np.arange(W, dtype=float)

    noise_cfg = (wcfg.get("noise") or {}) if isinstance(wcfg, dict) else {}

    baseline = _estimate_baseline_bins(
        prof,
        bin_size=int(noise_cfg.get("baseline_bin_size", 32)),
        quantile=float(noise_cfg.get("baseline_quantile", 0.25)),
    )
    residual = prof - baseline

    sigma, _ = _robust_sigma_quasi_empty(
        residual,
        empty_quantile=float(noise_cfg.get("empty_quantile", 0.7)),
        clip=float(noise_cfg.get("clip", 3.5)),
        n_iter=int(noise_cfg.get("n_iter", 3)),
    )
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    peak_snr = float(wcfg.get("peak_snr", 4.5))
    peak_prom_snr = float(wcfg.get("peak_prom_snr", 3.5))
    floor_k = float(wcfg.get("peak_floor_snr", 3.0))
    distance = int(wcfg.get("peak_dist", 8))

    min_height = max(float(peak_snr * sigma), float(floor_k * sigma))
    prominence = float(peak_prom_snr * sigma)

    pk = _find_peaks_simple(
        residual,
        min_height=min_height,
        prominence=prominence,
        distance=distance,
        max_peaks=int(wcfg.get("peak_max", 80)),
    )

    # header: calibration + QC
    hdr = fits.Header()
    hdr["NNEON"] = (len(neon_list), "Number of input neon frames")
    hdr["NSUB"] = (len(neon_list), "Number of neon frames combined")
    hdr["BIASSUB"] = (bool(did_sub), "Bias-subtracted before stacking")
    hdr["SB_OK"] = (bool(superbias is not None), "Superbias file was found")
    hdr["XSHMAX"] = (int(xshift_max), "Max abs X shift allowed [px]")
    hdr["SHIFTMTH"] = ("xcorr_int", "X-shift method (integer, xcorr)")
    hdr["SHIFTMD"] = (float(np.median(dxs)), "Median X shift [px]")
    hdr["SHIFTMX"] = (int(np.max(np.abs(dxs))), "Max |X shift| [px]")
    hdr["CCSCMED"] = (float(np.median(scores)), "Median xcorr score (peak/RMS)")
    hdr["CLIPK"] = (float(clip_k), "Outlier threshold used for CLIPFR estimate")
    hdr["CLIPFR"] = (float(clip_frac), "Estimated outlier fraction in aligned stack")
    hdr["PKN"] = (int(len(pk)), "Number of detected peak candidates (pre-refine)")
    hdr["PKSIG"] = (float(sigma), "Robust background sigma used for peak thresholds")
    hdr["PKHGT"] = (float(min_height), "Peak min height (above baseline)")
    hdr["PKPRM"] = (float(prominence), "Peak prominence (above baseline)")

    if ref_sig is not None:
        hdr["FSIG"] = (ref_sig.describe(), "FrameSignature of the calibration product")

    # refine peaks (subpixel centroid)
    rows: list[tuple[float, float, float, float, float]] = []
    hw = int(wcfg.get("gauss_half_win", 4))
    for i0 in pk:
        xc, amp, fwhm = _refine_peak_centroid(x, residual, int(i0), hw=hw)
        snr = float(amp / sigma) if sigma > 0 else 0.0
        rows.append((xc, amp, snr, fwhm, float(np.interp(xc, x, prof))))

    # write peak candidates CSV
    if rows:
        arr = np.array(rows, dtype=float)
        hdr_csv = "x_pix,amp,snr,fwhm_pix,profile_I"
        np.savetxt(peaks_csv, arr, delimiter=",", header=hdr_csv, comments="", fmt="%.6f")
    else:
        peaks_csv.write_text("x_pix,amp,snr,fwhm_pix,profile_I\n", encoding="utf-8")

    # line SNR QC
    snrs = [float(r[2]) for r in rows] if rows else []
    snr_med = float(np.median(snrs)) if snrs else 0.0
    snr_p90 = float(np.percentile(snrs, 90)) if len(snrs) >= 3 else (snr_med if snrs else 0.0)
    hdr["NPK"] = (int(len(snrs)), "Number of detected peak candidates (refined)")
    hdr["SNRMED"] = (float(snr_med), "Median line S/N among detected peaks")
    hdr["SNRP90"] = (float(snr_p90), "90th percentile line S/N among detected peaks")

    # saturation QC (per input; computed on raw frames)
    sat_fracs = [_saturation_fraction(img, sat_level) for img in frames_raw]
    sat_med = float(np.median(sat_fracs)) if sat_fracs else 0.0
    hdr["SATFRAC"] = (float(sat_med), "Median saturated-pixel fraction in inputs")
    if sat_level is not None:
        hdr["SATLVL"] = (float(sat_level), "Estimated saturation level [ADU]")

    # save superneon artifacts
    _save_fits(superneon_fits, superneon, header=hdr)
    _save_png(superneon_png, superneon, title="Super-Neon (stacked)")

    # --- machine-readable shifts + QC (Block 04) ---
    shifts_rows = []
    for i, (p, _img, _hdr, _sig) in enumerate(items):
        shifts_rows.append(
            {
                "path": str(p),
                "dx": int(dxs[i]),
                "xcorr_score": float(scores[i]) if i < len(scores) else None,
                "sat_frac": float(sat_fracs[i]) if i < len(sat_fracs) else 0.0,
            }
        )

    _write_csv(
        shifts_csv,
        header="path,dx,xcorr_score,sat_frac",
        rows=[
            [
                r["path"],
                r["dx"],
                "" if r["xcorr_score"] is None else f"{float(r['xcorr_score']):.6g}",
                f"{float(r['sat_frac']):.6g}",
            ]
            for r in shifts_rows
        ],
    )
    _write_json(
        shifts_json,
        payload={
            "n_frames": int(len(neon_list)),
            "signature": ref_sig.to_dict(),
            "shift_method": "xcorr_int",
            "xshift_max_abs": int(xshift_max),
            "rows": shifts_rows,
            "summary": {
                "shift_median_px": float(np.median(dxs)),
                "shift_maxabs_px": int(np.max(np.abs(dxs))),
                "xcorr_score_median": float(np.median(scores)),
                "sat_frac_median": float(sat_med),
            },
        },
    )
    _write_json(
        qc_json,
        payload={
            "product": "superneon",
            "n_frames": int(len(neon_list)),
            "shape": [int(H), int(W)],
            "signature": ref_sig.to_dict(),
            "bias_subtracted": bool(did_sub),
            "superbias_path": (str(superbias_path) if (superbias_path is not None) else ""),
            "alignment": {
                "method": "xcorr_int",
                "profile_y": [int(y0), int(y1)],
                "xshift_max_abs": int(xshift_max),
                "shift_median_px": float(np.median(dxs)),
                "shift_maxabs_px": int(np.max(np.abs(dxs))),
                "xcorr_score_median": float(np.median(scores)),
            },
            "saturation": {
                "sat_level_adu": (None if sat_level is None else float(sat_level)),
                "sat_frac_median": float(sat_med),
                "sat_fracs": [float(v) for v in sat_fracs],
            },
            "clipping": {
                "clip_k": float(clip_k),
                "clip_frac": float(clip_frac),
                "note": "diagnostic estimate vs median stack; does not affect output",
            },
            "lines": {
                "sigma_bg": float(sigma),
                "n_peaks": int(len(snrs)),
                "snr_median": float(snr_med),
                "snr_p90": float(snr_p90),
            },
            "files": {
                "superneon_fits": str(superneon_fits),
                "superneon_png": str(superneon_png),
                "peaks_candidates_csv": str(peaks_csv),
                "superneon_shifts_csv": str(shifts_csv),
                "superneon_shifts_json": str(shifts_json),
                "superneon_qc_json": str(qc_json),
            },
        },
    )

    return SuperNeonResult(
        superneon_fits=superneon_fits, peaks_csv=peaks_csv, superneon_png=superneon_png
    )


