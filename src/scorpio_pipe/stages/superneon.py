from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Any

import numpy as np
from astropy.io import fits

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


def _xcorr_shift_1d(ref: np.ndarray, cur: np.ndarray, max_abs: int = 6) -> int:
    """
    Целочисленный сдвиг cur относительно ref по максимуму корреляции.
    Ограничиваем поиск |shift|<=max_abs.
    """
    ref = np.asarray(ref, float)
    cur = np.asarray(cur, float)
    # нормировка чтобы DC не доминировал
    ref = ref - np.nanmedian(ref)
    cur = cur - np.nanmedian(cur)
    # корреляция через FFT
    n = int(2 ** np.ceil(np.log2(ref.size + cur.size)))
    fr = np.fft.rfft(ref, n=n)
    fc = np.fft.rfft(cur, n=n)
    cc = np.fft.irfft(fr * np.conj(fc), n=n)
    # cc индексы: 0..n-1, "нулевой" лаг в 0
    # переводим к лагам [-n/2..+n/2] удобнее через argmax около 0
    lags = np.arange(n)
    # приведём лаги к signed
    signed = (lags + n // 2) % n - n // 2
    m = np.abs(signed) <= int(max_abs)
    idx = np.argmax(cc[m])
    shift = int(signed[m][idx])
    return shift


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
    """
    Требует в cfg:
      - work_dir (str)
      - frames.neon (list[str])
      - calib.superbias_path (str)  [опционально]
      - wavesol.profile_y (tuple[int,int]) [опционально]
      - wavesol.xshift_max_abs (int) [опционально]
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

    # --- resolve superbias path robustly ---
    # Supported conventions:
    #   - calib/superbias.fits           (relative to work_dir)
    #   - work/run1/calib/superbias.fits (relative to project root)
    #   - absolute path
    sb_raw = str((cfg.get("calib", {}) or {}).get("superbias_path", "")).strip()
    superbias_path = Path(sb_raw) if sb_raw else Path("")
    if sb_raw and not superbias_path.is_absolute():
        base_cfg = Path(str(cfg.get("config_dir", "."))).resolve()
        candidates = [
            (work_dir / superbias_path).resolve(),
            (base_cfg / superbias_path).resolve(),
            _resolve_from_root(superbias_path),
        ]
        superbias_path = next((c for c in candidates if c.is_file()), candidates[0])

    superbias = None
    if bias_sub and superbias_path.is_file():
        superbias, _ = _load_fits(superbias_path)

    # грузим все neon (с вычитанием superbias, если совпадает shape)
    frames = []
    did_sub = False
    ref_hdr = None
    n_jobs = int((cfg.get("runtime", {}) or {}).get("n_jobs", 0) or 0)
    if n_jobs <= 0:
        n_jobs = max(1, min(8, os.cpu_count() or 1))

    def _read_and_sub(p: Path):
        img, hdr = _load_fits(p)
        if superbias is not None and superbias.shape == img.shape:
            return (img - superbias, hdr, True)
        return (img, hdr, False)

    if len(neon_list) >= 4 and n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            for img, hdr, did in ex.map(_read_and_sub, neon_list):
                if ref_hdr is None:
                    ref_hdr = hdr
                did_sub = did_sub or bool(did)
                frames.append(img)
    else:
        for p in neon_list:
            img, hdr, did = _read_and_sub(p)
            if ref_hdr is None:
                ref_hdr = hdr
            did_sub = did_sub or bool(did)
            frames.append(img)

    # выбираем полосу по Y для построения профиля и x-shift
    H, W = frames[0].shape
    log.info("SuperNeon: loaded %d frames (bias_sub=%s)", len(frames), did_sub)
    wcfg = cfg.get("wavesol", {}) or {}
    prof_y = wcfg.get("profile_y")
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

    # выравнивание всех остальных по X
    for i, img in enumerate(frames[1:], start=1):
        cur_prof = _profile_x(img, y0, y1)
        dx = _xcorr_shift_1d(ref_prof, cur_prof, max_abs=xshift_max)
        log.debug("SuperNeon shift frame #%d: dx=%d", i, dx)
        aligned.append(_shift_x_int(img, dx))

    superneon = _median_stack(aligned)

    # выходные файлы
    superneon_fits = outdir / "superneon.fits"
    superneon_png = outdir / "superneon.png"
    peaks_csv = outdir / "peaks_candidates.csv"

    # header: минимально полезные ключи
    hdr = fits.Header()
    hdr["NNEON"] = len(neon_list)
    hdr["SB_OK"] = bool(superbias is not None)
    hdr["SB_SUB"] = bool(did_sub)
    hdr["Y0PROF"] = y0
    hdr["Y1PROF"] = y1
    hdr["XSHMAX"] = xshift_max

    _save_fits(superneon_fits, superneon, header=hdr)
    _save_png(superneon_png, superneon, title="Super-Neon (stacked)")

    # кандидаты пиков по профилю I(x)
    prof = _profile_x(superneon, y0, y1)
    x = np.arange(W, dtype=float)

    wcfg = cfg.get("wavesol", {}) or {}
    noise_cfg = (wcfg.get("noise") or {}) if isinstance(wcfg, dict) else {}

    baseline = _estimate_baseline_bins(
        prof,
        bin_size=int(noise_cfg.get("baseline_bin_size", 32)),
        q=float(noise_cfg.get("baseline_quantile", 0.2)),
        smooth_bins=int(noise_cfg.get("baseline_smooth_bins", 5)),
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
    distance = int(wcfg.get("peak_distance", 3))

    # Автоподстройка: если слишком мало/много линий — мягко сдвигаем порог
    autotune = bool(wcfg.get("peak_autotune", True))
    target_min = int(wcfg.get("peak_target_min", 0) or 0)
    target_max = int(wcfg.get("peak_target_max", 0) or 0)
    snr_min = float(wcfg.get("peak_snr_min", 2.5))
    snr_max = float(wcfg.get("peak_snr_max", 12.0))
    relax = float(wcfg.get("peak_snr_relax", 0.85))
    boost = float(wcfg.get("peak_snr_boost", 1.15))
    max_tries = int(wcfg.get("peak_autotune_max_tries", 10))

    def _detect(snr_k: float):
        min_height = wcfg.get("peak_min_amp", None)
        if min_height is None:
            min_height = max(floor_k * sigma, float(snr_k) * sigma)

        prominence = wcfg.get("peak_prominence", None)
        if prominence is None:
            prominence = max(1.0 * sigma, peak_prom_snr * sigma)

        pk = _find_peaks_simple(
            residual,
            min_height=float(min_height),
            prominence=float(prominence),
            distance=distance,
        )
        return pk, float(min_height), float(prominence)

    used_snr = float(peak_snr)
    pk, min_height, prominence = _detect(used_snr)

    n_try = 0
    if autotune and target_min > 0:
        while len(pk) < target_min and used_snr > snr_min and n_try < max_tries:
            used_snr *= relax
            pk, min_height, prominence = _detect(used_snr)
            n_try += 1

    if autotune and target_max > 0:
        while len(pk) > target_max and used_snr < snr_max and n_try < max_tries:
            used_snr *= boost
            pk, min_height, prominence = _detect(used_snr)
            n_try += 1

    log.info(
        "task=superneon peaks sigma=%.3g snr=%.3g height=%.3g prom=%.3g distance=%d n_peaks=%d",
        sigma,
        used_snr,
        min_height,
        prominence,
        distance,
        int(len(pk)),
    )

    hdr["PKSIG"] = (float(sigma), "Robust background sigma used for peak thresholds")
    hdr["PKHGT"] = (float(min_height), "Peak min height (above median)")
    hdr["PKPRM"] = (float(prominence), "Peak prominence (above median)")

    rows = []
    hw = int(wcfg.get("gauss_half_win", 4))
    for i0 in pk:
        xc, amp, fwhm = _refine_peak_centroid(x, residual, int(i0), hw=hw)
        snr = float(amp / sigma) if sigma > 0 else 0.0
        rows.append((xc, amp, snr, fwhm, float(np.interp(xc, x, prof))))

    if rows:
        arr = np.array(rows, dtype=float)
        hdr_csv = "x_pix,amp,snr,fwhm_pix,profile_I"
        np.savetxt(
            peaks_csv, arr, delimiter=",", header=hdr_csv, comments="", fmt="%.6f"
        )
    else:
        peaks_csv.write_text("x_pix,amp,snr,fwhm_pix,profile_I\n", encoding="utf-8")

    return SuperNeonResult(
        superneon_fits=superneon_fits, peaks_csv=peaks_csv, superneon_png=superneon_png
    )
