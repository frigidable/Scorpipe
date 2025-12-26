from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Tuple
import numpy as np


def _project_root() -> Path:
    """Project root in source layout and PyInstaller (onefile) builds."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(str(meipass)).resolve()
    return Path(__file__).resolve().parents[3]


def _resolve_from_root(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (_project_root() / p).resolve()


def _load_neon_lines_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    try:
        data = np.genfromtxt(
            str(path), delimiter=",", names=True, dtype=None, encoding=None
        )
        names = [n.lower() for n in data.dtype.names]
        wl_key = None
        for cand in ("wavelength", "lambda", "wl", "lam", "lambda_a", "wavelength_a"):
            if cand in names:
                wl_key = data.dtype.names[names.index(cand)]
                break
        if wl_key is None:
            wl_key = data.dtype.names[0]
        lam = np.asarray(data[wl_key], dtype=float)

        w = np.ones_like(lam, dtype=float)
        for cand in ("intensity", "relint", "i", "strength"):
            if cand in names:
                ik = data.dtype.names[names.index(cand)]
                w = np.asarray(data[ik], dtype=float)
                break
    except Exception:
        arr = np.genfromtxt(str(path), delimiter=",", dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        lam = arr[:, 0].astype(float)
        w = arr[:, 1].astype(float) if arr.shape[1] >= 2 else np.ones_like(lam)

    if np.nanmax(lam) < 1000:
        lam = lam * 10.0

    m = np.isfinite(lam)
    lam = lam[m]
    w = w[m] if w.size == lam.size else np.ones_like(lam)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)

    idx = np.argsort(lam)
    return lam[idx], w[idx]


def _nearest_idx(sorted_arr: np.ndarray, x: np.ndarray) -> np.ndarray:
    i = np.searchsorted(sorted_arr, x)
    i0 = np.clip(i - 1, 0, len(sorted_arr) - 1)
    i1 = np.clip(i, 0, len(sorted_arr) - 1)
    d0 = np.abs(sorted_arr[i0] - x)
    d1 = np.abs(sorted_arr[i1] - x)
    return np.where(d1 < d0, i1, i0)


def _suggest_linear_pairs(
    px: np.ndarray,
    amp: np.ndarray,
    lam_lab: np.ndarray,
    w_lab: np.ndarray,
    tol_A: float,
    top_peaks: int,
    top_lines: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    ordp = np.argsort(-amp) if amp.size == px.size else np.arange(px.size)
    px_top = px[ordp[: min(top_peaks, px.size)]].astype(float)

    ordl = np.argsort(-w_lab) if w_lab.size == lam_lab.size else np.arange(lam_lab.size)
    lam_top = lam_lab[ordl[: min(top_lines, lam_lab.size)]].astype(float)
    w_top = w_lab[ordl[: min(top_lines, lam_lab.size)]].astype(float)

    sidx = np.argsort(lam_top)
    lam_top = lam_top[sidx]
    w_top = w_top[sidx]

    best = None  # (score,a,b,nmatch,rms)
    for i in range(len(px_top)):
        for j in range(i + 1, len(px_top)):
            dx = px_top[j] - px_top[i]
            if abs(dx) < 30:
                continue
            for p in range(len(lam_top)):
                for q in range(p + 1, len(lam_top)):
                    dlam = lam_top[q] - lam_top[p]
                    if abs(dlam) < 5:
                        continue
                    a = dlam / dx
                    if not (0.05 <= abs(a) <= 50.0):
                        continue
                    b = lam_top[p] - a * px_top[i]

                    lam_pred = a * px_top + b
                    kk = _nearest_idx(lam_top, lam_pred)
                    d = np.abs(lam_top[kk] - lam_pred)
                    ok = d <= tol_A
                    nmatch = int(np.sum(ok))
                    if nmatch < 6:
                        continue

                    rms = float(np.sqrt(np.mean(d[ok] ** 2)))
                    score = nmatch * 1000.0 + float(np.sum(w_top[kk[ok]])) - 30.0 * rms
                    if best is None or score > best[0]:
                        best = (score, float(a), float(b), nmatch, rms)

    if best is None:
        raise RuntimeError(
            "Не удалось построить первичную линейную привязку (auto-pairs). "
            "Проверь neon_lines.csv или увеличь wavesol.pair_tol_A."
        )

    _, a, b, nmatch, rms = best

    lam_pred_all = a * px.astype(float) + b
    idx_all = _nearest_idx(lam_lab, lam_pred_all)
    d_all = np.abs(lam_lab[idx_all] - lam_pred_all)

    used = set()
    pairs = []
    for x0, ii, dd in sorted(zip(px.astype(float), idx_all, d_all), key=lambda t: t[0]):
        if dd > tol_A:
            continue
        ii = int(ii)
        if ii in used:
            continue
        used.add(ii)
        pairs.append((x0, float(lam_lab[ii]), float(dd)))

    if len(pairs) < 8:
        raise RuntimeError(
            f"Слишком мало auto-пар: {len(pairs)} (нужно хотя бы ~8–10)."
        )

    x_pairs = np.array([p[0] for p in pairs], float)
    lam_pairs = np.array([p[1] for p in pairs], float)
    d_pairs = np.array([p[2] for p in pairs], float)
    meta = {"a": a, "b": b, "nmatch": nmatch, "rms_A": rms}
    return x_pairs, lam_pairs, d_pairs, meta


def prepare_lineid(cfg: dict[str, Any]) -> dict[str, Path]:
    """Prepare files for the LineID GUI.

    Writes:
      - manual_pairs_template.csv  (top peaks template for manual selection)
      - manual_pairs_auto.csv      (auto-suggested (x_pix, lambda) pairs)
      - lineid_report.txt          (human-readable summary)

    Returns a dict with these paths.
    """

    from scorpio_pipe.paths import resolve_work_dir
    from scorpio_pipe.wavesol_paths import wavesol_dir

    work_dir = resolve_work_dir(cfg)
    outdir = wavesol_dir(cfg)
    outdir.mkdir(parents=True, exist_ok=True)

    superneon_fits = outdir / "superneon.fits"
    peaks_csv = outdir / "peaks_candidates.csv"
    if not superneon_fits.exists() or not peaks_csv.exists():
        raise FileNotFoundError(
            "Нужны superneon.fits и peaks_candidates.csv. Сначала запусти doit superneon."
        )

    wcfg = cfg.get("wavesol", {}) or {}
    from scorpio_pipe.resource_utils import resolve_resource

    lines_res = resolve_resource(
        (wcfg.get("neon_lines_csv", "neon_lines.csv")),
        work_dir=work_dir,
        config_dir=cfg.get("config_dir"),
        project_root=cfg.get("project_root"),
        allow_package=True,
    )
    lines_path = lines_res.path
    if not lines_path.exists():
        raise FileNotFoundError(f"neon_lines.csv not found: {lines_path}")

    tol_A = float(wcfg.get("pair_tol_A", 2.5))
    top_peaks = int(wcfg.get("pair_top_peaks", 28))
    top_lines = int(wcfg.get("pair_top_lines", 120))

    pk = np.genfromtxt(peaks_csv, delimiter=",", names=True, dtype=float)
    px = np.asarray(pk["x_pix"], float)
    amp = np.asarray(pk["amp"], float)
    fwhm = np.asarray(pk["fwhm_pix"], float)

    lam_lab, w_lab = _load_neon_lines_csv(lines_path)

    tpl_path = outdir / "manual_pairs_template.csv"
    sel = np.argsort(-amp)[: min(30, amp.size)]
    with tpl_path.open("w", encoding="utf-8", newline="") as f:
        f.write("use,x_pix,lambda_A,amp,fwhm_pix,comment\n")
        for i in sel:
            f.write(f"1,{px[i]:.6f},,{amp[i]:.6f},{fwhm[i]:.6f},\n")

    auto_path = outdir / "manual_pairs_auto.csv"
    rep_path = outdir / "lineid_report.txt"

    x_pairs, lam_pairs, d_pairs, meta = _suggest_linear_pairs(
        px, amp, lam_lab, w_lab, tol_A, top_peaks, top_lines
    )

    idx_near = np.argmin(np.abs(px[None, :] - x_pairs[:, None]), axis=1)
    with auto_path.open("w", encoding="utf-8", newline="") as f:
        f.write("use,x_pix,lambda_A,delta_A,amp,fwhm_pix,comment\n")
        for k in range(len(x_pairs)):
            i = int(idx_near[k])
            f.write(
                f"1,{x_pairs[k]:.6f},{lam_pairs[k]:.6f},{d_pairs[k]:.6f},{amp[i]:.6f},{fwhm[i]:.6f},auto\n"
            )

    with rep_path.open("w", encoding="utf-8") as f:
        f.write("LINE-ID (auto) report\n")
        f.write(f"lines_csv: {lines_path}\n")
        f.write(f"tol_A: {tol_A}\n")
        f.write(f"pairs_auto: {len(x_pairs)}\n")
        f.write(f"linear_guess: a={meta['a']:.6f} A/px, b={meta['b']:.3f} A\n")
        f.write(f"linear_rms_A: {meta['rms_A']:.3f}\n")

    return {"template": tpl_path, "auto": auto_path, "report": rep_path}
