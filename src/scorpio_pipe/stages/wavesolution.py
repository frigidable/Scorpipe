from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import logging
import json
import numpy as np
from astropy.io import fits
from numpy.polynomial.chebyshev import chebvander2d, chebval2d

log = logging.getLogger(__name__)


# --------------------------- IO helpers ---------------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_hand_pairs(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read `hand_pairs.txt` written by LineID GUI.

    Returns
    -------
    x : (N,) float
    lam : (N,) float
    is_blend : (N,) bool
    """
    xs: list[float] = []
    lams: list[float] = []
    blends: list[bool] = []
    if not path.exists():
        return np.array([]), np.array([]), np.array([], bool)

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # allow inline comments
        blend = ("# blend" in s.lower())
        s = s.split("#", 1)[0].strip()
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            x0 = float(parts[0])
            lam = float(parts[1])
        except Exception:
            continue
        xs.append(x0)
        lams.append(lam)
        blends.append(blend)

    x = np.asarray(xs, float)
    lam = np.asarray(lams, float)
    is_blend = np.asarray(blends, bool)
    return x, lam, is_blend


# --------------------------- 1D solution ---------------------------

def robust_polyfit_1d(
    x: np.ndarray,
    y: np.ndarray,
    deg: int,
    *,
    weights: np.ndarray | None = None,
    sigma_clip: float = 3.0,
    maxiter: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Sigma-clipped polynomial fit.

    Returns (coeffs, used_mask) where coeffs are in np.polyval order.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)

    xw, yw = x[mask].copy(), y[mask].copy()
    ww = (np.ones_like(xw) if weights is None else np.asarray(weights, float)[mask].copy())
    ww = np.where(np.isfinite(ww) & (ww > 0), ww, 0.0)

    if xw.size < deg + 1:
        raise RuntimeError(f"Need at least {deg+1} points for deg={deg}, got {xw.size}")

    used = np.ones_like(xw, bool)
    for _ in range(maxiter):
        coeffs = np.polyfit(xw[used], yw[used], deg, w=ww[used])
        resid = yw - np.polyval(coeffs, xw)
        s = float(np.std(resid[used]))
        if not np.isfinite(s) or s <= 0:
            break
        new_used = (np.abs(resid) <= sigma_clip * s)
        if new_used.sum() == used.sum():
            break
        used = new_used

    coeffs = np.polyfit(xw[used], yw[used], deg, w=ww[used])
    # map used back to original mask
    used_mask = np.zeros_like(mask, bool)
    used_mask[np.where(mask)[0]] = used
    return coeffs, used_mask


def _plot_wavesol_1d(
    x: np.ndarray,
    lam: np.ndarray,
    coeffs: np.ndarray,
    used_mask: np.ndarray,
    outpng: Path,
    title: str,
) -> float:
    import matplotlib.pyplot as plt
    from scorpio_pipe.plot_style import mpl_style

    x_used = x[used_mask]
    lam_used = lam[used_mask]
    resid = lam_used - np.polyval(coeffs, x_used)
    rms = float(np.sqrt(np.mean(resid**2))) if resid.size else float("nan")

    with mpl_style():
        fig = plt.figure(figsize=(9.5, 6.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        ax1.set_title(f"{title}  (deg={len(coeffs)-1}, N={len(x_used)}, RMS={rms:.3f} Å)")
        ax1.scatter(x_used, lam_used, s=26, label="pairs (used)")
        if (~used_mask).any():
            ax1.scatter(x[~used_mask], lam[~used_mask], s=22, marker="x", label="clipped")
        xx = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 1200)
        ax1.plot(xx, np.polyval(coeffs, xx), lw=1.6, label="model")
        ax1.set_ylabel("Wavelength [Å]")
        ax1.legend(frameon=False, loc="best")

        ax2.axhline(0, linewidth=0.9)
        ax2.scatter(x_used, resid, s=20)
        ax2.set_xlabel("Pixel X")
        ax2.set_ylabel("Δλ [Å]")

        fig.savefig(outpng)
        plt.close(fig)

    return rms


# --------------------------- 2D tracing (xcorr) ---------------------------

def _parabolic_subpix(v: np.ndarray, i: int) -> float:
    """Sub-pixel parabola peak position around i."""
    if i <= 0 or i >= len(v) - 1:
        return 0.0
    y0, y1, y2 = float(v[i - 1]), float(v[i]), float(v[i + 1])
    denom = (y0 - 2.0 * y1 + y2)
    if denom == 0.0:
        return 0.0
    return 0.5 * (y0 - y2) / denom


def _trace_one_line_xcorr(
    img: np.ndarray,
    x0: float,
    y0: int,
    *,
    template_hw: int = 6,
    avg_half: int = 3,
    search_rad: int = 12,
    y_step: int = 1,
    amp_thresh: float = 20.0,
) -> tuple[list[float], list[float], list[float]]:
    """Trace one emission line in a 2D image by local X-correlation.

    Returns lists (xs, ys, scores) where score is max correlation at each row.
    """
    H, W = img.shape
    y0 = int(np.clip(y0, 0, H - 1))

    # template from several rows around y0
    x1 = int(max(0, np.floor(x0 - template_hw)))
    x2 = int(min(W, np.ceil(x0 + template_hw + 1)))
    y1 = int(max(0, y0 - avg_half))
    y2 = int(min(H, y0 + avg_half + 1))

    template = np.nanmean(img[y1:y2, x1:x2], axis=0)
    template = np.asarray(template, float)
    template = template - np.nanmedian(template)
    if not np.isfinite(template).all():
        template = np.nan_to_num(template)

    # go in two directions from y0
    xs: list[float] = [float(x0)]
    ys: list[float] = [float(y0)]
    scores: list[float] = [float(np.nanmax(template))]

    def _walk(direction: int) -> None:
        nonlocal xs, ys, scores
        x_prev = float(x0)
        y = y0 + direction * y_step
        while 0 <= y < H:
            # search window around x_prev
            c1 = int(max(0, np.floor(x_prev - search_rad - template_hw)))
            c2 = int(min(W, np.ceil(x_prev + search_rad + template_hw + 1)))
            row = np.asarray(img[y, c1:c2], float)
            if row.size < template.size + 2:
                y += direction * y_step
                continue
            row = row - np.nanmedian(row)
            row = np.nan_to_num(row)

            # correlation by sliding dot product (small kernel)
            corr = np.correlate(row, template, mode="valid")
            if corr.size == 0:
                y += direction * y_step
                continue
            i_max = int(np.argmax(corr))
            cmax = float(corr[i_max])
            if cmax < amp_thresh:
                y += direction * y_step
                continue

            sub = _parabolic_subpix(corr, i_max)
            x_found = (c1 + i_max + sub) + (len(template) - 1) / 2.0

            xs.append(float(x_found))
            ys.append(float(y))
            scores.append(cmax)

            x_prev = x_found
            y += direction * y_step

    _walk(+1)
    _walk(-1)
    return xs, ys, scores


def trace_lines_2d_cc(
    img2d: np.ndarray,
    lambda_list: Iterable[float],
    x_guesses: Iterable[float],
    *,
    template_hw: int = 6,
    avg_half: int = 3,
    search_rad: int = 12,
    y_step: int = 1,
    amp_thresh: float = 20.0,
    y0: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trace multiple lines; returns control points (x,y,lambda,score)."""
    H, _ = img2d.shape
    y0 = H // 2 if y0 is None else int(y0)

    xs_all: list[float] = []
    ys_all: list[float] = []
    lams_all: list[float] = []
    sc_all: list[float] = []

    for lam0, x0 in zip(lambda_list, x_guesses):
        xs, ys, scores = _trace_one_line_xcorr(
            img2d,
            float(x0),
            y0,
            template_hw=template_hw,
            avg_half=avg_half,
            search_rad=search_rad,
            y_step=y_step,
            amp_thresh=amp_thresh,
        )
        if len(xs) == 0:
            continue
        xs_all.extend(xs)
        ys_all.extend(ys)
        lams_all.extend([float(lam0)] * len(xs))
        sc_all.extend(scores)

    return (
        np.asarray(xs_all, float),
        np.asarray(ys_all, float),
        np.asarray(lams_all, float),
        np.asarray(sc_all, float),
    )


# --------------------------- 2D fit (Chebyshev) ---------------------------

def _scale_to_unit(v: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    v = np.asarray(v, float)
    vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        return v * 0.0, (vmin, 1.0)
    off = 0.5 * (vmax + vmin)
    sca = 0.5 * (vmax - vmin)
    return (v - off) / sca, (off, sca)


def robust_polyfit_2d_cheb(
    x: np.ndarray,
    y: np.ndarray,
    lam: np.ndarray,
    degx: int,
    degy: int,
    *,
    weights: np.ndarray | None = None,
    sigma_clip: float = 3.0,
    maxiter: int = 10,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    """Robust 2D Chebyshev fit: λ(x,y).

    Returns:
      C : (degx+1, degy+1) coefficient matrix for chebval2d
      meta : scaling dict with xo,xscl,yo,yscl
      used_mask : (N,) bool mask of used points
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    lam = np.asarray(lam, float)

    m0 = np.isfinite(x) & np.isfinite(y) & np.isfinite(lam)
    x = x[m0]; y = y[m0]; lam = lam[m0]

    if x.size < (degx + 1) * (degy + 1):
        raise RuntimeError(f"Not enough control points for degx={degx},degy={degy}: N={x.size}")

    xs, (xo, xscl) = _scale_to_unit(x)
    ys, (yo, yscl) = _scale_to_unit(y)

    V = chebvander2d(xs, ys, [degx, degy])  # (N, (degx+1)*(degy+1))
    if weights is None:
        ww = np.ones_like(lam)
    else:
        w_in = np.asarray(weights, float)
        ww = np.where(np.isfinite(w_in) & (w_in > 0), w_in, 0.0)
        ww = ww[m0]

    used = np.ones_like(lam, bool)
    for _ in range(maxiter):
        W = ww[used]
        A = V[used] * W[:, None]
        b = lam[used] * W
        coeff_vec, *_ = np.linalg.lstsq(A, b, rcond=None)
        model = V @ coeff_vec
        resid = lam - model
        s = float(np.std(resid[used]))
        if not np.isfinite(s) or s <= 0:
            break
        new_used = (np.abs(resid) <= sigma_clip * s)
        if new_used.sum() == used.sum():
            break
        used = new_used

    # final
    W = ww[used]
    A = V[used] * W[:, None]
    b = lam[used] * W
    coeff_vec, *_ = np.linalg.lstsq(A, b, rcond=None)

    C = coeff_vec.reshape((degx + 1, degy + 1))
    meta = {"xo": float(xo), "xscl": float(xscl), "yo": float(yo), "yscl": float(yscl)}
    return C, meta, used


def _plot_wavelength_matrix(lam_map: np.ndarray, outpng: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    from scorpio_pipe.plot_style import mpl_style

    with mpl_style():
        fig = plt.figure(figsize=(10.5, 4.8))
        ax = fig.add_subplot(111)
        im = ax.imshow(lam_map, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("X [px]")
        ax.set_ylabel("Y [px]")
        fig.colorbar(im, ax=ax, label="λ [Å]")
        fig.savefig(outpng)
        plt.close(fig)


def _plot_residuals_2d(xs: np.ndarray, ys: np.ndarray, resid: np.ndarray, outpng: Path, title: str) -> float:
    import matplotlib.pyplot as plt
    from scorpio_pipe.plot_style import mpl_style

    rms = float(np.sqrt(np.mean(resid**2))) if resid.size else float("nan")
    with mpl_style():
        fig = plt.figure(figsize=(9.5, 6.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        ax1.set_title(f"{title}  (RMS={rms:.3f} Å, N={len(resid)})")
        sc = ax1.scatter(xs, ys, c=resid, s=6)
        ax1.set_xlabel("X [px]")
        ax1.set_ylabel("Y [px]")
        fig.colorbar(sc, ax=ax1, label="Δλ [Å]")

        ax2.hist(resid, bins=40)
        ax2.set_xlabel("Δλ [Å]")
        ax2.set_ylabel("N")

        fig.savefig(outpng)
        plt.close(fig)
    return rms


# --------------------------- public stage ---------------------------

@dataclass
class WaveSolutionResult:
    poly1d_png: Path
    poly1d_json: Path
    poly1d_resid_csv: Path
    cheb2d_json: Path
    cheb2d_resid_csv: Path
    lambda_map_fits: Path
    lambda_map_png: Path
    resid2d_png: Path


def build_wavesolution(cfg: dict[str, Any]) -> WaveSolutionResult:
    """Build 1D and 2D dispersion solution from superneon + hand_pairs.

    Expected inputs in work_dir/wavesol:
      - superneon.fits
      - hand_pairs.txt (from LineID)
    Outputs are also written to work_dir/wavesol.
    """
    wcfg = cfg.get("wavesol", {}) if isinstance(cfg.get("wavesol"), dict) else {}
    work_dir = Path(str(cfg.get("work_dir", "."))).expanduser()
    if not work_dir.is_absolute():
        base = Path(str(cfg.get("config_dir", ".")))
        work_dir = (base / work_dir).resolve()
    else:
        work_dir = work_dir.resolve()
    # disperser-specific layout (so multiple gratings can live in one work_dir)
    from scorpio_pipe.wavesol_paths import wavesol_dir

    outdir = _ensure_dir(wavesol_dir(cfg))

    superneon_fits = outdir / "superneon.fits"

    # Allow overriding the hand pairs file (useful for alternative pair-sets,
    # built-in pair libraries, or cleaned subsets).
    hp_raw = str(wcfg.get("hand_pairs_path", "") or "").strip()
    if hp_raw:
        hp = Path(hp_raw)
        hand_pairs = hp if hp.is_absolute() else (work_dir / hp)
    else:
        hand_pairs = outdir / "hand_pairs.txt"
    if not superneon_fits.exists():
        raise FileNotFoundError(f"Missing: {superneon_fits} (run superneon first)")
    if not hand_pairs.exists():
        raise FileNotFoundError(f"Missing: {hand_pairs} (run LineID first)")

    x, lam, is_blend = _read_hand_pairs(hand_pairs)
    if x.size < 5:
        raise RuntimeError(f"Not enough pairs in {hand_pairs} (need >=5, got {x.size})")

    # 1D poly fit
    deg1d = int(wcfg.get("poly_deg_1d", 4))
    w_blend = np.where(is_blend, float(wcfg.get("blend_weight", 0.3)), 1.0)
    coeffs1d, used_mask = robust_polyfit_1d(
        x, lam, deg1d,
        weights=w_blend,
        sigma_clip=float(wcfg.get("poly_sigma_clip", 3.0)),
        maxiter=int(wcfg.get("poly_maxiter", 10)),
    )

    poly1d_png = outdir / "wavesolution_1d.png"
    rms1d = _plot_wavesol_1d(x, lam, coeffs1d, used_mask, poly1d_png, title="1D dispersion solution")

    poly1d_json = outdir / "wavesolution_1d.json"
    poly1d_json.write_text(json.dumps({
        "deg": int(deg1d),
        "coeffs_polyval": [float(c) for c in coeffs1d],  # highest order first
        "rms_A": float(rms1d),
        "n_pairs": int(x.size),
        "n_used": int(np.sum(used_mask)),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    poly1d_resid_csv = outdir / "residuals_1d.csv"
    resid_all = lam - np.polyval(coeffs1d, x)
    rows = np.column_stack([x, lam, resid_all, used_mask.astype(int), is_blend.astype(int)])
    np.savetxt(poly1d_resid_csv, rows, delimiter=",",
               header="x_pix,lambda_A,delta_lambda_A,used,blend",
               comments="", fmt="%.6f")

    # 2D: trace lines and fit cheb surface
    img2d = fits.getdata(superneon_fits).astype(float)
    y0 = wcfg.get("trace_y0", None)
    xs_cp, ys_cp, lams_cp, scores = trace_lines_2d_cc(
        img2d,
        lambda_list=lam,
        x_guesses=x,
        template_hw=int(wcfg.get("trace_template_hw", 6)),
        avg_half=int(wcfg.get("trace_avg_half", 3)),
        search_rad=int(wcfg.get("trace_search_rad", 12)),
        y_step=int(wcfg.get("trace_y_step", 1)),
        amp_thresh=float(wcfg.get("trace_amp_thresh", 20.0)),
        y0=(None if y0 is None else int(y0)),
    )

    # basic filtering: remove short traces per line
    min_pts = int(wcfg.get("trace_min_pts", 120))
    keep = np.ones_like(xs_cp, bool)
    for lam0 in np.unique(lams_cp):
        m = (np.abs(lams_cp - lam0) < 1e-6)
        if m.sum() < min_pts:
            keep[m] = False

    xs_cp = xs_cp[keep]; ys_cp = ys_cp[keep]; lams_cp = lams_cp[keep]; scores = scores[keep]
    if xs_cp.size < 50:
        raise RuntimeError("Too few 2D control points after filtering; try lowering trace_min_pts / amp_thresh.")

    degx = int(wcfg.get("cheb_degx", 5))
    degy = int(wcfg.get("cheb_degy", 3))
    C, meta, used2d = robust_polyfit_2d_cheb(
        xs_cp, ys_cp, lams_cp, degx, degy,
        weights=np.sqrt(np.clip(scores, 0, None)),
        sigma_clip=float(wcfg.get("cheb_sigma_clip", 3.0)),
        maxiter=int(wcfg.get("cheb_maxiter", 10)),
    )

    # residuals at control points
    xs_s = (xs_cp - meta["xo"]) / meta["xscl"]
    ys_s = (ys_cp - meta["yo"]) / meta["yscl"]
    lam_model = chebval2d(xs_s, ys_s, C)
    resid2d = lams_cp - lam_model

    cheb2d_json = outdir / "wavesolution_2d.json"
    cheb2d_json.write_text(json.dumps({
        "degx": int(degx), "degy": int(degy),
        "C": C.tolist(),
        "meta": meta,
        "n_points": int(xs_cp.size),
        "n_used": int(np.sum(used2d)),
        "rms_A": float(np.sqrt(np.mean(resid2d[used2d]**2))),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    cheb2d_resid_csv = outdir / "residuals_2d.csv"
    rows2 = np.column_stack([xs_cp, ys_cp, lams_cp, resid2d, used2d.astype(int), scores])
    np.savetxt(cheb2d_resid_csv, rows2, delimiter=",",
               header="x_pix,y_pix,lambda_A,delta_lambda_A,used,score",
               comments="", fmt="%.6f")

    # lambda map
    H, W = img2d.shape
    xg = (np.arange(W, dtype=float) - meta["xo"]) / meta["xscl"]
    yg = (np.arange(H, dtype=float) - meta["yo"]) / meta["yscl"]
    lam_map = chebval2d(xg[None, :], yg[:, None], C).astype(np.float32)

    lambda_map_fits = outdir / "lambda_map.fits"
    fits.writeto(lambda_map_fits, lam_map, overwrite=True)

    lambda_map_png = outdir / "wavelength_matrix.png"
    _plot_wavelength_matrix(lam_map, lambda_map_png, title="2D wavelength map λ(x,y)")

    resid2d_png = outdir / "residuals_2d.png"
    _plot_residuals_2d(xs_cp[used2d], ys_cp[used2d], resid2d[used2d], resid2d_png, title="2D fit residuals (control points)")

    return WaveSolutionResult(
        poly1d_png=poly1d_png,
        poly1d_json=poly1d_json,
        poly1d_resid_csv=poly1d_resid_csv,
        cheb2d_json=cheb2d_json,
        cheb2d_resid_csv=cheb2d_resid_csv,
        lambda_map_fits=lambda_map_fits,
        lambda_map_png=lambda_map_png,
        resid2d_png=resid2d_png,
    )
