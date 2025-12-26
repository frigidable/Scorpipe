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


# --------------------------- 2D fit (power + Chebyshev) ---------------------------

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
    x = x[m0]
    y = y[m0]
    lam = lam[m0]

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
    # keep both short and "program style" keys for compatibility
    meta = {
        "xo": float(xo), "xscl": float(xscl), "yo": float(yo), "yscl": float(yscl),
        "x_off": float(xo), "x_s": float(xscl), "y_off": float(yo), "y_s": float(yscl),
        "degx": int(degx), "degy": int(degy),
        "kind": "chebyshev",
    }
    return C, meta, used


def polyval2d_cheb(x: np.ndarray, y: np.ndarray, C: np.ndarray, meta: dict[str, float]) -> np.ndarray:
    """Evaluate Chebyshev 2D model with stored scaling."""
    xs = (np.asarray(x, float) - float(meta["x_off"])) / float(meta["x_s"])
    ys = (np.asarray(y, float) - float(meta["y_off"])) / float(meta["y_s"])
    return chebval2d(xs, ys, C)


def _terms_total_degree(deg: int) -> list[tuple[int, int]]:
    terms: list[tuple[int, int]] = []
    for i in range(int(deg) + 1):
        for j in range(int(deg) + 1 - i):
            terms.append((i, j))
    return terms


def robust_polyfit_2d_power(
    x: np.ndarray,
    y: np.ndarray,
    lam: np.ndarray,
    deg: int,
    *,
    weights: np.ndarray | None = None,
    sigma_clip: float = 3.0,
    maxiter: int = 10,
) -> tuple[np.ndarray, dict[str, float | int | list], np.ndarray]:
    """Robust 2D polynomial of total degree: λ(x,y).

    This follows the same logic as the reference "SKY_MODEL" script:
    - scale x and y to ~[-1,1]
    - weighted least squares
    - iterative sigma clipping
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    lam = np.asarray(lam, float)

    m0 = np.isfinite(x) & np.isfinite(y) & np.isfinite(lam)
    x = x[m0]
    y = y[m0]
    lam = lam[m0]

    deg = int(deg)
    terms = _terms_total_degree(deg)
    npar = len(terms)
    if x.size < npar:
        raise RuntimeError(f"Not enough control points for power deg={deg}: N={x.size}, npar={npar}")

    xs, (xo, xscl) = _scale_to_unit(x)
    ys, (yo, yscl) = _scale_to_unit(y)

    # Vandermonde
    A = np.vstack([(xs ** i) * (ys ** j) for (i, j) in terms]).T  # (N, npar)
    if weights is None:
        ww = np.ones_like(lam)
    else:
        w_in = np.asarray(weights, float)
        ww = np.where(np.isfinite(w_in) & (w_in > 0), w_in, 0.0)
        ww = ww[m0]

    used = np.ones_like(lam, bool)
    for _ in range(int(maxiter)):
        W = ww[used]
        Aw = A[used] * W[:, None]
        bw = lam[used] * W
        coeff, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        pred = A @ coeff
        resid = lam - pred
        s = float(np.std(resid[used]))
        if not np.isfinite(s) or s <= 0:
            break
        new_used = (np.abs(resid) <= float(sigma_clip) * s)
        if new_used.sum() == used.sum():
            break
        used = new_used

    # final solve
    W = ww[used]
    Aw = A[used] * W[:, None]
    bw = lam[used] * W
    coeff, *_ = np.linalg.lstsq(Aw, bw, rcond=None)

    meta: dict[str, float | int | list] = {
        "x_off": float(xo), "x_s": float(xscl),
        "y_off": float(yo), "y_s": float(yscl),
        "deg": int(deg),
        "terms": [(int(i), int(j)) for (i, j) in terms],
        "kind": "power",
    }
    return coeff, meta, used


def polyval2d_power(x: np.ndarray, y: np.ndarray, coeff: np.ndarray, meta: dict[str, float | int | list]) -> np.ndarray:
    """Evaluate total-degree 2D polynomial model with stored scaling."""
    xs = (np.asarray(x, float) - float(meta["x_off"])) / float(meta["x_s"])
    ys = (np.asarray(y, float) - float(meta["y_off"])) / float(meta["y_s"])
    terms = meta.get("terms")
    if not isinstance(terms, list):
        raise TypeError("meta['terms'] must be a list")
    # broadcast
    out = np.zeros_like(xs, dtype=float)
    for c, ij in zip(np.asarray(coeff, float), terms):
        i, j = int(ij[0]), int(ij[1])
        out += float(c) * (xs ** i) * (ys ** j)
    return out


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


def _plot_residuals_2d(ys: np.ndarray, lams: np.ndarray, resid: np.ndarray, outpng: Path, title: str) -> float:
    """IDL-like 2D residual plot: one curve per line, stacked by vertical offsets.

    Parameters
    ----------
    ys : array
        Y positions of control points.
    lams : array
        Laboratory wavelengths [Å] for each control point.
    resid : array
        Residuals (lambda_obs - lambda_model) [Å].
    """
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms
    from matplotlib.colors import LinearSegmentedColormap
    from scorpio_pipe.plot_style import mpl_style

    ys = np.asarray(ys, float)
    lams = np.asarray(lams, float)
    resid = np.asarray(resid, float)

    m = np.isfinite(ys) & np.isfinite(lams) & np.isfinite(resid)
    ys = ys[m]
    lams = lams[m]
    resid = resid[m]

    # prefer the same sign as the reference script: Δλ = λ_model - λ_lab
    dlam = -resid

    rms = float(np.sqrt(np.mean(dlam**2))) if dlam.size else float("nan")
    y_offset_step = float(max(0.5, min(2.5, 4.0 * (rms if np.isfinite(rms) else 1.0))))

    uniq = np.array(sorted(np.unique(lams)), dtype=float)
    cmap = LinearSegmentedColormap.from_list("blue_to_red", ["blue", "red"])
    colors = cmap(np.linspace(0.0, 1.0, max(1, len(uniq))))

    with mpl_style():
        fig, ax = plt.subplots(figsize=(9.0, 7.8))
        ax.set_xlabel("Y [px]")
        ax.set_ylabel("Δλ [Å]")
        ax.set_title(f"{title}  (RMS={rms:.3f} Å)")

        text_rows: list[tuple[float, tuple, float, float, float]] = []

        for k, lam0 in enumerate(uniq):
            mm = np.abs(lams - lam0) < 1e-6
            if not np.any(mm):
                continue
            yk = ys[mm]
            dk = dlam[mm]
            if yk.size < 3:
                continue

            ordy = np.argsort(yk)
            y_sorted = yk[ordy]
            d_sorted = dk[ordy]
            off = k * y_offset_step
            col = colors[k % len(colors)]

            ax.plot(y_sorted, d_sorted + off, lw=1.2, color=col)
            step_pts = max(1, len(y_sorted) // 150)
            ax.scatter(y_sorted[::step_pts], (d_sorted + off)[::step_pts], s=3, color=col, alpha=0.85, linewidths=0.0)

            good = np.isfinite(dk)
            if int(np.count_nonzero(good)) >= 2:
                mu = float(np.mean(dk[good]))
                sd = float(np.std(dk[good], ddof=1))
            else:
                mu, sd = np.nan, np.nan
            y_text = off + (float(np.nanmean(dk)) if np.any(np.isfinite(dk)) else 0.0)
            text_rows.append((y_text, col, float(lam0), mu, sd))

        # helper lines at label heights
        for y_text, col, *_ in text_rows:
            ax.axhline(y=y_text, xmin=0.0, xmax=1.0, color=col, linestyle=":", linewidth=0.9, alpha=0.35, zorder=0)

        fig.canvas.draw()
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        y_top = ax.get_ylim()[1]
        x_col1 = 1.02
        x_col2 = 1.02 + 0.18
        ax.text(x_col1, y_top + 0.1, "λ", transform=trans, ha="left", va="bottom", fontsize=11)
        ax.text(x_col2, y_top + 0.1, "Δλ [Å]", transform=trans, ha="left", va="bottom", fontsize=11)

        for y_text, col, lam0, mu, sd in text_rows:
            ax.text(x_col1, y_text, f"{lam0:7.2f} Å", color=col, transform=trans, ha="left", va="center", fontsize=10)
            s = f"{mu:+.2f} ± {sd:.2f}" if np.isfinite(mu) else "—"
            s = s.replace("-", "−")
            ax.text(x_col2, y_text, s, color=col, transform=trans, ha="left", va="center", fontsize=10)

        fig.tight_layout()
        fig.savefig(outpng, dpi=150, bbox_inches="tight", pad_inches=0.05)
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

    # 2D: trace lines and fit a 2D model (power + Chebyshev, like the reference program)
    img2d = fits.getdata(superneon_fits).astype(float)
    y0 = wcfg.get("trace_y0", None)
    # allow manual rejection of bad lamp lines for the 2D fit
    rej = []
    for key in ("rejected_lines_A", "ignore_lines_A"):
        v = wcfg.get(key, None)
        if isinstance(v, (list, tuple)):
            for t in v:
                try:
                    rej.append(float(t))
                except Exception:
                    pass
    rej = sorted(set(rej))

    # Trace all lines; apply manual rejection at the *fit* stage (not here),
    # so the interactive cleaner can still display rejected lines in grey.
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

    xs_cp = xs_cp[keep]
    ys_cp = ys_cp[keep]
    lams_cp = lams_cp[keep]
    scores = scores[keep]

    # edge crop (matches reference workflow: avoid low-SNR borders)
    H, W = img2d.shape
    crop_x = int(wcfg.get("edge_crop_x", 12))
    crop_y = int(wcfg.get("edge_crop_y", 12))
    if crop_x > 0 or crop_y > 0:
        m_edge = (
            (xs_cp >= crop_x) & (xs_cp <= (W - 1 - crop_x)) &
            (ys_cp >= crop_y) & (ys_cp <= (H - 1 - crop_y))
        )
        xs_cp, ys_cp, lams_cp, scores = xs_cp[m_edge], ys_cp[m_edge], lams_cp[m_edge], scores[m_edge]

    # Save control points for QC / interactive cleanup (after all basic filtering).
    control_points_csv = outdir / "control_points_2d.csv"
    if xs_cp.size:
        rows_cp = np.column_stack([xs_cp, ys_cp, lams_cp, scores])
        np.savetxt(control_points_csv, rows_cp, delimiter=",",
                   header="x_pix,y_pix,lambda_A,score", comments="", fmt="%.6f")

    if xs_cp.size < 50:
        raise RuntimeError("Too few 2D control points after filtering; try lowering trace_min_pts / amp_thresh.")

    # --------- fit both 2D models and select the best (or forced by config) ---------
    # Apply manual rejection at the *fit* stage, while still keeping all control points
    # for interactive visualization (rejected curves can be shown in grey).
    fit_mask = np.ones_like(xs_cp, bool)
    if rej:
        for r in rej:
            fit_mask &= (np.abs(lams_cp - float(r)) > 0.25)

    xs_fit, ys_fit, lams_fit, scores_fit = xs_cp[fit_mask], ys_cp[fit_mask], lams_cp[fit_mask], scores[fit_mask]

    if xs_fit.size < 50:
        raise RuntimeError(
            "Too few 2D control points after applying rejected lines; "
            "try removing fewer lines or lowering trace_min_pts / amp_thresh."
        )

    w2 = np.sqrt(np.clip(scores_fit, 0, None))

    # Power (total degree)
    pow_deg = int(wcfg.get("power_deg", max(int(wcfg.get("cheb_degx", 5)), int(wcfg.get("cheb_degy", 3)))))
    pow_coeff, pow_meta, pow_used_fit = robust_polyfit_2d_power(
        xs_fit, ys_fit, lams_fit, pow_deg,
        weights=w2,
        sigma_clip=float(wcfg.get("power_sigma_clip", wcfg.get("cheb_sigma_clip", 3.0))),
        maxiter=int(wcfg.get("power_maxiter", wcfg.get("cheb_maxiter", 10))),
    )
    pow_pred_fit = polyval2d_power(xs_fit, ys_fit, pow_coeff, pow_meta)
    pow_resid_fit = lams_fit - pow_pred_fit
    pow_rms = float(np.sqrt(np.mean(pow_resid_fit[pow_used_fit] ** 2))) if np.any(pow_used_fit) else float("nan")

    # Chebyshev
    degx = int(wcfg.get("cheb_degx", 5))
    degy = int(wcfg.get("cheb_degy", 3))
    cheb_C, cheb_meta, cheb_used_fit = robust_polyfit_2d_cheb(
        xs_fit, ys_fit, lams_fit, degx, degy,
        weights=w2,
        sigma_clip=float(wcfg.get("cheb_sigma_clip", 3.0)),
        maxiter=int(wcfg.get("cheb_maxiter", 10)),
    )
    cheb_pred_fit = polyval2d_cheb(xs_fit, ys_fit, cheb_C, cheb_meta)
    cheb_resid_fit = lams_fit - cheb_pred_fit
    cheb_rms = float(np.sqrt(np.mean(cheb_resid_fit[cheb_used_fit] ** 2))) if np.any(cheb_used_fit) else float("nan")

    # Expand used masks to all control points (rejected lines → used=0)
    pow_used = np.zeros_like(xs_cp, bool)
    cheb_used = np.zeros_like(xs_cp, bool)
    pow_used[fit_mask] = pow_used_fit
    cheb_used[fit_mask] = cheb_used_fit

    # Residuals for all points (for CSV / optional plotting)
    pow_pred = polyval2d_power(xs_cp, ys_cp, pow_coeff, pow_meta)
    pow_resid = lams_cp - pow_pred
    cheb_pred = polyval2d_cheb(xs_cp, ys_cp, cheb_C, cheb_meta)
    cheb_resid = lams_cp - cheb_pred

    model2d = str(wcfg.get("model2d", "auto")).strip().lower()
    if model2d in ("cheb", "chebyshev"):
        kind = "chebyshev"
    elif model2d in ("pow", "power", "poly"):
        kind = "power"
    else:
        kind = "power" if pow_rms <= cheb_rms else "chebyshev"

    if kind == "power":
        used2d = pow_used
        resid2d = pow_resid
        model_payload = {
            "kind": "power",
            "power": {"deg": int(pow_meta.get("deg", pow_deg)), "coeff": [float(c) for c in np.asarray(pow_coeff).tolist()], "meta": pow_meta, "rms_A": float(pow_rms), "n_used": int(np.sum(pow_used))},
            "chebyshev": {"degx": int(degx), "degy": int(degy), "C": cheb_C.tolist(), "meta": cheb_meta, "rms_A": float(cheb_rms), "n_used": int(np.sum(cheb_used))},
        }
        # lambda map with correct (x,y) shape
        YY, XX = np.mgrid[0:H, 0:W]
        lam_map = polyval2d_power(XX, YY, pow_coeff, pow_meta).astype(np.float32)
    else:
        used2d = cheb_used
        resid2d = cheb_resid
        model_payload = {
            "kind": "chebyshev",
            "power": {"deg": int(pow_meta.get("deg", pow_deg)), "coeff": [float(c) for c in np.asarray(pow_coeff).tolist()], "meta": pow_meta, "rms_A": float(pow_rms), "n_used": int(np.sum(pow_used))},
            "chebyshev": {"degx": int(degx), "degy": int(degy), "C": cheb_C.tolist(), "meta": cheb_meta, "rms_A": float(cheb_rms), "n_used": int(np.sum(cheb_used))},
        }
        YY, XX = np.mgrid[0:H, 0:W]
        lam_map = polyval2d_cheb(XX, YY, cheb_C, cheb_meta).astype(np.float32)

    wavesol2d_json = outdir / "wavesolution_2d.json"
    wavesol2d_json.write_text(json.dumps({
        **model_payload,
        "n_points": int(xs_cp.size),
        "n_used": int(np.sum(used2d)),
        "rms_A": float(np.sqrt(np.mean(resid2d[used2d] ** 2))) if np.any(used2d) else float("nan"),
        "rejected_lines_A": rej,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    resid2d_csv = outdir / "residuals_2d.csv"
    rows2 = np.column_stack([xs_cp, ys_cp, lams_cp, resid2d, used2d.astype(int), scores])
    np.savetxt(resid2d_csv, rows2, delimiter=",",
               header="x_pix,y_pix,lambda_A,delta_lambda_A,used,score",
               comments="", fmt="%.6f")

    lambda_map_fits = outdir / "lambda_map.fits"
    fits.writeto(lambda_map_fits, lam_map, overwrite=True)

    lambda_map_png = outdir / "wavelength_matrix.png"
    _plot_wavelength_matrix(lam_map, lambda_map_png, title="2D wavelength map λ(x,y)")

    resid2d_png = outdir / "residuals_2d.png"
    _plot_residuals_2d(ys_cp[used2d], lams_cp[used2d], resid2d[used2d], resid2d_png, title=f"2D fit residuals (kind={kind})")

    return WaveSolutionResult(
        poly1d_png=poly1d_png,
        poly1d_json=poly1d_json,
        poly1d_resid_csv=poly1d_resid_csv,
        cheb2d_json=wavesol2d_json,
        cheb2d_resid_csv=resid2d_csv,
        lambda_map_fits=lambda_map_fits,
        lambda_map_png=lambda_map_png,
        resid2d_png=resid2d_png,
    )
