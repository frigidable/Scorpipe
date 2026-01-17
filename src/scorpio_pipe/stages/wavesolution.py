from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import logging
import json
import numpy as np
from astropy.io import fits
from scorpio_pipe.provenance import add_provenance
from numpy.polynomial.chebyshev import chebvander2d, chebval2d

log = logging.getLogger(__name__)


# --------------------------- IO helpers ---------------------------


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_hand_pairs(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read `hand_pairs.txt` written by LineID GUI.

    Returns
    -------
    x : (N,) float
    lam : (N,) float
    is_blend : (N,) bool
    is_disabled : (N,) bool
    """
    xs: list[float] = []
    lams: list[float] = []
    blends: list[bool] = []
    disabled: list[bool] = []
    if not path.exists():
        return np.array([]), np.array([]), np.array([], bool), np.array([], bool)

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue

        s_low = s.lower()
        blend = "blend" in s_low
        is_dis = ("disabled" in s_low) or ("reject" in s_low) or ("rejected" in s_low)

        # Allow disabled pairs to be kept as commented numeric records, e.g.:
        #   # 123.4  5461.2  # disabled
        if s.startswith("#") and not is_dis:
            continue

        # Strip leading comment marker for numeric parsing.
        s_num = s.lstrip("#").strip()
        # allow inline comments
        s_num = s_num.split("#", 1)[0].strip()
        parts = s_num.split()
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
        disabled.append(bool(is_dis))

    x = np.asarray(xs, float)
    lam = np.asarray(lams, float)
    is_blend = np.asarray(blends, bool)
    is_disabled = np.asarray(disabled, bool)
    return x, lam, is_blend, is_disabled


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
    ww = (
        np.ones_like(xw) if weights is None else np.asarray(weights, float)[mask].copy()
    )
    ww = np.where(np.isfinite(ww) & (ww > 0), ww, 0.0)

    if xw.size < deg + 1:
        raise RuntimeError(
            f"Need at least {deg + 1} points for deg={deg}, got {xw.size}"
        )

    used = np.ones_like(xw, bool)
    for _ in range(maxiter):
        coeffs = np.polyfit(xw[used], yw[used], deg, w=ww[used])
        resid = yw - np.polyval(coeffs, xw)
        s = float(np.std(resid[used]))
        if not np.isfinite(s) or s <= 0:
            break
        new_used = np.abs(resid) <= sigma_clip * s
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
    disabled_mask: np.ndarray | None,
    outpng: Path,
    title: str,
) -> float:
    import matplotlib.pyplot as plt
    from scorpio_pipe.plot_style import mpl_style

    disabled_mask = (
        disabled_mask
        if disabled_mask is not None
        else np.zeros_like(used_mask, dtype=bool)
    )

    x_used = x[used_mask]
    lam_used = lam[used_mask]
    resid = lam_used - np.polyval(coeffs, x_used)
    rms = float(np.sqrt(np.mean(resid**2))) if resid.size else float("nan")

    with mpl_style():
        fig = plt.figure(figsize=(9.5, 6.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        n_dis = int(np.sum(disabled_mask))
        ax1.set_title(
            f"{title}  (deg={len(coeffs) - 1}, N_used={len(x_used)}, N_disabled={n_dis}, RMS={rms:.3f} Å)"
        )
        ax1.scatter(x_used, lam_used, s=26, label="pairs (used)")
        clipped_mask = (~used_mask) & (~disabled_mask)
        if np.any(clipped_mask):
            ax1.scatter(
                x[clipped_mask], lam[clipped_mask], s=22, marker="x", label="clipped"
            )
        if np.any(disabled_mask):
            ax1.scatter(
                x[disabled_mask],
                lam[disabled_mask],
                s=22,
                marker="o",
                alpha=0.5,
                label="disabled",
            )
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
    denom = y0 - 2.0 * y1 + y2
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
        raise RuntimeError(
            f"Not enough control points for degx={degx},degy={degy}: N={x.size}"
        )

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
        new_used = np.abs(resid) <= sigma_clip * s
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
        "xo": float(xo),
        "xscl": float(xscl),
        "yo": float(yo),
        "yscl": float(yscl),
        "x_off": float(xo),
        "x_s": float(xscl),
        "y_off": float(yo),
        "y_s": float(yscl),
        "degx": int(degx),
        "degy": int(degy),
        "kind": "chebyshev",
    }
    return C, meta, used


def polyval2d_cheb(
    x: np.ndarray, y: np.ndarray, C: np.ndarray, meta: dict[str, float]
) -> np.ndarray:
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
        raise RuntimeError(
            f"Not enough control points for power deg={deg}: N={x.size}, npar={npar}"
        )

    xs, (xo, xscl) = _scale_to_unit(x)
    ys, (yo, yscl) = _scale_to_unit(y)

    # Vandermonde
    A = np.vstack([(xs**i) * (ys**j) for (i, j) in terms]).T  # (N, npar)
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
        new_used = np.abs(resid) <= float(sigma_clip) * s
        if new_used.sum() == used.sum():
            break
        used = new_used

    # final solve
    W = ww[used]
    Aw = A[used] * W[:, None]
    bw = lam[used] * W
    coeff, *_ = np.linalg.lstsq(Aw, bw, rcond=None)

    meta: dict[str, float | int | list] = {
        "x_off": float(xo),
        "x_s": float(xscl),
        "y_off": float(yo),
        "y_s": float(yscl),
        "deg": int(deg),
        "terms": [(int(i), int(j)) for (i, j) in terms],
        "kind": "power",
    }
    return coeff, meta, used


def polyval2d_power(
    x: np.ndarray, y: np.ndarray, coeff: np.ndarray, meta: dict[str, float | int | list]
) -> np.ndarray:
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
        out += float(c) * (xs**i) * (ys**j)
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


def _plot_residuals_2d(
    ys_pix_all: np.ndarray,
    lams_all: np.ndarray,
    dlam_all: np.ndarray,
    used_mask: np.ndarray,
    rejected_lines_A: list[float] | None,
    outpng: Path,
    title: str,
    *,
    final_view: bool,
) -> float:
    """2D residuals diagnostic: stacked curves per line (ESO/IDL-like), with a stable convention.

    Convention:
        Δλ = λ_model − λ_lab

    The interactive 2D cleaner uses the same convention; this function is the single
    source of truth for the plot used in both the stage page and QC.
    """
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms
    from matplotlib.colors import LinearSegmentedColormap
    from scorpio_pipe.plot_style import mpl_style

    ys_pix_all = np.asarray(ys_pix_all, float)
    lams_all = np.asarray(lams_all, float)
    dlam_all = np.asarray(dlam_all, float)
    used_mask = np.asarray(used_mask, bool)

    m = np.isfinite(ys_pix_all) & np.isfinite(lams_all) & np.isfinite(dlam_all)
    ys_pix_all = ys_pix_all[m]
    lams_all = lams_all[m]
    dlam_all = dlam_all[m]
    used_mask = used_mask[m]

    rejected_lines_A = rejected_lines_A or []

    # Group by rounded λ to keep grouping stable across float formatting.
    lam_key = np.round(lams_all, 3)
    uniq_keys = np.array(sorted(np.unique(lam_key)), dtype=float)
    uniq_lams = []
    for k in uniq_keys:
        mm = lam_key == k
        if np.any(mm):
            uniq_lams.append(float(np.median(lams_all[mm])))
    uniq = np.array(uniq_lams, dtype=float)

    # Estimate typical Y sampling step for offset spacing.
    dy_steps: list[float] = []
    for lam0 in uniq:
        mm = np.abs(lams_all - lam0) < 1e-2
        y_u = np.unique(ys_pix_all[mm])
        if y_u.size >= 3:
            dy = np.diff(np.sort(y_u))
            dy = dy[dy > 0]
            if dy.size:
                dy_steps.append(float(np.median(dy)))
    y_step = float(np.median(dy_steps)) if dy_steps else 1.0
    y_offset_step = 0.9 * max(1.0, y_step)

    def _is_rejected_line(lam0: float) -> bool:
        return any(abs(float(lam0) - float(r)) <= 0.25 for r in rejected_lines_A)

    # RMS over used points of *active* lines (rejected lines excluded) in the final-view.
    active_point = np.ones_like(dlam_all, dtype=bool)
    if rejected_lines_A:
        for r in rejected_lines_A:
            active_point &= np.abs(lams_all - float(r)) > 0.25
    m_rms = used_mask & active_point
    rms = (
        float(np.sqrt(np.mean(dlam_all[m_rms] ** 2))) if np.any(m_rms) else float("nan")
    )

    cmap = LinearSegmentedColormap.from_list("blue_to_red", ["blue", "red"])
    colors = cmap(np.linspace(0.0, 1.0, max(1, len(uniq))))

    with mpl_style():
        fig, ax = plt.subplots(figsize=(9.0, 7.8))
        ax.set_xlabel("Δλ [Å]")
        ax.set_ylabel("Y [px] (stacked)")
        ax.set_title(f"{title}  (RMS={rms:.3f} Å)")

        text_rows: list[tuple[float, tuple, float, float, float]] = []

        for k, lam0 in enumerate(uniq):
            mm_line = np.abs(lams_all - lam0) < 1e-2
            if not np.any(mm_line):
                continue
            line_rej = _is_rejected_line(float(lam0))
            if final_view and (not line_rej):
                mm_plot = mm_line & used_mask
            else:
                # in audit view: show everything; in final-view: show rejected lines in grey
                mm_plot = mm_line
            if not np.any(mm_plot):
                continue

            yk = ys_pix_all[mm_plot]
            dk = dlam_all[mm_plot]
            if yk.size < 2:
                continue

            ordy = np.argsort(yk)
            y_sorted = yk[ordy]
            d_sorted = dk[ordy]

            off = k * y_offset_step
            y_disp = y_sorted + off
            col = colors[k % len(colors)]
            if line_rej:
                col = (0.5, 0.5, 0.5, 1.0)

            ax.plot(d_sorted, y_disp, lw=1.1, color=col)
            step_pts = max(1, len(y_disp) // 160)
            ax.scatter(
                d_sorted[::step_pts],
                y_disp[::step_pts],
                s=3,
                color=col,
                alpha=0.85,
                linewidths=0.0,
            )

            good = np.isfinite(dk)
            if int(np.count_nonzero(good)) >= 2:
                mu = float(np.mean(dk[good]))
                sd = float(np.std(dk[good], ddof=1))
            else:
                mu, sd = np.nan, np.nan

            y_text = off + (float(np.nanmedian(yk)) if np.any(np.isfinite(yk)) else 0.0)
            text_rows.append((y_text, col, float(lam0), mu, sd))

        # helper lines at label heights
        for y_text, col, *_ in text_rows:
            ax.axhline(
                y=y_text,
                xmin=0.0,
                xmax=1.0,
                color=col,
                linestyle=":",
                linewidth=0.9,
                alpha=0.35,
                zorder=0,
            )

        fig.canvas.draw()
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        y_top = ax.get_ylim()[1]
        x_col1 = 1.02
        x_col2 = 1.02 + 0.22
        ax.text(
            x_col1,
            y_top + 0.1,
            "λ",
            transform=trans,
            ha="left",
            va="bottom",
            fontsize=11,
        )
        ax.text(
            x_col2,
            y_top + 0.1,
            "Δλ [Å]",
            transform=trans,
            ha="left",
            va="bottom",
            fontsize=11,
        )

        for y_text, col, lam0, mu, sd in text_rows:
            ax.text(
                x_col1,
                y_text,
                f"{lam0:7.2f} Å",
                color=col,
                transform=trans,
                ha="left",
                va="center",
                fontsize=10,
            )
            s = f"{mu:+.2f} ± {sd:.2f}" if np.isfinite(mu) else "—"
            s = s.replace("-", "−")
            ax.text(
                x_col2,
                y_text,
                s,
                color=col,
                transform=trans,
                ha="left",
                va="center",
                fontsize=10,
            )

        fig.tight_layout()
        fig.savefig(outpng, dpi=150, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
    return rms


def _mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad


def _weighted_rms(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.sum(w[m] * x[m] ** 2) / np.sum(w[m])))


def _plot_residuals_vs_lambda(
    lam_key: np.ndarray,
    lam_all: np.ndarray,
    dlam_all: np.ndarray,
    used_mask: np.ndarray,
    rejected_lines_A: Sequence[float],
    outpng: Path,
    title: str,
) -> None:
    """QC plot: median Δλ per identified line vs λ.

    Notes
    -----
    - "used" points are derived from samples that survived the 2D fit/rejection mask.
    - "rejected" points are lines explicitly rejected by the user (in Å), shown for context.
    """
    # Local imports keep matplotlib optional for headless/CLI runs until QC is requested.
    from scorpio_pipe.plot_style import mpl_style
    import matplotlib.pyplot as plt

    lam_key = np.asarray(lam_key, dtype=float)
    lam_all = np.asarray(lam_all, dtype=float)
    dlam_all = np.asarray(dlam_all, dtype=float)
    used_mask = np.asarray(used_mask, dtype=bool)

    m_finite = np.isfinite(lam_all) & np.isfinite(dlam_all) & np.isfinite(lam_key)
    if not np.any(m_finite):
        return

    xs_used: list[float] = []
    ys_used: list[float] = []
    xs_rej: list[float] = []
    ys_rej: list[float] = []

    uniq = np.unique(lam_key[m_finite])
    rej = [float(r) for r in rejected_lines_A] if rejected_lines_A is not None else []

    # classify and aggregate per line
    for lk in uniq:
        m_line = m_finite & (lam_key == lk)
        if not np.any(m_line):
            continue

        lam0 = float(np.median(lam_all[m_line])) if np.any(m_line) else float(lk)
        line_rej = any(abs(lam0 - float(r)) <= 0.25 for r in rej)

        m_used = m_line & used_mask
        if np.any(m_used):
            xs_used.append(lam0)
            ys_used.append(float(np.median(dlam_all[m_used])))
        else:
            # Keep rejected / unused lines visible for context (median over all finite samples).
            xs_rej.append(lam0)
            ys_rej.append(float(np.median(dlam_all[m_line])))

        if line_rej and xs_rej and xs_rej[-1] != lam0:
            # If a rejected line had used points (rare), also mark it as rejected for visibility.
            xs_rej.append(lam0)
            ys_rej.append(float(np.median(dlam_all[m_line])))

    with mpl_style():
        fig, ax = plt.subplots(figsize=(8.5, 3.2))
        if xs_used:
            ax.scatter(xs_used, ys_used, label="used")
            # Trend on used points only
            if len(xs_used) >= 2:
                x = np.asarray(xs_used)
                y = np.asarray(ys_used)
                p = np.polyfit(x, y, deg=1)
                xx = np.linspace(x.min(), x.max(), 200)
                ax.plot(xx, p[0] * xx + p[1], label="linear trend")

        if xs_rej:
            ax.scatter(xs_rej, ys_rej, marker="x", label="rejected/unused")

        ax.axhline(0.0, lw=1.0)
        ax.set_xlabel("λ [Å]")
        ax.set_ylabel("median Δλ [Å]")
        ax.set_title(title)
        if xs_rej or (xs_used and len(xs_used) >= 2):
            ax.legend(loc="best", frameon=False)

        fig.tight_layout()
        fig.savefig(outpng, dpi=150)
        plt.close(fig)


def _plot_residuals_vs_y(
    y_all: np.ndarray,
    dlam_all: np.ndarray,
    used_mask: np.ndarray,
    outpng: Path,
    title: str,
) -> None:
    """QC plot: median Δλ vs slit coordinate Y."""
    from scorpio_pipe.plot_style import mpl_style
    import matplotlib.pyplot as plt

    y_all = np.asarray(y_all, dtype=float)
    dlam_all = np.asarray(dlam_all, dtype=float)
    used_mask = np.asarray(used_mask, dtype=bool)

    m = np.isfinite(y_all) & np.isfinite(dlam_all) & used_mask
    if not np.any(m):
        return

    yc = np.unique(y_all[m])
    med: list[float] = []
    for yy in yc:
        mi = m & (y_all == yy)
        if np.any(mi):
            med.append(float(np.median(dlam_all[mi])))

    with mpl_style():
        fig, ax = plt.subplots(figsize=(8.5, 3.2))
        if len(med) == len(yc) and len(yc) > 0:
            ax.plot(yc, med)
        ax.axhline(0.0, lw=1.0)
        ax.set_xlabel("Y [px]")
        ax.set_ylabel("median Δλ [Å]")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(outpng, dpi=150)
        plt.close(fig)


def _plot_residual_hist(
    dlam_all: np.ndarray,
    used_mask: np.ndarray,
    outpng: Path,
    title: str,
) -> None:
    """QC plot: histogram of Δλ normalized by σ_MAD."""
    from scorpio_pipe.plot_style import mpl_style
    import matplotlib.pyplot as plt

    d = np.asarray(dlam_all, dtype=float)
    used_mask = np.asarray(used_mask, dtype=bool)
    m = np.isfinite(d) & used_mask
    if not np.any(m):
        return

    sigma = _mad_sigma(d[m])
    z = d[m] / sigma if np.isfinite(sigma) and sigma > 0 else d[m]

    with mpl_style():
        fig, ax = plt.subplots(figsize=(6.8, 3.2))
        ax.hist(z, bins=40)
        ax.axvline(0.0, lw=1.0)
        ax.set_xlabel("Δλ / σ_MAD")
        ax.set_ylabel("N")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(outpng, dpi=150)
        plt.close(fig)


# --------------------------- public stage ---------------------------


@dataclass
class WaveSolutionResult:
    """Outputs + QC metrics for the *entire* wavelength-solution stage.

    This is what the GUI/QC uses as a single source of truth.
    All paths are relative to the work_dir/wavesol/<disperser>/ directory.
    """

    # Paths / artifacts
    wavesol_dir: Path
    lambda_map_fits: Path
    wavesolution_1d_png: Path
    wavesolution_1d_json: Path
    wavesolution_2d_json: Path
    control_points_csv: Path
    residuals_1d_csv: Path
    residuals_2d_csv: Path
    residuals_2d_png: Path
    residuals_2d_audit_png: Path
    residuals_vs_lambda_png: Path
    residuals_vs_y_png: Path
    residuals_hist_png: Path
    report_txt: Path

    # QC metrics (duplicated in JSON/report)
    rms1d_A: float
    rms1d_px: float
    wrms1d_A: float
    sigma1d_mad_A: float
    dispersion_A_per_px: float
    n_pairs_total: int
    n_pairs_used: int
    n_pairs_disabled: int

    model2d_kind: str
    rms2d_A: float
    wrms2d_A: float
    sigma2d_mad_A: float
    n_lines_total: int
    n_lines_used: int
    n_lines_rejected: int

    def as_dict(self) -> dict[str, object]:
        """JSON-safe representation used by the GUI runner."""
        return {
            "wavesol_dir": str(self.wavesol_dir),
            "lambda_map_fits": str(self.lambda_map_fits),
            "wavesolution_1d_png": str(self.wavesolution_1d_png),
            "wavesolution_1d_json": str(self.wavesolution_1d_json),
            "wavesolution_2d_json": str(self.wavesolution_2d_json),
            "control_points_csv": str(self.control_points_csv),
            "residuals_1d_csv": str(self.residuals_1d_csv),
            "residuals_2d_csv": str(self.residuals_2d_csv),
            "residuals_2d_png": str(self.residuals_2d_png),
            "residuals_2d_audit_png": str(self.residuals_2d_audit_png),
            "residuals_vs_lambda_png": str(self.residuals_vs_lambda_png),
            "residuals_vs_y_png": str(self.residuals_vs_y_png),
            "residuals_hist_png": str(self.residuals_hist_png),
            "report_txt": str(self.report_txt),
            "rms1d_A": float(self.rms1d_A),
            "rms1d_px": float(self.rms1d_px),
            "wrms1d_A": float(self.wrms1d_A),
            "sigma1d_mad_A": float(self.sigma1d_mad_A),
            "dispersion_A_per_px": float(self.dispersion_A_per_px),
            "n_pairs_total": int(self.n_pairs_total),
            "n_pairs_used": int(self.n_pairs_used),
            "n_pairs_disabled": int(self.n_pairs_disabled),
            "model2d_kind": str(self.model2d_kind),
            "rms2d_A": float(self.rms2d_A),
            "wrms2d_A": float(self.wrms2d_A),
            "sigma2d_mad_A": float(self.sigma2d_mad_A),
            "n_lines_total": int(self.n_lines_total),
            "n_lines_used": int(self.n_lines_used),
            "n_lines_rejected": int(self.n_lines_rejected),
        }


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

    # --- P0-H: ARC compatibility contract (must-match vs QC-only) ---
    # --- P0-K: re-apply global exclude as safety net (stale dataset_manifest) ---
    # --- P0-M: resolve lamp_type + line-list provenance ---
    # --- P0-N: validate per setup (do not assume the "first object") ---

    pre_flags: list[dict[str, Any]] = []
    compat_flags: list[dict[str, Any]] = []
    compat_meta: dict[str, Any] = {}
    lamp_meta: dict[str, Any] = {}

    from scorpio_pipe.calib_compat import CalibrationMismatchError

    # Input lists (may be empty in special workflows)
    obj_list = [Path(x) for x in (cfg.get("frames", {}) or {}).get("obj", [])]  # type: ignore[union-attr]
    arc_list = [Path(x) for x in (cfg.get("frames", {}) or {}).get("neon", [])]  # type: ignore[union-attr]

    # Safety-net exclude application (absolute policy).
    try:
        from scorpio_pipe.exclude_policy import filter_paths_by_exclude, resolve_exclude_set
        from scorpio_pipe.qc.flags import make_flag

        ex = resolve_exclude_set(cfg, data_dir=cfg.get("data_dir"))
        if ex.excluded_abs:
            obj_list, obj_dropped = filter_paths_by_exclude(obj_list, ex.excluded_abs)
            arc_list, arc_dropped = filter_paths_by_exclude(arc_list, ex.excluded_abs)
            if obj_dropped or arc_dropped:
                pre_flags.append(
                    make_flag(
                        "MANIFEST_EXCLUDE_APPLIED",
                        "WARN",
                        "Applied global exclude while building wavelength solution inputs.",
                        dropped_obj=[str(p) for p in obj_dropped],
                        dropped_arc=[str(p) for p in arc_dropped],
                    )
                )
    except Exception:
        # Exclude should never break the stage.
        pass

    # Resolve lamp type + line list (always recorded).
    try:
        from scorpio_pipe.lamp_contract import (
            LAMP_UNKNOWN,
            resolve_lamp_type,
            resolve_linelist_csv_path,
        )
        from scorpio_pipe.qc.flags import make_flag

        arc_hint = str(arc_list[0]) if arc_list else ""
        setup = (cfg.get("frames", {}) or {}).get("__setup__", {}) or {}
        instr_hint = str(setup.get("instrument") or "")

        lamp_res = resolve_lamp_type(cfg, arc_path=arc_hint, instrument_hint=instr_hint)
        linelist_path = resolve_linelist_csv_path(cfg, lamp_res.lamp_type)

        # Strict mode: require explicit override when lamp is unknown.
        strict_lamp = bool((wcfg.get("strict_lamp") if isinstance(wcfg, dict) else False) or cfg.get("strict"))
        if lamp_res.lamp_type == LAMP_UNKNOWN:
            pre_flags.append(
                make_flag(
                    "LAMP_UNKNOWN",
                    "WARN",
                    "Could not determine arc lamp type; line list choice may be wrong.",
                    "Set wavesol.lamp_type in config to override.",
                )
            )
            if strict_lamp:
                raise RuntimeError(
                    "Unknown lamp type for wavelength calibration. Set wavesol.lamp_type in config.yaml "
                    "(e.g. 'HeNeAr' or 'Ne') or disable strict_lamp."
                )

        if lamp_res.source == "default":
            pre_flags.append(
                make_flag(
                    "LAMP_DEFAULT_USED",
                    "INFO",
                    f"Using default lamp_type={lamp_res.lamp_type} for this instrument/setup.",
                )
            )

        lamp_meta = {
            "lamp_type": lamp_res.lamp_type,
            "lamp_source": lamp_res.source,
            "lamp_raw": lamp_res.lamp_raw,
            "linelist_csv": str(linelist_path),
            "linelist_reason": (
                "config" if str((wcfg or {}).get("linelist_csv") or "").strip() else "lamp_type_default"
            ),
        }
    except Exception as e:
        # Lamp metadata must never stop the fit; keep a warning.
        try:
            from scorpio_pipe.qc.flags import make_flag

            pre_flags.append(
                make_flag(
                    "LAMP_CONTRACT_FAILED",
                    "WARN",
                    f"Lamp/linelist resolution failed: {e}",
                )
            )
        except Exception:
            pass

    # Validate setup consistency across science frames (P0-N).
    try:
        if obj_list and arc_list:
            from scorpio_pipe.instruments import parse_frame_meta

            setup_groups: dict[tuple[Any, ...], list[str]] = {}
            for sp in obj_list:
                try:
                    with fits.open(sp, memmap=False) as hdul:  # type: ignore[attr-defined]
                        hdr = dict(hdul[0].header)
                        try:
                            if "SCI" in hdul:
                                hdr = dict(hdul["SCI"].header)
                        except Exception:
                            pass
                    meta = parse_frame_meta(hdr, strict=False)
                    key = (
                        meta.instrument,
                        meta.mode,
                        meta.disperser,
                        round(float(meta.slit_width_arcsec), 3),
                        int(meta.binning_x),
                        int(meta.binning_y),
                        int(meta.naxis1),
                        int(meta.naxis2),
                    )
                    setup_groups.setdefault(key, []).append(str(sp))
                except Exception:
                    # If we cannot parse one header, do not block the stage here.
                    continue

            if len(setup_groups) > 1:
                # Policy B: fail fast with a clear explanation.
                parts = []
                for k, paths in setup_groups.items():
                    parts.append({"setup_key": list(k), "n": len(paths), "examples": paths[:3]})
                raise RuntimeError(
                    "Multiple distinct science setups detected in this wavesolution run. "
                    "Build wavelength solutions per-setup (separate work dirs/configs), or run in a mode that "
                    "produces per-setup solutions. Details: "
                    + json.dumps(parts, ensure_ascii=False)
                )

            # ARC compatibility: check against the (single) setup representative.
            from scorpio_pipe.calib_compat import ensure_compatible_calib
            from scorpio_pipe.qc.flags import make_flag

            sci_path = Path(next(iter(next(iter(setup_groups.values()))))) if setup_groups else obj_list[0]
            with fits.open(sci_path, memmap=False) as hdul:  # type: ignore[attr-defined]
                sci_hdr = dict(hdul[0].header)
                try:
                    if "SCI" in hdul:
                        sci_hdr = dict(hdul["SCI"].header)
                except Exception:
                    pass

            arc_checks = []
            for ap in arc_list:
                arc_checks.append(
                    ensure_compatible_calib(
                        sci_hdr,
                        ap,
                        kind="arc",
                        strict=True,
                        allow_readout_diff=True,
                        stage_flags=compat_flags,
                    )
                )
            compat_meta = {
                "ref_science": str(sci_path),
                "n_science": int(len(obj_list)),
                "n_arcs": int(len(arc_list)),
                "arc_checks": arc_checks,
            }
    except CalibrationMismatchError:
        # must-match mismatch is a hard ERROR
        raise
    except Exception as e:
        # Do not fail the whole stage just because we could not read headers.
        try:
            from scorpio_pipe.qc.flags import make_flag

            compat_flags.append(
                make_flag(
                    "ARC_COMPAT_CHECK_FAILED",
                    "WARN",
                    f"Failed to verify ARC compatibility: {e}",
                )
            )
        except Exception:
            pass
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

    x_all, lam_all, is_blend_all, is_disabled_all = _read_hand_pairs(hand_pairs)
    active_pairs = (~is_disabled_all).astype(bool)
    if int(np.sum(active_pairs)) < 5:
        raise RuntimeError(
            f"Not enough ACTIVE pairs in {hand_pairs} (need >=5, got {int(np.sum(active_pairs))}; total {x_all.size})"
        )

    # 1D poly fit
    deg1d = int(wcfg.get("poly_deg_1d", 4))
    w_blend_all = np.where(is_blend_all, float(wcfg.get("blend_weight", 0.3)), 1.0)
    x_fit = x_all[active_pairs]
    lam_fit = lam_all[active_pairs]
    w_fit = w_blend_all[active_pairs]

    coeffs1d, used_fit = robust_polyfit_1d(
        x_fit,
        lam_fit,
        deg1d,
        weights=w_fit,
        sigma_clip=float(wcfg.get("poly_sigma_clip", 3.0)),
        maxiter=int(wcfg.get("poly_maxiter", 10)),
    )

    used_mask = np.zeros_like(x_all, dtype=bool)
    used_mask[active_pairs] = used_fit

    poly1d_png = outdir / "wavesolution_1d.png"
    rms1d = _plot_wavesol_1d(
        x_all,
        lam_all,
        coeffs1d,
        used_mask,
        is_disabled_all,
        poly1d_png,
        title="1D dispersion solution",
    )

    wavesol1d_json = outdir / "wavesolution_1d.json"
    resid_all = lam_all - np.polyval(coeffs1d, x_all)
    resid_used = resid_all[used_mask]
    # dispersion estimate (Å/px) from derivative of the fitted polynomial
    if np.any(used_mask):
        dlamdx = np.polyval(np.polyder(coeffs1d), x_all[used_mask])
        dlamdx_med = float(np.median(np.abs(dlamdx)))
    else:
        dlamdx_med = float("nan")
    rms1d_px = (
        float(rms1d / dlamdx_med)
        if np.isfinite(dlamdx_med) and dlamdx_med > 0
        else float("nan")
    )
    # robust sigma estimate (MAD → σ)
    if resid_used.size:
        med = float(np.median(resid_used))
        mad = float(np.median(np.abs(resid_used - med)))
        sig_mad = 1.4826 * mad
    else:
        sig_mad = float("nan")
    # weighted RMS (blend-weight aware)
    w_used = w_blend_all[used_mask]
    wrms1d = (
        float(np.sqrt(np.sum(w_used * (resid_used**2)) / np.sum(w_used)))
        if resid_used.size and np.sum(w_used) > 0
        else float("nan")
    )

    wavesol1d_json.write_text(
        json.dumps(
            {
                "deg": int(deg1d),
                "coeffs_polyval": [float(c) for c in coeffs1d],  # highest order first
                "rms_A": float(rms1d),
                "rms_px": float(rms1d_px),
                "wrms_A": float(wrms1d),
                "sigma_mad_A": float(sig_mad),
                "dispersion_A_per_px": float(dlamdx_med),
                "n_pairs": int(x_all.size),
                "n_used": int(np.sum(used_mask)),
                "n_disabled": int(np.sum(is_disabled_all)),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    poly1d_resid_csv = outdir / "residuals_1d.csv"
    rows = np.column_stack(
        [
            x_all,
            lam_all,
            resid_all,
            used_mask.astype(int),
            is_blend_all.astype(int),
            is_disabled_all.astype(int),
        ]
    )
    np.savetxt(
        poly1d_resid_csv,
        rows,
        delimiter=",",
        header="x_pix,lambda_A,delta_lambda_A,used,blend,disabled",
        comments="",
        fmt="%.6f",
    )

    # 2D: trace lines and fit a 2D model (power + Chebyshev, like the reference program)
    img2d = fits.getdata(superneon_fits, memmap=False).astype(float)
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

    # Control points are the *contract* between the interactive 2D cleaner and
    # the final stage run.  To ensure the "Final Run" produces results that
    # match what the user saw in the 2D cleaning view, we optionally reuse the
    # previously saved control points.
    control_points_csv = outdir / "control_points_2d.csv"

    def _load_control_points_csv(
        p: Path,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        try:
            arr = np.genfromtxt(p, delimiter=",", names=True, dtype=float)
            x = np.asarray(arr["x_pix"], float)
            y = np.asarray(arr["y_pix"], float)
            lam = np.asarray(arr["lambda_A"], float)
            score = np.asarray(arr["score"], float)
        except Exception:
            try:
                a = np.loadtxt(p, delimiter=",", dtype=float)
                if a.ndim == 1:
                    a = a[None, :]
                x, y, lam, score = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
            except Exception:
                return None
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(lam) & np.isfinite(score)
        x, y, lam, score = x[m], y[m], lam[m], score[m]
        if x.size < 50:
            return None
        return x, y, lam, score

    use_cached_cp = bool(wcfg.get("use_cached_control_points", True))
    cached = (
        _load_control_points_csv(control_points_csv)
        if (use_cached_cp and control_points_csv.exists())
        else None
    )
    if cached is not None:
        xs_cp, ys_cp, lams_cp, scores = cached
    else:
        # Trace all lines; apply manual rejection at the *fit* stage (not here),
        # so the interactive cleaner can still display rejected lines in grey.
        xs_cp, ys_cp, lams_cp, scores = trace_lines_2d_cc(
            img2d,
            lambda_list=lam_fit,
            x_guesses=x_fit,
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
        m = np.abs(lams_cp - lam0) < 1e-6
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
            (xs_cp >= crop_x)
            & (xs_cp <= (W - 1 - crop_x))
            & (ys_cp >= crop_y)
            & (ys_cp <= (H - 1 - crop_y))
        )
        xs_cp, ys_cp, lams_cp, scores = (
            xs_cp[m_edge],
            ys_cp[m_edge],
            lams_cp[m_edge],
            scores[m_edge],
        )

    # Save control points for QC / interactive cleanup (after all basic filtering).
    # This file is the *single source of truth* for the interactive 2D cleaner.
    # Always write it so that the GUI and the final stage outputs stay in lockstep.
    if xs_cp.size:
        rows_cp = np.column_stack([xs_cp, ys_cp, lams_cp, scores])
        np.savetxt(
            control_points_csv,
            rows_cp,
            delimiter=",",
            header="x_pix,y_pix,lambda_A,score",
            comments="",
            fmt="%.6f",
        )

    if xs_cp.size < 50:
        raise RuntimeError(
            "Too few 2D control points after filtering; try lowering trace_min_pts / amp_thresh."
        )

    # --------- fit both 2D models and select the best (or forced by config) ---------
    # Apply manual rejection at the *fit* stage, while still keeping all control points
    # for interactive visualization (rejected curves can be shown in grey).
    fit_mask = np.ones_like(xs_cp, bool)
    if rej:
        for r in rej:
            fit_mask &= np.abs(lams_cp - float(r)) > 0.25

    xs_fit, ys_fit, lams_fit, scores_fit = (
        xs_cp[fit_mask],
        ys_cp[fit_mask],
        lams_cp[fit_mask],
        scores[fit_mask],
    )

    if xs_fit.size < 50:
        raise RuntimeError(
            "Too few 2D control points after applying rejected lines; "
            "try removing fewer lines or lowering trace_min_pts / amp_thresh."
        )

    w2 = np.sqrt(np.clip(scores_fit, 0, None))

    # Power (total degree)
    pow_deg = int(
        wcfg.get(
            "power_deg",
            max(int(wcfg.get("cheb_degx", 5)), int(wcfg.get("cheb_degy", 3))),
        )
    )
    pow_coeff, pow_meta, pow_used_fit = robust_polyfit_2d_power(
        xs_fit,
        ys_fit,
        lams_fit,
        pow_deg,
        weights=w2,
        sigma_clip=float(
            wcfg.get("power_sigma_clip", wcfg.get("cheb_sigma_clip", 3.0))
        ),
        maxiter=int(wcfg.get("power_maxiter", wcfg.get("cheb_maxiter", 10))),
    )
    pow_pred_fit = polyval2d_power(xs_fit, ys_fit, pow_coeff, pow_meta)
    pow_resid_fit = pow_pred_fit - lams_fit
    pow_rms = (
        float(np.sqrt(np.mean(pow_resid_fit[pow_used_fit] ** 2)))
        if np.any(pow_used_fit)
        else float("nan")
    )

    # Chebyshev
    degx = int(wcfg.get("cheb_degx", 5))
    degy = int(wcfg.get("cheb_degy", 3))
    cheb_C, cheb_meta, cheb_used_fit = robust_polyfit_2d_cheb(
        xs_fit,
        ys_fit,
        lams_fit,
        degx,
        degy,
        weights=w2,
        sigma_clip=float(wcfg.get("cheb_sigma_clip", 3.0)),
        maxiter=int(wcfg.get("cheb_maxiter", 10)),
    )
    cheb_pred_fit = polyval2d_cheb(xs_fit, ys_fit, cheb_C, cheb_meta)
    cheb_resid_fit = cheb_pred_fit - lams_fit
    cheb_rms = (
        float(np.sqrt(np.mean(cheb_resid_fit[cheb_used_fit] ** 2)))
        if np.any(cheb_used_fit)
        else float("nan")
    )

    # Expand used masks to all control points (rejected lines → used=0)
    pow_used = np.zeros_like(xs_cp, bool)
    cheb_used = np.zeros_like(xs_cp, bool)
    pow_used[fit_mask] = pow_used_fit
    cheb_used[fit_mask] = cheb_used_fit

    # Residuals for all points (for CSV / optional plotting)
    pow_pred = polyval2d_power(xs_cp, ys_cp, pow_coeff, pow_meta)
    pow_resid = pow_pred - lams_cp
    cheb_pred = polyval2d_cheb(xs_cp, ys_cp, cheb_C, cheb_meta)
    cheb_resid = cheb_pred - lams_cp

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
            "power": {
                "deg": int(pow_meta.get("deg", pow_deg)),
                "coeff": [float(c) for c in np.asarray(pow_coeff).tolist()],
                "meta": pow_meta,
                "rms_A": float(pow_rms),
                "n_used": int(np.sum(pow_used)),
            },
            "chebyshev": {
                "degx": int(degx),
                "degy": int(degy),
                "C": cheb_C.tolist(),
                "meta": cheb_meta,
                "rms_A": float(cheb_rms),
                "n_used": int(np.sum(cheb_used)),
            },
        }
        # lambda map with correct (x,y) shape
        YY, XX = np.mgrid[0:H, 0:W]
        lam_map = polyval2d_power(XX, YY, pow_coeff, pow_meta).astype(np.float32)
    else:
        used2d = cheb_used
        resid2d = cheb_resid
        model_payload = {
            "kind": "chebyshev",
            "power": {
                "deg": int(pow_meta.get("deg", pow_deg)),
                "coeff": [float(c) for c in np.asarray(pow_coeff).tolist()],
                "meta": pow_meta,
                "rms_A": float(pow_rms),
                "n_used": int(np.sum(pow_used)),
            },
            "chebyshev": {
                "degx": int(degx),
                "degy": int(degy),
                "C": cheb_C.tolist(),
                "meta": cheb_meta,
                "rms_A": float(cheb_rms),
                "n_used": int(np.sum(cheb_used)),
            },
        }
        YY, XX = np.mgrid[0:H, 0:W]
        lam_map = polyval2d_cheb(XX, YY, cheb_C, cheb_meta).astype(np.float32)

    wavesol2d_json = outdir / "wavesolution_2d.json"
    wavesol2d_json.write_text(
        json.dumps(
            {
                **model_payload,
                "n_points": int(xs_cp.size),
                "n_used": int(np.sum(used2d)),
                "rms_A": float(np.sqrt(np.mean(resid2d[used2d] ** 2)))
                if np.any(used2d)
                else float("nan"),
                "rejected_lines_A": rej,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    resid2d_csv = outdir / "residuals_2d.csv"
    rows2 = np.column_stack(
        [xs_cp, ys_cp, lams_cp, resid2d, used2d.astype(int), scores]
    )
    np.savetxt(
        resid2d_csv,
        rows2,
        delimiter=",",
        header="x_pix,y_pix,lambda_A,delta_lambda_A,used,score",
        comments="",
        fmt="%.6f",
    )

    lambda_map_fits = outdir / "lambda_map.fits"
    lambda_map_png = outdir / "wavelength_matrix.png"
    _plot_wavelength_matrix(lam_map, lambda_map_png, title="2D wavelength map λ(x,y)")

    # --- 2D QC (ESO-like) ---
    mask_active_line = np.ones_like(lams_cp, dtype=bool)
    for r in rej:
        mask_active_line &= np.abs(lams_cp - r) > 0.25
    used_active = used2d & mask_active_line & np.isfinite(resid2d)
    resid_used = resid2d[used_active]

    rms2d_A = (
        float(np.sqrt(np.mean(resid_used**2))) if resid_used.size else float("nan")
    )
    wrms2d_A = _weighted_rms(resid2d[used_active], scores[used_active])
    sig_mad_2d_A = _mad_sigma(resid_used)
    resid2d_p95_A = (
        float(np.percentile(np.abs(resid_used), 95)) if resid_used.size else float("nan")
    )

    disp2d_A_per_px = float("nan")
    if lam_map.shape[1] > 1:
        gx = np.diff(lam_map.astype(float), axis=1)
        gx = gx[np.isfinite(gx)]
        if gx.size:
            disp2d_A_per_px = float(np.median(np.abs(gx)))

    rms2d_px = (
        float(rms2d_A / disp2d_A_per_px)
        if np.isfinite(disp2d_A_per_px) and disp2d_A_per_px > 0
        else float("nan")
    )
    sig_mad_2d_px = (
        float(sig_mad_2d_A / disp2d_A_per_px)
        if np.isfinite(disp2d_A_per_px) and disp2d_A_per_px > 0
        else float("nan")
    )

    resid2d_png = outdir / "residuals_2d.png"
    resid2d_audit_png = outdir / "residuals_2d_audit.png"
    resid_vs_lambda_png = outdir / "residuals_vs_lambda.png"
    resid_vs_y_png = outdir / "residuals_vs_y.png"
    resid_hist_png = outdir / "residuals_hist.png"
    report_txt = outdir / "wavesolution_report.txt"

    _plot_residuals_2d(
        ys_pix_all=ys_cp,
        lams_all=lams_cp,
        dlam_all=resid2d,
        used_mask=used2d,
        rejected_lines_A=rej,
        outpng=resid2d_png,
        title=f"2D residuals (kind={kind})",
        final_view=True,
    )
    _plot_residuals_2d(
        ys_pix_all=ys_cp,
        lams_all=lams_cp,
        dlam_all=resid2d,
        used_mask=used2d,
        rejected_lines_A=rej,
        outpng=resid2d_audit_png,
        title=f"2D residuals — audit (kind={kind})",
        final_view=False,
    )
    _plot_residuals_vs_lambda(
        lams_all=lams_cp,
        dlam_all=resid2d,
        used_mask=used2d,
        rejected_lines_A=rej,
        outpng=resid_vs_lambda_png,
        title="Δλ vs λ (per-line medians)",
    )
    _plot_residuals_vs_y(
        ys_pix_all=ys_cp,
        dlam_all=resid2d,
        used_mask=used2d,
        outpng=resid_vs_y_png,
        title="Δλ vs Y (diagnostic)",
    )
    _plot_residual_hist(
        dlam_all=resid2d,
        used_mask=used2d,
        outpng=resid_hist_png,
        title="Normalised residuals histogram",
    )

    # Update JSON report with metrics, and write a human-readable report.
    try:
        d2 = json.loads(wavesol2d_json.read_text(encoding="utf-8"))
    except Exception:
        d2 = {}
    # per-line counts
    lam_keys = np.round(lams_cp.astype(float), 3)
    uniq = np.unique(lam_keys)
    n_lines = int(uniq.size)
    n_rejected_lines = 0
    for lk in uniq:
        if any(abs(float(lk) - float(r)) <= 0.25 for r in rej):
            n_rejected_lines += 1
    d2.update(
        {
            "qc": {
                "rms_A": rms2d_A,
                "wrms_A": wrms2d_A,
                "robust_sigma_A": sig_mad_2d_A,
                "rms_px": rms2d_px,
                "robust_sigma_px": sig_mad_2d_px,
                "dispersion_A_per_px": disp2d_A_per_px,
                "n_lines": n_lines,
                "n_rejected_lines": int(n_rejected_lines),
                "n_active_lines": int(n_lines - n_rejected_lines),
                "n_points": int(lams_cp.size),
                "n_used_points": int(np.sum(used_active)),
            }
        }
    )
    wavesol2d_json.write_text(
        json.dumps(d2, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Text report (for papers / traceability)
    lines: list[str] = []
    lines.append("Wavelength solution report")
    lines.append(f"kind: {kind}")
    lines.append("")
    lines.append("2D QC metrics")
    lines.append(f"  RMS: {rms2d_A:.6g} Å  ({rms2d_px:.6g} px)")
    lines.append(f"  wRMS: {wrms2d_A:.6g} Å")
    lines.append(f"  robust σ(MAD): {sig_mad_2d_A:.6g} Å  ({sig_mad_2d_px:.6g} px)")
    lines.append(f"  dispersion (median |dλ/dx|): {disp2d_A_per_px:.6g} Å/px")
    lines.append(
        f"  lines: total={n_lines} active={n_lines - n_rejected_lines} rejected={n_rejected_lines}"
    )
    lines.append(f"  points: total={int(lams_cp.size)} used={int(np.sum(used_active))}")
    if rej:
        lines.append("  rejected_lines_A:")
        for r in sorted(rej):
            lines.append(f"    - {r}")
    lines.append("")
    # per-line table
    lines.append("Per-line stats (λ, status, N_used/N_total, median Δλ, RMS Δλ)")
    for lk in sorted(uniq):
        lam0 = float(lk)
        m_line = np.abs(lam_keys - lk) < 0.001
        is_rej = any(abs(lam0 - float(r)) <= 0.25 for r in rej)
        m_used = used2d & m_line & mask_active_line
        rr = resid2d[m_used]
        med = float(np.median(rr)) if rr.size else float("nan")
        rms_line = float(np.sqrt(np.mean(rr**2))) if rr.size else float("nan")
        lines.append(
            f"  {lam0:.3f}  {'REJECT' if is_rej else 'OK':6s}  {int(np.sum(m_used))}/{int(np.sum(m_line))}  {med:+.3g}  {rms_line:.3g}"
        )
    report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # finalize counts / robust stats for 2D
    n_lines_total = int(n_lines)
    n_lines_rejected = int(n_rejected_lines)
    n_lines_used = int(n_lines_total - n_lines_rejected)


    # Write lambda_map FITS with explicit wavelength metadata (no heuristics downstream).
    # This file is a *lookup image* λ(x,y) and not a rectified spectrum, but we still
    # store wavelength unit/reference and QC cards for strict downstream behavior.
    wave_unit_raw = str(wcfg.get("wave_unit", wcfg.get("waveunit", "Angstrom")) or "Angstrom").strip()
    s = wave_unit_raw.lower().replace("å", "angstrom")
    if s in {"a", "aa", "ang", "angs", "angstrom", "ångström", "angstroms"}:
        wave_unit = "Angstrom"
    elif s in {"nm", "nanometer", "nanometers"}:
        wave_unit = "nm"
    else:
        wave_unit = wave_unit_raw

    waveref = str(wcfg.get("wave_ref", wcfg.get("waveref", "air")) or "air").strip().lower()
    if waveref not in {"air", "vacuum"}:
        # keep but normalise to a safe default
        waveref = "air"

    # Convert internally-Angstrom lambda_map to requested output unit if needed
    unit_scale = 1.0
    if wave_unit == "nm":
        unit_scale = 0.1  # 1 Å = 0.1 nm
    lam_map_out = (lam_map * unit_scale).astype("float32")
    rms2d_out = float(rms2d_A) * unit_scale
    

    lam_hdr = fits.Header()
    lam_hdr["CTYPE1"] = ("WAVE", "Data are wavelengths (lookup map)")
    lam_hdr["CUNIT1"] = (wave_unit, "Wavelength unit")
    lam_hdr["WAVEUNIT"] = (wave_unit, "Wavelength unit (explicit)")
    lam_hdr["LAMUNIT"] = (wave_unit, "Wavelength unit (alias)")
    lam_hdr["WAVEREF"] = (waveref, "Wavelength reference (air/vacuum)")
    lam_hdr["LAMREF"] = (waveref, "Wavelength reference (alias)")
    lam_hdr["BUNIT"] = (wave_unit, "Unit of lambda_map pixel values")

    # QC summary (2D fit)
    lam_hdr["RMSA"] = (float(rms2d_out), "2D wavesol RMS in wavelength unit")
    lam_hdr["RMSPX"] = (float(rms2d_px), "2D wavesol RMS in pixels")
    lam_hdr["NLINE"] = (int(n_lines_used), "Number of active (used) lines")
    lam_hdr["NREJ"] = (int(n_lines_rejected), "Number of rejected lines")
    lam_hdr["NPTS"] = (int(lams_cp.size), "Number of traced control points")
    lam_hdr["NUSED"] = (int(np.sum(used_active)), "Number of used control points")

    lam_hdr = add_provenance(lam_hdr, cfg, stage="wavesolution")

    fits.PrimaryHDU(data=lam_map_out, header=lam_hdr).writeto(
        lambda_map_fits, overwrite=True
    )

    # --- strict contract validation + rectification model (formal artifact for sky/linearization) ---
    #
    # Downstream stages MUST not guess.
    # We therefore (1) validate lambda_map.fits strictly, and (2) write a stable
    # rectification_model.json that ties the solution to a frame signature and
    # fixes VAR/MASK propagation policies.

    from scorpio_pipe.frame_signature import FrameSignature
    from scorpio_pipe.qc.lambda_map import validate_lambda_map

    frame_sig = FrameSignature.from_path(superneon_fits)

    lam_diag = validate_lambda_map(
        lambda_map_fits,
        expected_shape=(int(lam_map_out.shape[0]), int(lam_map_out.shape[1])),
        expected_unit=wave_unit,
        expected_waveref=waveref,
    )

    rect_model_json = outdir / "rectification_model.json"

    # Build an explicit output wavelength grid for Linearization.
    lcfg = cfg.get("linearize", {}) if isinstance(cfg.get("linearize"), dict) else {}
    dw_raw = lcfg.get("dlambda_A", lcfg.get("dw", "auto"))
    dw = None
    try:
        if isinstance(dw_raw, str) and dw_raw.strip().lower() in {"auto", "data", "from_data"}:
            # Use the measured dispersion from 1D fit; convert to output unit.
            dw = float(dlamdx_med) * unit_scale
        else:
            dw = float(dw_raw)
    except Exception:
        dw = float(dlamdx_med) * unit_scale
    if not (np.isfinite(dw) and dw > 0):
        dw = float(dlamdx_med) * unit_scale

    # Optional explicit limits from config (kept compatible with Angstrom keys).
    wmin_cfg = lcfg.get("lambda_min_A", lcfg.get("wmin"))
    wmax_cfg = lcfg.get("lambda_max_A", lcfg.get("wmax"))
    if wave_unit == "nm":
        # If user supplied Angstrom-style keys, convert for consistency.
        try:
            if wmin_cfg is not None:
                wmin_cfg = float(wmin_cfg) * 0.1
            if wmax_cfg is not None:
                wmax_cfg = float(wmax_cfg) * 0.1
        except Exception:
            pass

    lam_start = float(wmin_cfg) if wmin_cfg is not None else float(lam_diag.lam_min)
    lam_stop = float(wmax_cfg) if wmax_cfg is not None else float(lam_diag.lam_max)
    if not (np.isfinite(lam_start) and np.isfinite(lam_stop) and lam_stop > lam_start):
        lam_start = float(lam_diag.lam_min)
        lam_stop = float(lam_diag.lam_max)

    nlam = int(max(16, np.ceil((lam_stop - lam_start) / float(dw))))
    lam_end = float(lam_start + float(dw) * nlam)

    y_crop_top = int(lcfg.get("y_crop_top", 0) or 0)
    y_crop_bottom = int(lcfg.get("y_crop_bottom", 0) or 0)
    ny_in, nx_in = int(lam_map_out.shape[0]), int(lam_map_out.shape[1])
    ny_out = int(max(1, ny_in - max(0, y_crop_top) - max(0, y_crop_bottom)))

    # Hash lambda_map for strict reproducibility.
    import hashlib
    from datetime import datetime, timezone

    def _sha256(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    from scorpio_pipe.maskbits import (
        NO_COVERAGE,
        BADPIX,
        COSMIC,
        SATURATED,
        USER,
        REJECTED,
        MASK_SCHEMA_VERSION,
    )
    from scorpio_pipe.version import PIPELINE_VERSION

    rect_model = {
        "model_version": "1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline_version": str(PIPELINE_VERSION),
        "stage": "wavesolution",
        "frame_signature": frame_sig.to_dict(),
        "input_shape": [ny_in, nx_in],
        "output_shape": [ny_out, int(nlam)],
        "y_crop": {"top": int(y_crop_top), "bottom": int(y_crop_bottom)},
        "lambda_map": {
            "path": "lambda_map.fits",
            "sha256": _sha256(lambda_map_fits),
            "shape": [ny_in, nx_in],
            "dtype": str(lam_map_out.dtype),
            "unit": str(wave_unit),
            "waveref": str(waveref),
            "range": [float(lam_diag.lam_min), float(lam_diag.lam_max)],
            "monotonic_sign": int(lam_diag.monotonic_sign),
            "monotonic_bad_frac": float(lam_diag.monotonic_bad_frac),
            "valid_frac": float(lam_diag.valid_frac),
        },
        "wavelength_grid": {
            "unit": str(wave_unit),
            "type": "linear",
            "lam_start": float(lam_start),
            "lam_step": float(dw),
            "nlam": int(nlam),
            "lam_end": float(lam_end),
        },
        "mapping": {
            "type": "lambda_map",
            "method": "bin_integral",
            "boundary_policy": "mask_no_coverage",
        },
        "var_policy": {
            "rule": "sum_a2_var",
            "formula": "VAR_out = Σ_k (a_k^2 * VAR_k)",
        },
        "mask_policy": {
            "combine": "OR",
            "fatal_bits": [
                {"name": "NO_COVERAGE", "value": int(NO_COVERAGE)},
                {"name": "BADPIX", "value": int(BADPIX)},
                {"name": "COSMIC", "value": int(COSMIC)},
                {"name": "SATURATED", "value": int(SATURATED)},
                {"name": "USER", "value": int(USER)},
                {"name": "REJECTED", "value": int(REJECTED)},
            ],
            "no_coverage_bit": {"name": "NO_COVERAGE", "value": int(NO_COVERAGE)},
            "schema_version": int(MASK_SCHEMA_VERSION),
        },
        "provenance": {
            "created_from": {
                "superneon_fits": str(Path(superneon_fits).name),
                "wavesolution_1d_json": str(Path(wavesol1d_json).name) if wavesol1d_json else None,
                "wavesolution_2d_json": str(Path(wavesol2d_json).name) if wavesol2d_json else None,
            },
        },
    }

    rect_model_json.write_text(json.dumps(rect_model, indent=2), encoding="utf-8")

    # --- stage done / QC flags (used by the pipeline gate) ---
    try:
        from scorpio_pipe.qc_thresholds import compute_thresholds
        from scorpio_pipe.qc.flags import make_flag, max_severity

        thr, _thr_meta = compute_thresholds(cfg)
        flags: list[dict[str, Any]] = []

        # P0-K/P0-M: carry over pre-flags (manifest exclude applied, lamp contract notes)
        try:
            if isinstance(pre_flags, list) and pre_flags:
                flags.extend(pre_flags)
        except Exception:
            pass

        # P0-H: prepend ARC compatibility warnings (CALIB_* / ARC_COMPAT_CHECK_FAILED)
        try:
            if isinstance(compat_flags, list) and compat_flags:
                flags.extend(compat_flags)
        except Exception:
            pass

        # 1D fit quality
        if np.isfinite(rms1d):
            if float(rms1d) >= float(thr.wavesol_1d_rms_bad):
                flags.append(
                    make_flag(
                        "WAVESOL_1D_RMS",
                        "ERROR",
                        f"1D RMS is high: {float(rms1d):.3g} Å",
                        value_A=float(rms1d),
                        bad_A=float(thr.wavesol_1d_rms_bad),
                        warn_A=float(thr.wavesol_1d_rms_warn),
                    )
                )
            elif float(rms1d) >= float(thr.wavesol_1d_rms_warn):
                flags.append(
                    make_flag(
                        "WAVESOL_1D_RMS",
                        "WARN",
                        f"1D RMS is above warn: {float(rms1d):.3g} Å",
                        value_A=float(rms1d),
                        warn_A=float(thr.wavesol_1d_rms_warn),
                    )
                )

        # 2D fit quality
        if np.isfinite(rms2d_A):
            if float(rms2d_A) >= float(thr.wavesol_2d_rms_bad):
                flags.append(
                    make_flag(
                        "WAVESOL_2D_RMS",
                        "ERROR",
                        f"2D RMS is high: {float(rms2d_A):.3g} Å",
                        value_A=float(rms2d_A),
                        bad_A=float(thr.wavesol_2d_rms_bad),
                        warn_A=float(thr.wavesol_2d_rms_warn),
                    )
                )
            elif float(rms2d_A) >= float(thr.wavesol_2d_rms_warn):
                flags.append(
                    make_flag(
                        "WAVESOL_2D_RMS",
                        "WARN",
                        f"2D RMS is above warn: {float(rms2d_A):.3g} Å",
                        value_A=float(rms2d_A),
                        warn_A=float(thr.wavesol_2d_rms_warn),
                    )
                )

        if np.isfinite(resid2d_p95_A):
            if float(resid2d_p95_A) >= float(thr.resid_2d_p95_bad):
                flags.append(
                    make_flag(
                        "WAVESOL_2D_P95",
                        "ERROR",
                        f"2D residual |dλ| P95 is high: {float(resid2d_p95_A):.3g} Å",
                        p95_A=float(resid2d_p95_A),
                        bad_A=float(thr.resid_2d_p95_bad),
                        warn_A=float(thr.resid_2d_p95_warn),
                    )
                )
            elif float(resid2d_p95_A) >= float(thr.resid_2d_p95_warn):
                flags.append(
                    make_flag(
                        "WAVESOL_2D_P95",
                        "WARN",
                        f"2D residual |dλ| P95 above warn: {float(resid2d_p95_A):.3g} Å",
                        p95_A=float(resid2d_p95_A),
                        warn_A=float(thr.resid_2d_p95_warn),
                    )
                )

        # Hard contract checks
        if "WAVEUNIT" not in lam_hdr or "WAVEREF" not in lam_hdr:
            flags.append(
                make_flag(
                    "WAVESOL_META",
                    "ERROR",
                    "lambda_map.fits is missing WAVEUNIT/WAVEREF metadata",
                )
            )

        sev = max_severity(flags)

        # --- quicklook for lambda_map (navigator-friendly) ---
        lambda_map_png = outdir / "lambda_map.png"
        try:
            from astropy.io import fits
            import numpy as np
            from scorpio_pipe.io.quicklook import write_quicklook_png

            with fits.open(lambda_map_fits, memmap=False) as hdul:
                lm = np.asarray(hdul[0].data, dtype=np.float64)
            # Lambda map is smooth; use a wider k to avoid banding.
            write_quicklook_png(lm, lambda_map_png, k=8.0, method="linear", meta={"kind": "lambda_map"})
        except Exception:
            pass

        done_payload: dict[str, Any] = {
            "stage": "wavesolution",
            "ok": bool(sev not in {"ERROR"}),
            "frame_signature": frame_sig.to_dict(),
            "lamp": lamp_meta or None,
            "compat": {"arc": compat_meta or None, "excluded_summary": excluded_summary or None},
            "lambda_map": lam_diag.as_dict(),
            "products": {
                "lambda_map_fits": str(lambda_map_fits),
                "lambda_map_png": str(lambda_map_png),
                "rectification_model_json": str(rect_model_json),
                "wavesolution_1d_json": str(wavesol1d_json),
                "wavesolution_2d_json": str(wavesol2d_json),
                "report_txt": str(report_txt),
                "residuals_2d_png": str(resid2d_png),
                "residuals_hist_png": str(resid_hist_png),
            },
            "metrics": {
                "rms1d_A": float(rms1d) if np.isfinite(rms1d) else None,
                "rms2d_A": float(rms2d_A) if np.isfinite(rms2d_A) else None,
                "resid2d_p95_A": float(resid2d_p95_A) if np.isfinite(resid2d_p95_A) else None,
                "dispersion_A_per_px": float(dlamdx_med) if np.isfinite(dlamdx_med) else None,
                "n_lines_used": int(n_lines_used),
                "n_lines_rejected": int(n_lines_rejected),
            },
            "qc": {
                "flags": flags,
                "max_severity": sev,
            },
            "versioning": {
                "wavesol_model_version": "1",
            },
        }

        # Canonical + legacy names
        (outdir / "done.json").write_text(json.dumps(done_payload, indent=2), encoding="utf-8")
        (outdir / "wavesolution_done.json").write_text(
            json.dumps(done_payload, indent=2), encoding="utf-8"
        )
        # P1-G navigator contract expects this alias.
        try:
            (outdir / "wavesol_done.json").write_text(
                json.dumps(done_payload, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

        # P1-B canonical marker
        status = "ok"
        if sev == "ERROR":
            status = "fail"
        elif sev == "WARN":
            status = "warn"

        wave_done = {
            "stage": "wavesolution",
            "status": status,
            "frame_signature": frame_sig.to_dict(),
            "lambda_map": lam_diag.as_dict(),
            "errors": {
                "rms_arc_fit": float(rms1d) if np.isfinite(rms1d) else None,
                "rms_2d_fit": float(rms2d_A) if np.isfinite(rms2d_A) else None,
                "dispersion_A_per_px": float(dlamdx_med) if np.isfinite(dlamdx_med) else None,
            },
            "flags": flags,
            "products": {
                "lambda_map_fits": "lambda_map.fits",
                "rectification_model_json": "rectification_model.json",
                "done_json": "done.json",
            },
            "versioning": {
                "wavesol_model_version": "1",
            },
        }
        (outdir / "wave_done.json").write_text(json.dumps(wave_done, indent=2), encoding="utf-8")
    except Exception:
        pass

    return WaveSolutionResult(
        wavesol_dir=outdir,
        lambda_map_fits=lambda_map_fits,
        wavesolution_1d_png=poly1d_png,
        wavesolution_1d_json=wavesol1d_json,
        wavesolution_2d_json=wavesol2d_json,
        control_points_csv=control_points_csv,
        residuals_1d_csv=poly1d_resid_csv,
        residuals_2d_csv=resid2d_csv,
        residuals_2d_png=resid2d_png,
        residuals_2d_audit_png=resid2d_audit_png,
        residuals_vs_lambda_png=resid_vs_lambda_png,
        residuals_vs_y_png=resid_vs_y_png,
        residuals_hist_png=resid_hist_png,
        report_txt=report_txt,
        # 1D metrics
        rms1d_A=float(rms1d),
        rms1d_px=float(rms1d_px),
        wrms1d_A=float(wrms1d),
        sigma1d_mad_A=float(sig_mad),
        dispersion_A_per_px=float(dlamdx_med),
        n_pairs_total=int(x_all.size),
        n_pairs_used=int(np.sum(used_mask)),
        n_pairs_disabled=int(np.sum(is_disabled_all)),
        # 2D metrics
        model2d_kind=str(kind),
        rms2d_A=float(rms2d_A),
        wrms2d_A=float(wrms2d_A),
        sigma2d_mad_A=float(sig_mad_2d_A),
        n_lines_total=n_lines_total,
        n_lines_used=n_lines_used,
        n_lines_rejected=n_lines_rejected,
    )
