from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from PySide6 import QtCore, QtWidgets


@dataclass
class Wave2DCleanConfig:
    model2d: str = "auto"  # auto | power | chebyshev
    power_deg: int = 5
    cheb_degx: int = 5
    cheb_degy: int = 3
    # Keep separate robustness controls so the interactive view can exactly
    # mirror the stage outputs.
    power_sigma_clip: float = 3.0
    power_maxiter: int = 10
    cheb_sigma_clip: float = 3.0
    cheb_maxiter: int = 10


class Wave2DLineCleanerDialog(QtWidgets.QDialog):
    """Interactive rejector for bad lamp lines in the 2D wavelength solution.

    - loads control points from CSV (produced by wavesolution stage)
    - lets the user toggle whole lines (by laboratory wavelength)
    - refits the 2D model on the fly and updates RMS in the window

    Notes
    -----
    This dialog is intentionally lightweight: it operates on the saved
    control points and does not re-trace the superneon.
    """

    def __init__(
        self,
        control_points_csv: Path,
        *,
        cfg: Wave2DCleanConfig | None = None,
        rejected_lines_A: list[float] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("2D Wavesolution — reject bad lines")
        self.setModal(True)
        self.resize(1100, 720)
        # User preference: open windows maximized by default.
        try:
            self.setWindowState(self.windowState() | QtCore.Qt.WindowMaximized)
        except Exception:
            pass

        self._cfg = cfg or Wave2DCleanConfig()
        self._rejected0 = sorted(set(float(x) for x in (rejected_lines_A or [])))

        self._x, self._y, self._lam, self._score = self._load_points(control_points_csv)
        if self._x.size == 0:
            raise RuntimeError("No control points in CSV")

        # state: per-line include/exclude
        # Group control points into spectral lines using rounded wavelength keys
        lam_key = np.round(self._lam, 3)
        keys, inv = np.unique(lam_key, return_inverse=True)
        centers = np.array([float(np.nanmedian(self._lam[inv == j])) for j in range(len(keys))], dtype=float)
        order = np.argsort(centers)
        rev = np.empty_like(order)
        rev[order] = np.arange(order.size)
        self._line_id = rev[inv]
        self._unique_lines = centers[order]
        self._active = {float(l0): (not any(abs(float(l0) - r) <= 0.25 for r in self._rejected0)) for l0 in self._unique_lines}

        # UI
        root = QtWidgets.QHBoxLayout(self)

        left = QtWidgets.QVBoxLayout()
        right = QtWidgets.QVBoxLayout()
        root.addLayout(left, 0)
        root.addLayout(right, 1)

        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        left.addWidget(self.lbl_info)

        self.list_lines = QtWidgets.QListWidget()
        self.list_lines.setMinimumWidth(320)
        left.addWidget(self.list_lines, 1)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_all = QtWidgets.QPushButton("Select all")
        self.btn_none = QtWidgets.QPushButton("Select none")
        btn_row.addWidget(self.btn_all)
        btn_row.addWidget(self.btn_none)
        left.addLayout(btn_row)

        # plotting
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
        from scorpio_pipe.plot_style import mpl_style

        self._mpl_style = mpl_style
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right.addWidget(self.toolbar)

        # View toggle: in "final" mode we hide rejected lines AND show only the
        # points actually used by the robust fit. This makes the plot visually
        # consistent with the stage's residuals_2d.png (QC).
        self.chk_hide_rejected = QtWidgets.QCheckBox("Fit-consistent view (inliers only)")
        self.chk_hide_rejected.setChecked(True)
        right.addWidget(self.chk_hide_rejected)

        right.addWidget(self.canvas, 1)

        self.lbl_rms = QtWidgets.QLabel("")
        self.lbl_rms.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        right.addWidget(self.lbl_rms)

        # buttons
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        right.addWidget(bb)

        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)

        self.btn_all.clicked.connect(self._select_all)
        self.btn_none.clicked.connect(self._select_none)
        self.list_lines.itemChanged.connect(self._on_item_changed)
        self.chk_hide_rejected.toggled.connect(lambda *_: self._recompute_and_redraw())
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)

        # cache for click-to-toggle in plot coordinates
        self._click_x: np.ndarray | None = None
        self._click_y: np.ndarray | None = None
        self._click_lam: np.ndarray | None = None

        self._populate_list()
        self._recompute_and_redraw()

    @staticmethod
    def _load_points(p: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        try:
            arr = np.genfromtxt(p, delimiter=",", names=True, dtype=float)
        except Exception:
            # fallback: no header
            arr = np.loadtxt(p, delimiter=",", dtype=float)
            if arr.ndim == 1:
                arr = arr[None, :]
            x, y, lam, score = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
            return x, y, lam, score

        x = np.asarray(arr["x_pix"], float)
        y = np.asarray(arr["y_pix"], float)
        lam = np.asarray(arr["lambda_A"], float)
        score = np.asarray(arr["score"], float)
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(lam) & np.isfinite(score)
        return x[m], y[m], lam[m], score[m]


    def save_plots(self, outdir: Path, stem: str = 'wavesol2d_clean') -> list[Path]:
        """Save two diagnostic PNGs to `outdir`.

        1) `*_audit.png` — all points shown; rejected lines stay grey (best for audit).
        2) `*_final.png` — fit-consistent view: active lines show only inliers.

        Returns list of saved paths.
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        saved: list[Path] = []
        prev_hide = bool(self.chk_hide_rejected.isChecked())

        # audit view (all points)
        try:
            self.chk_hide_rejected.setChecked(False)
            self._recompute_and_plot()
        except Exception:
            pass
        p1 = outdir / f"{stem}_audit.png"
        try:
            self.fig.savefig(p1, dpi=180, bbox_inches='tight')
            saved.append(p1)
        except Exception:
            pass

        # final view (inliers only for active lines)
        try:
            self.chk_hide_rejected.setChecked(True)
            self._recompute_and_plot()
        except Exception:
            pass
        p2 = outdir / f"{stem}_final.png"
        try:
            self.fig.savefig(p2, dpi=180, bbox_inches='tight')
            saved.append(p2)
        except Exception:
            pass

        # restore
        try:
            self.chk_hide_rejected.setChecked(prev_hide)
            self._recompute_and_plot()
        except Exception:
            pass

        return saved

    # Keep backwards compatibility: older code used a private helper with this name.
    def _recompute_and_plot(self) -> None:
        self._recompute_and_redraw()

    def rejected_lines(self) -> list[float]:
        return sorted([float(l0) for l0, on in self._active.items() if not on])

    def _populate_list(self) -> None:
        self.list_lines.blockSignals(True)
        self.list_lines.clear()
        for l0 in self._unique_lines:
            l0 = float(l0)
            n = int(np.sum(np.abs(self._lam - l0) < 1e-6))
            it = QtWidgets.QListWidgetItem(f"{l0:9.2f} Å   (N={n})")
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if self._active.get(l0, True) else QtCore.Qt.Unchecked)
            it.setData(QtCore.Qt.UserRole, l0)
            self.list_lines.addItem(it)
        self.list_lines.blockSignals(False)

    def _select_all(self) -> None:
        for l0 in self._unique_lines:
            self._active[float(l0)] = True
        self._populate_list()
        self._recompute_and_redraw()

    def _select_none(self) -> None:
        for l0 in self._unique_lines:
            self._active[float(l0)] = False
        self._populate_list()
        self._recompute_and_redraw()

    def _on_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        l0 = float(item.data(QtCore.Qt.UserRole))
        self._active[l0] = (item.checkState() == QtCore.Qt.Checked)
        self._recompute_and_redraw()

    def _on_plot_click(self, event) -> None:
        # Toggle the nearest displayed point's line (simple & practical)
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if self._click_x is None or self._click_y is None or self._click_lam is None:
            return
        if self._click_x.size == 0:
            return
        xx = self._click_x
        yy = self._click_y
        d2 = (xx - float(event.xdata)) ** 2 + (yy - float(event.ydata)) ** 2
        idx = int(np.argmin(d2))
        if idx < 0 or idx >= xx.size or (not np.isfinite(d2[idx])):
            return
        # only toggle if click is reasonably close
        # (threshold in pixels; convert from data -> display)
        try:
            ax = event.inaxes
            p_click = ax.transData.transform((event.xdata, event.ydata))
            p_pt = ax.transData.transform((xx[idx], yy[idx]))
            if float(np.hypot(*(p_click - p_pt))) > 12.0:
                return
        except Exception:
            pass
        l0 = float(self._click_lam[idx])
        self._active[l0] = not self._active.get(l0, True)
        self._populate_list()
        self._recompute_and_redraw()

    def _points_mask(self) -> np.ndarray:
        # mask points by active lines
        on = np.vectorize(lambda v: self._active.get(float(v), True), otypes=[bool])
        return on(self._lam)

    def _recompute_and_redraw(self) -> None:
        from scorpio_pipe.stages.wavesolution import (
            robust_polyfit_2d_cheb,
            robust_polyfit_2d_power,
            polyval2d_cheb,
            polyval2d_power,
        )

        # Fit only on active lines, but we can still visualize rejected lines.
        mask_fit = self._points_mask()
        x = self._x[mask_fit]
        y = self._y[mask_fit]
        lam = self._lam[mask_fit]
        score = self._score[mask_fit]

        if x.size < 30:
            self.lbl_rms.setText("Too few points selected")
            return

        w = np.sqrt(np.clip(score, 0, None))

        # fit both quickly
        pow_coeff, pow_meta, pow_used = robust_polyfit_2d_power(
            x, y, lam, int(self._cfg.power_deg),
            weights=w,
            sigma_clip=float(self._cfg.power_sigma_clip),
            maxiter=int(self._cfg.power_maxiter),
        )
        pow_pred = polyval2d_power(x, y, pow_coeff, pow_meta)
        pow_dlam = pow_pred - lam
        pow_rms = float(np.sqrt(np.mean(pow_dlam[pow_used] ** 2)))

        cheb_C, cheb_meta, cheb_used = robust_polyfit_2d_cheb(
            x, y, lam, int(self._cfg.cheb_degx), int(self._cfg.cheb_degy),
            weights=w,
            sigma_clip=float(self._cfg.cheb_sigma_clip),
            maxiter=int(self._cfg.cheb_maxiter),
        )
        cheb_pred = polyval2d_cheb(x, y, cheb_C, cheb_meta)
        cheb_dlam = cheb_pred - lam
        cheb_rms = float(np.sqrt(np.mean(cheb_dlam[cheb_used] ** 2)))

        mode = self._cfg.model2d.strip().lower()
        if mode in ("cheb", "chebyshev"):
            kind = "chebyshev"
        elif mode in ("pow", "power", "poly"):
            kind = "power"
        else:
            kind = "power" if pow_rms <= cheb_rms else "chebyshev"

        if kind == "power":
            used_fit = pow_used

            def model_fn(xx, yy):
                return polyval2d_power(xx, yy, pow_coeff, pow_meta)
        else:
            used_fit = cheb_used

            def model_fn(xx, yy):
                return polyval2d_cheb(xx, yy, cheb_C, cheb_meta)

        # Expand used mask to all control points (for QC-consistent plotting).
        used_all = np.zeros_like(self._x, dtype=bool)
        used_all[mask_fit] = np.asarray(used_fit, bool)

        # Residuals for plotting (all points), but in "final" mode we will show
        # only the inliers used by the robust fit.
        pred_all = model_fn(self._x, self._y)
        dlam_all = pred_all - self._lam

        # Plot must match the stage QC plot (residuals_2d.png) exactly:
        # x-axis is Δλ, y-axis is Y (stacked with offsets).
        final_view = bool(self.chk_hide_rejected.isChecked())

        _np = np
        import matplotlib.transforms as mtransforms
        from matplotlib.colors import LinearSegmentedColormap

        uniq = _np.asarray(self._unique_lines, float)
        cmap = LinearSegmentedColormap.from_list("blue_to_red", ["blue", "red"])
        colors = cmap(_np.linspace(0.0, 1.0, max(1, len(uniq))))

        # Offset step: same heuristic as the stage plotter (based on Y sampling).
        y_steps: list[float] = []
        for lam0 in uniq:
            m_line = _np.abs(self._lam - float(lam0)) < 1e-6
            ys = _np.sort(self._y[m_line])
            if ys.size > 3:
                dy = _np.diff(ys)
                dy = dy[dy > 0]
                if dy.size:
                    y_steps.append(float(_np.median(dy)))
        y_step = float(_np.median(y_steps)) if y_steps else 1.0
        y_offset_step = float(0.9 * max(1.0, y_step))

        # QC-like global stats (on inliers used by the robust fit).
        resid_used = dlam_all[used_all & _np.isfinite(dlam_all)]
        rms_used = float(_np.sqrt(_np.mean(resid_used ** 2))) if resid_used.size else float("nan")
        w_used = _np.asarray(self._score, float)[used_all & _np.isfinite(dlam_all)]
        wrms_used = (
            float(_np.sqrt(_np.sum(w_used * (resid_used ** 2)) / _np.sum(w_used)))
            if resid_used.size and _np.sum(w_used) > 0
            else float("nan")
        )
        if resid_used.size:
            med = float(_np.median(resid_used))
            mad = float(_np.median(_np.abs(resid_used - med)))
            sig_mad = float(1.4826 * mad)
        else:
            sig_mad = float("nan")

        with self._mpl_style():
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.set_xlabel("Δλ [Å]")
            ax.set_ylabel("Y [px] (stacked)")
            ax.set_title(
                f"2D residuals (RMS={rms_used:.3f} Å, σ_MAD={sig_mad:.3f} Å) — click to toggle a line"
            )
            ax.axvline(0.0, color="0.6", lw=1.0, alpha=0.6)

            text_rows: list[tuple[float, tuple, float, float, float]] = []
            click_x: list[float] = []
            click_y: list[float] = []
            click_l: list[float] = []

            for k, lam0 in enumerate(uniq):
                lam0 = float(lam0)
                m_line = _np.abs(self._lam - lam0) < 1e-6
                if not _np.any(m_line):
                    continue

                is_on = bool(self._active.get(lam0, True))
                line_rej = not is_on

                if final_view and (not line_rej):
                    m_disp = m_line & used_all
                    if int(_np.count_nonzero(m_disp)) < 3:
                        m_disp = m_line
                else:
                    m_disp = m_line

                ys = self._y[m_disp]
                ds = dlam_all[m_disp]
                if ys.size < 3:
                    continue
                ordy = _np.argsort(ys)
                y_sorted = ys[ordy]
                d_sorted = ds[ordy]
                y_disp = y_sorted + float(k) * y_offset_step

                col = colors[k % len(colors)]
                if line_rej:
                    col = (0.65, 0.65, 0.65, 0.9)

                ax.plot(d_sorted, y_disp, lw=1.2, color=col)

                step_pts = max(1, len(y_disp) // 150)
                xs_s = d_sorted[::step_pts]
                ys_s = y_disp[::step_pts]
                ax.scatter(xs_s, ys_s, s=3, color=col, alpha=0.85, linewidths=0.0)

                click_x.extend(xs_s.tolist())
                click_y.extend(ys_s.tolist())
                click_l.extend([lam0] * len(xs_s))

                # per-line stats (prefer inliers)
                d_stat = dlam_all[m_line & used_all]
                if d_stat.size < 2:
                    d_stat = ds
                good = _np.isfinite(d_stat)
                if int(_np.count_nonzero(good)) >= 2:
                    mu = float(_np.mean(d_stat[good]))
                    sd = float(_np.std(d_stat[good], ddof=1))
                else:
                    mu, sd = _np.nan, _np.nan

                y_text = float(_np.nanmedian(y_disp))
                text_rows.append((y_text, col, lam0, mu, sd))

            for y_text, col, *_ in text_rows:
                ax.axhline(y=y_text, xmin=0.0, xmax=1.0, color=col, linestyle=":", linewidth=0.9, alpha=0.35, zorder=0)

            self.fig.canvas.draw()
            trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
            y_top = ax.get_ylim()[1]
            x_col1 = 1.02
            x_col2 = 1.02 + 0.18
            ax.text(x_col1, y_top + 0.1, "λ", transform=trans, ha="left", va="bottom", fontsize=11)
            ax.text(x_col2, y_top + 0.1, "Δλ [Å]", transform=trans, ha="left", va="bottom", fontsize=11)

            for y_text, col, lam0, mu, sd in text_rows:
                ax.text(x_col1, y_text, f"{lam0:7.2f} Å", color=col, transform=trans, ha="left", va="center", fontsize=10)
                s = f"{mu:+.2f} ± {sd:.2f}" if _np.isfinite(mu) else "—"
                s = s.replace("-", "−")
                ax.text(x_col2, y_text, s, color=col, transform=trans, ha="left", va="center", fontsize=10)

            self.fig.tight_layout()

        # cache for click events
        self._click_x = _np.asarray(click_x, float)
        self._click_y = _np.asarray(click_y, float)
        self._click_lam = _np.asarray(click_l, float)

        self.canvas.draw()

        n_fit_total = int(_np.count_nonzero(mask_fit))
        n_used_total = int(_np.count_nonzero(used_all & _np.isfinite(dlam_all)))
        self.lbl_rms.setText(
            f"Model: {kind}   |   RMS: {rms_used:.4f} Å   |   wRMS: {wrms_used:.4f} Å   |   σ_MAD: {sig_mad:.4f} Å\n"
            f"Points used: {n_used_total}/{n_fit_total}   |   Lines rejected: {len(self.rejected_lines())}/{len(uniq)}"
        )
        self.lbl_info.setText(
            "Tip: uncheck a wavelength (line) to exclude it from the 2D fit.\n"
            "You can also click near a curve/point on the plot to toggle its line.\n"
            "Use 'Fit-consistent view (inliers only)' to compare the fit cleanly with the QC residual plot."
        )
