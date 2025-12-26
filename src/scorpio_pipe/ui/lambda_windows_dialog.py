from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits

from scorpio_pipe.fits_utils import open_fits_smart

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from scorpio_pipe.plot_style import mpl_style


@dataclass
class LambdaWindows:
    unit: str = "auto"  # "A" or "pix"
    windows_A: List[Tuple[float, float]] = None
    windows_pix: List[Tuple[int, int]] = None

    def __post_init__(self):
        if self.windows_A is None:
            self.windows_A = []
        if self.windows_pix is None:
            self.windows_pix = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit": self.unit,
            "windows_A": [[float(a), float(b)] for a, b in self.windows_A],
            "windows_pix": [[int(a), int(b)] for a, b in self.windows_pix],
        }


def _extract_sci_and_hdr(hdul: fits.HDUList) -> tuple[np.ndarray, fits.Header]:
    hdr = hdul[0].header
    sci = hdul[0].data
    if sci is None:
        if "SCI" in hdul:
            sci = hdul["SCI"].data
            hdr = hdul["SCI"].header.copy()
            # keep primary WCS if present
            for k in ("CRVAL1", "CDELT1", "CD1_1", "CRPIX1", "CUNIT1"):
                if k in hdul[0].header and k not in hdr:
                    hdr[k] = hdul[0].header[k]
    if sci is None:
        raise ValueError("SCI image not found in FITS")
    return np.asarray(sci, dtype=float), hdr


def _has_linear_wcs_x(hdr: fits.Header) -> bool:
    crval1 = hdr.get("CRVAL1")
    cdelt1 = hdr.get("CDELT1", hdr.get("CD1_1"))
    return crval1 is not None and cdelt1 is not None


class LambdaWindowsDialog(QDialog):
    """Interactive selection of λ-windows (or pixel windows) for cross-correlation.

    The dialog shows a 1D sky spectrum (median over sky rows) and lets the user select
    horizontal spans. The returned windows are either in Angstrom (if a linear WCS is
    present in the header), or in pixels otherwise.

    The list of windows is intended for:
    - flexure correction (Δλ) via sky-line cross-correlation
    - y-alignment windows for Stack2D
    """

    def __init__(
        self,
        fits_path: str | Path,
        roi: Optional[Dict[str, Any]] = None,
        *,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Select λ windows (cross-correlation)")
        # User preference: open windows maximized by default.
        try:
            self.setWindowState(self.windowState() | Qt.WindowMaximized)
        except Exception:
            pass
        self._fits_path = Path(fits_path)
        self._roi = roi or {}
        self.windows = LambdaWindows(unit="auto")

        # --- UI ---
        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        info = QLabel(
            "Выберите 2–6 диапазонов по X (λ или пиксели), где есть яркие и узкие линии (макс. S/N).\n"
            "Эти окна используются для стабильного определения сдвигов (Δλ и/или y-align).\n"
            "ЛКМ+drag — выделить, кнопки ниже — удалить/очистить." 
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        self.fig = Figure(figsize=(7.2, 3.2), dpi=110)
        self.canvas = FigureCanvasQTAgg(self.fig)
        lay.addWidget(self.canvas, 1)
        self.ax = self.fig.add_subplot(1, 1, 1)

        # Controls
        row = QHBoxLayout()
        self.lbl_units = QLabel("Units: —")
        row.addWidget(self.lbl_units)
        row.addStretch(1)
        self.btn_remove_last = QPushButton("Remove last")
        self.btn_clear = QPushButton("Clear")
        row.addWidget(self.btn_remove_last)
        row.addWidget(self.btn_clear)
        lay.addLayout(row)

        self.lbl_list = QLabel("Windows: <none>")
        self.lbl_list.setWordWrap(True)
        f = QFrame()
        f.setFrameShape(QFrame.StyledPanel)
        fl = QVBoxLayout(f)
        fl.addWidget(self.lbl_list)
        lay.addWidget(f)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        lay.addWidget(buttons)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.btn_remove_last.clicked.connect(self._remove_last)
        self.btn_clear.clicked.connect(self._clear)

        # Load + plot
        self._x, self._spec = self._make_spectrum()
        self._plot()

        # Span selector
        self._span = SpanSelector(
            self.ax,
            onselect=self._on_select,
            direction="horizontal",
            useblit=True,
            props=dict(alpha=0.25),
            interactive=True,
            drag_from_anywhere=True,
        )

    def _make_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._fits_path.exists():
            raise FileNotFoundError(str(self._fits_path))
        with open_fits_smart(self._fits_path, prefer_memmap=True) as hdul:
            sci, hdr = _extract_sci_and_hdr(hdul)

        ny, nx = sci.shape

        # Determine sky rows from ROI (if provided)
        sky_rows = None
        try:
            obj_y0 = int(self._roi.get("obj_y0", 0))
            obj_y1 = int(self._roi.get("obj_y1", 0))
            st0 = int(self._roi.get("sky_top_y0", 0))
            st1 = int(self._roi.get("sky_top_y1", 0))
            sb0 = int(self._roi.get("sky_bot_y0", 0))
            sb1 = int(self._roi.get("sky_bot_y1", 0))
            def _clip(v):
                return max(0, min(ny - 1, int(v)))
            obj_y0, obj_y1 = sorted((_clip(obj_y0), _clip(obj_y1)))
            st0, st1 = sorted((_clip(st0), _clip(st1)))
            sb0, sb1 = sorted((_clip(sb0), _clip(sb1)))
            sky_rows = np.zeros(ny, dtype=bool)
            sky_rows[st0 : st1 + 1] = True
            sky_rows[sb0 : sb1 + 1] = True
            sky_rows[obj_y0 : obj_y1 + 1] = False
            if sky_rows.sum() < 3:
                sky_rows = None
        except Exception:
            sky_rows = None

        if sky_rows is None:
            sky = sci
        else:
            sky = sci[sky_rows, :]

        spec = np.nanmedian(sky, axis=0)
        spec = np.asarray(spec, dtype=float)

        if _has_linear_wcs_x(hdr):
            crval1 = float(hdr.get("CRVAL1"))
            cdelt1 = float(hdr.get("CDELT1", hdr.get("CD1_1")))
            crpix1 = float(hdr.get("CRPIX1", 1.0))
            x = crval1 + ((np.arange(nx) + 1.0) - crpix1) * cdelt1
            self.windows.unit = "A"
            self.lbl_units.setText("Units: Å")
        else:
            x = np.arange(nx, dtype=float)
            self.windows.unit = "pix"
            self.lbl_units.setText("Units: pixels")

        # light smoothing for visuals
        try:
            k = 7
            if spec.size > k:
                from numpy.lib.stride_tricks import sliding_window_view
                sw = sliding_window_view(spec, k)
                sm = np.nanmedian(sw, axis=1)
                spec2 = spec.copy()
                pad = k // 2
                spec2[pad:-pad] = sm
                spec = spec2
        except Exception:
            pass

        return x, spec

    def _plot(self) -> None:
        self.ax.clear()
        with mpl_style():
            self.ax.plot(self._x, self._spec)
            self.ax.set_xlabel("Wavelength [Å]" if self.windows.unit == "A" else "X [pix]")
            self.ax.set_ylabel("Median sky flux")
            self.ax.grid(True, alpha=0.3)
            self.ax.set_title("Select windows with bright, narrow lines")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _on_select(self, xmin: float, xmax: float) -> None:
        a, b = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
        if self.windows.unit == "A":
            self.windows.windows_A.append((float(a), float(b)))
        else:
            self.windows.windows_pix.append((int(round(a)), int(round(b))))
        self._update_list()

    def _remove_last(self) -> None:
        if self.windows.unit == "A":
            if self.windows.windows_A:
                self.windows.windows_A.pop()
        else:
            if self.windows.windows_pix:
                self.windows.windows_pix.pop()
        self._update_list()

    def _clear(self) -> None:
        self.windows.windows_A.clear()
        self.windows.windows_pix.clear()
        self._update_list()

    def _update_list(self) -> None:
        if self.windows.unit == "A":
            lst = self.windows.windows_A
            if not lst:
                self.lbl_list.setText("Windows: <none>")
            else:
                parts = [f"{a:.1f}–{b:.1f}" for a, b in lst]
                self.lbl_list.setText("Windows [Å]: " + "; ".join(parts))
        else:
            lst = self.windows.windows_pix
            if not lst:
                self.lbl_list.setText("Windows: <none>")
            else:
                parts = [f"{a:d}–{b:d}" for a, b in lst]
                self.lbl_list.setText("Windows [pix]: " + "; ".join(parts))

    def result_windows(self) -> Dict[str, Any]:
        """Return windows in a config-friendly dict."""
        return self.windows.to_dict()
