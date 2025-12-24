from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from PySide6 import QtCore, QtGui, QtWidgets


@dataclass
class StretchParams:
    cmap: str = "gray"  # gray, viridis, magma
    p_lo: float = 1.0
    p_hi: float = 99.0
    gamma: float = 1.0


def _as_float(a: np.ndarray) -> np.ndarray:
    if a is None:
        return np.zeros((10, 10), dtype=np.float32)
    if a.ndim > 2:
        # a common pattern for some FITS is (1, ny, nx)
        a = np.squeeze(a)
    return np.array(a, dtype=np.float32, copy=False)


def _safe_percentiles(x: np.ndarray, p_lo: float, p_hi: float) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    p_lo = float(np.clip(p_lo, 0.0, 100.0))
    p_hi = float(np.clip(p_hi, 0.0, 100.0))
    if p_hi < p_lo:
        p_lo, p_hi = p_hi, p_lo
    lo = float(np.percentile(x, p_lo))
    hi = float(np.percentile(x, p_hi))
    if not np.isfinite(lo):
        lo = float(np.nanmin(x))
    if not np.isfinite(hi):
        hi = float(np.nanmax(x))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _apply_stretch(img: np.ndarray, params: StretchParams) -> np.ndarray:
    img = _as_float(img)
    lo, hi = _safe_percentiles(img, params.p_lo, params.p_hi)
    x = (img - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)

    gamma = float(params.gamma)
    if gamma <= 0:
        gamma = 1.0
    # display gamma: gamma>1 makes it brighter (as user expects)
    x = np.power(x, 1.0 / gamma)
    return x


def _to_qimage_gray(x01: np.ndarray) -> QtGui.QImage:
    u8 = (x01 * 255.0).astype(np.uint8)
    h, w = u8.shape
    # QImage needs bytes that live as long as image; keep a copy
    buf = u8.tobytes()
    qimg = QtGui.QImage(buf, w, h, w, QtGui.QImage.Format_Grayscale8)
    return qimg.copy()


def _to_qimage_cmap(x01: np.ndarray, cmap_name: str) -> QtGui.QImage:
    # Lazy import: matplotlib is heavy, avoid importing unless needed
    import matplotlib.cm as cm

    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(x01)  # float64 0..1, shape (h,w,4)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    h, w, _ = rgb.shape
    buf = rgb.tobytes()
    qimg = QtGui.QImage(buf, w, h, w * 3, QtGui.QImage.Format_RGB888)
    return qimg.copy()


class FitsPreviewWidget(QtWidgets.QWidget):
    """Simple FITS viewer with colormap + exposure controls.

    - Colormap: gray, viridis, magma
    - Exposure: low/high percentiles
    - Gamma: display gamma

    Designed to be lightweight and safe for large files.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._path: str | None = None
        self._data: np.ndarray | None = None
        self._params = StretchParams()

        # Controls
        self.cmb_cmap = QtWidgets.QComboBox()
        self.cmb_cmap.addItems(["gray", "viridis", "magma"])
        self.cmb_cmap.setCurrentText(self._params.cmap)

        self.sp_lo = QtWidgets.QDoubleSpinBox()
        self.sp_lo.setRange(0.0, 100.0)
        self.sp_lo.setDecimals(1)
        self.sp_lo.setSingleStep(0.5)
        self.sp_lo.setValue(self._params.p_lo)
        self.sp_lo.setSuffix(" %")

        self.sp_hi = QtWidgets.QDoubleSpinBox()
        self.sp_hi.setRange(0.0, 100.0)
        self.sp_hi.setDecimals(1)
        self.sp_hi.setSingleStep(0.5)
        self.sp_hi.setValue(self._params.p_hi)
        self.sp_hi.setSuffix(" %")

        self.sp_gamma = QtWidgets.QDoubleSpinBox()
        self.sp_gamma.setRange(0.1, 10.0)
        self.sp_gamma.setDecimals(2)
        self.sp_gamma.setSingleStep(0.1)
        self.sp_gamma.setValue(self._params.gamma)

        # Force dot as decimal separator regardless of system locale
        try:
            loc = QtCore.QLocale.c()
            self.sp_lo.setLocale(loc)
            self.sp_hi.setLocale(loc)
            self.sp_gamma.setLocale(loc)
        except Exception:
            pass

        self.btn_auto = QtWidgets.QPushButton("Auto")
        self.btn_auto.setToolTip("Reset exposure to 1–99% and gamma=1")

        self.lbl_path = QtWidgets.QLabel("No file")
        self.lbl_path.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self._img_label = QtWidgets.QLabel("No image")
        self._img_label.setAlignment(QtCore.Qt.AlignCenter)
        self._img_label.setMinimumSize(200, 200)
        self._img_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Cmap:"))
        top.addWidget(self.cmb_cmap)
        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel("Low:"))
        top.addWidget(self.sp_lo)
        top.addWidget(QtWidgets.QLabel("High:"))
        top.addWidget(self.sp_hi)
        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel("Gamma:"))
        top.addWidget(self.sp_gamma)
        top.addWidget(self.btn_auto)
        top.addStretch(1)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.lbl_path)
        lay.addWidget(self._img_label, 1)

        self.cmb_cmap.currentTextChanged.connect(self._on_controls)
        self.sp_lo.valueChanged.connect(self._on_controls)
        self.sp_hi.valueChanged.connect(self._on_controls)
        self.sp_gamma.valueChanged.connect(self._on_controls)
        self.btn_auto.clicked.connect(self._on_auto)

    def set_path(self, path):
        """Set current FITS file path (str or Path)."""
        if not path:
            self.clear()
            return
        self.load_fits(str(path))

    def load_fits(self, path: str):
        self._path = path
        self.lbl_path.setText(path)

        try:
            with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
                self._data = _as_float(hdul[0].data)
        except Exception as e:
            self._data = None
            self._img_label.setText(f"Failed to load FITS:\n{e}")
            return

        self._render()

    def clear(self):
        self._path = None
        self._data = None
        self.lbl_path.setText("No file")
        self._img_label.setText("No image")
        self._img_label.setPixmap(QtGui.QPixmap())

    def _on_auto(self):
        self.sp_lo.setValue(1.0)
        self.sp_hi.setValue(99.0)
        self.sp_gamma.setValue(1.0)

    def _on_controls(self, *_):
        if self._data is None:
            return
        self._render()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._data is not None:
            self._render()

    def _render(self):
        if self._data is None:
            return

        self._params.cmap = self.cmb_cmap.currentText()
        self._params.p_lo = float(self.sp_lo.value())
        self._params.p_hi = float(self.sp_hi.value())
        self._params.gamma = float(self.sp_gamma.value())

        x01 = _apply_stretch(self._data, self._params)

        try:
            if self._params.cmap == "gray":
                qimg = _to_qimage_gray(x01)
            else:
                qimg = _to_qimage_cmap(x01, self._params.cmap)
            pix = QtGui.QPixmap.fromImage(qimg)
        except Exception as e:
            self._img_label.setText(f"Render error:\n{e}")
            return

        # Scale to fit
        target = self._img_label.size()
        pix = pix.scaled(target, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self._img_label.setPixmap(pix)
        self._img_label.setText("")
