"""Lightweight FITS preview widget.

Goals ("DS9-like" baseline, without trying to re-implement DS9):

* Robust FITS loading (MEF, scaled integer data with BZERO/BSCALE/BLANK)
* Fast visual controls: colormap, percentile cut, gamma, scale function, invert
* Basic interaction: zoom (wheel), pan (drag), value-under-cursor readout

This widget is intentionally self-contained so the GUI stays responsive.
"""


from __future__ import annotations


from dataclasses import dataclass
import numpy as np
from astropy.io import fits

from PySide6 import QtCore, QtGui, QtWidgets


@dataclass
class StretchParams:
    cmap: str = "gray"
    p_lo: float = 1.0
    p_hi: float = 99.0
    gamma: float = 1.0
    scale: str = "linear"  # linear, log, sqrt, asinh
    invert: bool = False


def _as_2d_float(a: np.ndarray | None) -> np.ndarray:
    if a is None:
        return np.zeros((10, 10), dtype=np.float32)
    a = np.asarray(a)
    if a.ndim > 2:
        # common for some FITS: (1, ny, nx)
        a = np.squeeze(a)
    if a.ndim != 2:
        raise ValueError(f"Unsupported FITS image ndim={a.ndim} (expected 2D)")
    # NumPy 2.0: copy=False is *strict* and may raise if a copy is required.
    # We prefer robustness in the GUI, so we allow copies when needed.
    a = np.asarray(a, dtype=np.float32)
    return a


def _choose_image_hdu(hdul: fits.HDUList) -> tuple[int, np.ndarray]:
    """Pick the first HDU that looks like an image.

    Notes
    -----
    Some real-world FITS files can trigger exceptions when accessing ``hdu.data``
    (e.g. scaled integer data with BZERO/BSCALE + memmap=True). We must not
    swallow those specific errors, otherwise the caller cannot retry with
    ``memmap=False``.

    For other errors, we keep the last exception and surface it if no suitable
    image HDU is found, instead of hiding the root cause behind a generic
    message.
    """

    last_err: Exception | None = None

    for i, hdu in enumerate(hdul):
        try:
            data = getattr(hdu, "data", None)
            if data is None:
                continue

            # Some HDU.data objects are lazy wrappers; np.asarray may yield a
            # 0-d object array. Prefer the declared shape if available.
            try:
                a = np.asarray(data)
            except Exception as e:
                msg = str(e)
                if ("memory-mapped" in msg) or ("BZERO" in msg) or ("BSCALE" in msg) or ("BLANK" in msg):
                    raise
                last_err = e
                continue

            if a.ndim == 0 and hasattr(data, "shape"):
                shp = getattr(data, "shape", None)
                if isinstance(shp, tuple) and len(shp) >= 2:
                    try:
                        a2 = np.array(data)
                        if a2.ndim >= 2:
                            return i, a2
                    except Exception as e:
                        last_err = e
                        continue

            if a.ndim >= 2:
                return i, data

        except Exception as e:
            msg = str(e)
            if ("memory-mapped" in msg) or ("BZERO" in msg) or ("BSCALE" in msg) or ("BLANK" in msg):
                raise
            last_err = e
            continue

    # Give a more actionable error with HDU summary.
    info = ""
    try:
        import io

        s = io.StringIO()
        hdul.info(output=s)
        info = s.getvalue().strip()
    except Exception:
        info = ""

    if last_err is not None:
        msg = f"No image HDU with data found (last error: {last_err})."
    else:
        msg = "No image HDU with data found."

    if info:
        msg = msg + "\n\nHDUList.info():\n" + info

    raise ValueError(msg)


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


def _apply_scale(x01: np.ndarray, scale: str) -> np.ndarray:
    # x01 is expected in 0..1
    scale = (scale or "linear").lower().strip()
    if scale == "linear":
        return x01
    if scale == "sqrt":
        return np.sqrt(np.clip(x01, 0.0, 1.0))
    if scale == "log":
        # DS9-like behaviour: logarithmic stretch with a fixed softening constant.
        # 1000 gives a usable dynamic range for typical CCD data.
        a = 1000.0
        return np.log10(1.0 + a * np.clip(x01, 0.0, 1.0)) / np.log10(1.0 + a)
    if scale == "asinh":
        a = 10.0
        return np.arcsinh(a * np.clip(x01, 0.0, 1.0)) / np.arcsinh(a)
    # fallback
    return x01


def _apply_stretch(img: np.ndarray, params: StretchParams) -> np.ndarray:
    img = _as_2d_float(img)
    lo, hi = _safe_percentiles(img, params.p_lo, params.p_hi)
    x = (img - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)

    x = _apply_scale(x, params.scale)

    gamma = float(params.gamma)
    if gamma <= 0:
        gamma = 1.0
    # display gamma: gamma>1 => brighter (as users expect)
    x = np.power(np.clip(x, 0.0, 1.0), 1.0 / gamma)

    if params.invert:
        x = 1.0 - x
    return x


def _to_qimage_gray(x01: np.ndarray) -> QtGui.QImage:
    u8 = (np.clip(x01, 0.0, 1.0) * 255.0).astype(np.uint8)
    h, w = u8.shape
    buf = u8.tobytes()
    qimg = QtGui.QImage(buf, w, h, w, QtGui.QImage.Format_Grayscale8)
    return qimg.copy()


def _to_qimage_cmap(x01: np.ndarray, cmap_name: str) -> QtGui.QImage:
    # Lazy import: matplotlib is heavy, avoid importing unless needed
    import matplotlib.cm as cm

    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(np.clip(x01, 0.0, 1.0))  # float64 0..1, (h,w,4)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    h, w, _ = rgb.shape
    buf = rgb.tobytes()
    qimg = QtGui.QImage(buf, w, h, w * 3, QtGui.QImage.Format_RGB888)
    return qimg.copy()


class _ImageView(QtWidgets.QGraphicsView):
    """Pan/zoom image view with value-under-cursor."""

    cursorInfoChanged = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self._pix_item: QtWidgets.QGraphicsPixmapItem | None = None
        self._render_data: np.ndarray | None = None
        self._dragging = False
        self._last_pos: QtCore.QPoint | None = None

        self.setRenderHints(
            QtGui.QPainter.Antialiasing
            | QtGui.QPainter.SmoothPixmapTransform
            | QtGui.QPainter.TextAntialiasing
        )
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(25, 25, 25)))

    def set_image(self, pix: QtGui.QPixmap, render_data: np.ndarray | None) -> None:
        self.scene().clear()
        self._pix_item = self.scene().addPixmap(pix)
        self._pix_item.setZValue(0)
        self._render_data = render_data
        self.resetTransform()
        self.fitInView(self._pix_item, QtCore.Qt.KeepAspectRatio)
        self._emit_cursor_info(None)

    def clear(self) -> None:
        self.scene().clear()
        self._pix_item = None
        self._render_data = None
        self.resetTransform()
        self._emit_cursor_info(None)

    def fit_to_view(self) -> None:
        if self._pix_item is None:
            return
        self.resetTransform()
        self.fitInView(self._pix_item, QtCore.Qt.KeepAspectRatio)

    def zoom_1_to_1(self) -> None:
        if self._pix_item is None:
            return
        self.resetTransform()
        self.centerOn(self._pix_item)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self._pix_item is None:
            return
        # Standard smooth zoom.
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 0.8
        self.scale(factor, factor)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self._dragging = True
            self._last_pos = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton and self._dragging:
            self._dragging = False
            self._last_pos = None
            self.setCursor(QtCore.Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._dragging and self._last_pos is not None:
            delta = event.pos() - self._last_pos
            self._last_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
        else:
            super().mouseMoveEvent(event)

        self._emit_cursor_info(event.pos())

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        super().leaveEvent(event)
        self._emit_cursor_info(None)

    def _emit_cursor_info(self, pos: QtCore.QPoint | None) -> None:
        if self._pix_item is None or self._render_data is None or pos is None:
            self.cursorInfoChanged.emit("—")
            return

        sp = self.mapToScene(pos)
        x = int(np.floor(sp.x()))
        y = int(np.floor(sp.y()))
        h, w = self._render_data.shape
        if x < 0 or y < 0 or x >= w or y >= h:
            self.cursorInfoChanged.emit("—")
            return

        v = float(self._render_data[y, x])
        self.cursorInfoChanged.emit(f"x={x}  y={y}  value={v:.6g}")


class FitsPreviewWidget(QtWidgets.QWidget):
    """Simple FITS viewer with DS9-like baseline controls."""

    # Keep preview light: if a frame is very large, downsample for display.
    _MAX_DIM_FOR_RENDER = 2200

    def __init__(self, parent=None):
        super().__init__(parent)
        self._path: str | None = None
        self._data: np.ndarray | None = None
        self._data_disp: np.ndarray | None = None
        self._params = StretchParams()
        self._hdu_index: int | None = None
        self._fits_info: dict[str, object] | None = None

        # Controls
        self.cmb_cmap = QtWidgets.QComboBox()
        self.cmb_cmap.addItems(["gray", "viridis", "magma", "plasma", "inferno"])
        self.cmb_cmap.setCurrentText(self._params.cmap)

        self.cmb_scale = QtWidgets.QComboBox()
        self.cmb_scale.addItems(["linear", "log", "sqrt", "asinh"])
        self.cmb_scale.setCurrentText(self._params.scale)

        self.chk_invert = QtWidgets.QCheckBox("Invert")
        self.chk_invert.setChecked(self._params.invert)

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
        self.btn_auto.setToolTip("Reset stretch to 1–99%, linear, gamma=1")

        self.btn_fit = QtWidgets.QPushButton("Fit")
        self.btn_fit.setToolTip("Fit image to view")

        self.btn_1to1 = QtWidgets.QPushButton("1:1")
        self.btn_1to1.setToolTip("Reset zoom to 1:1")

        self.lbl_path = QtWidgets.QLabel("No file")
        self.lbl_path.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        # Two separate lines: stats (file-level) and cursor info (dynamic).
        self.lbl_stats = QtWidgets.QLabel("—")
        self.lbl_stats.setStyleSheet("color: #A0A0A0;")
        self.lbl_stats.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.lbl_info = QtWidgets.QLabel("—")
        self.lbl_info.setStyleSheet("color: #A0A0A0;")
        self.lbl_info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self._view = _ImageView()
        self._view.cursorInfoChanged.connect(self.lbl_info.setText)

        # Layout
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Cmap:"))
        top.addWidget(self.cmb_cmap)
        top.addSpacing(10)
        top.addWidget(QtWidgets.QLabel("Scale:"))
        top.addWidget(self.cmb_scale)
        top.addWidget(self.chk_invert)
        top.addSpacing(10)
        top.addWidget(QtWidgets.QLabel("Low:"))
        top.addWidget(self.sp_lo)
        top.addWidget(QtWidgets.QLabel("High:"))
        top.addWidget(self.sp_hi)
        top.addSpacing(10)
        top.addWidget(QtWidgets.QLabel("Gamma:"))
        top.addWidget(self.sp_gamma)
        top.addWidget(self.btn_auto)
        top.addSpacing(10)
        top.addWidget(self.btn_fit)
        top.addWidget(self.btn_1to1)
        top.addStretch(1)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.lbl_path)
        lay.addWidget(self.lbl_stats)
        lay.addWidget(self.lbl_info)
        lay.addWidget(self._view, 1)

        # Signals
        self.cmb_cmap.currentTextChanged.connect(self._on_controls)
        self.cmb_scale.currentTextChanged.connect(self._on_controls)
        self.chk_invert.toggled.connect(self._on_controls)
        self.sp_lo.valueChanged.connect(self._on_controls)
        self.sp_hi.valueChanged.connect(self._on_controls)
        self.sp_gamma.valueChanged.connect(self._on_controls)
        self.btn_auto.clicked.connect(self._on_auto)
        self.btn_fit.clicked.connect(self._view.fit_to_view)
        self.btn_1to1.clicked.connect(self._view.zoom_1_to_1)

    # ---------------------------- public API ----------------------------

    def set_path(self, path) -> None:
        """Set current FITS file path (str or Path)."""
        if not path:
            self.clear()
            return
        self.load_fits(str(path))

    def load_fits(self, path: str) -> None:
        self._path = path
        self.lbl_path.setText(path)
        self._hdu_index = None

        try:
            data = self._read_fits_image(path)
            self._data = _as_2d_float(data)
        except Exception as e:
            self._data = None
            self._data_disp = None
            self._fits_info = None
            self._view.clear()
            self.lbl_path.setText(path)
            self.lbl_stats.setText("—")
            self.lbl_info.setText(f"Failed to load FITS: {e}")
            # Put a readable message directly in the view area.
            sc = self._view.scene()
            sc.clear()
            item = sc.addText(f"Failed to load FITS:\n{e}")
            item.setDefaultTextColor(QtGui.QColor("#D0D0D0"))
            item.setPos(10, 10)
            return

        # Prepare display array (optional downsample)
        self._data_disp = self._make_display_array(self._data)

        # File-level stats/diagnostics (computed on the display array for speed).
        try:
            img = self._data_disp
            finite = img[np.isfinite(img)]
            if finite.size:
                p1 = float(np.percentile(finite, 1.0))
                p50 = float(np.percentile(finite, 50.0))
                p99 = float(np.percentile(finite, 99.0))
                vmin = float(np.min(finite))
                vmax = float(np.max(finite))
            else:
                p1 = p50 = p99 = vmin = vmax = float("nan")

            info = self._fits_info or {}
            mm = info.get("memmap_used", None)
            scale = "scaled" if info.get("has_scaling") else "unscaled"
            bzero = info.get("bzero")
            bscale = info.get("bscale")
            dtype = info.get("dtype", str(self._data.dtype))

            extra = []
            if mm is not None:
                extra.append(f"memmap={bool(mm)}")
            extra.append(scale)
            if bzero is not None or bscale is not None:
                extra.append(f"BZERO={bzero} BSCALE={bscale}")

            self.lbl_stats.setText(
                f"dtype={dtype}   min={vmin:.3g}  p01={p1:.3g}  median={p50:.3g}  p99={p99:.3g}  max={vmax:.3g}   |   "
                + "   ".join(extra)
            )
        except Exception:
            self.lbl_stats.setText("—")

        # Add a little context: HDU index, shape
        extra = ""
        if self._hdu_index is not None:
            extra += f"  [HDU {self._hdu_index}]"
        extra += f"  shape={self._data.shape[1]}×{self._data.shape[0]}"
        if self._data_disp is not None and self._data_disp.shape != self._data.shape:
            extra += f"  preview={self._data_disp.shape[1]}×{self._data_disp.shape[0]}"
        self.lbl_path.setText(path + extra)

        self._render()

    def clear(self) -> None:
        self._path = None
        self._data = None
        self._data_disp = None
        self._hdu_index = None
        self._fits_info = None
        self.lbl_path.setText("No file")
        self.lbl_stats.setText("—")
        self.lbl_info.setText("—")
        self._view.clear()

    # ---------------------------- FITS loading ----------------------------
    def _read_fits_image(self, path: str) -> np.ndarray:
        """Read FITS image robustly.

        Previewing FITS in a GUI should prefer *robustness* over maximum IO
        micro-optimizations. In practice, memmap=True can be fragile on some
        scaled integer files (BZERO/BSCALE/BLANK), and the performance gain is
        not important for typical SCORPIO frame sizes.

        Strategy:
          1) Try ``fits.open(..., memmap=False)`` and pick the first image-like HDU.
          2) If that fails, try common extension names ("SCI") and a broader ext range.
          3) As a last resort, retry with memmap=True (can help for huge files).
        """

        # Use the same loader as the pipeline to avoid "preview-only" surprises.
        from scorpio_pipe.fits_utils import read_image_smart

        arr, _hdr, info = read_image_smart(path, memmap="auto", dtype=np.float32)
        self._fits_info = info
        self._hdu_index = info.get("hdu_index") if isinstance(info, dict) else None
        return arr

    def _make_display_array(self, a: np.ndarray) -> np.ndarray:
        a = _as_2d_float(a)
        ny, nx = a.shape
        step = 1
        if max(nx, ny) > self._MAX_DIM_FOR_RENDER:
            step = int(np.ceil(max(nx, ny) / float(self._MAX_DIM_FOR_RENDER)))
            step = max(2, step)
        if step <= 1:
            return a
        return a[::step, ::step]

    # ---------------------------- rendering ----------------------------

    def _on_auto(self) -> None:
        self.sp_lo.setValue(1.0)
        self.sp_hi.setValue(99.0)
        self.sp_gamma.setValue(1.0)
        self.cmb_scale.setCurrentText("linear")
        self.chk_invert.setChecked(False)

    def _on_controls(self, *_):
        if self._data_disp is None:
            return
        self._render()

    def _render(self) -> None:
        if self._data_disp is None:
            return

        self._params.cmap = self.cmb_cmap.currentText()
        self._params.scale = self.cmb_scale.currentText()
        self._params.invert = bool(self.chk_invert.isChecked())
        self._params.p_lo = float(self.sp_lo.value())
        self._params.p_hi = float(self.sp_hi.value())
        self._params.gamma = float(self.sp_gamma.value())

        x01 = _apply_stretch(self._data_disp, self._params)

        try:
            if self._params.cmap == "gray":
                qimg = _to_qimage_gray(x01)
            else:
                qimg = _to_qimage_cmap(x01, self._params.cmap)
            pix = QtGui.QPixmap.fromImage(qimg)
        except Exception as e:
            self._view.clear()
            self.lbl_info.setText(f"Render error: {e}")
            return

        # Use the *display* data for cursor readout.
        self._view.set_image(pix, render_data=self._data_disp)
