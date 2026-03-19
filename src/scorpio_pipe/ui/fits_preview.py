"""Lightweight FITS preview widget.

Goals ("DS9-like" baseline, without trying to re-implement DS9):

* Robust FITS loading (MEF, scaled integer data with BZERO/BSCALE/BLANK)
* Fast visual controls: colormap, percentile cut, gamma, scale function, invert
* Basic interaction: zoom (wheel), pan (drag), value-under-cursor readout

This widget is intentionally self-contained so the GUI stays responsive.
"""

from __future__ import annotations


from dataclasses import dataclass
from pathlib import Path
import os
from collections import OrderedDict
from threading import Lock
import numpy as np
from astropy.io import fits

from PySide6 import QtCore, QtGui, QtWidgets

from scorpio_pipe.ui.async_worker import RunHandle, run_in_thread


@dataclass
class StretchParams:
    cmap: str = "gray"
    p_lo: float = 1.0
    p_hi: float = 99.0
    gamma: float = 1.0
    scale: str = "linear"  # linear, log, sqrt, asinh
    invert: bool = False


# ---------------------------- preview cache ----------------------------

# Cache key: (path, mtime_ns, size_bytes)
# Value: {"payload": dict, "bytes": int}
_PREVIEW_CACHE_LOCK = Lock()
_PREVIEW_CACHE: "OrderedDict[tuple[object, ...], dict[str, object]]" = OrderedDict()
_PREVIEW_CACHE_BYTES: int = 0

# Tunables (env override). Keep conservative to avoid bloating UI memory.
_PREVIEW_CACHE_MAX_ITEMS = int(os.environ.get("SCORPIPE_UI_PREVIEW_CACHE_ITEMS", "24"))
_PREVIEW_CACHE_MAX_MB = int(os.environ.get("SCORPIPE_UI_PREVIEW_CACHE_MB", "256"))
_PREVIEW_CACHE_MAX_BYTES = max(16, _PREVIEW_CACHE_MAX_MB) * 1024 * 1024


def _file_cache_key(path: str) -> tuple[object, ...] | None:
    try:
        st = os.stat(path)
        return (os.path.abspath(path), int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))), int(st.st_size))
    except Exception:
        return None


def _cache_get(key: tuple[object, ...]) -> dict[str, object] | None:
    with _PREVIEW_CACHE_LOCK:
        entry = _PREVIEW_CACHE.get(key)
        if not entry:
            return None
        try:
            _PREVIEW_CACHE.move_to_end(key)
        except Exception:
            pass
        payload = entry.get("payload")
        return payload if isinstance(payload, dict) else None


def _cache_put(key: tuple[object, ...], payload: dict[str, object], bytes_est: int) -> None:
    global _PREVIEW_CACHE_BYTES
    with _PREVIEW_CACHE_LOCK:
        # Update/insert
        old = _PREVIEW_CACHE.get(key)
        if old:
            try:
                _PREVIEW_CACHE_BYTES -= int(old.get("bytes", 0))
            except Exception:
                pass
            try:
                del _PREVIEW_CACHE[key]
            except Exception:
                pass

        _PREVIEW_CACHE[key] = {"payload": payload, "bytes": int(max(0, bytes_est))}
        _PREVIEW_CACHE_BYTES += int(max(0, bytes_est))

        # Evict LRU
        while (
            len(_PREVIEW_CACHE) > _PREVIEW_CACHE_MAX_ITEMS
            or _PREVIEW_CACHE_BYTES > _PREVIEW_CACHE_MAX_BYTES
        ):
            try:
                k, v = _PREVIEW_CACHE.popitem(last=False)
                _PREVIEW_CACHE_BYTES -= int(v.get("bytes", 0))
            except Exception:
                break



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
                if (
                    ("memory-mapped" in msg)
                    or ("BZERO" in msg)
                    or ("BSCALE" in msg)
                    or ("BLANK" in msg)
                ):
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
            if (
                ("memory-mapped" in msg)
                or ("BZERO" in msg)
                or ("BSCALE" in msg)
                or ("BLANK" in msg)
            ):
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


def _choose_image_hdu_by_header(hdul: fits.HDUList) -> int:
    """Pick first 2D image-like HDU without forcing full data load."""
    for i, hdu in enumerate(hdul):
        try:
            hdr = getattr(hdu, "header", None)
            if hdr is None:
                continue
            naxis = int(hdr.get("NAXIS", 0) or 0)
            if naxis < 2:
                continue
            naxis1 = hdr.get("NAXIS1")
            naxis2 = hdr.get("NAXIS2")
            if naxis1 is None or naxis2 is None:
                continue
            nx = int(naxis1)
            ny = int(naxis2)
            if nx <= 0 or ny <= 0:
                continue
            return i
        except Exception:
            continue
    # fallback (may touch data)
    i, _ = _choose_image_hdu(hdul)
    return int(i)


def _load_fits_preview_quick(path: str, max_dim: int) -> dict[str, object]:
    """Load a lightweight preview without reading the full SCI array.

    Strategy:
      * Read only header + a strided sample (every N-th row/col) via ``HDU.section``.
      * Apply BSCALE/BZERO/BLANK on the sampled cut only.
      * Fall back to the robust pipeline loader if the file is exotic.
    """

    import numpy as np

    # Open FITS with memmap; do NOT auto-scale, we apply scaling on the slice.
    mode = "section"
    hdul = fits.open(path, memmap=True, do_not_scale_image_data=True)
    try:
        hdu_i = _choose_image_hdu_by_header(hdul)
        hdu = hdul[hdu_i]
        hdr = hdu.header
        nx = int(hdr.get("NAXIS1") or 0)
        ny = int(hdr.get("NAXIS2") or 0)
        if nx <= 0 or ny <= 0:
            raise ValueError("Invalid NAXIS1/NAXIS2")

        step = 1
        if max(nx, ny) > int(max_dim):
            step = int(np.ceil(max(nx, ny) / float(max_dim)))
            step = max(2, step)

        # Read a strided view (best-effort). For some HDUs (compressed images)
        # .section may not be available.
        try:
            sect = getattr(hdu, "section", None)
            if sect is None:
                raise AttributeError("no section")
            raw = np.asarray(sect[::step, ::step])
        except Exception:
            # Fallback: this may load more, but still use slicing.
            raw_full = getattr(hdu, "data", None)
            if raw_full is None:
                raise
            raw = np.asarray(raw_full)[::step, ::step]
            mode = "slice"

        raw = np.asarray(raw)
        if raw.ndim > 2:
            raw = np.squeeze(raw)
        if raw.ndim != 2:
            raise ValueError(f"Unsupported FITS image ndim={raw.ndim} (expected 2D)")

        # Apply BLANK mask on integer arrays before scaling.
        blank = hdr.get("BLANK", None)
        if blank is not None and np.issubdtype(raw.dtype, np.integer):
            try:
                raw = raw.astype(np.float32)
                raw[raw == float(blank)] = np.nan
            except Exception:
                raw = raw.astype(np.float32)
        else:
            raw = raw.astype(np.float32)

        bzero = hdr.get("BZERO", None)
        bscale = hdr.get("BSCALE", None)
        has_scaling = (bzero is not None) or (bscale is not None)
        bzero_f = float(bzero) if bzero is not None else 0.0
        bscale_f = float(bscale) if bscale is not None else 1.0
        if has_scaling:
            raw = raw * bscale_f + bzero_f

        data_disp = _as_2d_float(raw)

        finite = data_disp[np.isfinite(data_disp)]
        if finite.size:
            p1 = float(np.percentile(finite, 1.0))
            p50 = float(np.percentile(finite, 50.0))
            p99 = float(np.percentile(finite, 99.0))
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
        else:
            p1 = p50 = p99 = vmin = vmax = float("nan")

        info = {
            "hdu_index": int(hdu_i),
            "memmap_used": True,
            "has_scaling": bool(has_scaling),
            "bzero": bzero,
            "bscale": bscale,
            "dtype_raw": str(raw.dtype),
            "dtype": "float32",
            "full_shape": (int(ny), int(nx)),
            "preview_shape": (int(data_disp.shape[0]), int(data_disp.shape[1])),
            "preview_step": int(step),
            "preview_mode": str(mode),
        }

        return {
            "data": data_disp,
            "data_disp": data_disp,
            "info": info,
            "stats": {"min": vmin, "p1": p1, "p50": p50, "p99": p99, "max": vmax},
        }

    finally:
        try:
            hdul.close()
        except Exception:
            pass



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
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
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

        # Async loading state
        self._load_id: int = 0
        self._load_job: RunHandle | None = None
        self._last_cache_key: tuple[object, ...] | None = None

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
        """Load and display a FITS file without blocking the UI."""

        # Best-effort: mark previous job stale to avoid queue storms.
        try:
            if self._load_job is not None:
                self._load_job.mark_stale()
        except Exception:
            pass

        # Ignore previous jobs by bumping load_id.
        self._load_id += 1
        my_id = self._load_id
        self._path = path
        self.lbl_path.setText(path)
        self._hdu_index = None

        # Immediate placeholder.
        self._data = None
        self._data_disp = None
        self._fits_info = None
        self._view.clear()
        self.lbl_stats.setText("Loading…")
        self.lbl_info.setText("—")

        # Cache: switching between files should not re-read huge arrays.
        cache_key = _file_cache_key(path)
        self._last_cache_key = cache_key

        def _apply(res_obj: object) -> None:
            if my_id != self._load_id:
                return
            from scorpio_pipe.ui.slow_action_profiler import profile_ui_action

            with profile_ui_action(
                "ui.fits_preview.apply",
                path_hint=str(path),
                extra={"path": str(path), "cache_hit": bool(res_obj is not None and res_obj is _cached_payload)},
            ):
                try:
                    d = res_obj if isinstance(res_obj, dict) else {}
                    self._data = d.get("data")
                    self._data_disp = d.get("data_disp")
                    info = d.get("info")
                    self._fits_info = info if isinstance(info, dict) else None
                    try:
                        self._hdu_index = (
                            int(self._fits_info.get("hdu_index"))
                            if self._fits_info and self._fits_info.get("hdu_index") is not None
                            else None
                        )
                    except Exception:
                        self._hdu_index = None

                    st = d.get("stats") if isinstance(d.get("stats"), dict) else {}
                    mm = (
                        (self._fits_info or {}).get("memmap_used", None)
                        if self._fits_info
                        else None
                    )
                    scale = (
                        "scaled"
                        if (self._fits_info or {}).get("has_scaling")
                        else "unscaled"
                    )
                    bzero = (self._fits_info or {}).get("bzero") if self._fits_info else None
                    bscale = (self._fits_info or {}).get("bscale") if self._fits_info else None
                    dtype = (self._fits_info or {}).get("dtype") if self._fits_info else None
                    dtype = dtype or (
                        str(self._data.dtype) if self._data is not None else "unknown"
                    )

                    extra = []
                    if mm is not None:
                        extra.append(f"memmap={bool(mm)}")
                    extra.append(scale)
                    if bzero is not None or bscale is not None:
                        extra.append(f"BZERO={bzero} BSCALE={bscale}")

                    # Quick preview info
                    try:
                        mode = (self._fits_info or {}).get("preview_mode")
                        step = (self._fits_info or {}).get("preview_step")
                        if mode:
                            extra.append(f"preview={mode}")
                        if step and int(step) > 1:
                            extra.append(f"step={int(step)}")
                    except Exception:
                        pass

                    self.lbl_stats.setText(
                        f"dtype={dtype}   min={st.get('min', float('nan')):.3g}  p01={st.get('p1', float('nan')):.3g}  "
                        f"median={st.get('p50', float('nan')):.3g}  p99={st.get('p99', float('nan')):.3g}  "
                        f"max={st.get('max', float('nan')):.3g}   |   " + "   ".join(extra)
                    )

                    extra2 = ""
                    if self._hdu_index is not None:
                        extra2 += f"  [HDU {self._hdu_index}]"

                    # Prefer full_shape from info (true file shape), fall back to loaded array.
                    full_shape = None
                    try:
                        fs = (self._fits_info or {}).get("full_shape")
                        if isinstance(fs, (tuple, list)) and len(fs) == 2:
                            full_shape = (int(fs[0]), int(fs[1]))
                    except Exception:
                        full_shape = None

                    if full_shape is not None:
                        extra2 += f"  shape={full_shape[1]}×{full_shape[0]}"
                    elif self._data is not None:
                        extra2 += f"  shape={self._data.shape[1]}×{self._data.shape[0]}"

                    if (
                        self._data is not None
                        and self._data_disp is not None
                        and full_shape is not None
                        and self._data_disp.shape != full_shape
                    ):
                        extra2 += f"  preview={self._data_disp.shape[1]}×{self._data_disp.shape[0]}"
                    elif (
                        self._data is not None
                        and self._data_disp is not None
                        and self._data_disp.shape != self._data.shape
                    ):
                        extra2 += f"  preview={self._data_disp.shape[1]}×{self._data_disp.shape[0]}"

                    self.lbl_path.setText(path + extra2)

                    self._render()
                except Exception as e:
                    self._view.clear()
                    self.lbl_stats.setText("—")
                    self.lbl_info.setText(f"Render error: {e}")
                    self.lbl_path.setText(path)

        def _fail(tb: str) -> None:
            if my_id != self._load_id:
                return
            self._data = None
            self._data_disp = None
            self._fits_info = None
            self._view.clear()
            self.lbl_path.setText(path)
            self.lbl_stats.setText("—")
            self.lbl_info.setText("—")
            sc = self._view.scene()
            sc.clear()
            item = sc.addText("Failed to load FITS:\n" + tb.splitlines()[-1])
            item.setDefaultTextColor(QtGui.QColor("#D0D0D0"))
            item.setPos(10, 10)

        # Try cache hit (apply in next tick to keep UI consistent).
        _cached_payload = _cache_get(cache_key) if cache_key else None
        if _cached_payload is not None:
            QtCore.QTimer.singleShot(0, lambda: _apply(_cached_payload))
            return

        def _job() -> object:
            payload = _load_fits_preview_quick(path, self._MAX_DIM_FOR_RENDER)
            try:
                dd = payload.get("data_disp") if isinstance(payload, dict) else None
                bytes_est = int(getattr(dd, "nbytes", 0)) if dd is not None else 0
                if cache_key is not None and isinstance(payload, dict):
                    _cache_put(cache_key, payload, bytes_est)
            except Exception:
                pass
            return payload

        self._load_job = run_in_thread(
            _job,
            name=f"fits-preview:{Path(path).name}",
            dedupe_key=f"fits-preview:{os.path.abspath(path)}",
        )
        self._load_job.worker.finished.connect(_apply)
        self._load_job.worker.failed.connect(_fail)

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

        from scorpio_pipe.ui.slow_action_profiler import profile_ui_action

        with profile_ui_action(
            "ui.fits_preview.render",
            path_hint=str(self._path or ""),
            extra={
                "path": str(self._path or ""),
                "cmap": str(self.cmb_cmap.currentText()),
                "scale": str(self.cmb_scale.currentText()),
            },
        ):
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