from __future__ import annotations

"""Embedded PDF viewer widget for PySide6.

Design goals:
  - Mouse: scroll, pan by dragging, Ctrl+wheel zoom
  - Keyboard: PgUp/PgDn/Home/End for pages, Ctrl+/- for zoom
  - Works even when QtPdfWidgets is missing (fallback to PyMuPDF)

This widget is meant to be used inside larger dialogs (e.g. LineID GUI).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets


# ------------------------------ API ------------------------------


@dataclass
class PdfViewerState:
    page0: int = 0
    zoom_pct: int = 120


class PdfViewer(QtWidgets.QWidget):
    """A compact PDF viewer with toolbar.

    Backend priority:
      1) QtPdfWidgets (vector rendering, best UX)
      2) PyMuPDF (fitz) rendered into QGraphicsView (still full mouse/keyboard)
    """

    def __init__(self, pdf_path: str | Path, *, state: Optional[PdfViewerState] = None, parent=None):
        super().__init__(parent)
        self.pdf_path = Path(pdf_path)
        self.state = state or PdfViewerState()

        self._backend: _BackendBase
        self._build()
        self._init_backend()
        self._connect_shortcuts()

    # -------------------------- UI --------------------------

    def _build(self) -> None:
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        tb = QtWidgets.QHBoxLayout()
        tb.setContentsMargins(6, 6, 6, 6)

        self.btn_prev = QtWidgets.QToolButton(text="◀")
        self.btn_next = QtWidgets.QToolButton(text="▶")
        self.btn_fitw = QtWidgets.QToolButton(text="Fit width")
        self.btn_fitp = QtWidgets.QToolButton(text="Fit page")
        self.lbl_page = QtWidgets.QLabel("–/–")

        self.slider_zoom = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_zoom.setRange(25, 400)
        self.slider_zoom.setSingleStep(5)
        self.slider_zoom.setValue(int(self.state.zoom_pct))
        self.slider_zoom.setToolTip("Zoom (%)")

        self.lbl_zoom = QtWidgets.QLabel(f"{int(self.state.zoom_pct)}%")
        self.lbl_zoom.setMinimumWidth(44)
        self.lbl_zoom.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

        tb.addWidget(self.btn_prev)
        tb.addWidget(self.btn_next)
        tb.addWidget(self.lbl_page)
        tb.addSpacing(12)
        tb.addWidget(self.btn_fitw)
        tb.addWidget(self.btn_fitp)
        tb.addStretch(1)
        tb.addWidget(QtWidgets.QLabel("Zoom:"))
        tb.addWidget(self.slider_zoom, 1)
        tb.addWidget(self.lbl_zoom)

        tbw = QtWidgets.QWidget()
        tbw.setLayout(tb)
        lay.addWidget(tbw)

        self.container = QtWidgets.QStackedWidget()
        lay.addWidget(self.container, 1)

        self.btn_prev.clicked.connect(lambda: self.go_page(self.page0 - 1))
        self.btn_next.clicked.connect(lambda: self.go_page(self.page0 + 1))
        self.btn_fitw.clicked.connect(self.fit_width)
        self.btn_fitp.clicked.connect(self.fit_page)
        self.slider_zoom.valueChanged.connect(self.set_zoom_pct)

    # ------------------------ Backend ------------------------

    def _init_backend(self) -> None:
        """Pick the best available backend."""
        err_msgs: list[str] = []

        try:
            self._backend = _QtPdfBackend(self.pdf_path, parent=self)
            self.container.addWidget(self._backend.widget)
            self.container.setCurrentWidget(self._backend.widget)
            self._after_backend_ready()
            return
        except Exception as e:
            err_msgs.append(f"QtPdfWidgets: {e}")

        try:
            self._backend = _MuPdfBackend(self.pdf_path, parent=self)
            self.container.addWidget(self._backend.widget)
            self.container.setCurrentWidget(self._backend.widget)
            self._after_backend_ready()
            return
        except Exception as e:
            err_msgs.append(f"PyMuPDF: {e}")

        # ultimate fallback: show a helpful message + external open button
        msg = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(msg)
        v.setContentsMargins(12, 12, 12, 12)
        lbl = QtWidgets.QLabel(
            "Не удалось встроить PDF в окно.\n\n"
            "Рекомендовано установить один из вариантов:\n"
            "  • PySide6 с QtPdfWidgets (PySide6 + PySide6-Addons)\n"
            "  • или PyMuPDF (pymupdf)\n\n"
            "Диагностика:\n  - " + "\n  - ".join(err_msgs)
        )
        lbl.setWordWrap(True)
        v.addWidget(lbl)
        btn = QtWidgets.QPushButton("Открыть PDF во внешнем просмотрщике")
        btn.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.pdf_path))))
        v.addWidget(btn)
        v.addStretch(1)
        self.container.addWidget(msg)
        self.container.setCurrentWidget(msg)

        # disable toolbar actions
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.btn_fitw.setEnabled(False)
        self.btn_fitp.setEnabled(False)
        self.slider_zoom.setEnabled(False)

    def _after_backend_ready(self) -> None:
        self.go_page(int(self.state.page0))
        self.set_zoom_pct(int(self.state.zoom_pct))
        self._sync_labels()

        # propagate state from backend to toolbar
        self._backend.page_changed.connect(lambda *_: self._sync_labels())
        self._backend.zoom_changed.connect(lambda *_: self._sync_labels())

    # --------------------------- Actions ---------------------------

    @property
    def page0(self) -> int:
        return int(self._backend.page0)

    @property
    def page_count(self) -> int:
        return int(self._backend.page_count)

    @property
    def zoom_pct(self) -> int:
        return int(self._backend.zoom_pct)

    def go_page(self, page0: int) -> None:
        if not hasattr(self, "_backend"):
            return
        self._backend.go_page(page0)
        self.state.page0 = int(self._backend.page0)
        self._sync_labels()

    def set_zoom_pct(self, zoom_pct: int) -> None:
        if not hasattr(self, "_backend"):
            return
        zoom_pct = int(max(25, min(400, int(zoom_pct))))
        self._backend.set_zoom_pct(zoom_pct)
        self.state.zoom_pct = int(self._backend.zoom_pct)
        self._sync_labels()

    def zoom_in(self) -> None:
        self.set_zoom_pct(self.zoom_pct + 10)

    def zoom_out(self) -> None:
        self.set_zoom_pct(self.zoom_pct - 10)

    def fit_width(self) -> None:
        if hasattr(self, "_backend"):
            self._backend.fit_width()
            self._sync_labels()

    def fit_page(self) -> None:
        if hasattr(self, "_backend"):
            self._backend.fit_page()
            self._sync_labels()

    # ------------------------- Shortcuts -------------------------

    def _connect_shortcuts(self) -> None:
        # navigation
        QtGui.QShortcut(QtGui.QKeySequence("PgUp"), self, activated=lambda: self.go_page(self.page0 - 1))
        QtGui.QShortcut(QtGui.QKeySequence("PgDown"), self, activated=lambda: self.go_page(self.page0 + 1))
        QtGui.QShortcut(QtGui.QKeySequence("Home"), self, activated=lambda: self.go_page(0))
        QtGui.QShortcut(QtGui.QKeySequence("End"), self, activated=lambda: self.go_page(self.page_count - 1))

        # zoom
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl++"), self, activated=self.zoom_in)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+="), self, activated=self.zoom_in)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+-"), self, activated=self.zoom_out)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+0"), self, activated=self.fit_width)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+9"), self, activated=self.fit_page)

    def _sync_labels(self) -> None:
        p = self.page0 + 1 if self.page_count > 0 else 0
        n = self.page_count
        self.lbl_page.setText(f"{p}/{n}" if n else "–/–")

        z = int(self.zoom_pct)
        self.lbl_zoom.setText(f"{z}%")
        # don't spam signals: update slider only if differs
        if self.slider_zoom.value() != z:
            self.slider_zoom.blockSignals(True)
            self.slider_zoom.setValue(z)
            self.slider_zoom.blockSignals(False)

        self.btn_prev.setEnabled(self.page0 > 0)
        self.btn_next.setEnabled(self.page0 < max(0, n - 1))


# ------------------------------ Backends ------------------------------


class _BackendBase(QtCore.QObject):
    page_changed = QtCore.Signal(int)
    zoom_changed = QtCore.Signal(int)

    def __init__(self, pdf_path: Path, parent=None):
        super().__init__(parent)
        self.pdf_path = pdf_path
        self.widget: QtWidgets.QWidget
        self.page0: int = 0
        self.page_count: int = 0
        self.zoom_pct: int = 120

    def go_page(self, page0: int) -> None:
        raise NotImplementedError

    def set_zoom_pct(self, zoom_pct: int) -> None:
        raise NotImplementedError

    def fit_width(self) -> None:
        # optional
        pass

    def fit_page(self) -> None:
        # optional
        pass


# ---------------- QtPdfWidgets backend ----------------


class _PdfView(QtWidgets.QWidget):
    """Wrapper around QPdfView to add Ctrl+wheel zoom."""

    zoomRequest = QtCore.Signal(int)  # delta in pct (positive/negative)

    def __init__(self, view: QtWidgets.QWidget, parent=None):
        super().__init__(parent)
        self._view = view
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(view, 1)

        # make focus land here, so PgUp/PgDn work even after mouse click
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        if ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            # typical UX: Ctrl+wheel zooms
            delta = ev.angleDelta().y()
            step = 10 if delta > 0 else -10
            self.zoomRequest.emit(step)
            ev.accept()
            return
        return super().wheelEvent(ev)


class _QtPdfBackend(_BackendBase):
    def __init__(self, pdf_path: Path, parent=None):
        super().__init__(pdf_path, parent=parent)
        from PySide6 import QtPdf, QtPdfWidgets  # type: ignore

        if not pdf_path.is_file():
            raise FileNotFoundError(str(pdf_path))

        self._QtPdf = QtPdf
        self._QtPdfWidgets = QtPdfWidgets

        self.doc = QtPdf.QPdfDocument(parent)
        st = self.doc.load(str(pdf_path))
        if st != QtPdf.QPdfDocument.Status.Ready:
            raise RuntimeError(f"QPdfDocument load status={st}")

        self.view = QtPdfWidgets.QPdfView()
        self.view.setDocument(self.doc)

        # Wrapper adds Ctrl+wheel zoom
        self.widget = _PdfView(self.view)
        self.widget.zoomRequest.connect(lambda d: self.set_zoom_pct(self.zoom_pct + d))

        self.nav = self.view.pageNavigator()
        self.page_count = int(self.doc.pageCount())
        self.page0 = int(self.nav.currentPage()) if self.page_count else 0

        try:
            self.nav.currentPageChanged.connect(self._on_page_changed)
        except Exception:
            pass

        # Default: fit width (looks good for atlases)
        try:
            self.view.setZoomMode(QtPdfWidgets.QPdfView.ZoomMode.FitToWidth)
        except Exception:
            pass

    def _on_page_changed(self, p: int) -> None:
        self.page0 = int(p)
        self.page_changed.emit(int(p))

    def go_page(self, page0: int) -> None:
        if self.page_count <= 0:
            return
        p = max(0, min(int(page0), self.page_count - 1))
        try:
            self.nav.jump(p, QtCore.QPointF(0, 0))
        except Exception:
            self.nav.jump(p)
        self.page0 = p
        self.page_changed.emit(p)

    def set_zoom_pct(self, zoom_pct: int) -> None:
        z = max(25, min(400, int(zoom_pct)))
        self.zoom_pct = z
        try:
            self.view.setZoomMode(self._QtPdfWidgets.QPdfView.ZoomMode.Custom)
        except Exception:
            pass
        try:
            self.view.setZoomFactor(float(z) / 100.0)
        except Exception:
            # if setZoomFactor is not available, keep default mode
            pass
        self.zoom_changed.emit(z)

    def fit_width(self) -> None:
        try:
            self.view.setZoomMode(self._QtPdfWidgets.QPdfView.ZoomMode.FitToWidth)
        except Exception:
            return
        # QPdfView doesn't expose the computed zoom factor; keep slider as-is

    def fit_page(self) -> None:
        try:
            self.view.setZoomMode(self._QtPdfWidgets.QPdfView.ZoomMode.FitInView)
        except Exception:
            return


# ---------------- PyMuPDF (fitz) backend ----------------


class _MuPdfGraphicsView(QtWidgets.QGraphicsView):
    zoomRequest = QtCore.Signal(int)  # delta in pct

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing |
            QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        if ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            delta = ev.angleDelta().y()
            step = 10 if delta > 0 else -10
            self.zoomRequest.emit(step)
            ev.accept()
            return
        return super().wheelEvent(ev)


class _MuPdfBackend(_BackendBase):
    def __init__(self, pdf_path: Path, parent=None):
        super().__init__(pdf_path, parent=parent)
        if not pdf_path.is_file():
            raise FileNotFoundError(str(pdf_path))

        import fitz  # PyMuPDF

        self._fitz = fitz
        self.doc = fitz.open(str(pdf_path))
        self.page_count = int(self.doc.page_count)
        self.page0 = 0

        self.view = _MuPdfGraphicsView()
        self.scene = QtWidgets.QGraphicsScene(self.view)
        self.view.setScene(self.scene)
        self.pix_item = self.scene.addPixmap(QtGui.QPixmap())
        self.widget = self.view

        # Cache is keyed by (page, zoom_quantized, effective_dpi_quantized)
        # so moving the window between monitors doesn't reuse low-res renders.
        self._cache: dict[tuple[int, int, int], QtGui.QPixmap] = {}
        self.view.zoomRequest.connect(lambda d: self.set_zoom_pct(self.zoom_pct + d))

        # Rendering quality policy:
        #   - PyMuPDF uses a 72 dpi baseline when scale=1.0.
        #   - On modern screens this looks soft. We therefore render at the
        #     *effective screen dpi* (logical dpi × devicePixelRatio), and set
        #     the pixmap DPR so Qt shows it at the correct physical size.
        # This gives a near "native" crispness, close to vector viewers.
        self._max_pixels: int = 24_000_000  # safety cap ~24 MPix (~70 MB RGB)

    def _screen_metrics(self) -> tuple[float, float]:
        """Return (logical_dpi, device_pixel_ratio)."""
        screen = self.view.screen() or QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            return 96.0, 1.0

        try:
            logical_dpi = float(screen.logicalDotsPerInch())
        except Exception:
            logical_dpi = 96.0

        try:
            dpr = float(screen.devicePixelRatio())
        except Exception:
            dpr = float(getattr(self.view, "devicePixelRatioF", lambda: 1.0)())

        if logical_dpi <= 0:
            logical_dpi = 96.0
        if dpr <= 0:
            dpr = 1.0
        return logical_dpi, dpr

    def _render(self) -> None:
        if self.page_count <= 0:
            return

        # cache by (page, zoom) quantized to 5% to avoid explosion
        zq = int(round(self.zoom_pct / 5) * 5)
        logical_dpi, dpr = self._screen_metrics()
        eff_dpi = float(logical_dpi) * float(dpr)
        key = (int(self.page0), int(zq), int(round(eff_dpi)))
        pm = self._cache.get(key)
        if pm is None:
            p = self.doc.load_page(int(self.page0))
            r = p.rect

            # Base: 72 dpi when scale=1.0. We want ~screen DPI at 100% zoom.
            scale = (eff_dpi / 72.0) * (float(zq) / 100.0)
            scale = max(0.25, float(scale))

            # Prevent gigantic pixmaps when user zooms in a lot.
            try:
                page_area_pt2 = float(max(1.0, r.width)) * float(max(1.0, r.height))
                scale_cap = (float(self._max_pixels) / page_area_pt2) ** 0.5
                if scale > scale_cap:
                    scale = float(scale_cap)
            except Exception:
                pass

            mat = self._fitz.Matrix(scale, scale)
            pix = p.get_pixmap(matrix=mat, alpha=False)

            # Convert to QImage (copy to detach from MuPDF-owned memory).
            if pix.n >= 4:
                fmt = QtGui.QImage.Format.Format_RGBA8888
            else:
                fmt = QtGui.QImage.Format.Format_RGB888
            qimg = QtGui.QImage(pix.samples, pix.width, pix.height, pix.stride, fmt).copy()
            pm = QtGui.QPixmap.fromImage(qimg)
            try:
                pm.setDevicePixelRatio(float(dpr))
            except Exception:
                pass

            # keep cache modest: current +/- 2 pages only
            self._cache[key] = pm
            for k in list(self._cache.keys()):
                if abs(int(k[0]) - int(self.page0)) > 2:
                    self._cache.pop(k, None)

        self.pix_item.setPixmap(pm)
        # Scene should be in device-independent units.
        try:
            dpr_pm = float(pm.devicePixelRatioF())
        except Exception:
            dpr_pm = float(getattr(pm, "devicePixelRatio", lambda: 1.0)())
        if dpr_pm <= 0:
            dpr_pm = 1.0
        w = float(pm.width()) / dpr_pm
        h = float(pm.height()) / dpr_pm
        self.scene.setSceneRect(QtCore.QRectF(0.0, 0.0, w, h))

    def go_page(self, page0: int) -> None:
        if self.page_count <= 0:
            return
        p = max(0, min(int(page0), self.page_count - 1))
        if p == self.page0:
            return
        self.page0 = p
        self._render()
        self.page_changed.emit(p)

    def set_zoom_pct(self, zoom_pct: int) -> None:
        z = max(25, min(400, int(zoom_pct)))
        if z == self.zoom_pct:
            return
        self.zoom_pct = z
        self._render()
        self.zoom_changed.emit(z)

    def fit_width(self) -> None:
        # best-effort: render at current zoom, then adjust to view width
        if self.page_count <= 0:
            return
        p = self.doc.load_page(int(self.page0))
        r = p.rect
        view_w = max(1, self.view.viewport().width())
        logical_dpi, _dpr = self._screen_metrics()
        # At 100% zoom the displayed width is: r.width/72 * logical_dpi.
        base_w = (float(max(1.0, r.width)) / 72.0) * float(max(1.0, logical_dpi))
        zoom = (float(view_w) / float(max(1.0, base_w))) * 100.0
        self.set_zoom_pct(int(round(zoom)))

    def fit_page(self) -> None:
        if self.page_count <= 0:
            return
        p = self.doc.load_page(int(self.page0))
        r = p.rect
        vw = max(1, self.view.viewport().width())
        vh = max(1, self.view.viewport().height())
        logical_dpi, _dpr = self._screen_metrics()
        base_w = (float(max(1.0, r.width)) / 72.0) * float(max(1.0, logical_dpi))
        base_h = (float(max(1.0, r.height)) / 72.0) * float(max(1.0, logical_dpi))
        sx = float(vw) / float(max(1.0, base_w))
        sy = float(vh) / float(max(1.0, base_h))
        self.set_zoom_pct(int(round(min(sx, sy) * 100.0)))
