from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from astropy.io import fits

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.widgets import SpanSelector

from scorpio_pipe.fits_utils import read_image_smart
from scorpio_pipe.plot_style import mpl_style
from scorpio_pipe.ui.async_worker import RunHandle, run_in_thread
from scorpio_pipe.ui.mpl_render import RenderThrottle


@dataclass
class ROI:
    obj_y0: int = 0
    obj_y1: int = 0
    sky_top_y0: int = 0
    sky_top_y1: int = 0
    sky_bot_y0: int = 0
    sky_bot_y1: int = 0

    def to_dict(self) -> Dict[str, int]:
        """Serialize to a JSON/YAML-friendly dict."""
        return {
            "obj_y0": int(self.obj_y0),
            "obj_y1": int(self.obj_y1),
            "sky_top_y0": int(self.sky_top_y0),
            "sky_top_y1": int(self.sky_top_y1),
            "sky_bot_y0": int(self.sky_bot_y0),
            "sky_bot_y1": int(self.sky_bot_y1),
        }


@dataclass(frozen=True)
class RoiHeaderInfo:
    ny: int
    nx: int
    hdu_index: int
    object_name: str = ""


@dataclass(frozen=True)
class RoiImagePayload:
    image: np.ndarray
    ny: int
    nx: int
    hdu_index: int
    mode: str
    x_step: int = 1
    source_note: str = ""
    object_name: str = ""

    @property
    def is_full_resolution(self) -> bool:
        return bool(self.mode == "full")


def _find_first_image_hdu(hdul: fits.HDUList) -> tuple[int, fits.Header]:
    for i, hdu in enumerate(hdul):
        try:
            hdr = getattr(hdu, "header", None)
            if hdr is None:
                continue
            naxis = int(hdr.get("NAXIS", 0) or 0)
            nx = int(hdr.get("NAXIS1", 0) or 0)
            ny = int(hdr.get("NAXIS2", 0) or 0)
            if naxis >= 2 and nx > 0 and ny > 0:
                return int(i), hdr
        except Exception:
            continue
    raise ValueError("ROI selection expects a FITS file with a 2D image HDU")


def _open_header_only(path: str | Path) -> tuple[fits.HDUList, int, fits.Header]:
    last_err: Exception | None = None
    for kwargs in (
        {"memmap": True, "do_not_scale_image_data": True},
        {"memmap": False, "do_not_scale_image_data": True},
        {"memmap": False},
    ):
        try:
            hdul = fits.open(path, **kwargs)
            idx, hdr = _find_first_image_hdu(hdul)
            return hdul, idx, hdr
        except Exception as exc:
            last_err = exc
            try:
                hdul.close()  # type: ignore[name-defined]
            except Exception:
                pass
    if last_err is not None:
        raise last_err
    raise ValueError(f"Failed to open FITS header: {path}")


def _object_name_from_header(hdr: fits.Header) -> str:
    for key in ("OBJECT", "OBJNAME", "TARGET"):
        try:
            val = hdr.get(key)
            if val is None:
                continue
            text = str(val).strip()
            if text:
                return text
        except Exception:
            continue
    return ""


def _apply_fits_scaling(raw: np.ndarray, hdr: fits.Header) -> np.ndarray:
    arr = np.asarray(raw)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Unsupported FITS image ndim={arr.ndim} (expected 2D)")

    blank = hdr.get("BLANK", None)
    if blank is not None and np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32)
        try:
            arr[arr == float(blank)] = np.nan
        except Exception:
            pass
    else:
        arr = arr.astype(np.float32)

    bzero = hdr.get("BZERO", None)
    bscale = hdr.get("BSCALE", None)
    if bzero is not None or bscale is not None:
        arr = arr * float(bscale if bscale is not None else 1.0) + float(
            bzero if bzero is not None else 0.0
        )
    return arr


def _read_roi_header_info(path: str | Path) -> RoiHeaderInfo:
    hdul, hdu_index, hdr = _open_header_only(path)
    try:
        ny = int(hdr.get("NAXIS2") or 0)
        nx = int(hdr.get("NAXIS1") or 0)
        if ny <= 0 or nx <= 0:
            raise ValueError("Invalid NAXIS1/NAXIS2 for ROI image")
        return RoiHeaderInfo(
            ny=ny,
            nx=nx,
            hdu_index=int(hdu_index),
            object_name=_object_name_from_header(hdr),
        )
    finally:
        try:
            hdul.close()
        except Exception:
            pass


def _load_roi_quick_peek(
    path: str | Path,
    *,
    max_preview_nx: int = 1024,
) -> RoiImagePayload:
    hdul, hdu_index, hdr = _open_header_only(path)
    try:
        hdu = hdul[int(hdu_index)]
        nx = int(hdr.get("NAXIS1") or 0)
        ny = int(hdr.get("NAXIS2") or 0)
        if nx <= 0 or ny <= 0:
            raise ValueError("Invalid NAXIS1/NAXIS2 for ROI quick peek")

        step = 1
        if nx > int(max_preview_nx):
            step = max(2, int(np.ceil(nx / float(max_preview_nx))))

        mode = "section"
        try:
            sect = getattr(hdu, "section", None)
            if sect is None:
                raise AttributeError("no section")
            raw = np.asarray(sect[:, ::step])
        except Exception:
            raw_full = getattr(hdu, "data", None)
            if raw_full is None:
                raise
            raw = np.asarray(raw_full)[:, ::step]
            mode = "slice"

        img = _apply_fits_scaling(raw, hdr)
        if step <= 1:
            note = "Quick peek already uses full X resolution"
        else:
            note = f"Quick peek: every {int(step)}-th X column ({mode})"
        return RoiImagePayload(
            image=img,
            ny=ny,
            nx=nx,
            hdu_index=int(hdu_index),
            mode="quick",
            x_step=int(step),
            source_note=note,
            object_name=_object_name_from_header(hdr),
        )
    finally:
        try:
            hdul.close()
        except Exception:
            pass


def _load_roi_full_resolution(path: str | Path) -> RoiImagePayload:
    img, hdr, info = read_image_smart(path, memmap="auto", dtype=np.float32)
    arr = np.asarray(img)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Unsupported FITS image ndim={arr.ndim} (expected 2D)")
    ny, nx = map(int, arr.shape)
    return RoiImagePayload(
        image=np.asarray(arr, dtype=np.float32),
        ny=ny,
        nx=nx,
        hdu_index=int(info.get("hdu_index", 0) or 0) if isinstance(info, dict) else 0,
        mode="full",
        x_step=1,
        source_note="Full resolution",
        object_name=_object_name_from_header(hdr),
    )


class SkyRoiDialog(QDialog):
    """Select OBJECT and SKY regions on a 2D frame without blocking the GUI.

    The dialog opens immediately from header metadata and loads a *quick peek*
    asynchronously. For ROI work we keep the full Y axis intact and only thin the
    X axis in the quick preview, so ROI Y coordinates stay in detector pixels.
    Full-resolution data are loaded only on demand.
    """

    _MAX_PREVIEW_NX = 1024

    def __init__(
        self,
        fits_path: str | Path,
        roi: Optional[Dict[str, Any]] = None,
        *,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Select regions: object (green) and sky (red)")
        self.setModal(True)
        # User preference: open windows maximized by default.
        try:
            self.setWindowState(self.windowState() | Qt.WindowMaximized)
        except Exception:
            pass

        self.fits_path = Path(fits_path)
        if not self.fits_path.exists():
            raise FileNotFoundError(self.fits_path)

        self._header_info = _read_roi_header_info(self.fits_path)
        self.ny = int(self._header_info.ny)
        self.nx = int(self._header_info.nx)

        self.roi = ROI()
        if roi:
            # tolerate partial dict
            for k in (
                "obj_y0",
                "obj_y1",
                "sky_top_y0",
                "sky_top_y1",
                "sky_bot_y0",
                "sky_bot_y1",
            ):
                if k in roi and roi[k] is not None:
                    setattr(self.roi, k, int(roi[k]))

        # if ROI is empty, suggest a sane default
        if (
            self.roi.obj_y0,
            self.roi.obj_y1,
            self.roi.sky_top_y0,
            self.roi.sky_top_y1,
            self.roi.sky_bot_y0,
            self.roi.sky_bot_y1,
        ) == (0, 0, 0, 0, 0, 0):
            ymid = self.ny // 2
            self.roi.obj_y0 = max(0, ymid - 20)
            self.roi.obj_y1 = min(self.ny - 1, ymid + 20)
            self.roi.sky_top_y0 = min(self.ny - 1, self.roi.obj_y1 + 30)
            self.roi.sky_top_y1 = min(self.ny - 1, self.roi.obj_y1 + 130)
            self.roi.sky_bot_y1 = max(0, self.roi.obj_y0 - 30)
            self.roi.sky_bot_y0 = max(0, self.roi.obj_y0 - 130)

        self.img: Optional[np.ndarray] = None
        self._render_mode = "loading"
        self._render_note = "Loading quick peek…"
        self._load_job: RunHandle | None = None
        self._load_generation = 0
        self._pending_error_text: str | None = None
        self._render_throttle = RenderThrottle(self, self._draw_now, max_fps=10.0, debounce_ms=180)

        # --- UI layout
        root = QVBoxLayout(self)

        header_name = f" — {self._header_info.object_name}" if self._header_info.object_name else ""
        self.lbl_status = QLabel(
            f"Frame: {self.nx} × {self.ny} px{header_name}\nLoading quick peek…"
        )
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        root.addWidget(self.progress)

        # Matplotlib canvas
        self.fig = Figure(figsize=(7.8, 3.4), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        root.addWidget(self.canvas)

        controls = QGridLayout()
        root.addLayout(controls)

        mode_box = QGroupBox("Selection mode")
        mode_layout = QVBoxLayout(mode_box)
        self.rb_obj = QRadioButton("Object region (green)")
        self.rb_top = QRadioButton("Sky top (red)")
        self.rb_bot = QRadioButton("Sky bottom (red)")
        self.rb_obj.setChecked(True)
        mode_layout.addWidget(self.rb_obj)
        mode_layout.addWidget(self.rb_top)
        mode_layout.addWidget(self.rb_bot)

        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.rb_obj, 0)
        self.mode_group.addButton(self.rb_top, 1)
        self.mode_group.addButton(self.rb_bot, 2)
        self.mode_group.buttonClicked.connect(self._on_mode_changed)

        controls.addWidget(mode_box, 0, 0, 2, 1)

        # numeric edits (fine-tuning)
        ranges_box = QGroupBox("Y ranges (pixels)")
        ranges = QGridLayout(ranges_box)

        def _mk_row(r: int, title: str):
            ranges.addWidget(QLabel(title), r, 0)
            sb0 = QSpinBox()
            sb1 = QSpinBox()
            for sb in (sb0, sb1):
                sb.setRange(0, self.ny - 1)
                sb.setKeyboardTracking(False)
                sb.valueChanged.connect(self._sync_from_spinboxes)
            ranges.addWidget(QLabel("y0"), r, 1)
            ranges.addWidget(sb0, r, 2)
            ranges.addWidget(QLabel("y1"), r, 3)
            ranges.addWidget(sb1, r, 4)
            return sb0, sb1

        self.sb_obj0, self.sb_obj1 = _mk_row(0, "Object")
        self.sb_top0, self.sb_top1 = _mk_row(1, "Sky top")
        self.sb_bot0, self.sb_bot1 = _mk_row(2, "Sky bottom")

        controls.addWidget(ranges_box, 0, 1, 2, 1)

        # helper buttons
        btns_box = QFrame()
        btns_box.setFrameShape(QFrame.NoFrame)
        btns_layout = QVBoxLayout(btns_box)
        btns_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_reset = QPushButton("Reset defaults")
        self.btn_reset.clicked.connect(self._reset_defaults)
        btns_layout.addWidget(self.btn_reset)

        self.btn_zoom_obj = QPushButton("Zoom to object")
        self.btn_zoom_obj.clicked.connect(self._zoom_to_object)
        btns_layout.addWidget(self.btn_zoom_obj)

        self.btn_full_res = QPushButton("Full resolution")
        self.btn_full_res.setEnabled(False)
        self.btn_full_res.clicked.connect(self._request_full_resolution)
        btns_layout.addWidget(self.btn_full_res)

        btns_layout.addStretch(1)
        controls.addWidget(btns_box, 0, 2, 2, 1)

        # dialog buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        root.addWidget(self.button_box)

        self._span: Optional[SpanSelector] = None
        self._patch_obj = None
        self._patch_top = None
        self._patch_bot = None

        self._sync_to_spinboxes()
        self._request_draw()
        self._start_quick_peek_load()

    # ---------- public

    def get_roi_dict(self) -> Dict[str, int]:
        return {
            "obj_y0": int(min(self.roi.obj_y0, self.roi.obj_y1)),
            "obj_y1": int(max(self.roi.obj_y0, self.roi.obj_y1)),
            "sky_top_y0": int(min(self.roi.sky_top_y0, self.roi.sky_top_y1)),
            "sky_top_y1": int(max(self.roi.sky_top_y0, self.roi.sky_top_y1)),
            "sky_bot_y0": int(min(self.roi.sky_bot_y0, self.roi.sky_bot_y1)),
            "sky_bot_y1": int(max(self.roi.sky_bot_y0, self.roi.sky_bot_y1)),
        }

    # ---------- async loading

    def _cancel_active_load(self) -> None:
        if self._load_job is None:
            return
        try:
            self._load_job.cancel()
        except Exception:
            pass
        self._load_job = None

    def _set_busy(self, busy: bool, message: str) -> None:
        try:
            self.progress.setVisible(bool(busy))
            self.progress.setRange(0, 0)
        except Exception:
            pass
        try:
            self.lbl_status.setText(message)
        except Exception:
            pass
        try:
            self.btn_full_res.setEnabled((not busy) and (self.img is not None) and (self._render_mode != "full"))
        except Exception:
            pass

    def _start_quick_peek_load(self) -> None:
        self._start_background_load(
            mode="quick",
            title=f"Frame: {self.nx} × {self.ny} px\nLoading quick peek…",
            loader=lambda: _load_roi_quick_peek(
                self.fits_path,
                max_preview_nx=self._MAX_PREVIEW_NX,
            ),
        )

    def _request_full_resolution(self) -> None:
        if self._render_mode == "full":
            return
        self._start_background_load(
            mode="full",
            title=f"Frame: {self.nx} × {self.ny} px\nLoading full resolution…",
            loader=lambda: _load_roi_full_resolution(self.fits_path),
        )

    def _start_background_load(self, *, mode: str, title: str, loader) -> None:
        self._cancel_active_load()
        self._load_generation += 1
        generation = int(self._load_generation)
        self._set_busy(True, title)

        def _done(payload: RoiImagePayload) -> None:
            if generation != self._load_generation:
                return
            self._load_job = None
            self._apply_image_payload(payload)

        def _fail(tb: str) -> None:
            if generation != self._load_generation:
                return
            self._load_job = None
            self._set_busy(False, f"Frame: {self.nx} × {self.ny} px\nFailed to load {mode} preview")
            self._request_draw(error_text=f"Failed to load {mode} preview\n{tb.splitlines()[-1] if tb else ''}")
            if mode != "full":
                try:
                    self.btn_full_res.setEnabled(True)
                except Exception:
                    pass

        h = run_in_thread(
            loader,
            name=f"sky-roi:{mode}:{self.fits_path.name}",
            dedupe_key=f"sky-roi:{self.fits_path.resolve()}",
        )
        self._load_job = h
        h.worker.finished.connect(_done)
        h.worker.failed.connect(_fail)

    def _apply_image_payload(self, payload: RoiImagePayload) -> None:
        self.img = np.asarray(payload.image, dtype=float)
        if self.img.ndim != 2:
            raise ValueError("ROI selection expects a 2D image")
        self.ny = int(payload.ny)
        self.nx = int(payload.nx)
        self._render_mode = str(payload.mode)
        self._render_note = str(payload.source_note or "")
        object_name = str(payload.object_name or self._header_info.object_name or "")
        msg = f"Frame: {self.nx} × {self.ny} px"
        if object_name:
            msg += f" — {object_name}"
        if self._render_note:
            msg += f"\n{self._render_note}"
        self._set_busy(False, msg)
        self.btn_full_res.setEnabled(not payload.is_full_resolution and int(payload.x_step) > 1)
        self._request_draw()
        self._install_selector()

    # ---------- drawing

    def _request_draw(self, *, error_text: str | None = None, debounce: bool = False) -> None:
        self._pending_error_text = error_text
        if debounce:
            self._render_throttle.request()
        else:
            self._render_throttle.request_immediate()

    def _draw_now(self) -> None:
        error_text = self._pending_error_text
        self._pending_error_text = None
        self.ax.clear()
        with mpl_style():
            if error_text:
                self.ax.set_axis_off()
                self.ax.text(
                    0.5,
                    0.5,
                    error_text,
                    transform=self.ax.transAxes,
                    ha="center",
                    va="center",
                )
                self.canvas.draw_idle()
                return

            if self.img is None:
                self.ax.set_axis_off()
                self.ax.text(
                    0.5,
                    0.5,
                    "Loading quick peek…",
                    transform=self.ax.transAxes,
                    ha="center",
                    va="center",
                )
                self.canvas.draw_idle()
                return

            # contrast
            finite = self.img[np.isfinite(self.img)]
            if finite.size:
                vmin, vmax = np.nanpercentile(finite, [5, 99])
            else:
                vmin, vmax = None, None
            self.ax.imshow(
                self.img,
                origin="lower",
                aspect="auto",
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                extent=(0.0, float(self.nx - 1), 0.0, float(self.ny - 1)),
            )
            title = "Linearized stack: select regions"
            if self._render_mode == "quick":
                title = "Quick peek: select regions"
            elif self._render_mode == "full":
                title = "Full resolution: select regions"
            self.ax.set_title(title)
            self.ax.set_xlabel("X (detector pixels)")
            self.ax.set_ylabel("Y (pixels)")
            self.ax.set_xlim(0, max(1, self.nx - 1))
            self.ax.set_ylim(0, max(1, self.ny - 1))

            # overlays
            self._patch_obj = self.ax.axhspan(
                self.roi.obj_y0, self.roi.obj_y1, color="#00aa00", alpha=0.18
            )
            self._patch_top = self.ax.axhspan(
                self.roi.sky_top_y0, self.roi.sky_top_y1, color="#cc0000", alpha=0.14
            )
            self._patch_bot = self.ax.axhspan(
                self.roi.sky_bot_y0, self.roi.sky_bot_y1, color="#cc0000", alpha=0.14
            )

        self.canvas.draw_idle()

    def _install_selector(self) -> None:
        if self.img is None:
            return
        if self._span is not None:
            try:
                self._span.disconnect_events()
            except Exception:
                pass
            self._span = None

        mode = self.mode_group.checkedId()
        if mode == 0:
            y0, y1 = self.roi.obj_y0, self.roi.obj_y1
            color = "#00aa00"
        elif mode == 1:
            y0, y1 = self.roi.sky_top_y0, self.roi.sky_top_y1
            color = "#cc0000"
        else:
            y0, y1 = self.roi.sky_bot_y0, self.roi.sky_bot_y1
            color = "#cc0000"

        def _on_select(vmin: float, vmax: float):
            v0 = int(round(min(vmin, vmax)))
            v1 = int(round(max(vmin, vmax)))
            v0 = max(0, min(self.ny - 1, v0))
            v1 = max(0, min(self.ny - 1, v1))

            if mode == 0:
                self.roi.obj_y0, self.roi.obj_y1 = v0, v1
            elif mode == 1:
                self.roi.sky_top_y0, self.roi.sky_top_y1 = v0, v1
            else:
                self.roi.sky_bot_y0, self.roi.sky_bot_y1 = v0, v1

            self._sync_to_spinboxes()
            self._request_draw(debounce=True)

        # direction='vertical' selects along Y
        self._span = SpanSelector(
            self.ax,
            onselect=_on_select,
            direction="vertical",
            useblit=True,
            interactive=True,
            props=dict(facecolor=color, alpha=0.22),
        )
        # set current extent
        try:
            self._span.extents = (float(y0), float(y1))
        except Exception:
            pass

    # ---------- events

    def _on_mode_changed(self) -> None:
        self._install_selector()

    def _sync_to_spinboxes(self) -> None:
        # block signals to avoid feedback loops
        for sb, v in (
            (self.sb_obj0, self.roi.obj_y0),
            (self.sb_obj1, self.roi.obj_y1),
            (self.sb_top0, self.roi.sky_top_y0),
            (self.sb_top1, self.roi.sky_top_y1),
            (self.sb_bot0, self.roi.sky_bot_y0),
            (self.sb_bot1, self.roi.sky_bot_y1),
        ):
            sb.blockSignals(True)
            sb.setValue(int(v))
            sb.blockSignals(False)

    def _sync_from_spinboxes(self) -> None:
        self.roi.obj_y0 = int(self.sb_obj0.value())
        self.roi.obj_y1 = int(self.sb_obj1.value())
        self.roi.sky_top_y0 = int(self.sb_top0.value())
        self.roi.sky_top_y1 = int(self.sb_top1.value())
        self.roi.sky_bot_y0 = int(self.sb_bot0.value())
        self.roi.sky_bot_y1 = int(self.sb_bot1.value())
        self._request_draw(debounce=True)

    def _reset_defaults(self) -> None:
        ymid = self.ny // 2
        self.roi.obj_y0 = max(0, ymid - 20)
        self.roi.obj_y1 = min(self.ny - 1, ymid + 20)
        self.roi.sky_top_y0 = min(self.ny - 1, self.roi.obj_y1 + 30)
        self.roi.sky_top_y1 = min(self.ny - 1, self.roi.obj_y1 + 130)
        self.roi.sky_bot_y1 = max(0, self.roi.obj_y0 - 30)
        self.roi.sky_bot_y0 = max(0, self.roi.obj_y0 - 130)
        self._sync_to_spinboxes()
        self._request_draw()
        self._install_selector()

    def _zoom_to_object(self) -> None:
        y0 = min(self.roi.obj_y0, self.roi.obj_y1)
        y1 = max(self.roi.obj_y0, self.roi.obj_y1)
        pad = max(20, int(0.3 * (y1 - y0 + 1)))
        self.ax.set_ylim(max(0, y0 - pad), min(self.ny - 1, y1 + pad))
        self.canvas.draw_idle()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._cancel_active_load()
        super().closeEvent(event)
