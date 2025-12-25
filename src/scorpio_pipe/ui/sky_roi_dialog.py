from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from astropy.io import fits

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.widgets import SpanSelector

from scorpio_pipe.plot_style import mpl_style


@dataclass
class ROI:
    obj_y0: int = 0
    obj_y1: int = 0
    sky_top_y0: int = 0
    sky_top_y1: int = 0
    sky_bot_y0: int = 0
    sky_bot_y1: int = 0


class SkyRoiDialog(QDialog):
    """Select OBJECT and SKY regions on a linearized 2D frame.

    The selection is a set of *Y ranges* spanning the full X extent:
    - object (green)
    - sky top (red)
    - sky bottom (red)
    """

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

        self.roi = ROI()
        if roi:
            # tolerate partial dict
            for k in ("obj_y0", "obj_y1", "sky_top_y0", "sky_top_y1", "sky_bot_y0", "sky_bot_y1"):
                if k in roi and roi[k] is not None:
                    setattr(self.roi, k, int(roi[k]))

        with fits.open(self.fits_path, memmap=False) as hdul:
            self.img = np.array(hdul[0].data, dtype=float)

        if self.img.ndim != 2:
            raise ValueError("ROI selection expects a 2D image")

        self.ny, self.nx = self.img.shape
        # if ROI is empty, suggest a sane default
        if (self.roi.obj_y0, self.roi.obj_y1, self.roi.sky_top_y0, self.roi.sky_top_y1, self.roi.sky_bot_y0, self.roi.sky_bot_y1) == (0, 0, 0, 0, 0, 0):
            ymid = self.ny // 2
            self.roi.obj_y0 = max(0, ymid - 20)
            self.roi.obj_y1 = min(self.ny - 1, ymid + 20)
            self.roi.sky_top_y0 = min(self.ny - 1, self.roi.obj_y1 + 30)
            self.roi.sky_top_y1 = min(self.ny - 1, self.roi.obj_y1 + 130)
            self.roi.sky_bot_y1 = max(0, self.roi.obj_y0 - 30)
            self.roi.sky_bot_y0 = max(0, self.roi.obj_y0 - 130)

        # --- UI layout
        root = QVBoxLayout(self)

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

        btns_layout.addStretch(1)
        controls.addWidget(btns_box, 0, 2, 2, 1)

        # dialog buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        root.addWidget(self.button_box)

        self._span: Optional[SpanSelector] = None
        self._patch_obj = None
        self._patch_top = None
        self._patch_bot = None

        self._sync_to_spinboxes()
        self._draw()
        self._install_selector()

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

    # ---------- drawing

    def _draw(self) -> None:
        self.ax.clear()
        with mpl_style():
            # contrast
            finite = self.img[np.isfinite(self.img)]
            if finite.size:
                vmin, vmax = np.nanpercentile(finite, [5, 99])
            else:
                vmin, vmax = None, None
            self.ax.imshow(self.img, origin="lower", aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
            self.ax.set_title("Linearized stack: select regions")
            self.ax.set_xlabel("X (wavelength bins)")
            self.ax.set_ylabel("Y (pixels)")

            # overlays
            self._patch_obj = self.ax.axhspan(self.roi.obj_y0, self.roi.obj_y1, color="#00aa00", alpha=0.18)
            self._patch_top = self.ax.axhspan(self.roi.sky_top_y0, self.roi.sky_top_y1, color="#cc0000", alpha=0.14)
            self._patch_bot = self.ax.axhspan(self.roi.sky_bot_y0, self.roi.sky_bot_y1, color="#cc0000", alpha=0.14)

        self.canvas.draw_idle()

    def _install_selector(self) -> None:
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
            self._draw()

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
        self._draw()

    def _reset_defaults(self) -> None:
        ymid = self.ny // 2
        self.roi.obj_y0 = max(0, ymid - 20)
        self.roi.obj_y1 = min(self.ny - 1, ymid + 20)
        self.roi.sky_top_y0 = min(self.ny - 1, self.roi.obj_y1 + 30)
        self.roi.sky_top_y1 = min(self.ny - 1, self.roi.obj_y1 + 130)
        self.roi.sky_bot_y1 = max(0, self.roi.obj_y0 - 30)
        self.roi.sky_bot_y0 = max(0, self.roi.obj_y0 - 130)
        self._sync_to_spinboxes()
        self._draw()
        self._install_selector()

    def _zoom_to_object(self) -> None:
        y0 = min(self.roi.obj_y0, self.roi.obj_y1)
        y1 = max(self.roi.obj_y0, self.roi.obj_y1)
        pad = max(20, int(0.3 * (y1 - y0 + 1)))
        self.ax.set_ylim(max(0, y0 - pad), min(self.ny - 1, y1 + pad))
        self.canvas.draw_idle()
