from __future__ import annotations

"""Small FITS preview utilities for the UI.

The Launcher uses these helpers to show a quicklook image and a compact header
view without pulling the whole pipeline machinery.
"""

from pathlib import Path

import numpy as np
from astropy.io import fits
from PySide6 import QtCore, QtGui, QtWidgets


def _safe_read_fits_data(path: Path) -> np.ndarray:
    """Read FITS data robustly for preview.

    For preview we prefer resilience over speed. We also downsample very large
    frames to keep the UI responsive.
    """

    try:
        data = fits.getdata(path, memmap=True)
    except Exception:
        # tolerate "tired" FITS
        with fits.open(path, memmap=False, ignore_missing_end=True, ignore_missing_simple=True) as hdul:
            data = hdul[0].data

    if data is None:
        return np.zeros((1, 1), dtype=float)
    data = np.asarray(data, dtype=float)

    # downsample if huge (keep aspect)
    ny, nx = data.shape[:2]
    max_n = 1200
    step = max(1, int(max(ny, nx) / max_n))
    if step > 1:
        data = data[::step, ::step]

    return data


def fits_to_qpixmap(path: Path, *, w: int = 900, h: int = 520) -> QtGui.QPixmap:
    data = _safe_read_fits_data(path)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # percentile stretch
    try:
        lo, hi = np.percentile(data, [1.0, 99.7])
    except Exception:
        lo, hi = float(np.min(data)), float(np.max(data))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(data)), float(np.max(data) if np.max(data) != np.min(data) else np.min(data) + 1.0)
    norm = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    img8 = (norm * 255.0).astype(np.uint8)

    qimg = QtGui.QImage(img8.data, img8.shape[1], img8.shape[0], img8.strides[0], QtGui.QImage.Format_Grayscale8)
    qimg = qimg.copy()  # detach from numpy memory
    pm = QtGui.QPixmap.fromImage(qimg)
    return pm.scaled(w, h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)


def fits_header_text(path: Path, *, key_whitelist: list[str] | None = None, max_lines: int = 250) -> str:
    try:
        hdr = fits.getheader(path)
    except Exception:
        with fits.open(path, memmap=False, ignore_missing_end=True, ignore_missing_simple=True) as hdul:
            hdr = hdul[0].header

    if key_whitelist:
        lines: list[str] = []
        for k in key_whitelist:
            if k in hdr:
                lines.append(f"{k:>10s} = {hdr.get(k)}")
        # add short tail
        rest = [k for k in hdr.keys() if k not in set(key_whitelist)]
        for k in rest[: max(0, max_lines - len(lines))]:
            try:
                lines.append(f"{k:>10s} = {hdr.get(k)}")
            except Exception:
                pass
        return "\n".join(lines[:max_lines])

    # compact full header
    txt = hdr.tostring(sep="\n", endcard=False, padding=False)
    lines = txt.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["… (truncated) …"]
    return "\n".join(lines)


class FitsPreviewWidget(QtWidgets.QWidget):
    """A quicklook preview (image + header) for a FITS file."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.lbl_title = QtWidgets.QLabel("—")
        self.lbl_title.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.lbl_title.setStyleSheet("font-weight: 600;")
        lay.addWidget(self.lbl_title)

        self.lbl_img = QtWidgets.QLabel()
        self.lbl_img.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_img.setMinimumHeight(260)
        self.lbl_img.setStyleSheet("background: #111; border-radius: 10px;")
        lay.addWidget(self.lbl_img, 1)

        self.txt_hdr = QtWidgets.QPlainTextEdit()
        self.txt_hdr.setReadOnly(True)
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.txt_hdr.setFont(mono)
        self.txt_hdr.setMaximumBlockCount(4000)
        self.txt_hdr.setMinimumHeight(180)
        lay.addWidget(self.txt_hdr)

        self._path: Path | None = None

    def clear(self) -> None:
        self._path = None
        self.lbl_title.setText("—")
        self.lbl_img.clear()
        self.txt_hdr.setPlainText("")

    def set_path(self, path: Path) -> None:
        self._path = Path(path)
        self.lbl_title.setText(str(self._path))
        try:
            self.lbl_img.setPixmap(fits_to_qpixmap(self._path))
        except Exception:
            self.lbl_img.setText("(preview failed)")
        try:
            keys = [
                "OBJECT",
                "IMAGETYP",
                "OBSTYPE",
                "EXPTIME",
                "DATE-OBS",
                "INSTRUME",
                "GRISM",
                "GRATING",
                "DISPERSER",
                "SLIT",
                "CCDSUM",
                "BINNING",
                "NAXIS1",
                "NAXIS2",
            ]
            self.txt_hdr.setPlainText(fits_header_text(self._path, key_whitelist=keys))
        except Exception:
            self.txt_hdr.setPlainText("(header read failed)")
