from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PySide6 import QtCore, QtGui, QtWidgets

import numpy as np
from astropy.io import fits


@dataclass(frozen=True)
class QCItem:
    title: str
    rel_path: str
    kind: str  # "image" | "fits" | "text"


DEFAULT_QC_ITEMS: list[QCItem] = [
    QCItem("Manifest (JSON)", "report/manifest.json", "text"),
    QCItem("Superbias (FITS)", "calib/superbias.fits", "fits"),
    # legacy flat layout (still supported)
    QCItem("Superneon (PNG)", "wavesol/superneon.png", "image"),
    QCItem("Peaks candidates (CSV)", "wavesol/peaks_candidates.csv", "text"),
    QCItem("Hand pairs (TXT)", "wavesol/hand_pairs.txt", "text"),
    QCItem("1D wavesolution (PNG)", "wavesol/wavesolution_1d.png", "image"),
    QCItem("1D residuals (CSV)", "wavesol/residuals_1d.csv", "text"),
    QCItem("2D wavelength matrix (PNG)", "wavesol/wavelength_matrix.png", "image"),
    QCItem("2D residuals (PNG)", "wavesol/residuals_2d.png", "image"),
    QCItem("Lambda map (FITS)", "wavesol/lambda_map.fits", "fits"),
    QCItem("2D fit summary (JSON)", "wavesol/wavesolution_2d.json", "text"),
]


def _collect_dynamic_items(work_dir: Path) -> list[QCItem]:
    """Collect QC artifacts from disperser subfolders.

    New layout: work_dir/wavesol/<slug>/...
    """
    out: list[QCItem] = []
    base = work_dir / "wavesol"
    if not base.exists():
        return out

    # add per-disperser folders as mini-sections
    for sub in sorted([p for p in base.iterdir() if p.is_dir()]):
        slug = sub.name
        # key artifacts
        candidates = [
            ("Superneon", sub / "superneon.png", "image"),
            ("Peaks", sub / "peaks_candidates.csv", "text"),
            ("Pairs", sub / "hand_pairs.txt", "text"),
            ("1D", sub / "wavesolution_1d.png", "image"),
            ("2D map", sub / "wavelength_matrix.png", "image"),
            ("2D resid", sub / "residuals_2d.png", "image"),
            ("Lambda map", sub / "lambda_map.fits", "fits"),
        ]
        for title, p, kind in candidates:
            if p.exists():
                out.append(QCItem(f"[{slug}] {title}", str(p.relative_to(work_dir)), kind))

        # any alternative pair files
        for p in sorted(sub.glob("hand_pairs*.txt")):
            if p.name == "hand_pairs.txt":
                continue
            out.append(QCItem(f"[{slug}] {p.name}", str(p.relative_to(work_dir)), "text"))

    return out


def _read_text_file(path: Path, max_chars: int = 250_000) -> str:
    try:
        s = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        s = path.read_text(encoding="latin-1", errors="replace")
    if len(s) > max_chars:
        s = s[:max_chars] + "\n\n... (truncated) ..."
    return s


def _fits_to_qpixmap(path: Path, *, w: int = 1200, h: int = 600) -> QtGui.QPixmap:
    data = fits.getdata(path).astype(float)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # percentile stretch
    lo, hi = np.percentile(data, [1.0, 99.7])
    if hi <= lo:
        lo, hi = float(np.min(data)), float(np.max(data) if np.max(data) != np.min(data) else np.min(data) + 1.0)
    norm = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    img8 = (norm * 255.0).astype(np.uint8)

    # grayscale QImage
    qimg = QtGui.QImage(img8.data, img8.shape[1], img8.shape[0], img8.strides[0], QtGui.QImage.Format_Grayscale8)
    qimg = qimg.copy()  # detach from numpy memory
    pm = QtGui.QPixmap.fromImage(qimg)
    return pm.scaled(w, h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)


class QCViewer(QtWidgets.QMainWindow):
    def __init__(self, work_dir: Path, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Scorpio Pipe — QC Viewer")
        self.resize(1200, 720)

        self._work_dir = Path(work_dir)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        lay = QtWidgets.QVBoxLayout(central)
        lay.setContentsMargins(8, 8, 8, 8)

        # toolbar
        tb = QtWidgets.QHBoxLayout()
        lay.addLayout(tb)

        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_open_folder = QtWidgets.QPushButton("Open folder")
        self.btn_copy_path = QtWidgets.QPushButton("Copy path")
        self.lbl_root = QtWidgets.QLabel(str(self._work_dir))
        self.lbl_root.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        tb.addWidget(self.btn_refresh)
        tb.addWidget(self.btn_open_folder)
        tb.addWidget(self.btn_copy_path)
        tb.addSpacing(10)
        tb.addWidget(QtWidgets.QLabel("Work dir:"))
        tb.addWidget(self.lbl_root, 1)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Horizontal)
        lay.addWidget(splitter, 1)

        # left: list
        left = QtWidgets.QWidget()
        llay = QtWidgets.QVBoxLayout(left)
        llay.setContentsMargins(0, 0, 0, 0)
        self.list_items = QtWidgets.QListWidget()
        self.list_items.setAlternatingRowColors(True)
        llay.addWidget(self.list_items, 1)
        splitter.addWidget(left)

        # right: viewer
        right = QtWidgets.QWidget()
        rlay = QtWidgets.QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)

        self.lbl_title = QtWidgets.QLabel("—")
        self.lbl_title.setStyleSheet("font-weight: 600;")
        rlay.addWidget(self.lbl_title)

        self.stack = QtWidgets.QStackedWidget()
        rlay.addWidget(self.stack, 1)

        self.img_label = QtWidgets.QLabel()
        self.img_label.setAlignment(QtCore.Qt.AlignCenter)
        self.img_label.setMinimumHeight(300)
        self.img_label.setStyleSheet("background: #111; border-radius: 8px;")
        self.stack.addWidget(self.img_label)

        self.text_view = QtWidgets.QPlainTextEdit()
        self.text_view.setReadOnly(True)
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.text_view.setFont(mono)
        self.stack.addWidget(self.text_view)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # signals
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_open_folder.clicked.connect(self._open_folder)
        self.btn_copy_path.clicked.connect(self._copy_selected_path)
        self.list_items.currentRowChanged.connect(self._show_current)

        self.refresh()

    def refresh(self) -> None:
        self.list_items.clear()
        self._items: list[tuple[QCItem, Path]] = []
        # 1) legacy flat layout
        for it in DEFAULT_QC_ITEMS:
            p = self._work_dir / it.rel_path
            if p.exists():
                self._items.append((it, p))
                self.list_items.addItem(it.title)

        # 2) new disperser subfolders
        for it in _collect_dynamic_items(self._work_dir):
            p = self._work_dir / it.rel_path
            if p.exists():
                self._items.append((it, p))
                self.list_items.addItem(it.title)
        if self.list_items.count() == 0:
            self.list_items.addItem("(No QC artifacts found yet)")
            self._items = []
        self.list_items.setCurrentRow(0)

    def _open_folder(self) -> None:
        folder = self._work_dir
        if self._items and self.list_items.currentRow() >= 0:
            _, p = self._items[self.list_items.currentRow()]
            folder = p.parent
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))

    def _copy_selected_path(self) -> None:
        if not self._items or self.list_items.currentRow() < 0:
            return
        _, p = self._items[self.list_items.currentRow()]
        QtWidgets.QApplication.clipboard().setText(str(p))

    def _show_current(self, idx: int) -> None:
        if not self._items or idx < 0 or idx >= len(self._items):
            self.lbl_title.setText("—")
            self.img_label.clear()
            self.text_view.setPlainText("")
            return

        it, p = self._items[idx]
        self.lbl_title.setText(f"{it.title} — {p.name}")

        suffix = p.suffix.lower()
        kind = it.kind
        if suffix in (".png", ".jpg", ".jpeg", ".webp"):
            kind = "image"
        elif suffix in (".fits", ".fit", ".fts"):
            kind = "fits"
        else:
            kind = "text"

        if kind == "text":
            self.stack.setCurrentWidget(self.text_view)
            self.text_view.setPlainText(_read_text_file(p))
        elif kind == "fits":
            self.stack.setCurrentWidget(self.img_label)
            pm = _fits_to_qpixmap(p)
            self.img_label.setPixmap(pm)
        else:
            self.stack.setCurrentWidget(self.img_label)
            pm = QtGui.QPixmap(str(p))
            self.img_label.setPixmap(pm.scaled(1200, 650, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
