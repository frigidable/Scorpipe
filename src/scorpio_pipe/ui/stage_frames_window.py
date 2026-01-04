from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from scorpio_pipe.inspect import KIND_ORDER
from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.workspace_paths import stage_dir
from scorpio_pipe.wavesol_paths import wavesol_dir
from scorpio_pipe.ui.fits_preview import FitsPreviewWidget
from scorpio_pipe.ui.frame_browser import FrameBrowser


@dataclass(frozen=True)
class FileItem:
    path: Path


class _FilesBrowser(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self._paths: list[Path] = []

        self.list = QtWidgets.QListWidget(self)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        self._fits_preview = FitsPreviewWidget(self)

        self._img_label = QtWidgets.QLabel(self)
        self._img_label.setAlignment(QtCore.Qt.AlignCenter)
        self._img_label.setScaledContents(True)

        self._stack = QtWidgets.QStackedWidget(self)
        self._stack.addWidget(self._fits_preview)
        self._stack.addWidget(self._img_label)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        splitter.addWidget(self.list)
        splitter.addWidget(self._stack)
        splitter.setStretchFactor(1, 1)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self.list.currentRowChanged.connect(self._on_select)

    def set_files(self, paths: list[Path]) -> None:
        self._paths = [p for p in paths if p is not None]
        self.list.clear()
        for p in self._paths:
            self.list.addItem(p.name)
        if self._paths:
            self.list.setCurrentRow(0)

    def _on_select(self, row: int) -> None:
        if row < 0 or row >= len(self._paths):
            return
        path = self._paths[row]
        suf = path.suffix.lower()
        if suf in {".fits", ".fit", ".fts"}:
            self._stack.setCurrentWidget(self._fits_preview)
            self._fits_preview.load_fits(path)
        elif suf in {".png", ".jpg", ".jpeg"}:
            self._stack.setCurrentWidget(self._img_label)
            pm = QtGui.QPixmap(str(path))
            self._img_label.setPixmap(pm)
        else:
            self._stack.setCurrentWidget(self._img_label)
            self._img_label.setText(f"No preview for: {path.name}")


class StageFramesWindow(QtWidgets.QMainWindow):
    useSetupRequested = QtCore.Signal(object)  # SelectedFrame

    """Non-modal per-stage browser for produced FITS/PNG files."""

    def __init__(self, stage_key: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.stage_key = stage_key
        self.setWindowTitle(f"Frames — {stage_key}")
        self.resize(1100, 720)
        # User preference: always open windows maximized by default.
        try:
            self.setWindowState(self.windowState() | QtCore.Qt.WindowMaximized)
        except Exception:
            pass

        self._cfg: dict | None = None
        self._inspect_df = None
        self._data_dir: Path | None = None

        self.tabs = QtWidgets.QTabWidget(self)
        self.setCentralWidget(self.tabs)

        tb = self.addToolBar("Frames")
        tb.setMovable(False)
        act_refresh = QtGui.QAction("Refresh", self)
        act_refresh.triggered.connect(self.refresh)
        tb.addAction(act_refresh)

        self._last_work_dir: Path | None = None

    def set_context(
        self, cfg: dict | None, *, inspect_df=None, data_dir: Path | None = None
    ) -> None:
        self._cfg = cfg
        self._inspect_df = inspect_df
        self._data_dir = data_dir
        self.refresh()

    def refresh(self) -> None:
        self.tabs.clear()
        if self.stage_key == "project":
            self._add_project()
            return
        if not self._cfg:
            lab = QtWidgets.QLabel("Config is not loaded yet.")
            lab.setAlignment(QtCore.Qt.AlignCenter)
            self.tabs.addTab(lab, "Info")
            return
        work_dir = resolve_work_dir(self._cfg)
        self._last_work_dir = work_dir

        if self.stage_key == "calib":
            # Prefer new layout (calibs/), keep legacy (calib/) fallback.
            sb_new = work_dir / "calibs" / "superbias.fits"
            sb_old = work_dir / "calib" / "superbias.fits"
            sf_new = work_dir / "calibs" / "superflat.fits"
            sf_old = work_dir / "calib" / "superflat.fits"
            self._add_files_tab("Superbias", [sb_new if sb_new.exists() else sb_old])
            self._add_files_tab("Superflat", [sf_new if sf_new.exists() else sf_old])

            calib_dir = work_dir / "calibs"
            if not calib_dir.exists():
                calib_dir = work_dir / "calib"
            self._add_scan_tab("All calib", calib_dir)
        elif self.stage_key == "cosmics":
            self._add_cosmics_tabs(work_dir)
        elif self.stage_key == "flatfield":
            self._add_scan_tab("Flatfield", work_dir / "flatfield")
        elif self.stage_key == "superneon":
            wsd = wavesol_dir(self._cfg)
            self._add_files_tab("Superneon", [wsd / "superneon.fits"])
            self._add_scan_tab("All wavesol products", wsd)
        elif self.stage_key == "lineid":
            wsd = wavesol_dir(self._cfg)
            self._add_files_tab("hand_pairs.txt", [wsd / "hand_pairs.txt"])
            self._add_scan_tab("Wavesol folder", wsd)
        elif self.stage_key == "wavesol":
            wsd = wavesol_dir(self._cfg)
            self._add_scan_tab("Wavesol folder", wsd)
        else:
            self._add_scan_tab("Work", work_dir)

    def _add_project(self) -> None:
        if self._inspect_df is None:
            lab = QtWidgets.QLabel("No inspection results yet. Click Inspect first.")
            lab.setAlignment(QtCore.Qt.AlignCenter)
            self.tabs.addTab(lab, "Info")
            return
        fb = FrameBrowser(self)
        # FrameBrowser expects the inspection table with absolute paths.
        # The inspect step already provides absolute `path`, so no base_dir is needed.
        fb.set_frames_df(self._inspect_df)
        fb.useSetupRequested.connect(lambda sel: self.useSetupRequested.emit(sel))
        self.tabs.addTab(fb, "Night frames")

    def _add_files_tab(self, name: str, paths: list[Path]) -> None:
        w = _FilesBrowser(self)
        w.set_files([p for p in paths if p.exists()])
        if not any(p.exists() for p in paths):
            empty = QtWidgets.QLabel("No files yet.")
            empty.setAlignment(QtCore.Qt.AlignCenter)
            self.tabs.addTab(empty, name)
            return
        self.tabs.addTab(w, name)

    def _add_scan_tab(self, name: str, folder: Path) -> None:
        folder = Path(folder)
        if not folder.exists():
            empty = QtWidgets.QLabel(f"Folder does not exist yet: {folder}")
            empty.setAlignment(QtCore.Qt.AlignCenter)
            self.tabs.addTab(empty, name)
            return
        paths = []
        for ext in ("*.fits", "*.fit", "*.fts", "*.png", "*.jpg", "*.jpeg"):
            paths.extend(sorted(folder.rglob(ext)))
        w = _FilesBrowser(self)
        w.set_files(paths)
        self.tabs.addTab(w, name)

    def _add_cosmics_tabs(self, work_dir: Path) -> None:
        kinds = []
        frames = (self._cfg or {}).get("frames") or {}
        for k in KIND_ORDER:
            if k in frames and frames.get(k):
                kinds.append(k)
        if not kinds:
            kinds = ["obj", "sky", "sunsky", "neon"]

        for k in kinds:
            canonical = stage_dir(work_dir, "cosmics") / k
            legacy = work_dir / "cosmics" / k
            base = canonical if canonical.exists() else legacy

            self._add_scan_tab(f"{k.upper()} clean", base / "clean")
            # Historical name was "masks"; current is "masks_fits".
            masks_dir = base / "masks_fits"
            if not masks_dir.exists():
                masks_dir = base / "masks"
            self._add_scan_tab(f"{k.upper()} masks", masks_dir)
            self._add_scan_tab(f"{k.upper()} outputs", base)