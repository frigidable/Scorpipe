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
from scorpio_pipe.ui.async_worker import run_in_thread, RunHandle
from scorpio_pipe.ui.fs_scan import scan_tree, ScanResult
from scorpio_pipe.ui.slow_action_profiler import profile_ui_action
from scorpio_pipe.artifact_index import (
    artifact_index_cache_stamp,
    load_artifact_index,
    list_files_for_tasks,
    tasks_for_gui_stage,
)


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

        layout = QtWidgets.QVBoxLayout(self)
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
        self._scan_cache: dict[Path, tuple[float, ScanResult]] = {}

        # PERF-20.1: prefer engine-written artifact_index.json over os.walk scans.
        self._artifact_index_cache: tuple[Path, int, dict] | None = None
        # At most ONE scan job per tab index.
        # This prevents handle accumulation (and signal storms) on repeated refresh.
        self._scan_jobs_by_tab: dict[int, RunHandle[ScanResult]] = {}
        # Keep references to callbacks to be able to disconnect them on close/refresh.
        self._scan_job_callbacks: dict[int, tuple[object, object]] = {}

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        # Best-effort: cancel + disconnect any in-flight scans to avoid late UI updates
        # and keep object graph collectible after the window closes.
        self._cancel_all_scan_jobs()
        super().closeEvent(ev)

    def set_context(
        self, cfg: dict | None, *, inspect_df=None, data_dir: Path | None = None
    ) -> None:
        self._cfg = cfg
        self._inspect_df = inspect_df
        self._data_dir = data_dir
        self.refresh()

    def refresh(self) -> None:
        self.tabs.clear()

        # Cancel/mark stale any previous in-flight scans to avoid UI churn.
        # We also disconnect callbacks so the jobs do not keep references alive.
        self._cancel_all_scan_jobs(mark_stale=True)

        if self.stage_key == "project":
            self._add_project()
            return
        if not self._cfg:
            lab = QtWidgets.QLabel("Config is not loaded yet.")
            lab.setAlignment(QtCore.Qt.AlignCenter)
            self.tabs.addTab(lab, "Info")
            return

        work_dir = resolve_work_dir(self._cfg)
        with profile_ui_action(
            f"ui.stage_frames.refresh:{self.stage_key}",
            work_dir=work_dir,
            extra={"stage_key": str(self.stage_key)},
        ):
            self._last_work_dir = work_dir
            # Clear index cache when work_dir changes.
            try:
                if self._artifact_index_cache is not None and self._artifact_index_cache[0] != work_dir:
                    self._artifact_index_cache = None
            except Exception:
                self._artifact_index_cache = None

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
                self._add_scan_tab("Flatfield", stage_dir(work_dir, "flatfield"))
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
                self._add_scan_tab(
                    f"{self.stage_key} folder", stage_dir(work_dir, self.stage_key)
                )

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

        # Container: we populate it asynchronously to keep UI responsive on big runs.
        container = QtWidgets.QWidget(self)
        lay = QtWidgets.QVBoxLayout(container)
        lay.setContentsMargins(6, 6, 6, 6)

        info = QtWidgets.QLabel(f"Scanning…\n{folder}")
        info.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(info)
        tab_idx = int(self.tabs.addTab(container, name))

        # --- PERF-20.2 / PERF-72.1: prefer engine-written artifact index
        # (append-only JSONL when available; legacy JSON as fallback).
        # scan_tree remains only a fallback when the index is missing. ---
        try:
            work_dir = self._last_work_dir
            if work_dir is not None:
                idx_path, idx_mtime_ns = artifact_index_cache_stamp(work_dir)
                idx = None
                if idx_mtime_ns >= 0 and self._artifact_index_cache is not None:
                    if self._artifact_index_cache[0] == work_dir and self._artifact_index_cache[1] == idx_mtime_ns:
                        idx = self._artifact_index_cache[2]
                if idx is None and idx_mtime_ns >= 0:
                    idx = load_artifact_index(work_dir)
                    if isinstance(idx, dict):
                        self._artifact_index_cache = (work_dir, idx_mtime_ns, idx)

                if isinstance(idx, dict) and isinstance(idx.get('stages'), dict):
                    # Collect any indexed files under this folder across all stages.
                    files = []
                    try:
                        basep = folder.resolve()
                    except Exception:
                        basep = folder

                    stages_dict = idx.get('stages') or {}
                    if isinstance(stages_dict, dict):
                        for ent in stages_dict.values():
                            if not isinstance(ent, dict):
                                continue
                            fl = ent.get('files') or []
                            if not isinstance(fl, list):
                                continue
                            for rel in fl:
                                try:
                                    p = (work_dir / Path(str(rel))).resolve()
                                    if not p.exists():
                                        continue
                                    try:
                                        rp = p.resolve()
                                        if basep not in rp.parents and rp != basep:
                                            continue
                                    except Exception:
                                        pass
                                    files.append(p)
                                except Exception:
                                    continue

                    # De-dup + stable ordering
                    seen = set()
                    uniq = []
                    for p in sorted(files, key=lambda x: x.name):
                        sp = str(p)
                        if sp in seen:
                            continue
                        seen.add(sp)
                        uniq.append(p)

                    if uniq:
                        self._populate_scan_container(
                            container,
                            ScanResult(paths=uniq, truncated=False, scanned_files=len(uniq)),
                            folder,
                        )
                    else:
                        info.setText(f"No indexed files yet.\n{folder}")
                    return
        except Exception:
            # Fall back to scan_tree below when index is absent/broken.
            pass

        # Fast cache: keyed by folder mtime (best-effort).
        try:
            stamp = float(folder.stat().st_mtime)
        except Exception:
            stamp = -1.0

        cached = self._scan_cache.get(folder)
        if cached is not None and cached[0] == stamp:
            res = cached[1]
            self._populate_scan_container(container, res, folder)
            return

        def _do_scan() -> ScanResult:
            return scan_tree(folder)

        # Ensure at most one scan job per tab.
        self._cancel_scan_job_for_tab(tab_idx, mark_stale=True)

        h = run_in_thread(
            _do_scan,
            name=f"scan_tree:{folder}",
            dedupe_key=f"scan_tree:tab:{tab_idx}",
        )
        self._scan_jobs_by_tab[tab_idx] = h

        def _done(res_obj: object) -> None:
            # Ignore late results for a replaced/cancelled job.
            try:
                if self._scan_jobs_by_tab.get(tab_idx) is not h:
                    return
            except Exception:
                return

            self._detach_scan_job_callbacks(tab_idx)
            try:
                self._scan_jobs_by_tab.pop(tab_idx, None)
            except Exception:
                pass
            try:
                res = res_obj if isinstance(res_obj, ScanResult) else ScanResult(paths=[], truncated=False, scanned_files=0)
                if stamp >= 0:
                    self._scan_cache[folder] = (stamp, res)
                self._populate_scan_container(container, res, folder)
            except Exception:
                # leave the placeholder text
                pass

        def _failed(tb: str) -> None:
            try:
                if self._scan_jobs_by_tab.get(tab_idx) is not h:
                    return
            except Exception:
                return

            self._detach_scan_job_callbacks(tab_idx)
            try:
                self._scan_jobs_by_tab.pop(tab_idx, None)
            except Exception:
                pass
            try:
                info.setText(f"Scan failed:\n{folder}\n\n{tb}")
            except Exception:
                pass

        h.worker.finished.connect(_done)
        h.worker.failed.connect(_failed)
        # Store callbacks so we can disconnect on close/refresh.
        self._scan_job_callbacks[tab_idx] = (_done, _failed)

    def _detach_scan_job_callbacks(self, tab_idx: int) -> None:
        cb = self._scan_job_callbacks.pop(int(tab_idx), None)
        h = self._scan_jobs_by_tab.get(int(tab_idx))
        if cb is None or h is None:
            return
        try:
            done_cb, fail_cb = cb
            try:
                h.worker.finished.disconnect(done_cb)
            except Exception:
                pass
            try:
                h.worker.failed.disconnect(fail_cb)
            except Exception:
                pass
        except Exception:
            pass

    def _cancel_scan_job_for_tab(self, tab_idx: int, *, mark_stale: bool = False) -> None:
        tab_idx = int(tab_idx)
        h = self._scan_jobs_by_tab.get(tab_idx)
        if h is None:
            return
        # Disconnect first so the job cannot keep callbacks (and widgets) alive.
        self._detach_scan_job_callbacks(tab_idx)
        try:
            if mark_stale:
                h.mark_stale()
            else:
                h.cancel()
        except Exception:
            pass
        try:
            self._scan_jobs_by_tab.pop(tab_idx, None)
        except Exception:
            pass

    def _cancel_all_scan_jobs(self, *, mark_stale: bool = False) -> None:
        # Cancel/disconnect all in-flight scans.
        try:
            for tab_idx in list(self._scan_jobs_by_tab.keys()):
                self._cancel_scan_job_for_tab(tab_idx, mark_stale=mark_stale)
        except Exception:
            pass
        try:
            self._scan_jobs_by_tab.clear()
        except Exception:
            pass
        try:
            self._scan_job_callbacks.clear()
        except Exception:
            pass

    def _populate_scan_container(self, container: QtWidgets.QWidget, res: ScanResult, folder: Path) -> None:
        try:
            lay = container.layout()
            if lay is None:
                return
            # Clear
            while lay.count():
                it = lay.takeAt(0)
                w = it.widget()
                if w is not None:
                    w.deleteLater()

            if res.truncated:
                warn = QtWidgets.QLabel(
                    f"⚠️ Список ограничен (показано {len(res.paths)} файлов).\n"
                    f"Отсканировано файлов: {res.scanned_files}. Используйте фильтры/точечные вкладки."
                )
                warn.setWordWrap(True)
                lay.addWidget(warn)

            w = _FilesBrowser(self)
            w.set_files(res.paths)
            lay.addWidget(w)
        except Exception:
            pass

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
