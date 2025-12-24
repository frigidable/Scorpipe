from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from PySide6 import QtCore, QtGui, QtWidgets

from scorpio_pipe.inspect import inspect_dataset, InspectResult
from scorpio_pipe.autocfg import build_autoconfig
from scorpio_pipe.config import load_config
from scorpio_pipe.ui.qt_log import install as install_qt_log
from scorpio_pipe.ui.pipeline_runner import (
    load_context,
    run_sequence,
    run_lineid_prepare,
    run_wavesolution,
)
from scorpio_pipe.ui.qc_viewer import QCViewer
from scorpio_pipe.ui.pair_rejector import clean_pairs_interactively
from scorpio_pipe.ui.frame_browser import FrameBrowser, SelectedFrame
from scorpio_pipe.ui.outputs_panel import OutputsPanel
from scorpio_pipe.ui.inspector_dock import InspectorDock
from scorpio_pipe.ui.run_plan_dialog import RunPlanDialog
from scorpio_pipe.ui.config_diff import ConfigDiffDialog
from scorpio_pipe.wavesol_paths import slugify_disperser, wavesol_dir
from scorpio_pipe.pairs_library import (
    list_pair_sets,
    find_builtin_pairs_for_disperser,
    save_user_pair_set,
    copy_pair_set_to_workdir,
    user_pairs_root,
    export_pair_set,
    export_user_library_zip,
    export_user_library_folder,
)
from scorpio_pipe.ui.theme import apply_theme, load_ui_settings
from scorpio_pipe.instrument_db import find_grism, load_instrument_db


# --------------------------- tiny widgets ---------------------------


class HelpButton(QtWidgets.QToolButton):
    def __init__(self, text_ru: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._text_ru = text_ru
        self.setText("?")
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFixedSize(22, 22)
        self.setToolTip("Описание")
        self.clicked.connect(self._show)

    def _show(self) -> None:
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), self._text_ru, self)


@dataclass(frozen=True)
class ParamSpec:
    key: str
    label_en: str
    help_ru: str
    kind: str  # "int" | "float"
    rng: tuple[float, float]
    step: float


def _box(title: str) -> QtWidgets.QGroupBox:
    g = QtWidgets.QGroupBox(title)
    g.setFlat(False)
    return g


def _hline() -> QtWidgets.QFrame:
    fr = QtWidgets.QFrame()
    fr.setFrameShape(QtWidgets.QFrame.HLine)
    fr.setFrameShadow(QtWidgets.QFrame.Sunken)
    return fr


def _safe_parse_yaml(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        obj = yaml.safe_load(text) or {}
        if not isinstance(obj, dict):
            return None, "YAML root must be a mapping (dict)"
        return obj, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _yaml_dump(cfg: dict[str, Any]) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)


def _rel_to_workdir(work_dir: Path, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(work_dir.resolve()))
    except Exception:
        return str(p)


def _detect_pipeline_root() -> Path:
    """Return a writable app root directory.

    - Frozen (PyInstaller): directory of the executable
    - Source/editable: project root (folder containing pyproject.toml)
    """
    try:
        if getattr(sys, "frozen", False):
            # In a Windows installer build the executable usually lives under
            # Program Files which is not writable for a normal user.
            # Prefer the install folder for the default workspace suggestion.
            # If it is not writable, we will fall back in _suggest_work_dir().
            return Path(sys.executable).resolve().parent
    except Exception:
        pass

    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "pyproject.toml").is_file() or (parent / "scripts" / "windows" / "setup.bat").is_file():
            return parent
    return Path.cwd().resolve()


# --------------------------- main window ---------------------------


class LauncherWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        try:
            from scorpio_pipe.version import PIPELINE_VERSION
            self.setWindowTitle(f"Scorpio Pipe v{PIPELINE_VERSION}")
        except Exception:
            self.setWindowTitle("Scorpio Pipe")
        self.resize(1240, 780)

        self._settings = load_ui_settings()

        self._pipeline_root = _detect_pipeline_root()

        # Status bar: always show the writable app root (important for frozen builds).
        try:
            self.statusBar().showMessage(f"App root: {self._pipeline_root}")
        except Exception:
            pass

        self._inspect: InspectResult | None = None
        self._cfg_path: Path | None = None
        self._cfg: dict[str, Any] | None = None
        self._yaml_saved_text: str = ""
        self._qc: QCViewer | None = None
        self._yaml_saved_text: str = ""

        # -------------- central layout --------------
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        # Split main
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        outer.addWidget(splitter, 1)

        # Left: steps
        self.steps = QtWidgets.QListWidget()
        self.steps.setFixedWidth(270)

        # Step list with status icons (commercial UX: you always see progress).
        self._step_items = []  # list[QtWidgets.QListWidgetItem]
        for title in [
            "1  Project & data",
            "2  Config & setup",
            "3  Calibrations",
            "4  Clean Cosmics",
            "5  Flat-fielding (optional)",
            "6  SuperNeon",
            "7  Line ID",
            "8  Wavelength solution",
        ]:
            it = QtWidgets.QListWidgetItem(title)
            it.setIcon(self._icon_status("idle"))
            self.steps.addItem(it)
            self._step_items.append(it)
        self.steps.setCurrentRow(0)
        splitter.addWidget(self.steps)

        # Right: pages
        self.stack = QtWidgets.QStackedWidget()
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Create pages
        self.page_project = self._build_page_project()
        self.page_config = self._build_page_config()
        self.page_calib = self._build_page_calib()
        self.page_cosmics = self._build_page_cosmics()
        self.page_flatfield = self._build_page_flatfield()
        self.page_superneon = self._build_page_superneon()
        self.page_lineid = self._build_page_lineid()
        self.page_wavesol = self._build_page_wavesol()
        for p in [
            self.page_project,
            self.page_config,
            self.page_calib,
            self.page_cosmics,
            self.page_flatfield,
            self.page_superneon,
            self.page_lineid,
            self.page_wavesol,
        ]:
            self.stack.addWidget(p)

        self.steps.currentRowChanged.connect(self._on_step_changed)

        # Dock: log (collapsible)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.dock_log = QtWidgets.QDockWidget("Log", self)
        self.dock_log.setObjectName("dock_log")
        self.dock_log.setWidget(self.log_view)
        self.dock_log.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        # Keep log panel stable: resize only, no float/close/move
        self.dock_log.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.dock_log.setFloating(False)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.dock_log)
        self.dock_log.setMinimumHeight(160)

        # Dock: inspector (always visible on the right)
        self.dock_inspector = InspectorDock(self)
        self.dock_inspector.panel.openQCRequested.connect(self._open_qc_viewer_for)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dock_inspector)

        # Attach log handler
        # Route Python logging into the UI log panel.
        install_qt_log(self.log_view, logger_name="scorpio_pipe")

        try:
            import logging as _logging
            from scorpio_pipe.version import PIPELINE_VERSION

            _logging.getLogger("scorpio_pipe").info("Scorpio Pipe v%s", PIPELINE_VERSION)
        except Exception:
            pass

        self.statusBar().showMessage("Ready")

        # -------------- menu / toolbar --------------
        self._build_menus()
        self._build_toolbar()

        self._build_statusbar_widgets()
        self._install_shortcuts()

        # Initial state
        self._update_enables()
        # refresh derived UI state
        try:
            self._update_setup_hint()
        except Exception:
            pass
        try:
            self._refresh_pair_sets_combo()
            self._refresh_pairs_label()
        except Exception:
            pass
        try:
            if hasattr(self, 'lbl_wavesol_dir') and self._cfg:
                self.lbl_wavesol_dir.setText(f"wavesol: {wavesol_dir(self._cfg)}")
        except Exception:
            pass

    # --------------------------- step status ---------------------------

    _STATUS_ICONS: dict[str, QtGui.QIcon] | None = None

    def _icon_status(self, status: str) -> QtGui.QIcon:
        """Small colored status dot used in the step list."""
        status = (status or "idle").strip().lower()
        if self._STATUS_ICONS is None:
            self._STATUS_ICONS = {}
            def _dot(color: QtGui.QColor) -> QtGui.QIcon:
                pm = QtGui.QPixmap(14, 14)
                pm.fill(QtCore.Qt.transparent)
                p = QtGui.QPainter(pm)
                p.setRenderHint(QtGui.QPainter.Antialiasing, True)
                p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 60), 1.0))
                p.setBrush(color)
                p.drawEllipse(1, 1, 12, 12)
                p.end()
                return QtGui.QIcon(pm)
            self._STATUS_ICONS.update({
                "idle": _dot(QtGui.QColor(140, 140, 140)),
                "running": _dot(QtGui.QColor(47, 111, 237)),
                "ok": _dot(QtGui.QColor(46, 160, 67)),
                "warn": _dot(QtGui.QColor(230, 159, 0)),
                "fail": _dot(QtGui.QColor(220, 50, 47)),
            })
        return self._STATUS_ICONS.get(status, self._STATUS_ICONS["idle"])

    def _set_step_status(self, idx: int, status: str) -> None:
        try:
            if hasattr(self, "_step_items") and 0 <= idx < len(self._step_items):
                it = self._step_items[idx]
                it.setIcon(self._icon_status(status))
        except Exception:
            pass

    def _open_in_explorer(self, path: Path) -> None:
        try:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))
        except Exception as e:
            self._log_exception(e)



    # --------------------------- page: project ---------------------------

    def _build_page_project(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setSpacing(12)

        g = _box("Project")
        lay.addWidget(g)
        gl = QtWidgets.QFormLayout(g)
        gl.setLabelAlignment(QtCore.Qt.AlignLeft)

        # Data dir
        row = QtWidgets.QHBoxLayout()
        self.edit_data_dir = QtWidgets.QLineEdit()
        self.edit_data_dir.setPlaceholderText("Folder with FITS frames")
        self.btn_pick_data_dir = QtWidgets.QToolButton(text="…")
        self.btn_pick_data_dir.setCursor(QtCore.Qt.PointingHandCursor)
        row.addWidget(self.edit_data_dir, 1)
        row.addWidget(self.btn_pick_data_dir)
        gl.addRow("Data directory", row)

        # Config path
        row2 = QtWidgets.QHBoxLayout()
        self.edit_cfg_path = QtWidgets.QLineEdit()
        self.edit_cfg_path.setPlaceholderText("config.yaml (will be created in Work dir)")
        self.btn_open_cfg = QtWidgets.QToolButton(text="Open…")
        self.btn_open_cfg.setCursor(QtCore.Qt.PointingHandCursor)
        row2.addWidget(self.edit_cfg_path, 1)
        row2.addWidget(self.btn_open_cfg)
        gl.addRow("Config file", row2)

        # Inspect + quick preview (Frames Browser for this stage)
        self.btn_inspect = QtWidgets.QPushButton("Inspect dataset")
        self.btn_inspect.setProperty("primary", True)
        self.btn_frames_project = QtWidgets.QPushButton("Frames…")
        self.btn_frames_project.setToolTip("Open Frames Browser for the Project stage")
        self.lbl_inspect = QtWidgets.QLabel("—")
        self.lbl_inspect.setWordWrap(True)

        row_ins = QtWidgets.QHBoxLayout()
        row_ins.addWidget(self.btn_inspect)
        row_ins.addWidget(self.btn_frames_project)
        row_ins.addStretch(1)
        # QFormLayout.addRow() does not accept (QLayout, QWidget).
        # Put the action buttons on their own full-width row, then the status line below.
        gl.addRow(row_ins)
        gl.addRow(self.lbl_inspect)

        # Overview (filled after Inspect)
        self.g_overview = _box("Dataset overview")
        lay.addWidget(self.g_overview)
        ovl = QtWidgets.QHBoxLayout(self.g_overview)
        self.lbl_overview_counts = QtWidgets.QLabel("Run Inspect to see summary…")
        self.lbl_overview_counts.setWordWrap(True)
        ovl.addWidget(self.lbl_overview_counts, 2)
        # Objects list (multi-select for batch operations)
        right = QtWidgets.QVBoxLayout()
        self.list_overview_objects = QtWidgets.QListWidget()
        self.list_overview_objects.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list_overview_objects.setMaximumHeight(200)
        self.list_overview_objects.setToolTip("Double-click an object to select it for setup. Use multi-select for batch.")
        right.addWidget(self.list_overview_objects, 1)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_batch_configs = QtWidgets.QPushButton("Batch: build configs")
        self.btn_batch_run = QtWidgets.QPushButton("Batch: run")
        btn_row.addWidget(self.btn_batch_configs)
        btn_row.addWidget(self.btn_batch_run)
        right.addLayout(btn_row)

        ovl.addLayout(right, 1)
        self.list_overview_objects.itemDoubleClicked.connect(self._jump_to_object_from_overview)
        self.btn_batch_configs.clicked.connect(self._batch_build_configs)
        self.btn_batch_run.clicked.connect(self._batch_run)

        # (Frames Browser is exposed via a compact button and a global toolbar action.)

        # Actions
        act = QtWidgets.QHBoxLayout()
        lay.addLayout(act)
        self.btn_to_config = QtWidgets.QPushButton("Go to Config →")
        act.addStretch(1)
        act.addWidget(self.btn_to_config)
        # no stretch: the frame browser gets the remaining vertical space

        # signals
        self.btn_pick_data_dir.clicked.connect(self._pick_data_dir)
        self.btn_open_cfg.clicked.connect(self._open_existing_cfg)
        self.btn_inspect.clicked.connect(self._do_inspect)
        self.btn_frames_project.clicked.connect(lambda: self._open_frames_window('project'))
        self.btn_to_config.clicked.connect(lambda: self.steps.setCurrentRow(1))
        self.edit_data_dir.textChanged.connect(lambda *_: self._update_enables())
        self.edit_data_dir.textChanged.connect(lambda *_: self._refresh_statusbar())

        return w

    def _pick_data_dir(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select data directory", str(Path.home()))
        if d:
            self.edit_data_dir.setText(d)

    def _do_inspect(self) -> None:
        data_dir = Path(self.edit_data_dir.text()).expanduser()
        if not data_dir.exists():
            self._log_error("Data directory does not exist")
            return
        self._set_step_status(0, "running")
        try:
            self._log_info(f"Inspect: {data_dir}")
            self._inspect = inspect_dataset(data_dir)
            msg = f"FITS opened/found: {self._inspect.n_opened}/{self._inspect.n_found}"
            if self._inspect.nightlog_path:
                msg += f"\nNightlog: {self._inspect.nightlog_path} (rows={self._inspect.n_nightlog_rows})"
            if self._inspect.open_errors:
                msg += "\nOpen errors (first):\n" + "\n".join(self._inspect.open_errors)
            self.lbl_inspect.setText(msg)
            self._populate_objects_from_inspect()
            self._refresh_overview_from_inspect()
            # If work dir is still empty, auto-suggest a sensible default.
            try:
                if (not self.edit_work_dir.text().strip()) and (not getattr(self, "_workdir_user_edited", False)):
                    self._suggest_work_dir()
            except Exception:
                pass
            self._set_step_status(0, "ok")
        except Exception as e:
            self._set_step_status(0, "fail")
            self._log_exception(e)
    def _refresh_overview_from_inspect(self) -> None:
        """Fill the Project page overview panel from InspectResult."""
        try:
            if self._inspect is None:
                self.lbl_overview_counts.setText("—")
                self.list_overview_objects.clear()
                return
            df = getattr(self._inspect, 'table', None)
            if df is None or df.empty:
                self.lbl_overview_counts.setText("No frames found")
                self.list_overview_objects.clear()
                return

            # Counts by kind
            vc = df['kind'].value_counts(dropna=False).to_dict() if 'kind' in df.columns else {}
            total = int(len(df))
            lines = [f"Total frames: {total}"]
            for k in ['obj', 'sky', 'sunsky', 'neon', 'flat', 'bias']:
                lines.append(f"{k}: {int(vc.get(k, 0))}")

            # Quick setup diversity hints (dispersers/slits/binning)
            def _uniq(col: str) -> int:
                try:
                    if col in df.columns:
                        return int(df[col].dropna().astype(str).nunique())
                except Exception:
                    return 0
                return 0

            lines.append('')
            lines.append(f"Dispersers: {_uniq('disperser')}  |  Slits: {_uniq('slit')}  |  Binning: {_uniq('binning')}")

            self.lbl_overview_counts.setText('\n'.join(lines))

            # Objects list with counts
            self.list_overview_objects.blockSignals(True)
            self.list_overview_objects.clear()
            if 'kind' in df.columns and 'object' in df.columns:
                df_obj = df[df['kind'] == 'obj']
                if not df_obj.empty:
                    g = df_obj.groupby('object').size().sort_values(ascending=False)
                    for obj, n in g.items():
                        it = QtWidgets.QListWidgetItem(f"{obj}  ({int(n)})")
                        it.setData(QtCore.Qt.ItemDataRole.UserRole, str(obj))
                        self.list_overview_objects.addItem(it)
            self.list_overview_objects.blockSignals(False)
        except Exception as e:
            self._log_exception(e)

    def _jump_to_object_from_overview(self, item: QtWidgets.QListWidgetItem) -> None:
        """Select the object in Config page when user double-clicks it in overview."""
        try:
            obj = item.data(QtCore.Qt.ItemDataRole.UserRole) or item.text()
            obj = str(obj).split('  (')[0].strip()
            if hasattr(self, 'combo_object'):
                idx = self.combo_object.findText(obj)
                if idx >= 0:
                    self.combo_object.setCurrentIndex(idx)
                else:
                    self.combo_object.setCurrentText(obj)
            self.steps.setCurrentRow(1)
        except Exception as e:
            self._log_exception(e)

    def _selected_overview_objects(self) -> list[str]:
        """Get selected objects from the overview list (unique, in visual order)."""
        try:
            items = self.list_overview_objects.selectedItems() if hasattr(self, 'list_overview_objects') else []
        except Exception:
            items = []
        out: list[str] = []
        for it in items:
            try:
                obj = it.data(QtCore.Qt.ItemDataRole.UserRole) or it.text()
                obj = str(obj).split('  (')[0].strip()
                if obj and obj not in out:
                    out.append(obj)
            except Exception:
                continue
        return out

    def _batch_build_configs(self) -> None:
        if self._inspect is None:
            self._log_error('Run Inspect first')
            return
        objs = self._selected_overview_objects()
        if not objs:
            self._log_error('Select one or more objects in Dataset overview')
            return

        from scorpio_pipe.workdir import RunSignature, pick_smart_run_dir
        data_dir = Path(self.edit_data_dir.text()).expanduser()
        root = getattr(self, '_pipeline_root', None) or (data_dir.parent if data_dir.exists() else Path.home())
        dmy = self._infer_night_date_parts()
        if dmy is None:
            now = datetime.now()
            dd, mm, yyyy = int(now.day), int(now.month), int(now.year)
        else:
            dd, mm, yyyy = dmy
        from scorpio_pipe.app_paths import pick_workspace_root
        base = pick_workspace_root(Path(root) if root else None) / f'{dd:02d}_{mm:02d}_{yyyy:04d}'

        made: list[Path] = []
        self._log_info(f'Batch: building configs for {len(objs)} objects → {base}')
        for obj in objs:
            try:
                sig = RunSignature(obj, '', '', '')
                wd = pick_smart_run_dir(base, sig, prefer_flat=True)
                wd.mkdir(parents=True, exist_ok=True)
                cfg_path = wd / 'config.yaml'
                ac = build_autoconfig(self._inspect.table, data_dir, obj, wd)
                cfg_path.write_text(ac.to_yaml_text(), encoding='utf-8')
                made.append(cfg_path)
                self._log_info(f'  ✔ {obj}: {cfg_path}')
            except Exception as e:
                self._log_exception(e)

        if made:
            self._log_info(f'Batch done: {len(made)} configs created')
            try:
                self.edit_cfg_path.setText(str(made[0]))
                self._load_config(made[0])
            except Exception:
                pass
            try:
                self._open_in_explorer(base)
            except Exception:
                pass

    def _batch_run(self) -> None:
        """Run non-interactive steps for multiple objects sequentially.

        NOTE: the LineID interactive step is NOT executed; this prepares line candidates
        so you can open LineID later per object and then run Wavelength solution.
        """
        if self._inspect is None:
            self._log_error('Run Inspect first')
            return
        objs = self._selected_overview_objects()
        if not objs:
            self._log_error('Select one or more objects in Dataset overview')
            return
        # Ensure configs exist (build if needed)
        self._batch_build_configs()

        # Collect configs under the night work root
        data_dir = Path(self.edit_data_dir.text()).expanduser()
        root = getattr(self, '_pipeline_root', None) or (data_dir.parent if data_dir.exists() else Path.home())
        dmy = self._infer_night_date_parts()
        if dmy is None:
            now = datetime.now()
            dd, mm, yyyy = int(now.day), int(now.month), int(now.year)
        else:
            dd, mm, yyyy = dmy
        from scorpio_pipe.app_paths import pick_workspace_root
        base = pick_workspace_root(Path(root) if root else None) / f'{dd:02d}_{mm:02d}_{yyyy:04d}'

        cfgs: list[tuple[str, Path]] = []
        for obj in objs:
            # Find the newest config.yaml whose path contains the object slug/dir created by pick_smart_run_dir
            # Fallback: first matching by name anywhere under base.
            best: Path | None = None
            try:
                cands: list[Path] = []
                for cp in base.rglob('config.yaml'):
                    if obj.lower() in str(cp.parent).lower():
                        cands.append(cp)
                if cands:
                    best = max(cands, key=lambda p: p.stat().st_mtime)
            except Exception:
                pass
            if best and best.exists():
                cfgs.append((obj, best))

        if not cfgs:
            self._log_error('No configs found to run')
            return

        tasks = ['manifest', 'superbias', 'cosmics', 'superneon', 'lineid_prepare', 'qc_report']
        pd = QtWidgets.QProgressDialog('Batch running…', 'Cancel', 0, len(cfgs), self)
        pd.setWindowModality(QtCore.Qt.WindowModal)
        pd.setMinimumDuration(0)
        for i, (obj, cfg_path) in enumerate(cfgs, start=1):
            if pd.wasCanceled():
                self._log_info('Batch run canceled by user')
                break
            pd.setValue(i-1)
            pd.setLabelText(f'{obj}: running non-interactive steps…')
            QtWidgets.QApplication.processEvents()
            try:
                self._log_info(f'=== BATCH RUN: {obj} ===')
                run_sequence(cfg_path, tasks, resume=True, force=False)
                self._log_info(f'  ✔ {obj}: done')
            except Exception as e:
                self._log_exception(e)
        pd.setValue(len(cfgs))
        self._log_info('Batch run finished. Next: open LineID per object, then build Wavelength solution.')

    def _use_setup_from_frame(self, sel: SelectedFrame) -> None:
        """Fill Config page setup fields from a selected inspected frame."""
        try:
            if not hasattr(self, "combo_object"):
                return
            if sel.object:
                self.combo_object.setCurrentText(sel.object)

            def _ensure_set(combo: QtWidgets.QComboBox, value: str) -> None:
                if not value:
                    return
                if combo.findText(value) < 0:
                    combo.addItem(value)
                combo.setCurrentText(value)

            _ensure_set(self.combo_disperser, sel.disperser)
            _ensure_set(self.combo_slit, sel.slit)
            _ensure_set(self.combo_binning, sel.binning)
            self._update_setup_hint()
            self.steps.setCurrentRow(1)
            self._log_info(
                f"Setup from frame: object='{sel.object}', disperser='{sel.disperser}', slit='{sel.slit}', binning='{sel.binning}'"
            )
        except Exception as e:
            self._log_exception(e)




    def _open_frames_window(self, stage_key: str) -> None:
        """Open a non-modal per-stage Frames Browser window."""
        try:
            from scorpio_pipe.ui.stage_frames_window import StageFramesWindow

            if not hasattr(self, '_frames_windows'):
                self._frames_windows = {}

            win = self._frames_windows.get(stage_key)
            if win is None:
                win = StageFramesWindow(stage_key, parent=self)
                if stage_key == 'project':
                    try:
                        win.useSetupRequested.connect(self._use_setup_from_frame)
                    except Exception:
                        pass
                self._frames_windows[stage_key] = win

            inspect_df = None
            data_dir = None
            try:
                if getattr(self, '_inspect', None) is not None:
                    inspect_df = getattr(self._inspect, 'table', None)
                    data_dir = getattr(self._inspect, 'data_dir', None)
            except Exception:
                pass

            if stage_key == 'project':
                win.set_context(getattr(self, '_cfg', None), inspect_df=inspect_df, data_dir=data_dir)
            else:
                win.set_context(getattr(self, '_cfg', None))

            win.show()
            win.raise_()
            win.activateWindow()
        except Exception as e:
            self._log_exception(e)
    def _open_existing_cfg(self) -> None:
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open config", str(Path.home()), "YAML (*.yaml *.yml)")
        if not fn:
            return
        self.edit_cfg_path.setText(fn)
        self._load_config(Path(fn))
        self.steps.setCurrentRow(1)

    # --------------------------- page: config ---------------------------

    def _build_page_config(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setSpacing(12)

        # Two-column layout: left = target setup, right = config editor
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        lay.addWidget(splitter, 1)

        # --- setup (left) ---
        g_setup = _box("Target & setup")
        g_setup.setMinimumWidth(430)
        splitter.addWidget(g_setup)
        fl = QtWidgets.QFormLayout(g_setup)

        # Object
        row_obj = QtWidgets.QHBoxLayout()
        self.combo_object = QtWidgets.QComboBox()
        self.combo_object.setEditable(True)
        self.combo_object.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.combo_object.setMaxVisibleItems(25)
        self.combo_object.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContentsOnFirstShow)
        row_obj.addWidget(self.combo_object, 1)
        row_obj.addWidget(HelpButton("Имя научного объекта (как в ночном логе / в FITS-заголовке)."))
        fl.addRow("Object", row_obj)

        # Disperser
        row_disp = QtWidgets.QHBoxLayout()
        self.combo_disperser = QtWidgets.QComboBox()
        self.combo_disperser.setEditable(False)
        row_disp.addWidget(self.combo_disperser, 1)
        row_disp.addWidget(HelpButton("Выбор решётки/дисперсера. Если у объекта несколько решёток — выбери нужную."))
        fl.addRow("Disperser", row_disp)

        # Slit
        row_slit = QtWidgets.QHBoxLayout()
        self.combo_slit = QtWidgets.QComboBox()
        self.combo_slit.setEditable(False)
        row_slit.addWidget(self.combo_slit, 1)
        row_slit.addWidget(HelpButton("Щель (ширина). Если в данных встречается несколько щелей — выбери нужную."))
        fl.addRow("Slit", row_slit)

        # Binning
        row_bin = QtWidgets.QHBoxLayout()
        self.combo_binning = QtWidgets.QComboBox()
        self.combo_binning.setEditable(False)
        row_bin.addWidget(self.combo_binning, 1)
        row_bin.addWidget(HelpButton("Биннинг ПЗС. Мы используем его как часть setup, чтобы не смешивать разные режимы."))
        fl.addRow("Binning", row_bin)

        # Work dir
        row_wd = QtWidgets.QHBoxLayout()
        self.edit_work_dir = QtWidgets.QLineEdit()
        # If the user edits the field manually, we stop auto-suggesting paths.
        self._workdir_user_edited = False
        self.edit_work_dir.textEdited.connect(lambda *_: setattr(self, "_workdir_user_edited", True))
        self.btn_pick_work_dir = QtWidgets.QToolButton(text="…")
        self.btn_pick_work_dir.setCursor(QtCore.Qt.PointingHandCursor)
        row_wd.addWidget(self.edit_work_dir, 1)
        row_wd.addWidget(self.btn_pick_work_dir)
        row_wd.addWidget(HelpButton("Папка, куда будут записаны продукты пайплайна (calib/, wavesol/, report/...)."))
        fl.addRow("Work directory", row_wd)

        # Create/Load config actions
        row_cfg = QtWidgets.QHBoxLayout()
        self.btn_suggest_workdir = QtWidgets.QPushButton("Suggest")
        self.btn_make_cfg = QtWidgets.QPushButton("Create new config")
        self.btn_make_cfg.setProperty("primary", True)
        self.btn_reload_cfg = QtWidgets.QPushButton("Reload")
        row_cfg.addWidget(self.btn_suggest_workdir)
        row_cfg.addStretch(1)
        row_cfg.addWidget(self.btn_make_cfg)
        row_cfg.addWidget(self.btn_reload_cfg)
        fl.addRow(row_cfg)

        # Setup hint
        self.lbl_setup_hint = QtWidgets.QLabel("—")
        self.lbl_setup_hint.setWordWrap(True)
        self.lbl_setup_hint.setStyleSheet("color: #A0A0A0;")
        fl.addRow("Setup info", self.lbl_setup_hint)

        # tabs: Parameters / YAML (right)
        tabs = QtWidgets.QTabWidget()
        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 900])

        # --- quick params (tab) ---
        tab_params = QtWidgets.QWidget()
        tabs.addTab(tab_params, "Parameters")
        tab_params_l = QtWidgets.QVBoxLayout(tab_params)
        tab_params_l.setContentsMargins(0, 0, 0, 0)

        g_q = _box("Quick parameters")
        tab_params_l.addWidget(g_q)
        ql = QtWidgets.QGridLayout(g_q)
        ql.setHorizontalSpacing(10)
        ql.setVerticalSpacing(10)

        self._param_specs: list[ParamSpec] = [
            ParamSpec(
                "wavesol.y_half",
                "Profile half-height (y_half)",
                "Полувысота окна по Y для извлечения 1D-профиля (медиана по полосе y0±y_half).",
                "int",
                (1, 400),
                1,
            ),
            ParamSpec(
                "wavesol.peak_snr",
                "Peak SNR threshold",
                "Минимальное отношение сигнал/шум для автодетекции пиков в 1D-профиле супернеона.",
                "float",
                (0.0, 1000.0),
                0.5,
            ),
            ParamSpec(
                "wavesol.peak_prom_snr",
                "Peak prominence (SNR)",
                "Минимальная prominence в единицах σ (устойчивость к плавному фону).",
                "float",
                (0.0, 1000.0),
                0.5,
            ),
            ParamSpec(
                "wavesol.gui_min_amp_sigma_k",
                "Auto min amplitude (k·σ)",
                "Коэффициент k для автоматического порога амплитуды: threshold = median(noise) + k·σ(noise).",
                "float",
                (0.0, 50.0),
                0.5,
            ),
            ParamSpec(
                "wavesol.poly_deg_1d",
                "1D polynomial degree",
                "Степень полинома для λ(x) по ручным парам.",
                "int",
                (2, 10),
                1,
            ),
        ]

        self._param_widgets: dict[str, QtWidgets.QWidget] = {}
        for r, spec in enumerate(self._param_specs):
            lbl = QtWidgets.QLabel(spec.label_en)
            btn = HelpButton(spec.help_ru)
            if spec.kind == "int":
                w_inp = QtWidgets.QSpinBox()
                w_inp.setRange(int(spec.rng[0]), int(spec.rng[1]))
                w_inp.setSingleStep(int(spec.step))
            else:
                w_inp = QtWidgets.QDoubleSpinBox()
                w_inp.setRange(float(spec.rng[0]), float(spec.rng[1]))
                w_inp.setSingleStep(float(spec.step))
                w_inp.setDecimals(3)
            w_inp.setMinimumWidth(120)
            self._param_widgets[spec.key] = w_inp
            ql.addWidget(lbl, r, 0)
            ql.addWidget(w_inp, r, 1)
            ql.addWidget(btn, r, 2)

        self.btn_apply_quick = QtWidgets.QPushButton("Apply to config")
        self.btn_apply_quick.setProperty("primary", True)
        ql.addWidget(self.btn_apply_quick, len(self._param_specs), 0, 1, 3)

        tab_params_l.addStretch(1)

        # --- YAML editor (tab) ---
        tab_yaml = QtWidgets.QWidget()
        tabs.addTab(tab_yaml, "YAML")
        tab_yaml_l = QtWidgets.QVBoxLayout(tab_yaml)
        tab_yaml_l.setContentsMargins(0, 0, 0, 0)

        g_yaml = _box("Config YAML (editable)")
        tab_yaml_l.addWidget(g_yaml, 1)
        yl = QtWidgets.QVBoxLayout(g_yaml)

        self.editor_yaml = QtWidgets.QPlainTextEdit()
        self.editor_yaml.setPlaceholderText("Create or open a config to edit it here…")
        yl.addWidget(self.editor_yaml, 1)

        bar = QtWidgets.QHBoxLayout()
        yl.addLayout(bar)
        self.btn_validate_yaml = QtWidgets.QPushButton("Validate")
        self.btn_diff_cfg = QtWidgets.QPushButton("Diff")
        self.btn_save_cfg = QtWidgets.QPushButton("Save")
        self.btn_save_cfg.setProperty("primary", True)
        self.lbl_cfg_state = QtWidgets.QLabel("—")
        bar.addWidget(self.btn_validate_yaml)
        bar.addWidget(self.btn_diff_cfg)
        bar.addWidget(self.btn_save_cfg)
        bar.addStretch(1)
        bar.addWidget(self.lbl_cfg_state)

        # signals
        self.combo_object.currentTextChanged.connect(self._on_object_changed)
        self.combo_disperser.currentTextChanged.connect(self._on_disperser_changed)
        self.combo_slit.currentTextChanged.connect(self._update_setup_hint)
        self.combo_binning.currentTextChanged.connect(self._update_setup_hint)
        self.btn_pick_work_dir.clicked.connect(self._pick_work_dir)
        self.btn_suggest_workdir.clicked.connect(self._suggest_work_dir)
        self.btn_make_cfg.clicked.connect(self._do_make_cfg)
        self.btn_reload_cfg.clicked.connect(self._do_reload_cfg)
        self.btn_apply_quick.clicked.connect(self._apply_quick_params)
        self.btn_validate_yaml.clicked.connect(self._validate_yaml)
        self.btn_diff_cfg.clicked.connect(self._show_cfg_diff)
        self.btn_save_cfg.clicked.connect(self._do_save_cfg)
        self.editor_yaml.textChanged.connect(self._on_yaml_changed)

        lay.addWidget(_hline())
        foot = QtWidgets.QHBoxLayout()
        lay.addLayout(foot)
        self.btn_to_calib = QtWidgets.QPushButton("Go to Calibrations →")
        foot.addStretch(1)
        foot.addWidget(self.btn_to_calib)
        self.btn_to_calib.clicked.connect(lambda: self.steps.setCurrentRow(2))

        return w

    def _pick_work_dir(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select work directory", str(Path.home()))
        if d:
            self.edit_work_dir.setText(d)
            self._workdir_user_edited = True
            self._workdir_user_edited = True

    def _infer_night_date_parts(self) -> tuple[int, int, int] | None:
        """Infer night date as (dd, mm, yyyy) for the work/ structure.

        Preferred source is the nightlog filename (e.g. s251216.txt -> 16/12/2025).
        Falls back to DATE-OBS from the first opened FITS header if available.
        """
        try:
            if self._inspect is not None:
                p = getattr(self._inspect, "nightlog_path", None)
                if p:
                    import re

                    stem = Path(str(p)).stem.lower()
                    m = re.search(r"s(\d{2})(\d{2})(\d{2})", stem)
                    if m:
                        yy, mm, dd = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                        yyyy = (2000 + yy) if yy < 80 else (1900 + yy)
                        return dd, mm, yyyy
        except Exception:
            pass

        # fallback: DATE-OBS like 'YYYY-MM-DDThh:mm:ss' (or 'YYYY.MM.DD ...')
        try:
            if self._inspect is not None:
                df = getattr(self._inspect, "table", None)
                if df is not None and not df.empty and "date_obs" in df.columns:
                    import re

                    # take the most common date prefix if possible
                    vals = df["date_obs"].dropna().astype(str).tolist()
                    if vals:
                        # normalize to first 10 chars when looks like a date
                        prefixes: list[str] = []
                        for v in vals:
                            v = str(v).strip()
                            m = re.search(r"(\d{4})[-./](\d{2})[-./](\d{2})", v)
                            if m:
                                prefixes.append(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
                        if prefixes:
                            # choose the most common (robust to a few bad headers)
                            from collections import Counter

                            s = Counter(prefixes).most_common(1)[0][0]
                        else:
                            s = str(vals[0])
                        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
                        if m:
                            yyyy = int(m.group(1))
                            mm = int(m.group(2))
                            dd = int(m.group(3))
                            return dd, mm, yyyy
        except Exception:
            pass

        return None

    
    def _suggest_work_dir(self) -> None:
        """Suggest a default (smart) work directory.

        Base convention: `<workspace_root>/<dd_mm_yyyy>/`.

        Strategy (C-2 safe):
          - try `<install_dir>/workspace/<night>/...` when writable
          - otherwise fallback to `<LocalAppData>/Scorpipe/workspace/<night>/...`

        Smart mode:
          - If the night folder is empty -> use it (single-run, no extra nesting).
          - If it already contains a different run -> use `<night>/<object>/<disperser>/`
            (and add suffixes to avoid collisions).
        """

        from scorpio_pipe.workdir import RunSignature, pick_smart_run_dir
        from scorpio_pipe.app_paths import pick_workspace_root

        dmy = self._infer_night_date_parts()
        if dmy is None:
            now = datetime.now()
            dd, mm, yyyy = int(now.day), int(now.month), int(now.year)
        else:
            dd, mm, yyyy = dmy

        night_dir = f"{dd:02d}_{mm:02d}_{yyyy:04d}"
        ws_root = pick_workspace_root(getattr(self, "_pipeline_root", None))
        base = ws_root / night_dir

        # If we already know target+setup, apply the smart collision-free picker.
        try:
            obj = (self.combo_object.currentText() or "").strip()
            disp = (self.combo_disperser.currentText() or "").strip()
            slit = (getattr(self, "combo_slit", None).currentText() if hasattr(self, "combo_slit") else "") or ""
            slit = str(slit).strip()
            binning = (getattr(self, "combo_binning", None).currentText() if hasattr(self, "combo_binning") else "") or ""
            binning = str(binning).strip()

            if obj:
                sig = RunSignature(obj, disp, slit, binning)
                wd = pick_smart_run_dir(base, sig, prefer_flat=True)
            else:
                wd = base
        except Exception:
            wd = base

        self.edit_work_dir.setText(str(wd))

    def _populate_objects_from_inspect(self) -> None:
        self.combo_object.blockSignals(True)
        self.combo_object.clear()
        if self._inspect is not None:
            self.combo_object.addItems(self._inspect.objects)
        self.combo_object.blockSignals(False)
        if self.combo_object.count() > 0:
            self.combo_object.setCurrentIndex(0)
        self._update_dispersers_from_inspect()

    def _update_dispersers_from_inspect(self) -> None:
        self.combo_disperser.blockSignals(True)
        self.combo_disperser.clear()
        if self._inspect is None:
            self.combo_disperser.blockSignals(False)
            return
        obj = (self.combo_object.currentText() or "").strip()
        if not obj:
            self.combo_disperser.blockSignals(False)
            return
        df = self._inspect.table
        if df is None or df.empty:
            self.combo_disperser.blockSignals(False)
            return
        obj_n = "".join(ch for ch in obj.upper() if ch.isalnum())
        sci = df[(df["kind"] == "obj") & (df["object_norm"] == obj_n)]
        if "disperser" in sci.columns:
            vals = sorted([v for v in sci["disperser"].dropna().astype(str).unique().tolist() if v.strip()])
        else:
            vals = []
        if not vals:
            vals = [""]
        self.combo_disperser.addItems(vals)
        self.combo_disperser.blockSignals(False)
        self.combo_disperser.setCurrentIndex(0)
        self._update_slit_binning_from_inspect()
        self._update_setup_hint()

    def _update_slit_binning_from_inspect(self) -> None:
        # Populate slit/binning lists from Inspect table for the current object+disperser.
        if not hasattr(self, "combo_slit") or not hasattr(self, "combo_binning"):
            return
        self.combo_slit.blockSignals(True)
        self.combo_binning.blockSignals(True)
        self.combo_slit.clear()
        self.combo_binning.clear()

        if self._inspect is None:
            self.combo_slit.addItem("")
            self.combo_binning.addItem("")
            self.combo_slit.blockSignals(False)
            self.combo_binning.blockSignals(False)
            return

        obj = (self.combo_object.currentText() or "").strip()
        disp = (self.combo_disperser.currentText() or "").strip()
        if not obj:
            self.combo_slit.addItem("")
            self.combo_binning.addItem("")
            self.combo_slit.blockSignals(False)
            self.combo_binning.blockSignals(False)
            return

        df = self._inspect.table
        obj_n = "".join(ch for ch in obj.upper() if ch.isalnum())
        sci = df[(df["kind"] == "obj") & (df["object_norm"] == obj_n)].copy()
        if disp and "disperser" in sci.columns:
            sci = sci[sci["disperser"].astype(str) == disp]

        # slit
        slits: list[str] = []
        if "slit" in sci.columns:
            slits = sorted([v for v in sci["slit"].dropna().astype(str).unique().tolist() if v.strip()])
        if not slits:
            slits = [""]
        self.combo_slit.addItems(slits)

        # binning
        bins: list[str] = []
        if "binning" in sci.columns:
            bins = sorted([v for v in sci["binning"].dropna().astype(str).unique().tolist() if v.strip()])
        if not bins:
            bins = [""]
        self.combo_binning.addItems(bins)

        self.combo_slit.setCurrentIndex(0)
        self.combo_binning.setCurrentIndex(0)
        self.combo_slit.blockSignals(False)
        self.combo_binning.blockSignals(False)

    def _update_setup_hint(self) -> None:
        if not hasattr(self, "lbl_setup_hint"):
            return
        if self._inspect is None:
            self.lbl_setup_hint.setText("—")
            return

        obj = (self.combo_object.currentText() or "").strip()
        disp = (self.combo_disperser.currentText() or "").strip()
        slit = (self.combo_slit.currentText() or "").strip() if hasattr(self, "combo_slit") else ""
        binning = (self.combo_binning.currentText() or "").strip() if hasattr(self, "combo_binning") else ""

        df = self._inspect.table
        obj_n = "".join(ch for ch in obj.upper() if ch.isalnum())
        sci = df[(df["kind"] == "obj") & (df["object_norm"] == obj_n)].copy()
        if disp and "disperser" in sci.columns:
            sci = sci[sci["disperser"].astype(str) == disp]
        if slit and "slit" in sci.columns:
            sci = sci[sci["slit"].astype(str) == slit]
        if binning and "binning" in sci.columns:
            sci = sci[sci["binning"].astype(str) == binning]

        instr = ""
        if "instrument" in sci.columns:
            uniq = [v for v in sci["instrument"].dropna().astype(str).unique().tolist() if v.strip()]
            if len(uniq) == 1:
                instr = uniq[0]
            elif len(uniq) > 1:
                instr = " / ".join(sorted(uniq))

        # Instrument DB hint (best-effort)
        hint_parts: list[str] = []
        if instr:
            hint_parts.append(f"Instrument: {instr}")
        if disp:
            hint_parts.append(f"Disperser: {disp}")
        if slit:
            hint_parts.append(f"Slit: {slit}")
        if binning:
            hint_parts.append(f"Binning: {binning}")

        # attempt to look up disperser spec
        spec = None
        if instr and disp:
            spec = find_grism(instr, disp)
        if spec is None and disp:
            # try known instruments (if header didn't give us a clean instrument label)
            for name in ("SCORPIO-2", "SCORPIO"):
                spec = find_grism(name, disp)
                if spec is not None:
                    break

        if spec is not None:
            if spec.range_A:
                hint_parts.append(f"Range≈[{spec.range_A[0]:.0f}–{spec.range_A[1]:.0f}] Å")
            if spec.dispersion_A_per_pix:
                hint_parts.append(f"Dispersion≈{spec.dispersion_A_per_pix:.3f} Å/pix")
            if spec.resolution_fwhm_A:
                hint_parts.append(f"FWHM≈{spec.resolution_fwhm_A:.2f} Å")
            if spec.notes:
                hint_parts.append(f"Notes: {spec.notes}")

        if not hint_parts:
            self.lbl_setup_hint.setText("—")
        else:
            self.lbl_setup_hint.setText(" · ".join(hint_parts))

    def _on_object_changed(self, *_: object) -> None:
        self._update_dispersers_from_inspect()
        try:
            if (not self.edit_work_dir.text().strip()) and (not getattr(self, "_workdir_user_edited", False)):
                self._suggest_work_dir()
        except Exception:
            pass
        self._update_enables()

    def _on_disperser_changed(self, *_: object) -> None:
        self._update_slit_binning_from_inspect()
        self._update_setup_hint()
        try:
            if (not self.edit_work_dir.text().strip()) and (not getattr(self, "_workdir_user_edited", False)):
                self._suggest_work_dir()
        except Exception:
            pass
        self._update_enables()

    # --------------------------- config file ops ---------------------------

    def _do_make_cfg(self) -> None:
        if self._inspect is None:
            self._log_error("Run Inspect first")
            return

        data_dir = Path(self.edit_data_dir.text()).expanduser()
        obj = (self.combo_object.currentText() or "").strip()
        disp = (self.combo_disperser.currentText() or "").strip() or None
        slit = (getattr(self, "combo_slit", None).currentText() if hasattr(self, "combo_slit") else "")
        slit = (slit or "").strip() or None
        binning = (getattr(self, "combo_binning", None).currentText() if hasattr(self, "combo_binning") else "")
        binning = (binning or "").strip() or None

        work_dir_txt = self.edit_work_dir.text().strip()
        if not work_dir_txt:
            self._log_error("Work directory is empty")
            return
        work_dir = Path(work_dir_txt).expanduser()

        if not obj:
            self._log_error("Object is empty")
            return

        # Smart mode: if the user points to a night folder dd_mm_yyyy, avoid collisions automatically.
        try:
            import re as _re
            from scorpio_pipe.workdir import RunSignature, pick_smart_run_dir

            if _re.match(r"^\d{2}_\d{2}_\d{4}$", work_dir.name):
                setup = getattr(self._inspect, "setup", {}) or {}
                sig = RunSignature(obj, disp or "", slit or "", binning or "")
                smart = pick_smart_run_dir(work_dir, sig, prefer_flat=True)
                if smart != work_dir:
                    self._log_info(f"Smart work dir: {work_dir} → {smart}")
                    work_dir = smart
                    self.edit_work_dir.setText(str(work_dir))
        except Exception:
            pass

        self._set_step_status(1, "running")
        try:
            work_dir.mkdir(parents=True, exist_ok=True)
            cfg_path = work_dir / "config.yaml"
            ac = build_autoconfig(
                self._inspect.table,
                data_dir,
                obj,
                work_dir,
                disperser=disp,
                slit=slit,
                binning=binning,
            )
            cfg_path.write_text(ac.to_yaml_text(), encoding="utf-8")
            self._log_info(f"Wrote config: {cfg_path}")
            self.edit_cfg_path.setText(str(cfg_path))
            self._load_config(cfg_path)
            self._set_step_status(1, "ok")
        except Exception as e:
            self._set_step_status(1, "fail")
            self._log_exception(e)

    def _do_reload_cfg(self) -> None:

        if not self._cfg_path:
            # try from edit
            s = self.edit_cfg_path.text().strip()
            if s:
                self._load_config(Path(s))
            return
        self._load_config(self._cfg_path)

    def _load_config(self, cfg_path: Path) -> None:
        cfg_path = cfg_path.expanduser().resolve()
        try:
            cfg = load_config(cfg_path)
        except Exception as e:
            self._log_exception(e)
            return

        self._cfg_path = cfg_path
        self._cfg = cfg
        self.lbl_cfg_state.setText(f"Loaded: {cfg_path}")
        self.editor_yaml.blockSignals(True)
        yaml_txt = _yaml_dump(cfg)
        self.editor_yaml.setPlainText(yaml_txt)
        self.editor_yaml.blockSignals(False)
        self._yaml_saved_text = yaml_txt

        # fill quick widgets from cfg
        for spec in self._param_specs:
            val = self._get_cfg_value(spec.key)
            w = self._param_widgets.get(spec.key)
            if w is None:
                continue
            if isinstance(w, QtWidgets.QSpinBox):
                try:
                    w.setValue(int(val))
                except Exception:
                    pass
            elif isinstance(w, QtWidgets.QDoubleSpinBox):
                try:
                    w.setValue(float(val))
                except Exception:
                    pass

        # also update object/disperser/workdir if available
        try:
            self.edit_work_dir.setText(str(Path(str(cfg.get("work_dir", ""))).expanduser()))
            # Preserve user choice from an existing config (don't auto-suggest over it)
            if self.edit_work_dir.text().strip():
                self._workdir_user_edited = True
            setup = (cfg.get("frames", {}) or {}).get("__setup__", {})
            if isinstance(setup, dict):
                d = str(setup.get("disperser", "") or "")
                # ensure exists in combo
                if d and self.combo_disperser.findText(d) < 0:
                    self.combo_disperser.addItem(d)
                if d:
                    self.combo_disperser.setCurrentText(d)
        except Exception:
            pass

        self._update_enables()
        self._refresh_statusbar()
        self._refresh_inspector()

        # refresh derived UI state
        try:
            self._update_setup_hint()
        except Exception:
            pass
        try:
            self._refresh_pair_sets_combo()
            self._refresh_pairs_label()
        except Exception:
            pass
        try:
            if hasattr(self, 'lbl_wavesol_dir') and self._cfg:
                self.lbl_wavesol_dir.setText(f"wavesol: {wavesol_dir(self._cfg)}")
        except Exception:
            pass

    def _on_yaml_changed(self) -> None:
        # just mark state; we parse on demand
        cur = self.editor_yaml.toPlainText()
        if self._yaml_saved_text and (cur == self._yaml_saved_text):
            self.lbl_cfg_state.setText("No changes")
            return
        if self._cfg_path:
            self.lbl_cfg_state.setText(f"Edited (not saved): {self._cfg_path.name}")
        else:
            self.lbl_cfg_state.setText("Edited")

    def _show_cfg_diff(self) -> None:
        try:
            new_text = self.editor_yaml.toPlainText()
            old_text = self._yaml_saved_text
            if self._cfg_path and self._cfg_path.exists():
                try:
                    old_text = self._cfg_path.read_text(encoding="utf-8")
                except Exception:
                    pass
            if not old_text:
                old_text = ""
            title = "Config diff"
            if self._cfg_path:
                title += f" — {self._cfg_path.name}"
            ConfigDiffDialog(title, old_text, new_text, parent=self).exec()
        except Exception as e:
            self._log_exception(e)

    def _validate_yaml(self) -> None:
        cfg, err = _safe_parse_yaml(self.editor_yaml.toPlainText())
        if err:
            self._log_error(f"YAML invalid: {err}")
            return
        self._log_info("YAML ok")
        self._cfg = cfg

    def _sync_cfg_from_editor(self) -> bool:
        cfg, err = _safe_parse_yaml(self.editor_yaml.toPlainText())
        if err:
            self._log_error(f"YAML invalid: {err}")
            return False
        self._cfg = cfg
        self._refresh_outputs_panels()
        self._sync_stage_controls_from_cfg()
        return True

    def _refresh_outputs_panels(self) -> None:
        mapping = [
            ("outputs_calib", None),
            ("outputs_cosmics", "cosmics"),
            ("outputs_flatfield", "flatfield"),
            ("outputs_superneon", "wavesol"),
            ("outputs_lineid", "wavesol"),
            ("outputs_wavesol", "wavesol"),
        ]
        for attr, stage in mapping:
            try:
                if hasattr(self, attr):
                    panel = getattr(self, attr)
                    if isinstance(panel, OutputsPanel):
                        panel.set_context(self._cfg, stage=stage)
            except Exception:
                pass

    def _do_save_cfg(self) -> None:
        if not self._cfg_path:
            # infer from work_dir
            wd = Path(self.edit_work_dir.text()).expanduser()
            if wd:
                wd.mkdir(parents=True, exist_ok=True)
                self._cfg_path = (wd / "config.yaml").resolve()
        if not self._cfg_path:
            self._log_error("No config path")
            return
        if not self._sync_cfg_from_editor():
            return
        txt = self.editor_yaml.toPlainText()
        self._cfg_path.write_text(txt, encoding="utf-8")
        self._yaml_saved_text = txt
        self._log_info(f"Saved: {self._cfg_path}")
        self.lbl_cfg_state.setText(f"Saved: {self._cfg_path.name}")
        self._update_enables()
        self._refresh_statusbar()
        self._refresh_inspector()

    def _get_cfg_value(self, dotted: str, default: Any | None = None) -> Any:
        cfg = self._cfg or {}
        cur: Any = cfg
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def _set_cfg_value(self, dotted: str, value: Any) -> None:
        if self._cfg is None:
            self._cfg = {}
        cur: Any = self._cfg
        parts = dotted.split(".")
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value

    def _apply_quick_params(self) -> None:
        if not self._sync_cfg_from_editor():
            return
        for spec in self._param_specs:
            w = self._param_widgets.get(spec.key)
            if w is None:
                continue
            if isinstance(w, QtWidgets.QSpinBox):
                v: Any = int(w.value())
            elif isinstance(w, QtWidgets.QDoubleSpinBox):
                v = float(w.value())
            else:
                continue
            self._set_cfg_value(spec.key, v)

        # reflect into editor
        self.editor_yaml.blockSignals(True)
        self.editor_yaml.setPlainText(_yaml_dump(self._cfg or {}))
        self.editor_yaml.blockSignals(False)
        self.lbl_cfg_state.setText("Applied quick params (not saved)")

    # --------------------------- page: calibrations ---------------------------



    def _cfg_get(self, cfg: dict[str, Any] | None, path: list[str], default: Any = None) -> Any:
        cur: Any = cfg or {}
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    def _editor_patch_cfg(self, mutator) -> None:
        """Safely patch YAML in the editor and resync internal config."""
        # The single source of truth is the YAML editor shown on the Config page.
        txt = self.editor_yaml.toPlainText()
        cfg, err = _safe_parse_yaml(txt)
        if err or not isinstance(cfg, dict):
            self._log_warn(f"Cannot update YAML: {err or 'invalid YAML'}")
            return
        try:
            mutator(cfg)
        except Exception as e:
            self._log_exception(e)
            return
        new_txt = _yaml_dump(cfg)
        # Avoid re-entrant loops: block editor signals while setting text
        try:
            blocker = QtCore.QSignalBlocker(self.editor_yaml)
        except Exception:
            blocker = None
        self.editor_yaml.setPlainText(new_txt)
        if blocker is not None:
            del blocker
        # Resync internal config + validate
        self._sync_cfg_from_editor()

    def _cfg_set_path(self, path: list[str], value: Any) -> None:
        def mut(cfg: dict[str, Any]) -> None:
            cur = cfg
            for k in path[:-1]:
                if k not in cur or not isinstance(cur[k], dict):
                    cur[k] = {}
                cur = cur[k]
            cur[path[-1]] = value
        self._editor_patch_cfg(mut)

    def _cfg_set_apply_to(self, block_key: str, enabled: list[str]) -> None:
        # Preserve a stable order in YAML for readability
        order = ['obj', 'sky', 'sunsky', 'neon', 'flat', 'bias']
        enabled_sorted = [k for k in order if k in enabled] + [k for k in enabled if k not in order]
        self._cfg_set_path([block_key, 'apply_to'], enabled_sorted)

    def _sync_stage_controls_from_cfg(self) -> None:
        cfg = getattr(self, '_cfg', None)
        if not isinstance(cfg, dict):
            return

        # --- Cosmics ---
        if hasattr(self, 'chk_cosmics_obj'):
            apply_to = set(self._cfg_get(cfg, ['cosmics', 'apply_to'], []) or [])
            for name, cb in [
                ('obj', getattr(self, 'chk_cosmics_obj', None)),
                ('sky', getattr(self, 'chk_cosmics_sky', None)),
                ('sunsky', getattr(self, 'chk_cosmics_sunsky', None)),
                ('neon', getattr(self, 'chk_cosmics_neon', None)),
            ]:
                if cb is None:
                    continue
                with QtCore.QSignalBlocker(cb):
                    cb.setChecked(name in apply_to)
            if hasattr(self, 'spin_cosmics_k'):
                with QtCore.QSignalBlocker(self.spin_cosmics_k):
                    self.spin_cosmics_k.setValue(int(self._cfg_get(cfg, ['cosmics', 'k'], 5) or 5))
            if hasattr(self, 'chk_cosmics_bias'):
                with QtCore.QSignalBlocker(self.chk_cosmics_bias):
                    self.chk_cosmics_bias.setChecked(bool(self._cfg_get(cfg, ['cosmics', 'bias_subtract'], True)))
            if hasattr(self, 'chk_cosmics_png'):
                with QtCore.QSignalBlocker(self.chk_cosmics_png):
                    self.chk_cosmics_png.setChecked(bool(self._cfg_get(cfg, ['cosmics', 'save_png'], True)))

        # --- Flatfield ---
        if hasattr(self, 'chk_flat_enabled'):
            with QtCore.QSignalBlocker(self.chk_flat_enabled):
                self.chk_flat_enabled.setChecked(bool(self._cfg_get(cfg, ['flatfield', 'enabled'], False)))
            apply_to = set(self._cfg_get(cfg, ['flatfield', 'apply_to'], []) or [])
            for name, cb in [
                ('obj', getattr(self, 'chk_flat_obj', None)),
                ('sky', getattr(self, 'chk_flat_sky', None)),
                ('sunsky', getattr(self, 'chk_flat_sunsky', None)),
                ('neon', getattr(self, 'chk_flat_neon', None)),
            ]:
                if cb is None:
                    continue
                with QtCore.QSignalBlocker(cb):
                    cb.setChecked(name in apply_to)
            if hasattr(self, 'chk_flat_bias'):
                with QtCore.QSignalBlocker(self.chk_flat_bias):
                    self.chk_flat_bias.setChecked(bool(self._cfg_get(cfg, ['flatfield', 'bias_subtract'], True)))
            if hasattr(self, 'chk_flat_png'):
                with QtCore.QSignalBlocker(self.chk_flat_png):
                    self.chk_flat_png.setChecked(bool(self._cfg_get(cfg, ['flatfield', 'save_png'], True)))

    def _build_page_calib(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setSpacing(12)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        lay.addWidget(splitter, 1)

        # left: controls
        left = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout(left)
        l.setSpacing(12)

        g = _box("Calibrations")
        l.addWidget(g)
        gl = QtWidgets.QVBoxLayout(g)
        self.lbl_calib = QtWidgets.QLabel(
            "Build report/manifest.json and calib/superbias.fits.\n"
            "Use QC Viewer to check intermediate products at each step."
        )
        self.lbl_calib.setWordWrap(True)
        gl.addWidget(self.lbl_calib)

        row = QtWidgets.QHBoxLayout()
        self.btn_run_calib = QtWidgets.QPushButton("Run: Manifest + Superbias")
        self.btn_run_calib.setProperty("primary", True)
        self.btn_qc_calib = QtWidgets.QPushButton("QC")
        self.btn_frames_calib = QtWidgets.QPushButton("Frames…")
        self.btn_frames_calib.setToolTip("Open Frames Browser for the Calibrations stage")
        row.addWidget(self.btn_run_calib)
        row.addWidget(self.btn_qc_calib)
        row.addWidget(self.btn_frames_calib)
        row.addStretch(1)
        gl.addLayout(row)
        l.addStretch(1)

        splitter.addWidget(left)

        # right: outputs
        self.outputs_calib = OutputsPanel()
        self.outputs_calib.set_context(self._cfg, stage=None)
        splitter.addWidget(self.outputs_calib)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([650, 480])

        lay.addWidget(_hline())
        foot = QtWidgets.QHBoxLayout()
        lay.addLayout(foot)
        self.btn_to_cosmics = QtWidgets.QPushButton("Go to Cosmics →")
        foot.addStretch(1)
        foot.addWidget(self.btn_to_cosmics)

        self.btn_run_calib.clicked.connect(self._do_run_calib)
        self.btn_qc_calib.clicked.connect(self._open_qc_viewer)
        self.btn_frames_calib.clicked.connect(lambda: self._open_frames_window('calib'))
        self.btn_to_cosmics.clicked.connect(lambda: self.steps.setCurrentRow(3))
        return w

    def _do_run_calib(self) -> None:
        if not self._ensure_cfg_saved():
            return
        self._set_step_status(2, "running")
        try:
            ctx = load_context(self._cfg_path)
            run_sequence(ctx, ["manifest", "superbias"])
            self._log_info("Calibrations done")
            try:
                if hasattr(self, "outputs_calib"):
                    self.outputs_calib.set_context(self._cfg, stage=None)
            except Exception:
                pass
            self._set_step_status(2, "ok")
            self._maybe_auto_qc()
        except Exception as e:
            self._set_step_status(2, "fail")
            self._log_exception(e)

    # --------------------------- page: cosmics ---------------------------

    def _build_page_cosmics(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setSpacing(12)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        lay.addWidget(splitter, 1)

        left = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout(left)
        l.setSpacing(12)

        g = _box("Clean Cosmics")
        l.addWidget(g)
        gl = QtWidgets.QVBoxLayout(g)
        lbl = QtWidgets.QLabel(
            "Clean cosmic rays in object/sky frames using a simple median filter.\n"
            "Outputs are written under work_dir/cosmics/."
        )
        lbl.setWordWrap(True)
        gl.addWidget(lbl)

        # --- per-stage params ---
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignLeft)

        # Apply-to
        apply_row = QtWidgets.QHBoxLayout()
        self.chk_cosmics_obj = QtWidgets.QCheckBox("obj")
        self.chk_cosmics_sky = QtWidgets.QCheckBox("sky")
        self.chk_cosmics_sunsky = QtWidgets.QCheckBox("sunsky")
        self.chk_cosmics_neon = QtWidgets.QCheckBox("neon")
        for cb in (self.chk_cosmics_obj, self.chk_cosmics_sky, self.chk_cosmics_sunsky, self.chk_cosmics_neon):
            apply_row.addWidget(cb)
        apply_row.addStretch(1)
        form.addRow("Apply to", apply_row)

        # Parameters
        self.spin_cosmics_k = QtWidgets.QSpinBox()
        self.spin_cosmics_k.setRange(1, 99)
        self.spin_cosmics_k.setValue(5)
        self.spin_cosmics_k.setToolTip("Median window size (pixels). Larger is more aggressive.")
        form.addRow("Median window (k)", self.spin_cosmics_k)

        self.chk_cosmics_bias = QtWidgets.QCheckBox("Subtract superbias before cleaning")
        self.chk_cosmics_png = QtWidgets.QCheckBox("Save QC PNGs (coverage/sum)")
        form.addRow("", self.chk_cosmics_bias)
        form.addRow("", self.chk_cosmics_png)

        gl.addLayout(form)

        row = QtWidgets.QHBoxLayout()
        self.btn_run_cosmics = QtWidgets.QPushButton("Run: Clean cosmics")
        self.btn_run_cosmics.setProperty("primary", True)
        self.btn_qc_cosmics = QtWidgets.QPushButton("QC")
        self.btn_frames_cosmics = QtWidgets.QPushButton("Frames…")
        self.btn_frames_cosmics.setToolTip("Open Frames Browser for the Cosmics stage")
        row.addWidget(self.btn_run_cosmics)
        row.addWidget(self.btn_qc_cosmics)
        row.addWidget(self.btn_frames_cosmics)
        row.addStretch(1)
        gl.addLayout(row)
        l.addStretch(1)

        splitter.addWidget(left)
        self.outputs_cosmics = OutputsPanel()
        self.outputs_cosmics.set_context(self._cfg, stage="cosmics")
        splitter.addWidget(self.outputs_cosmics)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([650, 480])

        lay.addWidget(_hline())
        foot = QtWidgets.QHBoxLayout()
        lay.addLayout(foot)
        self.btn_to_flatfield = QtWidgets.QPushButton("Go to Flat-fielding →")
        foot.addStretch(1)
        foot.addWidget(self.btn_to_flatfield)

        self.btn_run_cosmics.clicked.connect(self._do_run_cosmics)
        self.btn_qc_cosmics.clicked.connect(self._open_qc_viewer)
        self.btn_frames_cosmics.clicked.connect(lambda: self._open_frames_window('cosmics'))
        self.btn_to_flatfield.clicked.connect(lambda: self.steps.setCurrentRow(4))

        # wire per-stage controls → YAML
        def _apply_to_from_ui() -> list[str]:
            out: list[str] = []
            if self.chk_cosmics_obj.isChecked():
                out.append('obj')
            if self.chk_cosmics_sky.isChecked():
                out.append('sky')
            if self.chk_cosmics_sunsky.isChecked():
                out.append('sunsky')
            if self.chk_cosmics_neon.isChecked():
                out.append('neon')
            return out

        for cb in (self.chk_cosmics_obj, self.chk_cosmics_sky, self.chk_cosmics_sunsky, self.chk_cosmics_neon):
            cb.toggled.connect(lambda *_: self._cfg_set_apply_to('cosmics', _apply_to_from_ui()))
        self.spin_cosmics_k.valueChanged.connect(lambda v: self._cfg_set_path(['cosmics', 'k'], int(v)))
        self.chk_cosmics_bias.toggled.connect(lambda v: self._cfg_set_path(['cosmics', 'bias_subtract'], bool(v)))
        self.chk_cosmics_png.toggled.connect(lambda v: self._cfg_set_path(['cosmics', 'save_png'], bool(v)))

        # initial sync from YAML
        self._sync_stage_controls_from_cfg()
        return w

    def _do_run_cosmics(self) -> None:
        if not self._ensure_cfg_saved():
            return
        self._set_step_status(3, "running")
        try:
            ctx = load_context(self._cfg_path)
            run_sequence(ctx, ["cosmics"])
            self._log_info("Cosmics cleaning done")
            try:
                if hasattr(self, "outputs_cosmics"):
                    self.outputs_cosmics.set_context(self._cfg, stage="cosmics")
            except Exception:
                pass
            self._set_step_status(3, "ok")
            self._maybe_auto_qc()
        except Exception as e:
            self._set_step_status(3, "fail")
            self._log_exception(e)

    # --------------------------- page: flatfield ---------------------------

    def _build_page_flatfield(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setSpacing(12)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        lay.addWidget(splitter, 1)

        left = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout(left)
        l.setSpacing(12)

        g = _box("Flat-fielding (optional)")
        l.addWidget(g)
        gl = QtWidgets.QVBoxLayout(g)
        lbl = QtWidgets.QLabel(
            "Apply flat-field correction after Cosmics.\n"
            "For each object, only flats with matching OBJECT in the nightlog are used.\n"
            "Superbias is subtracted before building and applying the flat."
        )
        lbl.setWordWrap(True)
        gl.addWidget(lbl)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignLeft)

        self.chk_flat_enabled = QtWidgets.QCheckBox("Enable flat-fielding")
        self.chk_flat_enabled.setToolTip("If disabled, the stage is skipped.")
        form.addRow("", self.chk_flat_enabled)

        apply_row = QtWidgets.QHBoxLayout()
        self.chk_flat_obj = QtWidgets.QCheckBox("obj")
        self.chk_flat_sky = QtWidgets.QCheckBox("sky")
        self.chk_flat_sunsky = QtWidgets.QCheckBox("sunsky")
        self.chk_flat_neon = QtWidgets.QCheckBox("neon")
        for cb in (self.chk_flat_obj, self.chk_flat_sky, self.chk_flat_sunsky, self.chk_flat_neon):
            apply_row.addWidget(cb)
        apply_row.addStretch(1)
        form.addRow("Apply to", apply_row)

        self.chk_flat_bias = QtWidgets.QCheckBox("Subtract superbias")
        self.chk_flat_bias.setToolTip("Recommended: keep enabled for stable flat-fielding")
        self.chk_flat_png = QtWidgets.QCheckBox("Save QC PNGs")
        form.addRow("", self.chk_flat_bias)
        form.addRow("", self.chk_flat_png)

        gl.addLayout(form)

        row = QtWidgets.QHBoxLayout()
        self.btn_run_flatfield = QtWidgets.QPushButton("Run: Flat-fielding")
        self.btn_run_flatfield.setProperty("primary", True)
        self.btn_qc_flatfield = QtWidgets.QPushButton("QC")
        self.btn_frames_flatfield = QtWidgets.QPushButton("Frames…")
        self.btn_frames_flatfield.setToolTip("Open Frames Browser for the Flat-fielding stage")
        row.addWidget(self.btn_run_flatfield)
        row.addWidget(self.btn_qc_flatfield)
        row.addWidget(self.btn_frames_flatfield)
        row.addStretch(1)
        gl.addLayout(row)

        l.addStretch(1)
        splitter.addWidget(left)

        self.outputs_flatfield = OutputsPanel()
        self.outputs_flatfield.set_context(self._cfg, stage="flatfield")
        splitter.addWidget(self.outputs_flatfield)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([650, 480])

        lay.addWidget(_hline())
        foot = QtWidgets.QHBoxLayout()
        lay.addLayout(foot)
        self.btn_to_superneon = QtWidgets.QPushButton("Go to SuperNeon →")
        foot.addStretch(1)
        foot.addWidget(self.btn_to_superneon)

        # actions
        self.btn_run_flatfield.clicked.connect(self._do_run_flatfield)
        self.btn_qc_flatfield.clicked.connect(self._open_qc_viewer)
        self.btn_frames_flatfield.clicked.connect(lambda: self._open_frames_window('flatfield'))
        self.btn_to_superneon.clicked.connect(lambda: self.steps.setCurrentRow(5))

        # wire controls → YAML
        def _apply_to_from_ui() -> list[str]:
            out: list[str] = []
            if self.chk_flat_obj.isChecked():
                out.append('obj')
            if self.chk_flat_sky.isChecked():
                out.append('sky')
            if self.chk_flat_sunsky.isChecked():
                out.append('sunsky')
            if self.chk_flat_neon.isChecked():
                out.append('neon')
            return out

        self.chk_flat_enabled.toggled.connect(lambda v: self._cfg_set_path(['flatfield', 'enabled'], bool(v)))
        for cb in (self.chk_flat_obj, self.chk_flat_sky, self.chk_flat_sunsky, self.chk_flat_neon):
            cb.toggled.connect(lambda *_: self._cfg_set_apply_to('flatfield', _apply_to_from_ui()))
        self.chk_flat_bias.toggled.connect(lambda v: self._cfg_set_path(['flatfield', 'bias_subtract'], bool(v)))
        self.chk_flat_png.toggled.connect(lambda v: self._cfg_set_path(['flatfield', 'save_png'], bool(v)))

        self._sync_stage_controls_from_cfg()
        return w

    def _do_run_flatfield(self) -> None:
        if not self._ensure_cfg_saved():
            return
        self._set_step_status(4, "running")
        try:
            ctx = load_context(self._cfg_path)
            run_sequence(ctx, ["flatfield"])
            self._log_info("Flat-fielding done")
            try:
                if hasattr(self, "outputs_flatfield"):
                    self.outputs_flatfield.set_context(self._cfg, stage="flatfield")
            except Exception:
                pass
            self._set_step_status(4, "ok")
            self._maybe_auto_qc()
        except Exception as e:
            self._set_step_status(4, "fail")
            self._log_exception(e)

    # --------------------------- page: superneon ---------------------------

    def _build_page_superneon(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setSpacing(12)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        lay.addWidget(splitter, 1)

        left = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout(left)
        l.setSpacing(12)

        g = _box("SuperNeon")
        l.addWidget(g)
        gl = QtWidgets.QVBoxLayout(g)
        lbl = QtWidgets.QLabel(
            "Stack all NEON frames into a single superneon image,\n"
            "detect candidate peaks in 1D profile, and write QC PNG/CSV."
        )
        lbl.setWordWrap(True)
        gl.addWidget(lbl)

        row = QtWidgets.QHBoxLayout()
        self.btn_run_superneon = QtWidgets.QPushButton("Run: SuperNeon")
        self.btn_run_superneon.setProperty("primary", True)
        self.btn_qc_superneon = QtWidgets.QPushButton("QC")
        self.btn_frames_superneon = QtWidgets.QPushButton("Frames…")
        self.btn_frames_superneon.setToolTip("Open Frames Browser for the SuperNeon stage")
        row.addWidget(self.btn_run_superneon)
        row.addWidget(self.btn_qc_superneon)
        row.addWidget(self.btn_frames_superneon)
        row.addStretch(1)
        gl.addLayout(row)
        l.addStretch(1)

        splitter.addWidget(left)
        self.outputs_superneon = OutputsPanel()
        self.outputs_superneon.set_context(self._cfg, stage="wavesol")
        splitter.addWidget(self.outputs_superneon)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([650, 480])

        lay.addWidget(_hline())
        foot = QtWidgets.QHBoxLayout()
        lay.addLayout(foot)
        self.btn_to_lineid = QtWidgets.QPushButton("Go to LineID →")
        foot.addStretch(1)
        foot.addWidget(self.btn_to_lineid)

        self.btn_run_superneon.clicked.connect(self._do_run_superneon)
        self.btn_qc_superneon.clicked.connect(self._open_qc_viewer)
        self.btn_frames_superneon.clicked.connect(lambda: self._open_frames_window('superneon'))
        self.btn_to_lineid.clicked.connect(lambda: self.steps.setCurrentRow(6))
        return w

    def _do_run_superneon(self) -> None:
        if not self._ensure_cfg_saved():
            return
        self._set_step_status(5, "running")
        try:
            ctx = load_context(self._cfg_path)
            run_sequence(ctx, ["superneon"])
            self._log_info("SuperNeon done")
            try:
                if hasattr(self, "outputs_superneon"):
                    self.outputs_superneon.set_context(self._cfg, stage="wavesol")
            except Exception:
                pass
            self._set_step_status(5, "ok")
            self._maybe_auto_qc()
        except Exception as e:
            self._set_step_status(5, "fail")
            self._log_exception(e)

    # --------------------------- page: lineid ---------------------------

    def _build_page_lineid(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setSpacing(12)

        g = _box("LineID (manual pairs)")
        lay.addWidget(g)
        gl = QtWidgets.QVBoxLayout(g)

        self.lbl_lineid = QtWidgets.QLabel(
            "Interactive identification: match peaks to reference lines and write hand_pairs.txt.\n"
            "Tip: if you have multiple gratings, each one gets its own wavesol/<grating>/ folder."
        )
        self.lbl_lineid.setWordWrap(True)
        gl.addWidget(self.lbl_lineid)

        self.lbl_pairs = QtWidgets.QLabel("Pairs: —")
        self.lbl_pairs.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        gl.addWidget(self.lbl_pairs)

        # Main actions
        row1 = QtWidgets.QHBoxLayout()
        self.btn_open_lineid = QtWidgets.QPushButton("Open LineID GUI")
        self.btn_open_lineid.setProperty("primary", True)
        self.btn_qc_lineid = QtWidgets.QPushButton("QC")
        self.btn_frames_lineid = QtWidgets.QPushButton("Frames…")
        self.btn_frames_lineid.setToolTip("Open Frames Browser for the LineID stage")
        row1.addWidget(self.btn_open_lineid)
        row1.addWidget(self.btn_qc_lineid)
        row1.addWidget(self.btn_frames_lineid)
        row1.addStretch(1)
        gl.addLayout(row1)

        # Pairs management
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Pair sets:"))
        self.combo_pair_sets = QtWidgets.QComboBox()
        self.combo_pair_sets.setMinimumWidth(280)
        self.combo_pair_sets.setToolTip("Select a pair set (built-in or from your user library).")
        self.btn_use_pair_set = QtWidgets.QPushButton("Use selected")
        self.btn_copy_pair_set = QtWidgets.QPushButton("Copy selected → workdir")
        self.btn_save_workdir_pairs = QtWidgets.QPushButton("Save workdir → library")
        self.btn_open_pairs_library = QtWidgets.QPushButton("Library folder")
        # Export options (single file or the whole user library)
        self.btn_export_pairs = QtWidgets.QToolButton()
        self.btn_export_pairs.setText("Export…")
        self.btn_export_pairs.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        mexp = QtWidgets.QMenu(self.btn_export_pairs)
        self.act_export_selected_pair_set = mexp.addAction("Export selected pair set…")
        self.act_export_current_pairs = mexp.addAction("Export current pairs (workdir)…")
        mexp.addSeparator()
        self.act_export_user_library_zip = mexp.addAction("Export full user library (.zip)…")
        self.btn_export_pairs.setMenu(mexp)
        row2.addWidget(self.combo_pair_sets, 1)
        row2.addWidget(self.btn_use_pair_set)
        row2.addWidget(self.btn_copy_pair_set)
        row2.addWidget(self.btn_save_workdir_pairs)
        row2.addWidget(self.btn_open_pairs_library)
        row2.addWidget(self.btn_export_pairs)
        row2.addStretch(1)
        gl.addLayout(row2)

        g_out = _box("Outputs")
        lay.addWidget(g_out)
        ol = QtWidgets.QVBoxLayout(g_out)
        self.outputs_lineid = OutputsPanel()
        self.outputs_lineid.set_context(self._cfg, stage="wavesol")
        ol.addWidget(self.outputs_lineid)


        lay.addStretch(1)
        foot = QtWidgets.QHBoxLayout()
        lay.addLayout(foot)
        self.btn_to_wavesol = QtWidgets.QPushButton("Go to Wavelength solution →")
        foot.addStretch(1)
        foot.addWidget(self.btn_to_wavesol)

        self.btn_open_lineid.clicked.connect(self._do_open_lineid)
        self.btn_qc_lineid.clicked.connect(self._open_qc_viewer)
        self.btn_frames_lineid.clicked.connect(lambda: self._open_frames_window('lineid'))
        self.btn_use_pair_set.clicked.connect(self._do_use_pair_set)
        self.btn_copy_pair_set.clicked.connect(self._do_copy_pair_set)
        self.btn_save_workdir_pairs.clicked.connect(self._do_save_workdir_pairs)
        self.btn_open_pairs_library.clicked.connect(self._do_open_pairs_library)
        self.act_export_selected_pair_set.triggered.connect(self._do_export_selected_pair_set)
        self.act_export_current_pairs.triggered.connect(self._do_export_current_pairs)
        self.act_export_user_library_zip.triggered.connect(self._do_export_user_library_zip)
        self.btn_to_wavesol.clicked.connect(lambda: self.steps.setCurrentRow(7))
        return w

    def _current_pairs_path(self) -> Path | None:
        if not self._cfg:
            return None
        wd = Path(str(self._cfg.get("work_dir", ""))).expanduser()
        if not wd.is_absolute() and self._cfg_path:
            wd = (self._cfg_path.parent / wd).resolve()

        outdir = wavesol_dir(self._cfg)
        wcfg = self._cfg.get("wavesol", {}) if isinstance(self._cfg.get("wavesol"), dict) else {}
        hp_raw = str(wcfg.get("hand_pairs_path", "") or "").strip()
        if hp_raw:
            hp = Path(hp_raw)
            return hp if hp.is_absolute() else (wd / hp).resolve()
        return (outdir / "hand_pairs.txt")

    def _do_open_lineid(self) -> None:
        if not self._ensure_cfg_saved():
            return
        try:
            ctx = load_context(self._cfg_path)
            # if config points to built-in pairs (absolute), avoid overwriting: use workdir file.
            pairs = self._current_pairs_path()
            if pairs and pairs.is_absolute():
                # switch to default workdir path
                self._set_cfg_value("wavesol.hand_pairs_path", "")
                self.editor_yaml.blockSignals(True)
                self.editor_yaml.setPlainText(_yaml_dump(self._cfg or {}))
                self.editor_yaml.blockSignals(False)
                self._do_save_cfg()
                ctx = load_context(self._cfg_path)

            out = run_lineid_prepare(ctx)
            self._log_info(f"LineID wrote: {out}")
            self._refresh_pairs_label()
            self._maybe_auto_qc()
        except Exception as e:
            self._log_exception(e)

    def _do_use_builtin_pairs(self) -> None:
        if not self._cfg:
            self._log_error("Load or create config first")
            return
        disp = ""
        try:
            setup = (self._cfg.get("frames", {}) or {}).get("__setup__", {})
            if isinstance(setup, dict):
                disp = str(setup.get("disperser", "") or "")
        except Exception:
            disp = ""
        it = find_builtin_pairs_for_disperser(disp)
        if it is None:
            self._log_info("No built-in pairs for this disperser yet")
            return

        # point config to builtin (absolute path) — read-only mode
        self._set_cfg_value("wavesol.hand_pairs_path", str(it))
        self.editor_yaml.blockSignals(True)
        self.editor_yaml.setPlainText(_yaml_dump(self._cfg or {}))
        self.editor_yaml.blockSignals(False)
        self._log_info(f"Using built-in pairs: {it}")
        self._refresh_pairs_label()

    def _do_copy_builtin_pairs(self) -> None:
        if not self._ensure_cfg_saved():
            return
        if not self._cfg:
            return
        disp = ""
        setup = (self._cfg.get("frames", {}) or {}).get("__setup__", {})
        if isinstance(setup, dict):
            disp = str(setup.get("disperser", "") or "")
        it = find_builtin_pairs_for_disperser(disp)
        if it is None:
            self._log_info("No built-in pairs for this disperser yet")
            return
        dst = self._current_pairs_path()
        if dst is None:
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(Path(it).read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        # switch to default workdir path
        self._set_cfg_value("wavesol.hand_pairs_path", "")
        self.editor_yaml.blockSignals(True)
        self.editor_yaml.setPlainText(_yaml_dump(self._cfg or {}))
        self.editor_yaml.blockSignals(False)
        self._do_save_cfg()
        self._log_info(f"Copied built-in pairs → {dst}")
        self._refresh_pairs_label()

    def _refresh_pairs_label(self) -> None:
        p = self._current_pairs_path()
        self.lbl_pairs.setText(f"Pairs: {p}" if p else "Pairs: —")
        self._refresh_pair_sets_combo()
        try:
            if hasattr(self, "outputs_lineid"):
                self.outputs_lineid.set_context(self._cfg, stage="wavesol")
        except Exception:
            pass


    def _current_disperser(self) -> str:
        disp = ""
        if self._cfg:
            setup = (self._cfg.get("frames", {}) or {}).get("__setup__", {})
            if isinstance(setup, dict):
                disp = str(setup.get("disperser", "") or "").strip()
        return disp

    def _refresh_pair_sets_combo(self) -> None:
        if not hasattr(self, "combo_pair_sets"):
            return
        try:
            disp = self._current_disperser()
            hp_raw = ""
            if self._cfg and isinstance(self._cfg.get("wavesol"), dict):
                hp_raw = str((self._cfg.get("wavesol") or {}).get("hand_pairs_path", "") or "").strip()

            self.combo_pair_sets.blockSignals(True)
            self.combo_pair_sets.clear()
            self.combo_pair_sets.addItem("Workdir: hand_pairs.txt (default)", {"origin": "workdir", "path": ""})

            items = []
            if disp:
                for ps in list_pair_sets(disp):
                    prefix = "Built-in" if ps.origin == "builtin" else "Library"
                    self.combo_pair_sets.addItem(f"{prefix}: {ps.label}", {"origin": ps.origin, "path": str(ps.path)})
                    items.append(str(ps.path))

            # preselect current
            if hp_raw:
                want = str(Path(hp_raw).expanduser())
                for i in range(1, self.combo_pair_sets.count()):
                    d = self.combo_pair_sets.itemData(i) or {}
                    if str(d.get("path", "")) == want:
                        self.combo_pair_sets.setCurrentIndex(i)
                        break
            else:
                self.combo_pair_sets.setCurrentIndex(0)
        finally:
            self.combo_pair_sets.blockSignals(False)

    def _selected_pair_set(self) -> tuple[str, str]:
        if not hasattr(self, "combo_pair_sets"):
            return "workdir", ""
        data = self.combo_pair_sets.currentData() or {}
        origin = str(data.get("origin", "workdir"))
        path = str(data.get("path", ""))
        return origin, path

    def _do_use_pair_set(self) -> None:
        if not self._cfg:
            self._log_error("Load or create config first")
            return
        origin, path = self._selected_pair_set()
        if origin == "workdir" or not path:
            self._set_cfg_value("wavesol.hand_pairs_path", "")
            self.editor_yaml.blockSignals(True)
            self.editor_yaml.setPlainText(_yaml_dump(self._cfg or {}))
            self.editor_yaml.blockSignals(False)
            self._do_save_cfg()
            self._log_info("Using workdir hand_pairs.txt")
            self._refresh_pairs_label()
            return

        self._set_cfg_value("wavesol.hand_pairs_path", path)
        self.editor_yaml.blockSignals(True)
        self.editor_yaml.setPlainText(_yaml_dump(self._cfg or {}))
        self.editor_yaml.blockSignals(False)
        self._do_save_cfg()
        self._log_info(f"Using pair set: {path}")
        self._refresh_pairs_label()

    def _do_copy_pair_set(self) -> None:
        if not self._cfg:
            self._log_error("Load or create config first")
            return
        origin, path = self._selected_pair_set()
        if origin == "workdir" or not path:
            self._log_error("Select a built-in or library pair set first")
            return
        disp = self._current_disperser()
        if not disp:
            self._log_error("Config has no disperser in frames.__setup__")
            return
        # workdir must be resolved against cfg_path
        wd = Path(str(self._cfg.get("work_dir", ""))).expanduser()
        if not wd.is_absolute() and self._cfg_path:
            wd = (self._cfg_path.parent / wd).resolve()

        try:
            dst = copy_pair_set_to_workdir(disp, wd, Path(path))
        except Exception as e:
            self._log_exception(e)
            return

        self._set_cfg_value("wavesol.hand_pairs_path", "")
        self.editor_yaml.blockSignals(True)
        self.editor_yaml.setPlainText(_yaml_dump(self._cfg or {}))
        self.editor_yaml.blockSignals(False)
        self._do_save_cfg()
        self._log_info(f"Copied pair set → {dst}")
        self._refresh_pairs_label()

    def _do_save_workdir_pairs(self) -> None:
        if not self._cfg:
            self._log_error("Load or create config first")
            return
        disp = self._current_disperser()
        if not disp:
            self._log_error("Config has no disperser in frames.__setup__")
            return
        # always save the workdir hand_pairs.txt (not an absolute built-in path)
        pairs_path = wavesol_dir(self._cfg) / "hand_pairs.txt"
        if not pairs_path.exists():
            self._log_error(f"No workdir pairs file: {pairs_path}")
            return

        default_label = f"{slugify_disperser(disp)}_pairs"
        label, ok = QtWidgets.QInputDialog.getText(self, "Save pairs", "Name for this pair set:", text=default_label)
        if not ok:
            return
        try:
            dest = save_user_pair_set(disp, pairs_path, label=label)
            self._log_info(f"Saved to library: {dest}")
            self._refresh_pair_sets_combo()
        except Exception as e:
            self._log_exception(e)

    def _do_open_pairs_library(self) -> None:
        disp = self._current_disperser()
        root = user_pairs_root()
        if disp:
            root = root / slugify_disperser(disp)
        root.mkdir(parents=True, exist_ok=True)
        self._open_in_explorer(root)

    def _do_export_selected_pair_set(self) -> None:
        disp = self._current_disperser()
        if not disp:
            self._log_error("Config has no disperser in frames.__setup__")
            return
        origin, path = self._selected_pair_set()
        if not path:
            self._log_error("No external pair set selected (choose 'Built-in: …' or 'Library: …')")
            return

        src = Path(path).expanduser()
        default_name = src.name if src.suffix else (src.name + ".txt")
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export pair set",
            str(Path.home() / default_name),
            "Text files (*.txt);;All files (*)",
        )
        if not fn:
            return
        try:
            export_pair_set(src, Path(fn))
            self._log_info(f"Exported pair set -> {fn}")
        except Exception as e:
            self._log_exception(e)

    def _do_export_current_pairs(self) -> None:
        if not self._cfg:
            self._log_error("Load or create config first")
            return
        pairs_path = wavesol_dir(self._cfg) / "hand_pairs.txt"
        if not pairs_path.exists():
            self._log_error(f"No workdir pairs file: {pairs_path}")
            return

        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export current pairs",
            str(Path.home() / pairs_path.name),
            "Text files (*.txt);;All files (*)",
        )
        if not fn:
            return
        try:
            export_pair_set(pairs_path, Path(fn))
            self._log_info(f"Exported current pairs -> {fn}")
        except Exception as e:
            self._log_exception(e)

    def _do_export_user_library_zip(self) -> None:
        root = user_pairs_root()
        root.mkdir(parents=True, exist_ok=True)

        default = f"scorpio_pairs_library_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export pair library (zip)",
            str(Path.home() / default),
            "Zip archive (*.zip);;All files (*)",
        )
        if not fn:
            return
        try:
            export_user_library_zip(Path(fn), include_builtin=False)
            self._log_info(f"Exported pair library -> {fn}")
        except Exception as e:
            self._log_exception(e)
    # --------------------------- page: wavesolution ---------------------------

    def _build_page_wavesol(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setSpacing(12)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        lay.addWidget(splitter, 1)

        left = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout(left)
        l.setSpacing(12)

        g = _box("Wavelength solution (1D + 2D)")
        l.addWidget(g)
        gl = QtWidgets.QVBoxLayout(g)

        lbl = QtWidgets.QLabel(
            "Build 1D λ(x) from hand pairs and a 2D λ(x,y) map from traced lines.\n"
            "You can clean (1) the pair list and (2) the 2D lamp lines used for the surface fit."
        )
        lbl.setWordWrap(True)
        gl.addWidget(lbl)

        self.lbl_wavesol_dir = QtWidgets.QLabel("wavesol: —")
        self.lbl_wavesol_dir.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        gl.addWidget(self.lbl_wavesol_dir)

        row = QtWidgets.QHBoxLayout()
        self.btn_clean_pairs = QtWidgets.QPushButton("Clean pairs…")
        self.btn_clean_wavesol2d = QtWidgets.QPushButton("Clean 2D lines…")
        self.btn_run_wavesol = QtWidgets.QPushButton("Run: Wavelength solution")
        self.btn_run_wavesol.setProperty("primary", True)
        self.btn_qc_wavesol = QtWidgets.QPushButton("QC")
        self.btn_frames_wavesol = QtWidgets.QPushButton("Frames…")
        self.btn_frames_wavesol.setToolTip("Open Frames Browser for the Wavelength solution stage")
        row.addWidget(self.btn_clean_pairs)
        row.addWidget(self.btn_clean_wavesol2d)
        row.addWidget(self.btn_run_wavesol)
        row.addWidget(self.btn_qc_wavesol)
        row.addWidget(self.btn_frames_wavesol)
        row.addStretch(1)
        gl.addLayout(row)
        l.addStretch(1)

        splitter.addWidget(left)
        self.outputs_wavesol = OutputsPanel()
        self.outputs_wavesol.set_context(self._cfg, stage="wavesol")
        splitter.addWidget(self.outputs_wavesol)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([650, 480])

        self.btn_clean_pairs.clicked.connect(self._do_clean_pairs)
        self.btn_clean_wavesol2d.clicked.connect(self._do_clean_wavesol2d)
        self.btn_run_wavesol.clicked.connect(self._do_wavesolution)
        self.btn_qc_wavesol.clicked.connect(self._open_qc_viewer)
        self.btn_frames_wavesol.clicked.connect(lambda: self._open_frames_window('wavesol'))
        return w

    def _do_clean_pairs(self) -> None:
        if not self._ensure_cfg_saved():
            return
        if not self._sync_cfg_from_editor():
            return
        p = self._current_pairs_path()
        if p is None or not p.exists():
            self._log_error("Pairs file not found (run LineID first)")
            return
        deg = int(self._get_cfg_value("wavesol.poly_deg_1d", 4) or 4)
        out = clean_pairs_interactively(p, poly_deg=deg, parent=self)
        if out is None:
            return
        # If user saved to a new file, point config to it (relative if possible)
        cfg = self._cfg or {}
        wd = Path(str(cfg.get("work_dir", "."))).expanduser()
        if self._cfg_path and not wd.is_absolute():
            wd = (self._cfg_path.parent / wd).resolve()
        rel = _rel_to_workdir(wd, out)
        if rel != str(out):
            self._set_cfg_value("wavesol.hand_pairs_path", rel)
        else:
            self._set_cfg_value("wavesol.hand_pairs_path", str(out))
        self.editor_yaml.blockSignals(True)
        self.editor_yaml.setPlainText(_yaml_dump(self._cfg or {}))
        self.editor_yaml.blockSignals(False)
        self._do_save_cfg()
        self._log_info(f"Pairs cleaned: {out}")
        self._refresh_pairs_label()
        try:
            if hasattr(self, "outputs_wavesol"):
                self.outputs_wavesol.set_context(self._cfg, stage="wavesol")
        except Exception:
            pass

    def _do_clean_wavesol2d(self) -> None:
        """Interactive rejection of bad lamp lines used in 2D fit.

        Works on the saved control points (wavesol/control_points_2d.csv).
        It updates wavesol.rejected_lines_A in config.
        """
        if not self._ensure_cfg_saved():
            return
        if not self._sync_cfg_from_editor():
            return
        try:
            from scorpio_pipe.wavesol_paths import wavesol_dir
            from scorpio_pipe.ui.wavesol_2d_cleaner import Wave2DCleanConfig, Wave2DLineCleanerDialog
        except Exception as e:
            self._log_exception(e)
            return

        cfg = self._cfg or {}
        outdir = Path(wavesol_dir(cfg))
        cp_csv = outdir / "control_points_2d.csv"
        if not cp_csv.exists():
            self._log_error("control_points_2d.csv not found (run Wavelength solution once first)")
            return

        # read current rejected list
        cur = self._get_cfg_value("wavesol.rejected_lines_A", [])
        rejected = []
        if isinstance(cur, (list, tuple)):
            for x in cur:
                try:
                    rejected.append(float(x))
                except Exception:
                    pass

        # configure model degrees from current config
        wcfg = (cfg.get("wavesol", {}) if isinstance(cfg.get("wavesol"), dict) else {})
        dlg_cfg = Wave2DCleanConfig(
            model2d=str(wcfg.get("model2d", "auto")),
            power_deg=int(wcfg.get("power_deg", max(int(wcfg.get("cheb_degx", 5)), int(wcfg.get("cheb_degy", 3))))),
            cheb_degx=int(wcfg.get("cheb_degx", 5)),
            cheb_degy=int(wcfg.get("cheb_degy", 3)),
            sigma_clip=float(wcfg.get("cheb_sigma_clip", 3.0)),
            maxiter=int(wcfg.get("cheb_maxiter", 10)),
        )

        dlg = Wave2DLineCleanerDialog(cp_csv, cfg=dlg_cfg, rejected_lines_A=rejected, parent=self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return

        new_rej = dlg.rejected_lines()
        self._set_cfg_value("wavesol.rejected_lines_A", new_rej)
        self.editor_yaml.blockSignals(True)
        self.editor_yaml.setPlainText(_yaml_dump(self._cfg or {}))
        self.editor_yaml.blockSignals(False)
        self._do_save_cfg()
        self._log_info(f"Updated wavesol.rejected_lines_A (N={len(new_rej)})")
        try:
            if hasattr(self, "outputs_wavesol"):
                self.outputs_wavesol.set_context(self._cfg, stage="wavesol")
        except Exception:
            pass

        # save a couple of diagnostic plots for reports/audit
        try:
            saved = dlg.save_plots(outdir, stem='wavesol2d_clean')
            if saved:
                self._log_info('Saved 2D-clean plots:\n' + '\n'.join(f'  {s}' for s in saved))
        except Exception as e:
            self._log_exception(e)

    def _do_wavesolution(self) -> None:
        if not self._ensure_cfg_saved():
            return
        self._set_step_status(7, "running")
        try:
            ctx = load_context(self._cfg_path)
            out = run_wavesolution(ctx)
            self._log_info("Wavelength solution done")
            try:
                if hasattr(self, "outputs_wavesol"):
                    self.outputs_wavesol.set_context(self._cfg, stage="wavesol")
            except Exception:
                pass
            self._set_step_status(7, "ok")
            self._log_info("Outputs:\n" + "\n".join(f"  {k}: {v}" for k, v in out.items()))
            self._maybe_auto_qc()
        except Exception as e:
            self._set_step_status(7, "fail")
            self._log_exception(e)

    # --------------------------- misc helpers ---------------------------

    def _ensure_cfg_saved(self) -> bool:
        if not self._cfg_path:
            # infer
            s = self.edit_cfg_path.text().strip()
            if s:
                self._cfg_path = Path(s).expanduser().resolve()
        if not self._cfg_path:
            self._log_error("No config file. Create or open config first.")
            return False
        if not self._sync_cfg_from_editor():
            return False
        # ensure cfg is on disk for runner
        try:
            self._cfg_path.parent.mkdir(parents=True, exist_ok=True)
            self._cfg_path.write_text(self.editor_yaml.toPlainText(), encoding="utf-8")
        except Exception as e:
            self._log_exception(e)
            return False
        # keep internal cfg in sync
        try:
            self._cfg = load_config(self._cfg_path)
        except Exception:
            pass
        self._refresh_pairs_label()
        if self._cfg:
            try:
                self.lbl_wavesol_dir.setText(f"wavesol: {wavesol_dir(self._cfg)}")
            except Exception:
                pass
        return True

    def _open_qc_viewer(self) -> None:
        if not self._ensure_cfg_saved():
            return
        cfg = self._cfg or {}
        wd = Path(str(cfg.get("work_dir", ""))).expanduser()
        if self._cfg_path and not wd.is_absolute():
            wd = (self._cfg_path.parent / wd).resolve()
        if self._qc is None:
            self._qc = QCViewer(wd)
        else:
            self._qc.set_work_dir(wd)
        self._qc.show()
        self._qc.raise_()
        self._qc.activateWindow()

    # --------------------------- navigation glue ---------------------------

    def _current_work_dir_resolved(self) -> Path | None:
        """Return absolute work_dir from current config/cfg_path (if possible)."""
        cfg = self._cfg or {}
        wd_raw = str(cfg.get('work_dir', '') or '').strip()
        if not wd_raw:
            return None
        wd = Path(wd_raw).expanduser()
        if (not wd.is_absolute()) and self._cfg_path is not None:
            wd = (self._cfg_path.parent / wd).resolve()
        return wd

    def _current_stage_for_step(self, idx: int) -> str | None:
        # Map UI steps to product stages for the inspector outputs view
        return {
            0: None,
            1: None,
            2: 'calib',
            3: 'cosmics',
            4: 'flatfield',
            5: 'wavesol',
            6: 'wavesol',
            7: 'wavesol',
        }.get(int(idx), None)

    def _on_step_changed(self, idx: int) -> None:
        try:
            self.stack.setCurrentIndex(int(idx))
        except Exception:
            pass
        self._refresh_inspector()
        self._refresh_statusbar()

    def _refresh_inspector(self) -> None:
        try:
            stage = self._current_stage_for_step(self.steps.currentRow())
        except Exception:
            stage = None
        wd = self._current_work_dir_resolved()
        try:
            if hasattr(self, 'dock_inspector') and self.dock_inspector is not None:
                self.dock_inspector.panel.set_context(self._cfg, stage=stage, work_dir=wd)
        except Exception:
            pass

    def _open_qc_viewer_for(self, work_dir: Path) -> None:
        """Open QC viewer and point it to the given work directory."""
        wd = Path(work_dir).expanduser().resolve()
        if self._qc is None:
            self._qc = QCViewer(wd)
        else:
            self._qc.set_work_dir(wd)
        self._qc.show()
        self._qc.raise_()
        self._qc.activateWindow()

    # --------------------------- statusbar / shortcuts ---------------------------

    def _build_statusbar_widgets(self) -> None:
        sb = self.statusBar()
        sb.setSizeGripEnabled(True)

        def _mk_btn(caption: str, cb) -> QtWidgets.QToolButton:
            b = QtWidgets.QToolButton()
            b.setText(caption)
            b.setCursor(QtCore.Qt.PointingHandCursor)
            b.clicked.connect(cb)
            b.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
            return b

        self._sb_btn_data = _mk_btn('Data: —', self._open_data_folder)
        self._sb_btn_work = _mk_btn('Work: —', self._open_work_folder)
        self._sb_btn_cfg = _mk_btn('Cfg: —', self._open_cfg_folder)
        self._sb_btn_report = _mk_btn('Report', self._open_report_html)

        sb.addPermanentWidget(self._sb_btn_data)
        sb.addPermanentWidget(self._sb_btn_work)
        sb.addPermanentWidget(self._sb_btn_cfg)
        sb.addPermanentWidget(self._sb_btn_report)

        try:
            self.edit_data_dir.textChanged.connect(lambda *_: self._refresh_statusbar())
            self.edit_cfg_path.textChanged.connect(lambda *_: self._refresh_statusbar())
            if hasattr(self, 'edit_work_dir'):
                self.edit_work_dir.textChanged.connect(lambda *_: self._refresh_statusbar())
        except Exception:
            pass

        self._refresh_statusbar()

    def _short_path(self, p: Path | None) -> str:
        if not p:
            return '—'
        s = str(p)
        # keep it readable in a status bar
        return (p.name or s)

    def _refresh_statusbar(self) -> None:
        try:
            d = Path(self.edit_data_dir.text()).expanduser() if self.edit_data_dir.text().strip() else None
        except Exception:
            d = None
        wd = self._current_work_dir_resolved()
        cfgp = None
        try:
            cfgp = Path(self.edit_cfg_path.text()).expanduser() if self.edit_cfg_path.text().strip() else None
        except Exception:
            cfgp = None

        try:
            self._sb_btn_data.setText(f'Data: {self._short_path(d)}')
            self._sb_btn_data.setToolTip(str(d) if d else '')
            self._sb_btn_work.setText(f'Work: {self._short_path(wd)}')
            self._sb_btn_work.setToolTip(str(wd) if wd else '')
            self._sb_btn_cfg.setText(f'Cfg: {self._short_path(cfgp)}')
            self._sb_btn_cfg.setToolTip(str(cfgp) if cfgp else '')
        except Exception:
            pass

    def _open_data_folder(self) -> None:
        try:
            p = Path(self.edit_data_dir.text()).expanduser()
            if p.exists():
                self._open_in_explorer(p)
        except Exception:
            pass

    def _open_work_folder(self) -> None:
        wd = self._current_work_dir_resolved()
        if wd and wd.exists():
            self._open_in_explorer(wd)

    def _open_cfg_folder(self) -> None:
        try:
            p = Path(self.edit_cfg_path.text()).expanduser()
            if p.exists():
                self._open_in_explorer(p.parent)
        except Exception:
            pass

    def _open_report_html(self) -> None:
        wd = self._current_work_dir_resolved()
        if not wd:
            return
        p = wd / 'report' / 'index.html'
        if p.exists():
            try:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(p)))
            except Exception:
                pass
        else:
            # fallback: open folder
            if (wd / 'report').exists():
                self._open_in_explorer(wd / 'report')

    def _install_shortcuts(self) -> None:
        # Keep the UI fast: shortcuts call existing handlers.
        def _sc(seq: str, cb) -> None:
            s = QtGui.QShortcut(QtGui.QKeySequence(seq), self)
            s.activated.connect(cb)

        _sc('Ctrl+I', self._do_inspect)
        _sc('Ctrl+S', self._do_save_cfg)
        _sc('Ctrl+R', self._run_all_steps)
        _sc('Ctrl+P', self._open_run_plan)
        _sc('Ctrl+Q', self._open_qc_viewer)
        _sc('Ctrl+O', self._open_data_folder)
        _sc('Ctrl+W', self._open_work_folder)

    def _maybe_auto_qc(self) -> None:
        if getattr(self, "act_auto_qc", None) is not None and self.act_auto_qc.isChecked():
            self._open_qc_viewer()

    # --------------------------- menus / toolbar ---------------------------

    def _build_menus(self) -> None:
        mb = self.menuBar()

        # FILE
        m_file = mb.addMenu("File")
        act_open_data = m_file.addAction("Open data folder…")
        act_open_cfg = m_file.addAction("Open config.yaml…")
        act_save_cfg = m_file.addAction("Save config")
        m_file.addSeparator()
        act_exit = m_file.addAction("Exit")

        act_open_data.triggered.connect(self._browse_data_dir)
        act_open_cfg.triggered.connect(self._menu_open_cfg)
        act_save_cfg.triggered.connect(self._do_save_cfg)
        act_exit.triggered.connect(self.close)

        # VIEW
        m_view = mb.addMenu("View")
        act_qc = m_view.addAction("QC Viewer")
        act_qc.triggered.connect(self._open_qc_viewer)

        self.act_toggle_log = m_view.addAction("Log panel")
        self.act_toggle_log.setCheckable(True)
        self.act_toggle_log.setChecked(True)
        self.act_toggle_log.triggered.connect(lambda checked: self.dock_log.setVisible(checked))
        self.dock_log.visibilityChanged.connect(lambda v: self.act_toggle_log.setChecked(v))

        m_view.addSeparator()
        self.act_theme_dark = m_view.addAction("Theme: Dark")
        self.act_theme_light = m_view.addAction("Theme: Light")
        self.act_theme_dark.triggered.connect(lambda: self._set_theme("dark"))
        self.act_theme_light.triggered.connect(lambda: self._set_theme("light"))

        # TOOLS
        m_tools = mb.addMenu("Tools")
        act_instr = m_tools.addAction("Instrument database")
        act_instr.triggered.connect(self._open_instrument_db)

        # HELP
        m_help = mb.addMenu("Help")
        act_manual = m_help.addAction("Quick manual")
        act_about = m_help.addAction("About")
        act_manual.triggered.connect(self._open_manual)
        act_about.triggered.connect(self._open_about)

        # persistent settings
        auto_qc = bool(self._settings.value("ui/auto_qc", True, type=bool))
        self.act_auto_qc = QtGui.QAction("Auto QC", self)
        self.act_auto_qc.setCheckable(True)
        self.act_auto_qc.setChecked(auto_qc)
        self.act_auto_qc.triggered.connect(lambda v: self._settings.setValue("ui/auto_qc", bool(v)))

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setFloatable(False)

        act_run_all = QtGui.QAction("Run all", self)
        act_run_all.triggered.connect(self._run_all_steps)
        tb.addAction(act_run_all)

        act_plan = QtGui.QAction("Plan", self)
        act_plan.triggered.connect(self._open_run_plan)
        tb.addAction(act_plan)

        act_qc = QtGui.QAction("QC", self)
        act_qc.triggered.connect(self._open_qc_viewer)
        tb.addAction(act_qc)

        # Frames Browser (per-stage, non-modal)
        frames_btn = QtWidgets.QToolButton()
        frames_btn.setText("Frames")
        frames_btn.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        frames_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        m = QtWidgets.QMenu(frames_btn)
        stage_menu = [
            ("project", "Project"),
            ("calib", "Superbias"),
            ("cosmics", "Cosmics"),
            ("flatfield", "Flat-field"),
            ("superneon", "SuperNeon"),
            ("lineid", "LineID"),
            ("wavesol", "Wavesol"),
        ]
        for key, title in stage_menu:
            act = m.addAction(title)
            act.triggered.connect(lambda _=False, k=key: self._open_frames_window(k))
        frames_btn.setMenu(m)

        def _stage_for_current_step() -> str:
            idx = int(self.steps.currentRow()) if hasattr(self, 'steps') else 0
            mapping = {0: 'project', 2: 'calib', 3: 'cosmics', 4: 'flatfield', 5: 'superneon', 6: 'lineid', 7: 'wavesol'}
            return mapping.get(idx, 'project')

        frames_btn.clicked.connect(lambda: self._open_frames_window(_stage_for_current_step()))
        tb.addWidget(frames_btn)

        tb.addSeparator()
        tb.addAction(self.act_auto_qc)

        tb.addSeparator()
        act_dark = QtGui.QAction("Dark", self)
        act_light = QtGui.QAction("Light", self)
        act_dark.triggered.connect(lambda: self._set_theme("dark"))
        act_light.triggered.connect(lambda: self._set_theme("light"))
        tb.addAction(act_dark)
        tb.addAction(act_light)

    def _open_run_plan(self) -> None:
        try:
            if not self._sync_cfg_from_editor():
                return
            if not self._cfg:
                self._log_error("No config loaded")
                return
            RunPlanDialog(self._cfg, parent=self).exec()
        except Exception as e:
            self._log_exception(e)

    def _set_theme(self, mode: str) -> None:
        try:
            app = QtWidgets.QApplication.instance()
            if app is not None:
                apply_theme(app, mode=mode)
            self._settings.setValue("ui/theme", mode)
        except Exception as e:
            self._log_exception(e)

    def _browse_data_dir(self) -> None:
        """Menu action: open (or pick) the raw data folder."""
        try:
            raw = (self.edit_data_dir.text() or "").strip()
            if raw:
                p = Path(raw).expanduser()
                if p.exists():
                    self._open_in_explorer(p)
                    return
            # fall back to dialog
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Open data folder", str(Path.home()))
            if d:
                self.edit_data_dir.setText(d)
                self._open_in_explorer(Path(d))
        except Exception as e:
            self._log_exception(e)



    def _menu_open_cfg(self) -> None:
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open config", str(Path.home()), "YAML (*.yaml *.yml)")
        if fn:
            self.edit_cfg_path.setText(fn)
            self._load_config(Path(fn))

    def _open_instrument_db(self) -> None:
        try:
            from scorpio_pipe.ui.instrument_browser import InstrumentBrowserDialog
            dlg = InstrumentBrowserDialog(parent=self)
            dlg.exec()
        except Exception as e:
            self._log_exception(e)

    def _open_manual(self) -> None:
        try:
            from scorpio_pipe.ui.simple_text_viewer import TextViewerDialog
            here = Path(__file__).resolve().parent.parent  # scorpio_pipe/
            manual = here / "resources" / "docs" / "MANUAL.md"
            if not manual.exists():
                raise FileNotFoundError(str(manual))
            TextViewerDialog("Scorpio Pipe — Quick manual", manual.read_text(encoding="utf-8"), parent=self).exec()
        except Exception as e:
            self._log_exception(e)

    def _open_about(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "About Scorpio Pipe",
            "Scorpio Pipe\n\nA reduction pipeline for SCORPIO/SCORPIO-2 long-slit spectroscopy.\n"
            "This build focuses on a clean, guided UI, QC at each step, and reproducible configs.",
        )

    def _run_all_steps(self) -> None:
        """Run the full pipeline in the intended order.

        We keep it conservative: stop on the first error to avoid burying the user.
        """
        if not self._ensure_cfg_saved():
            return
        chain = [
            (2, "Calibrations", self._do_run_calib),
            (3, "Cosmics", self._do_run_cosmics),
            (4, "SuperNeon", self._do_run_superneon),
            (5, "LineID", self._do_open_lineid),
            (6, "Wavelength solution", self._do_wavesolution),
        ]
        for row, name, fn in chain:
            try:
                self.steps.setCurrentRow(row)
                QtWidgets.QApplication.processEvents()
                self._log_info(f"--- RUN: {name} ---")
                fn()
            except Exception as e:
                self._log_exception(e)
                break

    def _update_enables(self) -> None:
        has_inspect = self._inspect is not None
        self.btn_make_cfg.setEnabled(has_inspect)
        self.btn_suggest_workdir.setEnabled(True)
        self.btn_run_calib.setEnabled(self._cfg_path is not None or bool(self.edit_cfg_path.text().strip()))
        self.btn_run_cosmics.setEnabled(self.btn_run_calib.isEnabled())
        if hasattr(self, 'btn_run_flatfield'):
            self.btn_run_flatfield.setEnabled(self.btn_run_calib.isEnabled())
        self.btn_run_superneon.setEnabled(self.btn_run_calib.isEnabled())
        self.btn_open_lineid.setEnabled(self.btn_run_calib.isEnabled())
        self.btn_run_wavesol.setEnabled(self.btn_run_calib.isEnabled())

    # --------------------------- logging helpers ---------------------------

    def _log_info(self, msg: str) -> None:
        self.log_view.appendPlainText(msg)

    def _log_error(self, msg: str) -> None:
        self.log_view.appendPlainText("[ERROR] " + msg)

    def _log_exception(self, e: BaseException) -> None:
        self._log_error(f"{type(e).__name__}: {e}")
        tb = traceback.format_exc(limit=12)
        self.log_view.appendPlainText(tb)
