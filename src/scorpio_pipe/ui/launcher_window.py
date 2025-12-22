from __future__ import annotations

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


# --------------------------- main window ---------------------------


class LauncherWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scorpio Pipe")
        self.resize(1240, 780)

        self._settings = load_ui_settings()

        self._inspect: InspectResult | None = None
        self._cfg_path: Path | None = None
        self._cfg: dict[str, Any] | None = None
        self._qc: QCViewer | None = None

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
            "4  SuperNeon",
            "5  Line ID",
            "6  Wavelength solution",
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
        self.page_superneon = self._build_page_superneon()
        self.page_lineid = self._build_page_lineid()
        self.page_wavesol = self._build_page_wavesol()
        for p in [
            self.page_project,
            self.page_config,
            self.page_calib,
            self.page_superneon,
            self.page_lineid,
            self.page_wavesol,
        ]:
            self.stack.addWidget(p)

        self.steps.currentRowChanged.connect(self.stack.setCurrentIndex)

        # Dock: log (collapsible)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.dock_log = QtWidgets.QDockWidget("Log", self)
        self.dock_log.setObjectName("dock_log")
        self.dock_log.setWidget(self.log_view)
        self.dock_log.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea | QtCore.Qt.TopDockWidgetArea)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.dock_log)
        self.dock_log.setMinimumHeight(160)

        # Attach log handler
        install_qt_log(self.log_view)

        self.statusBar().showMessage("Ready")

        # -------------- menu / toolbar --------------
        self._build_menus()
        self._build_toolbar()

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

        # Inspect
        self.btn_inspect = QtWidgets.QPushButton("Inspect dataset")
        self.btn_inspect.setProperty("primary", True)
        self.lbl_inspect = QtWidgets.QLabel("—")
        self.lbl_inspect.setWordWrap(True)
        gl.addRow(self.btn_inspect, self.lbl_inspect)

        # Actions
        act = QtWidgets.QHBoxLayout()
        lay.addLayout(act)
        self.btn_to_config = QtWidgets.QPushButton("Go to Config →")
        act.addStretch(1)
        act.addWidget(self.btn_to_config)
        lay.addStretch(1)

        # signals
        self.btn_pick_data_dir.clicked.connect(self._pick_data_dir)
        self.btn_open_cfg.clicked.connect(self._open_existing_cfg)
        self.btn_inspect.clicked.connect(self._do_inspect)
        self.btn_to_config.clicked.connect(lambda: self.steps.setCurrentRow(1))
        self.edit_data_dir.textChanged.connect(lambda *_: self._update_enables())

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
            self._set_step_status(0, "ok")
        except Exception as e:
            self._set_step_status(0, "fail")
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

        # --- setup ---
        g_setup = _box("Target & setup")
        lay.addWidget(g_setup)
        fl = QtWidgets.QFormLayout(g_setup)

        # Object
        row_obj = QtWidgets.QHBoxLayout()
        self.combo_object = QtWidgets.QComboBox()
        self.combo_object.setEditable(True)
        self.combo_object.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
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
        lay.addWidget(_hline())

        # tabs: Parameters / YAML
        tabs = QtWidgets.QTabWidget()
        lay.addWidget(tabs, 1)

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
        self.btn_save_cfg = QtWidgets.QPushButton("Save")
        self.btn_save_cfg.setProperty("primary", True)
        self.lbl_cfg_state = QtWidgets.QLabel("—")
        bar.addWidget(self.btn_validate_yaml)
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

    def _suggest_work_dir(self) -> None:
        data_dir = Path(self.edit_data_dir.text()).expanduser()
        obj = (self.combo_object.currentText() or "object").strip() or "object"
        disp = (self.combo_disperser.currentText() or "").strip()
        root = data_dir.parent if data_dir.exists() else Path.home()
        obj_slug = slugify_disperser(obj)
        disp_slug = slugify_disperser(disp)
        wd = root / "scorpio_work" / obj_slug / disp_slug
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
        self._update_enables()

    def _on_disperser_changed(self, *_: object) -> None:
        self._update_slit_binning_from_inspect()
        self._update_setup_hint()
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
        work_dir = Path(self.edit_work_dir.text()).expanduser()
        if not obj:
            self._log_error("Object is empty")
            return
        if not work_dir:
            self._log_error("Work directory is empty")
            return

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
        self.editor_yaml.setPlainText(_yaml_dump(cfg))
        self.editor_yaml.blockSignals(False)

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
        if self._cfg_path:
            self.lbl_cfg_state.setText(f"Edited (not saved): {self._cfg_path.name}")
        else:
            self.lbl_cfg_state.setText("Edited")

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
        return True

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
        self._cfg_path.write_text(self.editor_yaml.toPlainText(), encoding="utf-8")
        self._log_info(f"Saved: {self._cfg_path}")
        self.lbl_cfg_state.setText(f"Saved: {self._cfg_path.name}")
        self._update_enables()

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

    def _build_page_calib(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setSpacing(12)

        g = _box("Calibrations")
        lay.addWidget(g)
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
        row.addWidget(self.btn_run_calib)
        row.addWidget(self.btn_qc_calib)
        row.addStretch(1)
        gl.addLayout(row)

        lay.addStretch(1)
        foot = QtWidgets.QHBoxLayout()
        lay.addLayout(foot)
        self.btn_to_superneon = QtWidgets.QPushButton("Go to SuperNeon →")
        foot.addStretch(1)
        foot.addWidget(self.btn_to_superneon)

        self.btn_run_calib.clicked.connect(self._do_run_calib)
        self.btn_qc_calib.clicked.connect(self._open_qc_viewer)
        self.btn_to_superneon.clicked.connect(lambda: self.steps.setCurrentRow(3))
        return w

    def _do_run_calib(self) -> None:
        if not self._ensure_cfg_saved():
            return
        self._set_step_status(2, "running")
        try:
            ctx = load_context(self._cfg_path)
            run_sequence(ctx, ["manifest", "superbias"])
            self._log_info("Calibrations done")
            self._set_step_status(2, "ok")
            self._maybe_auto_qc()
        except Exception as e:
            self._set_step_status(2, "fail")
            self._log_exception(e)

    # --------------------------- page: superneon ---------------------------

    def _build_page_superneon(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setSpacing(12)

        g = _box("SuperNeon")
        lay.addWidget(g)
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
        row.addWidget(self.btn_run_superneon)
        row.addWidget(self.btn_qc_superneon)
        row.addStretch(1)
        gl.addLayout(row)

        lay.addStretch(1)
        foot = QtWidgets.QHBoxLayout()
        lay.addLayout(foot)
        self.btn_to_lineid = QtWidgets.QPushButton("Go to LineID →")
        foot.addStretch(1)
        foot.addWidget(self.btn_to_lineid)

        self.btn_run_superneon.clicked.connect(self._do_run_superneon)
        self.btn_qc_superneon.clicked.connect(self._open_qc_viewer)
        self.btn_to_lineid.clicked.connect(lambda: self.steps.setCurrentRow(4))
        return w

    def _do_run_superneon(self) -> None:
        if not self._ensure_cfg_saved():
            return
        self._set_step_status(3, "running")
        try:
            ctx = load_context(self._cfg_path)
            run_sequence(ctx, ["superneon"])
            self._log_info("SuperNeon done")
            self._set_step_status(3, "ok")
            self._maybe_auto_qc()
        except Exception as e:
            self._set_step_status(3, "fail")
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
        row1.addWidget(self.btn_open_lineid)
        row1.addWidget(self.btn_qc_lineid)
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


        lay.addStretch(1)
        foot = QtWidgets.QHBoxLayout()
        lay.addLayout(foot)
        self.btn_to_wavesol = QtWidgets.QPushButton("Go to Wavelength solution →")
        foot.addStretch(1)
        foot.addWidget(self.btn_to_wavesol)

        self.btn_open_lineid.clicked.connect(self._do_open_lineid)
        self.btn_qc_lineid.clicked.connect(self._open_qc_viewer)
        self.btn_use_pair_set.clicked.connect(self._do_use_pair_set)
        self.btn_copy_pair_set.clicked.connect(self._do_copy_pair_set)
        self.btn_save_workdir_pairs.clicked.connect(self._do_save_workdir_pairs)
        self.btn_open_pairs_library.clicked.connect(self._do_open_pairs_library)
        self.act_export_selected_pair_set.triggered.connect(self._do_export_selected_pair_set)
        self.act_export_current_pairs.triggered.connect(self._do_export_current_pairs)
        self.act_export_user_library_zip.triggered.connect(self._do_export_user_library_zip)
        self.btn_to_wavesol.clicked.connect(lambda: self.steps.setCurrentRow(5))
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

        g = _box("Wavelength solution (1D + 2D)")
        lay.addWidget(g)
        gl = QtWidgets.QVBoxLayout(g)

        lbl = QtWidgets.QLabel(
            "Build 1D λ(x) from hand pairs and a 2D Chebyshev λ(x,y) map from traced lines.\n"
            "You can clean the pair list before fitting (reject bad lines)."
        )
        lbl.setWordWrap(True)
        gl.addWidget(lbl)

        self.lbl_wavesol_dir = QtWidgets.QLabel("wavesol: —")
        self.lbl_wavesol_dir.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        gl.addWidget(self.lbl_wavesol_dir)

        row = QtWidgets.QHBoxLayout()
        self.btn_clean_pairs = QtWidgets.QPushButton("Clean pairs…")
        self.btn_run_wavesol = QtWidgets.QPushButton("Run: Wavelength solution")
        self.btn_run_wavesol.setProperty("primary", True)
        self.btn_qc_wavesol = QtWidgets.QPushButton("QC")
        row.addWidget(self.btn_clean_pairs)
        row.addWidget(self.btn_run_wavesol)
        row.addWidget(self.btn_qc_wavesol)
        row.addStretch(1)
        gl.addLayout(row)

        lay.addStretch(1)
        self.btn_clean_pairs.clicked.connect(self._do_clean_pairs)
        self.btn_run_wavesol.clicked.connect(self._do_wavesolution)
        self.btn_qc_wavesol.clicked.connect(self._open_qc_viewer)
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

    def _do_wavesolution(self) -> None:
        if not self._ensure_cfg_saved():
            return
        self._set_step_status(5, "running")
        try:
            ctx = load_context(self._cfg_path)
            out = run_wavesolution(ctx)
            self._log_info("Wavelength solution done")
            self._set_step_status(5, "ok")
            self._log_info("Outputs:\n" + "\n".join(f"  {k}: {v}" for k, v in out.items()))
            self._maybe_auto_qc()
        except Exception as e:
            self._set_step_status(5, "fail")
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

        act_qc = QtGui.QAction("QC", self)
        act_qc.triggered.connect(self._open_qc_viewer)
        tb.addAction(act_qc)

        tb.addSeparator()
        tb.addAction(self.act_auto_qc)

        tb.addSeparator()
        act_dark = QtGui.QAction("Dark", self)
        act_light = QtGui.QAction("Light", self)
        act_dark.triggered.connect(lambda: self._set_theme("dark"))
        act_light.triggered.connect(lambda: self._set_theme("light"))
        tb.addAction(act_dark)
        tb.addAction(act_light)

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
            (3, "SuperNeon", self._do_run_superneon),
            (4, "LineID", self._do_open_lineid),
            (5, "Wavelength solution", self._do_wavesolution),
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
