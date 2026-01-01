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
from scorpio_pipe.ui.cosmics_manual import CosmicsManualDialog
from scorpio_pipe.ui.frame_browser import SelectedFrame
from scorpio_pipe.ui.outputs_panel import OutputsToolDialog
from scorpio_pipe.ui.config_defaults import schema_default
from scorpio_pipe.ui.param_metadata import get_param_meta
from scorpio_pipe.ui.delayed_tooltip import install_delayed_tooltip
from scorpio_pipe.ui.run_plan_dialog import RunPlanDialog
from scorpio_pipe.ui.config_diff import ConfigDiffDialog
from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.wavesol_paths import wavesol_dir
from scorpio_pipe.stage_registry import iter_gui_stages
from scorpio_pipe.pairs_library import (
    list_pair_sets,
    save_user_pair_set,
    copy_pair_set_to_workdir,
    user_pairs_root,
    export_pair_set,
    export_user_library_zip,
)
from scorpio_pipe.ui.theme import apply_theme, load_ui_settings
from scorpio_pipe.instrument_db import find_grism
from scorpio_pipe.qc.gate import QCGateError



# --------------------------- tiny widgets ---------------------------


class ParamLabel(QtWidgets.QLabel):
    """Focusable, eliding label with delayed tooltip.

    Tooltip shows after ~200 ms on hover or keyboard focus.
    """

    def __init__(self, text: str, tooltip: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._full_text = str(text)
        self._tooltip_text = str(tooltip or "")
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setCursor(QtCore.Qt.WhatsThisCursor)
        self.setToolTip("" if not self._tooltip_text else self._tooltip_text)
        self.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)

        self._show_delay_ms = 200
        self._hide_delay_ms = 200

        self._t_show = QtCore.QTimer(self)
        self._t_show.setSingleShot(True)
        self._t_show.timeout.connect(self._show_tooltip_now)

        self._t_hide = QtCore.QTimer(self)
        self._t_hide.setSingleShot(True)
        self._t_hide.timeout.connect(lambda: QtWidgets.QToolTip.hideText())

        self._update_elide()

    def set_full_text(self, text: str) -> None:
        self._full_text = str(text)
        self._update_elide()

    def _update_elide(self) -> None:
        fm = self.fontMetrics()
        el = fm.elidedText(self._full_text, QtCore.Qt.TextElideMode.ElideRight, max(10, self.width()))
        super().setText(el)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_elide()

    def enterEvent(self, event: QtCore.QEvent) -> None:
        super().enterEvent(event)
        self._t_hide.stop()
        if self._tooltip_text:
            self._t_show.start(self._show_delay_ms)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        super().leaveEvent(event)
        self._t_show.stop()
        if self._tooltip_text:
            self._t_hide.start(self._hide_delay_ms)

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        super().focusInEvent(event)
        self._t_hide.stop()
        if self._tooltip_text:
            self._t_show.start(self._show_delay_ms)

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        super().focusOutEvent(event)
        self._t_show.stop()
        if self._tooltip_text:
            self._t_hide.start(self._hide_delay_ms)

    def _show_tooltip_now(self) -> None:
        if not self._tooltip_text:
            return
        pos = self.mapToGlobal(QtCore.QPoint(self.width() // 2, self.height()))
        QtWidgets.QToolTip.showText(pos, self._tooltip_text, self)


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


def _collapsible(
    title: str, *, expanded: bool = False
) -> tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout, QtWidgets.QToolButton]:
    """Return (widget, content_layout, header_button) for a simple collapsible section."""
    root = QtWidgets.QWidget()
    v = QtWidgets.QVBoxLayout(root)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(6)

    btn = QtWidgets.QToolButton()
    btn.setText(title)
    btn.setCheckable(True)
    btn.setChecked(bool(expanded))
    btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
    btn.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
    btn.setCursor(QtCore.Qt.PointingHandCursor)
    btn.setStyleSheet("QToolButton { border: none; font-weight: 600; padding: 2px 0; }")

    content = QtWidgets.QWidget()
    content.setVisible(bool(expanded))
    content_lay = QtWidgets.QVBoxLayout(content)
    content_lay.setContentsMargins(10, 6, 0, 0)
    content_lay.setSpacing(8)

    def _toggle(checked: bool) -> None:
        content.setVisible(bool(checked))
        btn.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)

    btn.toggled.connect(_toggle)

    v.addWidget(btn)
    v.addWidget(content)
    return root, content_lay, btn


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


def _safe_int(v: Any, default: int) -> int:
    """Best-effort int conversion for UI sync.

    The YAML editor is user-editable, so values can be stored as strings.
    This helper prevents GUI callbacks from crashing on non-numeric input.
    """

    if v is None:
        return default
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int,)):
        return int(v)
    try:
        # tolerate "3", "3.0", "  3  "
        return int(float(str(v).strip()))
    except Exception:
        return default


def _safe_float(v: Any, default: float) -> float:
    """Best-effort float conversion for UI sync."""
    if v is None:
        return default
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip())
    except Exception:
        return default


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
        if (parent / "pyproject.toml").is_file() or (
            parent / "scripts" / "windows" / "setup.bat"
        ).is_file():
            return parent
    return Path.cwd().resolve()


# --------------------------- main window ---------------------------


class LauncherWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        try:
            from scorpio_pipe.version import PIPELINE_VERSION

            self.setWindowTitle(f"Scorpio Pipe {PIPELINE_VERSION}")
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

        # Stage parameters: pending (not yet applied) changes per stage.
        # We only write into YAML/config when the user presses "Apply".
        self._stage_pending: dict[str, dict[str, Any]] = {}
        self._stage_dirty: dict[str, bool] = {}
        self._stage_apply_btns: dict[str, QtWidgets.QPushButton] = {}
        self._stage_dirty_labels: dict[str, QtWidgets.QLabel] = {}

        # Param UI: default-icon reset bindings (refreshed after config sync)
        self._param_default_buttons: list[callable] = []

        # Outputs panels registered per stage page (for in-page drawers).

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
        self._stage_keys = [s.key for s in iter_gui_stages()]
        self._stage_row = {k: i for i, k in enumerate(self._stage_keys)}
        self._stage_titles = [s.title for s in iter_gui_stages()]
        for title in self._stage_titles:
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

        # Per-page Outputs drawer state
        # Outputs are shown in a detached tool window (not inside stage pages).

        # Create pages
        self.page_project = self._build_page_project()
        self.page_config = self._build_page_config()
        self.page_calib = self._build_page_calib()
        self.page_cosmics = self._build_page_cosmics()
        self.page_flatfield = self._build_page_flatfield()
        self.page_superneon = self._build_page_superneon()
        self.page_lineid = self._build_page_lineid()
        self.page_wavesol = self._build_page_wavesol()
        self.page_linearize = self._build_page_linearize()
        self.page_sky = self._build_page_sky()
        self.page_stack2d = self._build_page_stack2d()
        self.page_extract1d = self._build_page_extract1d()
        for p in [
            self.page_project,
            self.page_config,
            self.page_calib,
            self.page_cosmics,
            self.page_flatfield,
            self.page_superneon,
            self.page_lineid,
            self.page_wavesol,
            self.page_linearize,
            self.page_sky,
            self.page_stack2d,
            self.page_extract1d,
        ]:
            self.stack.addWidget(p)

        # Map canonical stage list (01..12) to actual UI pages.
        _page_by_stage_key = {
            "project": 0,
            "setup": 1,
            "biascorr": 2,
            "flatfield": 4,
            "cosmics": 3,
            "superneon": 5,
            "arclineid": 6,
            "wavesol": 7,
            "linearize": 8,
            "sky": 9,
            "stack2d": 10,
            "extract1d": 11,
        }
        self._stage_page_index = [_page_by_stage_key[k] for k in self._stage_keys]

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

        # Attach log handler
        # Route Python logging into the UI log panel.
        install_qt_log(self.log_view, logger_name="scorpio_pipe")

        try:
            import logging as _logging
            from scorpio_pipe.version import PIPELINE_VERSION

            _logging.getLogger("scorpio_pipe").info("Scorpio Pipe %s", PIPELINE_VERSION)
        except Exception:
            pass

        self.statusBar().showMessage("Ready")

        # -------------- menu / toolbar --------------
        self._build_menus()
        self._build_toolbar()

        # Outputs: a detached tool window (keeps stage pages layout stable).
        self.outputs_tool = OutputsToolDialog(parent=self)
        self.outputs_tool.hide()
        try:
            self.outputs_tool.visibilityChanged.connect(
                self._on_outputs_tool_visibility_changed
            )
        except Exception:
            pass

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
            if hasattr(self, "lbl_wavesol_dir") and self._cfg:
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

            self._STATUS_ICONS.update(
                {
                    "idle": _dot(QtGui.QColor(140, 140, 140)),
                    "running": _dot(QtGui.QColor(47, 111, 237)),
                    "ok": _dot(QtGui.QColor(46, 160, 67)),
                    "warn": _dot(QtGui.QColor(230, 159, 0)),
                    "fail": _dot(QtGui.QColor(220, 50, 47)),
                }
            )
        return self._STATUS_ICONS.get(status, self._STATUS_ICONS["idle"])

    def _set_step_status(self, idx: int, status: str) -> None:
        try:
            if hasattr(self, "_step_items") and 0 <= idx < len(self._step_items):
                it = self._step_items[idx]
                it.setIcon(self._icon_status(status))
        except Exception:
            pass

    def _stage_row_index(self, stage_key: str) -> int:
        try:
            return int(getattr(self, '_stage_row', {}).get(stage_key, -1))
        except Exception:
            return -1

    def _open_in_explorer(self, path: Path) -> None:
        try:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))
        except Exception as e:
            self._log_exception(e)

    # --------------------------- small UI helpers ---------------------------

    def _format_tooltip(self, text: str, default_line: str | None = None) -> str:
        """Normalize tooltip text.

        Rules:
        - max 4 non-empty lines
        - concrete, no wall-of-text
        - if ``default_line`` provided and the text does not mention Default/По умолчанию,
          ensure a Default line is present.
        """
        raw = [ln.strip() for ln in (text or "").splitlines()]
        lines = [ln for ln in raw if ln]
        if not lines and not default_line:
            return ""

        def _has_default(ls: list[str]) -> bool:
            joined = " ".join(ls).lower()
            return ("default" in joined) or ("по умолч" in joined)

        if default_line and not _has_default(lines):
            if len(lines) >= 4:
                lines = lines[:3]
            lines.append(default_line)

        lines = lines[:4]
        return "\n".join(lines)

    def _param_label(
        self,
        title: str,
        help_ru: str,
        cfg_path: str | None = None,
        units: str | None = None,
    ) -> QtWidgets.QWidget:
        """Label column for parameter rows.

        - Tooltip on hover/focus (label itself)
        - Fixed width, single-line, ellipsis
        """
        # Prefer centralized UI metadata when available.
        typical: str | None = None
        if cfg_path:
            meta = get_param_meta(cfg_path)
            if meta is not None:
                title = meta.label or title
                help_ru = meta.tooltip or help_ru
                units = meta.units if meta.units else units
                typical = (meta.typical or None)

        # Units are shown next to the parameter name in square brackets.
        # Do not invent units: only schema/metadata may provide them.
        t = str(title)
        if units:
            u = str(units).strip()
            if u and ("[" not in t):
                t = f"{t} [{u}]"

        default_line = None
        if cfg_path:
            try:
                dv = schema_default(cfg_path)
                dv_s = self._fmt_default_value(dv)
                default_line = f"Default: {dv_s}"
                if typical:
                    default_line = f"{default_line} • Usually: {typical}"
            except Exception:
                # Not all labels have a schema entry (e.g. composite/pseudo rows).
                default_line = None

        lbl = ParamLabel(t, self._format_tooltip(help_ru, default_line=default_line))
        lbl.setFixedWidth(240)
        lbl.setMinimumHeight(28)
        return lbl

    def _param_field(
        self,
        field: QtWidgets.QWidget,
        cfg_path: str | None = None,
        *,
        show_default_icon: bool = True,
    ) -> QtWidgets.QWidget:
        """Wrap a parameter control with a compact "default" icon.

        The icon is always visible:
        - when value == schema default → grey, not clickable
        - when value != schema default → highlighted, click resets to the schema default
        """
        # QFormLayout supports adding QLayout directly, but for consistent
        # rendering (and for the default-icon wrapper) we normalize layouts
        # into widgets here.
        if isinstance(field, QtWidgets.QLayout):
            w = QtWidgets.QWidget()
            w.setLayout(field)
            field = w

        root = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(field, 1)

        if not show_default_icon:
            return root

        btn = QtWidgets.QToolButton()
        btn.setAutoRaise(True)
        btn.setFixedSize(18, 18)
        _base_icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload)

        def _grey_icon(icon: QtGui.QIcon) -> QtGui.QIcon:
            try:
                pm = icon.pixmap(16, 16)
                out = QtGui.QPixmap(pm.size())
                out.fill(QtCore.Qt.transparent)
                p = QtGui.QPainter(out)
                p.setOpacity(0.35)
                p.drawPixmap(0, 0, pm)
                p.end()
                return QtGui.QIcon(out)
            except Exception:
                return icon

        _greyed_icon = _grey_icon(_base_icon)
        btn.setIcon(_greyed_icon)
        btn.setToolTip("Default")
        btn.setCursor(QtCore.Qt.ArrowCursor)
        btn.setProperty("active", False)
        btn.setStyleSheet(
            "QToolButton { border: 1px solid transparent; border-radius: 4px; padding: 0px; }"
            "QToolButton:hover { border-color: rgba(127,127,127,120); }"
            "QToolButton[active=\"true\"] { background: rgba(0,120,215,40); border-color: rgba(0,120,215,140); }"
        )
        h.addWidget(btn, 0, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        # Collect bindable widgets (the control itself or descendants for composite rows).
        bindables: list[tuple[QtWidgets.QWidget, object, object, object, str]] = []

        _MISSING = object()
        cfg_default: object = _MISSING
        if cfg_path:
            try:
                cfg_default = schema_default(cfg_path)
            except Exception:
                cfg_default = _MISSING

        def _bind_one(w: QtWidgets.QWidget) -> None:

            if isinstance(w, QtWidgets.QSpinBox):

                get = w.value

                setv = w.setValue

                if cfg_default is _MISSING:

                    dv_value = int(w.value())

                    dv_disp = str(dv_value)

                elif cfg_default is None:

                    # Many optional numeric params use 0 / min as 'auto' in UI.

                    if w.specialValueText():

                        dv_value = int(w.minimum())

                        dv_disp = str(w.specialValueText())

                    elif int(w.minimum()) == 0:

                        dv_value = 0

                        dv_disp = 'auto'

                    else:

                        dv_value = int(w.minimum())

                        dv_disp = str(dv_value)

                else:

                    dv_value = int(cfg_default) if isinstance(cfg_default, (int, float)) else int(w.value())

                    dv_disp = str(dv_value)

            elif isinstance(w, QtWidgets.QDoubleSpinBox):

                get = w.value

                setv = w.setValue

                if cfg_default is _MISSING:

                    dv_value = float(w.value())

                    dv_disp = str(dv_value)

                elif cfg_default is None:

                    if w.specialValueText():

                        dv_value = float(w.minimum())

                        dv_disp = str(w.specialValueText())

                    elif float(w.minimum()) == 0.0:

                        dv_value = 0.0

                        dv_disp = 'auto'

                    else:

                        dv_value = float(w.minimum())

                        dv_disp = str(dv_value)

                else:

                    dv_value = float(cfg_default) if isinstance(cfg_default, (int, float)) else float(w.value())

                    dv_disp = str(dv_value)

            elif isinstance(w, QtWidgets.QComboBox):

                get = w.currentText

                setv = w.setCurrentText

                dv_value = str(cfg_default) if isinstance(cfg_default, str) else str(w.currentText())

                # If the schema default is not in options, fall back to current.

                if dv_value and w.findText(dv_value) < 0:

                    dv_value = str(w.currentText())

                dv_disp = str(dv_value)

            elif isinstance(w, QtWidgets.QCheckBox):

                get = w.isChecked

                setv = w.setChecked

                if cfg_default is _MISSING:

                    dv_value = bool(w.isChecked())

                elif isinstance(cfg_default, bool):

                    dv_value = bool(cfg_default)

                elif isinstance(cfg_default, (list, tuple, set)):

                    token = w.property('list_token')

                    if token is None:

                        token = (w.text() or '').strip()

                    dv_value = str(token) in {str(x) for x in cfg_default}

                else:

                    dv_value = bool(w.isChecked())

                dv_disp = 'on' if dv_value else 'off'

            elif isinstance(w, QtWidgets.QLineEdit):

                get = w.text

                setv = w.setText

                if cfg_default is _MISSING:

                    dv_value = str(w.text())

                elif cfg_default is None:

                    dv_value = ''

                else:

                    dv_value = str(cfg_default)

                dv_disp = str(dv_value) if dv_value != '' else 'empty'

            else:

                return

            bindables.append((w, get, setv, dv_value, dv_disp))


        # Try to bind the field itself first
        _bind_one(field)
        # Then bind any descendants
        for w in field.findChildren(QtWidgets.QWidget):
            _bind_one(w)

        def _values_equal(a: object, b: object) -> bool:
            try:
                if isinstance(a, float) or isinstance(b, float):
                    return abs(float(a) - float(b)) <= 1e-12
            except Exception:
                pass
            return a == b

        def _is_default() -> bool:
            if not bindables:
                return True
            for _, get, _, dv, _ in bindables:
                try:
                    if not _values_equal(get(), dv):
                        return False
                except Exception:
                    return False
            return True

        def _update_btn() -> None:
            active = not _is_default()
            btn.setProperty("active", bool(active))
            btn.setIcon(_base_icon if active else _greyed_icon)
            btn.setCursor(QtCore.Qt.PointingHandCursor if active else QtCore.Qt.ArrowCursor)
            try:
                btn.style().unpolish(btn)
                btn.style().polish(btn)
            except Exception:
                pass
            # Always keep tooltip visible; show defaults when we can.
            if bindables:
                if len(bindables) == 1:
                    tip = f"Default: {bindables[0][4]}"
                    if cfg_path:
                        meta = get_param_meta(cfg_path)
                        if meta is not None and meta.typical:
                            tip = f"{tip} • Usually: {meta.typical}"
                    btn.setToolTip(tip)
                else:
                    # Compact summary for composite controls (e.g. apply_to).
                    parts = []
                    for w, _, _, _, dv_display in bindables[:4]:
                        token = w.property("list_token")
                        token = token if token is not None else ((w.text() or w.objectName()) or "")
                        parts.append(f"{token}={dv_display}")
                    suffix = "" if len(bindables) <= 4 else f" (+{len(bindables)-4})"
                    btn.setToolTip("Default: " + ", ".join(parts) + suffix)
            else:
                btn.setToolTip("Default")

        def _reset() -> None:
            if not bool(btn.property("active")):
                return
            for _, _, setv, dv, _ in bindables:
                try:
                    setv(dv)
                except Exception:
                    pass
            _update_btn()

        btn.clicked.connect(_reset)

        # Track value changes
        for w, _, _, _, _ in bindables:
            try:
                if isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                    w.valueChanged.connect(lambda *_: _update_btn())
                elif isinstance(w, QtWidgets.QComboBox):
                    w.currentTextChanged.connect(lambda *_: _update_btn())
                elif isinstance(w, QtWidgets.QCheckBox):
                    w.toggled.connect(lambda *_: _update_btn())
                elif isinstance(w, QtWidgets.QLineEdit):
                    w.textChanged.connect(lambda *_: _update_btn())
            except Exception:
                pass

        # Store for later refresh after config sync (signals might be blocked).
        self._param_default_buttons.append(_update_btn)
        _update_btn()
        return root

    def _small_note(self, text: str) -> QtWidgets.QWidget:
        """A small muted note used in parameter panels."""
        lbl = QtWidgets.QLabel(text)
        lbl.setWordWrap(True)
        # Keep it visually secondary but readable.
        lbl.setStyleSheet("QLabel { font-size: 11px; color: rgba(0,0,0,160); }")
        return lbl

    def _force_dot_locale(self, *widgets: QtWidgets.QWidget) -> None:
        """Force dot as decimal separator for numeric inputs.

        Even if the OS locale uses comma (e.g. de_DE), we want "1.23" everywhere
        to avoid config/YAML confusion.
        """
        try:
            loc = QtCore.QLocale.c()
        except Exception:
            loc = None
        if loc is None:
            return
        for w in widgets:
            try:
                w.setLocale(loc)
            except Exception:
                pass

    def _collapsible(
        self, title: str, content: QtWidgets.QWidget, checked: bool = False
    ) -> QtWidgets.QWidget:
        """Wrap `content` into a collapsible container.

        Convenience wrapper to keep call-sites compact.
        """
        root, content_lay, _ = _collapsible(title, expanded=checked)
        content_lay.addWidget(content)
        return root

    def _mk_basic_advanced_tabs(
        self, basic: QtWidgets.QWidget, advanced: QtWidgets.QWidget | None
    ) -> QtWidgets.QWidget:
        """Compact Basic/Advanced parameter container.

        Scroll is allowed *only* inside the parameter area. If a stage has no
        meaningful Advanced parameters, we omit the Advanced tab entirely.
        """

        def _wrap_scroll(w: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
            sa = QtWidgets.QScrollArea()
            sa.setWidgetResizable(True)
            sa.setFrameShape(QtWidgets.QFrame.NoFrame)
            sa.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            sa.setWidget(w)
            return sa

        def _is_effectively_empty(w: QtWidgets.QWidget) -> bool:
            """Return True when `advanced` contains no interactive controls.

            Many stages keep a placeholder note label instead of real advanced
            parameters. We omit the Advanced tab in that case.
            """
            # Any interactive widgets present? then it's not empty.
            interactive = (
                QtWidgets.QSpinBox,
                QtWidgets.QDoubleSpinBox,
                QtWidgets.QComboBox,
                QtWidgets.QCheckBox,
                QtWidgets.QLineEdit,
                QtWidgets.QSlider,
                QtWidgets.QDateTimeEdit,
            )
            # NOTE: PySide6.QObject.findChildren does *not* accept a tuple of types
            # (unlike some PyQt variants). Keep this compatible by querying per-type.
            for t in interactive:
                if w.findChildren(t):
                    # Presence of any interactive control means Advanced is meaningful,
                    # even if that control is currently hidden by a mode/method.
                    return False

            # Otherwise, treat a single note label as empty.
            lay = w.layout()
            if lay is None:
                return True
            if lay.count() == 0:
                return True
            # If the only content is a label-like note, consider empty.
            only_widgets: list[QtWidgets.QWidget] = []
            for i in range(lay.count()):
                it = lay.itemAt(i)
                if it is None:
                    continue
                ww = it.widget()
                if ww is None:
                    continue
                only_widgets.append(ww)
            if len(only_widgets) == 1 and isinstance(only_widgets[0], QtWidgets.QLabel):
                return True
            return False

        if advanced is None or _is_effectively_empty(advanced):
            return _wrap_scroll(basic)

        tabs = QtWidgets.QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setTabPosition(QtWidgets.QTabWidget.North)
        tabs.addTab(_wrap_scroll(basic), "Basic")
        tabs.addTab(_wrap_scroll(advanced), "Advanced")
        return tabs

    def _mk_scroll_panel(self, inner: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
        """Wrap an arbitrary panel in a vertical scroll area.

        We already scroll *inside* Basic/Advanced tabs, but many stages have
        extra controls (buttons, notes, ROI widgets). Wrapping the whole left
        column makes the UI robust on small screens and matches DS9-like
        "always reachable" controls philosophy.
        """
        sa = QtWidgets.QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QtWidgets.QFrame.NoFrame)
        sa.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        sa.setWidget(inner)
        return sa

    def _current_outputs_stage(self) -> str | None:
        """Stage key for Outputs viewer.

        Outputs are tied to pipeline stages (01..12). For UI-only pages
        (Project/Setup) there are no products, so stage=None.
        """

        try:
            row = int(self.steps.currentRow())
        except Exception:
            row = -1
        if row < 0 or row >= len(getattr(self, "_stage_keys", [])):
            return None
        stage_key = str(self._stage_keys[row])
        if stage_key in {"project", "setup"}:
            return None
        return stage_key

    def _update_outputs_tool_context(self) -> None:
        try:
            if getattr(self, "outputs_tool", None) is None:
                return
            if not self.outputs_tool.isVisible():
                return
            stage = self._current_outputs_stage()
            if getattr(self, "_cfg", None) is None:
                return
            self.outputs_tool.set_context(self._cfg, stage=stage)
        except Exception:
            pass

    def _on_outputs_tool_visibility_changed(self, visible: bool) -> None:
        # Keep toolbar action in sync (avoid recursion).
        if not hasattr(self, "act_outputs"):
            return
        self.act_outputs.blockSignals(True)
        try:
            self.act_outputs.setChecked(bool(visible))
        finally:
            self.act_outputs.blockSignals(False)

    def _sync_outputs_action_state(self) -> None:
        """Enable/disable Outputs action and keep its checked state in sync."""
        if not hasattr(self, "act_outputs"):
            return
        enabled = getattr(self, "_cfg", None) is not None
        try:
            self.act_outputs.setEnabled(bool(enabled))
        except Exception:
            pass
        if not enabled:
            self.act_outputs.blockSignals(True)
            try:
                self.act_outputs.setChecked(False)
            finally:
                self.act_outputs.blockSignals(False)
            return
        # When enabled, reflect real visibility of the tool window.
        shown = bool(getattr(self, "outputs_tool", None) is not None and self.outputs_tool.isVisible())
        self.act_outputs.blockSignals(True)
        try:
            self.act_outputs.setChecked(shown)
        finally:
            self.act_outputs.blockSignals(False)

    def _toggle_outputs(self, checked: bool) -> None:
        if getattr(self, "outputs_tool", None) is None:
            return
        if bool(checked):
            self._update_outputs_tool_context()
            try:
                self.outputs_tool.show()
                self.outputs_tool.raise_()
                self.outputs_tool.activateWindow()
            except Exception:
                self.outputs_tool.show()
        else:
            try:
                self.outputs_tool.hide()
            except Exception:
                pass

    def _mk_stage_apply_row(self, stage: str) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(10)
        state_lbl = QtWidgets.QLabel("Применено")
        state_lbl.setStyleSheet("QLabel { font-size: 11px; opacity: 0.9; }")
        btn = QtWidgets.QPushButton("Apply")
        btn.setProperty("primary", True)
        btn.clicked.connect(lambda: self._stage_apply(stage))
        h.addStretch(1)
        h.addWidget(state_lbl)
        h.addWidget(btn)
        self._register_stage_apply_controls(stage, btn, state_lbl)
        return w

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
        self.edit_cfg_path.setPlaceholderText(
            "config.yaml (will be created in Work dir)"
        )
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
        self.list_overview_objects.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        self.list_overview_objects.setMaximumHeight(200)
        self.list_overview_objects.setToolTip(
            "Double-click an object to select it for setup. Use multi-select for batch."
        )
        right.addWidget(self.list_overview_objects, 1)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_batch_configs = QtWidgets.QPushButton("Batch: build configs")
        self.btn_batch_run = QtWidgets.QPushButton("Batch: run")
        btn_row.addWidget(self.btn_batch_configs)
        btn_row.addWidget(self.btn_batch_run)
        right.addLayout(btn_row)

        ovl.addLayout(right, 1)
        self.list_overview_objects.itemDoubleClicked.connect(
            self._jump_to_object_from_overview
        )
        self.btn_batch_configs.clicked.connect(self._batch_build_configs)
        self.btn_batch_run.clicked.connect(self._batch_run)

        # (Frames Browser is exposed via a compact button and a global toolbar action.)

        # Actions
        # no stretch: the frame browser gets the remaining vertical space

        # signals
        self.btn_pick_data_dir.clicked.connect(self._pick_data_dir)
        self.btn_open_cfg.clicked.connect(self._open_existing_cfg)
        self.btn_inspect.clicked.connect(self._do_inspect)
        self.btn_frames_project.clicked.connect(
            lambda: self._open_frames_window("project")
        )
        self.edit_data_dir.textChanged.connect(lambda *_: self._update_enables())
        self.edit_data_dir.textChanged.connect(lambda *_: self._refresh_statusbar())

        return w

    def _pick_data_dir(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select data directory", str(Path.home())
        )
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
                if (not self.edit_work_dir.text().strip()) and (
                    not getattr(self, "_workdir_user_edited", False)
                ):
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
            df = getattr(self._inspect, "table", None)
            if df is None or df.empty:
                self.lbl_overview_counts.setText("No frames found")
                self.list_overview_objects.clear()
                return

            # Counts by kind
            vc = (
                df["kind"].value_counts(dropna=False).to_dict()
                if "kind" in df.columns
                else {}
            )
            total = int(len(df))
            lines = [f"Total frames: {total}"]
            for k in ["obj", "sky", "sunsky", "neon", "flat", "bias"]:
                lines.append(f"{k}: {int(vc.get(k, 0))}")

            # Quick setup diversity hints (dispersers/slits/binning)
            def _uniq(col: str) -> int:
                try:
                    if col in df.columns:
                        return int(df[col].dropna().astype(str).nunique())
                except Exception:
                    return 0
                return 0

            lines.append("")
            lines.append(
                f"Dispersers: {_uniq('disperser')}  |  Slits: {_uniq('slit')}  |  Binning: {_uniq('binning')}"
            )

            self.lbl_overview_counts.setText("\n".join(lines))

            # Objects list with counts
            self.list_overview_objects.blockSignals(True)
            self.list_overview_objects.clear()
            if "kind" in df.columns and "object" in df.columns:
                df_obj = df[df["kind"] == "obj"]
                if not df_obj.empty:
                    g = df_obj.groupby("object").size().sort_values(ascending=False)
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
            obj = str(obj).split("  (")[0].strip()
            if hasattr(self, "combo_object"):
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
            items = (
                self.list_overview_objects.selectedItems()
                if hasattr(self, "list_overview_objects")
                else []
            )
        except Exception:
            items = []
        out: list[str] = []
        for it in items:
            try:
                obj = it.data(QtCore.Qt.ItemDataRole.UserRole) or it.text()
                obj = str(obj).split("  (")[0].strip()
                if obj and obj not in out:
                    out.append(obj)
            except Exception:
                continue
        return out

    def _batch_build_configs(self) -> None:
        if self._inspect is None:
            self._log_error("Run Inspect first")
            return
        objs = self._selected_overview_objects()
        if not objs:
            self._log_error("Select one or more objects in Dataset overview")
            return

        from scorpio_pipe.workdir import RunSignature, pick_smart_run_dir

        data_dir = Path(self.edit_data_dir.text()).expanduser()
        root = getattr(self, "_pipeline_root", None) or (
            data_dir.parent if data_dir.exists() else Path.home()
        )
        dmy = self._infer_night_date_parts()
        if dmy is None:
            now = datetime.now()
            dd, mm, yyyy = int(now.day), int(now.month), int(now.year)
        else:
            dd, mm, yyyy = dmy
        from scorpio_pipe.app_paths import pick_workspace_root

        base = (
            pick_workspace_root(Path(root) if root else None)
            / f"{dd:02d}_{mm:02d}_{yyyy:04d}"
        )

        made: list[Path] = []
        self._log_info(f"Batch: building configs for {len(objs)} objects → {base}")
        for obj in objs:
            try:
                sig = RunSignature(obj, "", "", "")
                wd = pick_smart_run_dir(base, sig, prefer_flat=True)
                wd.mkdir(parents=True, exist_ok=True)
                cfg_path = wd / "config.yaml"
                ac = build_autoconfig(self._inspect.table, data_dir, obj, wd)
                cfg_path.write_text(ac.to_yaml_text(), encoding="utf-8")
                made.append(cfg_path)
                self._log_info(f"  ✔ {obj}: {cfg_path}")
            except Exception as e:
                self._log_exception(e)

        if made:
            self._log_info(f"Batch done: {len(made)} configs created")
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
            self._log_error("Run Inspect first")
            return
        objs = self._selected_overview_objects()
        if not objs:
            self._log_error("Select one or more objects in Dataset overview")
            return
        # Ensure configs exist (build if needed)
        self._batch_build_configs()

        # Collect configs under the night work root
        data_dir = Path(self.edit_data_dir.text()).expanduser()
        root = getattr(self, "_pipeline_root", None) or (
            data_dir.parent if data_dir.exists() else Path.home()
        )
        dmy = self._infer_night_date_parts()
        if dmy is None:
            now = datetime.now()
            dd, mm, yyyy = int(now.day), int(now.month), int(now.year)
        else:
            dd, mm, yyyy = dmy
        from scorpio_pipe.app_paths import pick_workspace_root

        base = (
            pick_workspace_root(Path(root) if root else None)
            / f"{dd:02d}_{mm:02d}_{yyyy:04d}"
        )

        cfgs: list[tuple[str, Path]] = []
        for obj in objs:
            # Find the newest config.yaml whose path contains the object slug/dir created by pick_smart_run_dir
            # Fallback: first matching by name anywhere under base.
            best: Path | None = None
            try:
                cands: list[Path] = []
                for cp in base.rglob("config.yaml"):
                    if obj.lower() in str(cp.parent).lower():
                        cands.append(cp)
                if cands:
                    best = max(cands, key=lambda p: p.stat().st_mtime)
            except Exception:
                pass
            if best and best.exists():
                cfgs.append((obj, best))

        if not cfgs:
            self._log_error("No configs found to run")
            return

        tasks = [
            "manifest",
            "superbias",
            "cosmics",
            "superneon",
            "lineid_prepare",
            "qc_report",
        ]
        pd = QtWidgets.QProgressDialog("Batch running…", "Cancel", 0, len(cfgs), self)
        pd.setWindowModality(QtCore.Qt.WindowModal)
        pd.setMinimumDuration(0)
        for i, (obj, cfg_path) in enumerate(cfgs, start=1):
            if pd.wasCanceled():
                self._log_info("Batch run canceled by user")
                break
            pd.setValue(i - 1)
            pd.setLabelText(f"{obj}: running non-interactive steps…")
            QtWidgets.QApplication.processEvents()
            try:
                self._log_info(f"=== BATCH RUN: {obj} ===")
                run_sequence(cfg_path, tasks, resume=True, force=False)
                self._log_info(f"  ✔ {obj}: done")
            except Exception as e:
                self._log_exception(e)
        pd.setValue(len(cfgs))
        self._log_info(
            "Batch run finished. Next: open LineID per object, then build Wavelength solution."
        )

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

            if not hasattr(self, "_frames_windows"):
                self._frames_windows = {}

            win = self._frames_windows.get(stage_key)
            if win is None:
                win = StageFramesWindow(stage_key, parent=self)
                if stage_key == "project":
                    try:
                        win.useSetupRequested.connect(self._use_setup_from_frame)
                    except Exception:
                        pass
                self._frames_windows[stage_key] = win

            inspect_df = None
            data_dir = None
            try:
                if getattr(self, "_inspect", None) is not None:
                    inspect_df = getattr(self._inspect, "table", None)
                    data_dir = getattr(self._inspect, "data_dir", None)
            except Exception:
                pass

            if stage_key == "project":
                win.set_context(
                    getattr(self, "_cfg", None),
                    inspect_df=inspect_df,
                    data_dir=data_dir,
                )
            else:
                win.set_context(getattr(self, "_cfg", None))

            try:
                win.showMaximized()
            except Exception:
                win.show()
            win.raise_()
            win.activateWindow()
        except Exception as e:
            self._log_exception(e)

    def _open_existing_cfg(self) -> None:
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open config", str(Path.home()), "YAML (*.yaml *.yml)"
        )
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
        self.combo_object.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToContentsOnFirstShow
        )
        row_obj.addWidget(self.combo_object, 1)
        fl.addRow(self._param_label("Object", "Имя научного объекта (как в ночном логе / FITS header).\nЕсли пусто: autodetect из header.\nОбычно: как в логе ночи."), row_obj)

        # Disperser
        row_disp = QtWidgets.QHBoxLayout()
        self.combo_disperser = QtWidgets.QComboBox()
        self.combo_disperser.setEditable(False)
        row_disp.addWidget(self.combo_disperser, 1)
        fl.addRow(self._param_label("Disperser", "Решётка/дисперсер для этого run.\nОпределяет дисперсию и набор калибровок.\nЕсли в ночи несколько — выбери нужную."), row_disp)

        # Slit
        row_slit = QtWidgets.QHBoxLayout()
        self.combo_slit = QtWidgets.QComboBox()
        self.combo_slit.setEditable(False)
        row_slit.addWidget(self.combo_slit, 1)
        fl.addRow(self._param_label("Slit", "Ширина щели.\nВлияет на спектральное разрешение и подбор line-list.\nЕсли встречается несколько — выбери нужную."), row_slit)

        # Binning
        row_bin = QtWidgets.QHBoxLayout()
        self.combo_binning = QtWidgets.QComboBox()
        self.combo_binning.setEditable(False)
        row_bin.addWidget(self.combo_binning, 1)
        fl.addRow(self._param_label("Binning", "Биннинг CCD.\nВлияет на масштаб [px] и шум.\nНе смешивай разные binning в одном run."), row_bin)

        # Workspace root (top-level base)
        row_ws = QtWidgets.QHBoxLayout()
        self.edit_workspace_root = QtWidgets.QLineEdit()
        try:
            from scorpio_pipe.app_paths import pick_workspace_root

            self.edit_workspace_root.setText(str(pick_workspace_root(getattr(self, "_pipeline_root", None))))
        except Exception:
            self.edit_workspace_root.setText("")
        self.btn_pick_workspace_root = QtWidgets.QToolButton(text="…")
        self.btn_pick_workspace_root.setCursor(QtCore.Qt.PointingHandCursor)
        row_ws.addWidget(self.edit_workspace_root, 1)
        row_ws.addWidget(self.btn_pick_workspace_root)
        fl.addRow(self._param_label("Workspace root", "Верхняя база workspace.\nLayout: <Night>/<Object>_<Disperser>_<Run>/<NN_stage>/.\nЗдесь живут все проекты."), row_ws)

        # Run controls
        row_run = QtWidgets.QHBoxLayout()
        self.spin_run_id = QtWidgets.QSpinBox()
        self.spin_run_id.setRange(0, 99)
        self.spin_run_id.setSpecialValueText("Auto")
        self.spin_run_id.setToolTip("0 = Auto (следующий свободный _01/_02/...)")
        self.btn_init_run = QtWidgets.QPushButton("Initialize run")
        row_run.addWidget(self.spin_run_id)
        row_run.addWidget(self.btn_init_run)
        fl.addRow(self._param_label("Run", "Номер run в рамках ночи/объекта/решётки.\nAuto = следующий свободный _01/_02/...\nInitialize создаёт/выбирает run-folder."), row_run)

        # Recent runs (last 10) + Open…
        row_recent = QtWidgets.QHBoxLayout()
        self.combo_recent_runs = QtWidgets.QComboBox()
        self.combo_recent_runs.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.btn_switch_run = QtWidgets.QPushButton("Switch")
        self.btn_open_run = QtWidgets.QPushButton("Open…")
        row_recent.addWidget(self.combo_recent_runs, 1)
        row_recent.addWidget(self.btn_switch_run)
        row_recent.addWidget(self.btn_open_run)
        fl.addRow(self._param_label("Recent runs", "Последние 10 run-folder.\nSwitch — открыть выбранный.\nOpen… — выбрать любой через проводник."), row_recent)

        # Work dir (run-root)
        row_wd = QtWidgets.QHBoxLayout()
        self.edit_work_dir = QtWidgets.QLineEdit()
        # If the user edits the field manually, we stop auto-suggesting paths.
        self._workdir_user_edited = False
        self.edit_work_dir.textEdited.connect(
            lambda *_: setattr(self, "_workdir_user_edited", True)
        )
        self.btn_pick_work_dir = QtWidgets.QToolButton(text="…")
        self.btn_pick_work_dir.setCursor(QtCore.Qt.PointingHandCursor)
        row_wd.addWidget(self.edit_work_dir, 1)
        row_wd.addWidget(self.btn_pick_work_dir)
        fl.addRow(self._param_label("Run folder", "Текущий run-folder (root для стадий).\nВнутри создаются папки <NN_stage>/.\nМетаданные — в manifest/."), row_wd)

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

        # right: YAML editor (manual edit)
        g_yaml = _box("Config YAML (editable)")
        splitter.addWidget(g_yaml)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 900])

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
        self.btn_prev_state = QtWidgets.QPushButton("Previous state")
        self.btn_prev_state.setToolTip(
            "Restore the last saved session/config snapshot (ui/history)."
        )
        self.lbl_cfg_state = QtWidgets.QLabel("—")
        bar.addWidget(self.btn_validate_yaml)
        bar.addWidget(self.btn_diff_cfg)
        bar.addWidget(self.btn_save_cfg)
        bar.addWidget(self.btn_prev_state)
        bar.addStretch(1)
        bar.addWidget(self.lbl_cfg_state)

        # signals
        self.combo_object.currentTextChanged.connect(self._on_object_changed)
        self.combo_disperser.currentTextChanged.connect(self._on_disperser_changed)
        self.combo_slit.currentTextChanged.connect(self._update_setup_hint)
        self.combo_binning.currentTextChanged.connect(self._update_setup_hint)
        self.btn_pick_workspace_root.clicked.connect(self._pick_workspace_root)
        self.btn_init_run.clicked.connect(self._initialize_run)
        self.btn_switch_run.clicked.connect(self._switch_recent_run)
        self.btn_open_run.clicked.connect(self._open_run_dialog)
        # Keep recent list fresh when user changes workspace root.
        self.edit_workspace_root.editingFinished.connect(self._refresh_recent_runs)
        self.btn_pick_work_dir.clicked.connect(self._pick_work_dir)
        self.btn_suggest_workdir.clicked.connect(self._suggest_work_dir)
        self.btn_make_cfg.clicked.connect(self._do_make_cfg)
        self.btn_reload_cfg.clicked.connect(self._do_reload_cfg)
        self.btn_validate_yaml.clicked.connect(self._validate_yaml)
        self.btn_diff_cfg.clicked.connect(self._show_cfg_diff)
        self.btn_save_cfg.clicked.connect(self._do_save_cfg)
        self.btn_prev_state.clicked.connect(self._show_prev_state_menu)
        self.editor_yaml.textChanged.connect(self._on_yaml_changed)

        # Initial population of Recent runs.
        QtCore.QTimer.singleShot(0, self._refresh_recent_runs)


        return w

    def _pick_work_dir(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select run folder", str(Path.home())
        )
        if d:
            self.edit_work_dir.setText(d)
            self._workdir_user_edited = True
            # a manual pick disables auto-suggest

    def _pick_workspace_root(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select workspace root", str(Path.home())
        )
        if d:
            self.edit_workspace_root.setText(d)
            self._refresh_recent_runs()

    def _refresh_recent_runs(self) -> None:
        """Refresh combo box with the last 10 run folders under workspace_root."""

        try:
            from scorpio_pipe.run_root import format_run_label, scan_recent_runs

            ws = Path((self.edit_workspace_root.text() or "").strip()).expanduser()
            if not ws:
                self.combo_recent_runs.clear()
                self._recent_run_roots = []
                return

            runs = scan_recent_runs(ws, limit=10)
            self._recent_run_roots = runs
            self.combo_recent_runs.blockSignals(True)
            self.combo_recent_runs.clear()
            for idx, p in enumerate(runs):
                label = format_run_label(ws, p)
                if idx == 0:
                    label = f"★ {label}"
                self.combo_recent_runs.addItem(label, str(p))
            self.combo_recent_runs.blockSignals(False)
        except Exception:
            self.combo_recent_runs.clear()
            self._recent_run_roots = []

    def _switch_recent_run(self) -> None:
        i = int(self.combo_recent_runs.currentIndex())
        if i < 0:
            return
        try:
            p = Path(str(self.combo_recent_runs.currentData() or ""))
        except Exception:
            return
        if not p:
            return
        self._open_run_root(p)

    def _open_run_dialog(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Open run folder", str(Path.home()))
        if d:
            self._open_run_root(Path(d))

    def _open_run_root(self, run_root: Path) -> None:
        run_root = Path(run_root).expanduser()
        if not run_root.exists():
            self._log_error(f"Run folder does not exist: {run_root}")
            return

        # Ensure minimal run skeleton (manifest/qc/ui) and passport/session.
        try:
            from scorpio_pipe.work_layout import ensure_work_layout

            ensure_work_layout(run_root)
            from scorpio_pipe.run_passport import ensure_run_passport
            from scorpio_pipe.ui.session_store import load_session, save_session

            ensure_run_passport(run_root)
            save_session(run_root, load_session(run_root))
        except Exception:
            pass

        # Legacy layout hint (do not migrate automatically)
        try:
            from scorpio_pipe.run_root import detect_legacy_layout

            if detect_legacy_layout(run_root):
                QtWidgets.QMessageBox.information(
                    self,
                    "Old layout detected",
                    "This folder looks like a legacy workspace (products/...).\n"
                    "It will be opened as-is; no files will be moved automatically.",
                )
        except Exception:
            pass

        # Legacy stage numbering/slugs hint (v5.38.5 and earlier)
        try:
            from scorpio_pipe.workspace_migrate import (
                detect_legacy_stage_layout,
                migrate_run_to_v5386,
            )

            info = detect_legacy_stage_layout(run_root)
            if info.is_legacy:
                details = "\n".join(f"- {n}" for n in (info.found or ()))
                box = QtWidgets.QMessageBox(self)
                box.setIcon(QtWidgets.QMessageBox.Icon.Information)
                box.setWindowTitle("Legacy run layout")
                box.setText(info.reason)
                if details:
                    box.setInformativeText("Detected:\n" + details)

                btn_open = box.addButton("Open as-is", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
                btn_migrate = box.addButton(
                    "Migrate copy", QtWidgets.QMessageBox.ButtonRole.ActionRole
                )
                box.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
                box.exec()

                clicked = box.clickedButton()
                if clicked == btn_migrate:
                    new_root = migrate_run_to_v5386(run_root)
                    run_root = Path(new_root)
                elif clicked != btn_open:
                    return
        except Exception:
            pass

        # P2 safety belt: validate workspace layout + run.json (passport).
        try:
            from scorpio_pipe.run_validate import RunLayoutError, validate_run_dir
            from scorpio_pipe.run_passport import rewrite_run_passport_from_dir

            v = validate_run_dir(run_root, strict=False)
            if v.mismatches:
                # Offer a one-click fix: rewrite run.json to match folder naming.
                box = QtWidgets.QMessageBox(self)
                box.setIcon(QtWidgets.QMessageBox.Warning)
                box.setWindowTitle("run.json mismatch")
                box.setText(
                    "This run folder has a valid structure, but fields in run.json do not match "
                    "the folder name. This may lead to confusing UI/runner behavior."
                )
                box.setInformativeText("\n".join(v.mismatches))
                btn_fix = box.addButton(
                    "Fix run.json", QtWidgets.QMessageBox.ButtonRole.ActionRole
                )
                btn_continue = box.addButton(
                    "Open as-is", QtWidgets.QMessageBox.ButtonRole.AcceptRole
                )
                box.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
                box.exec()
                if box.clickedButton() == btn_fix:
                    try:
                        rewrite_run_passport_from_dir(run_root)
                    except Exception as e:
                        QtWidgets.QMessageBox.critical(
                            self,
                            "Failed to fix run.json",
                            str(e),
                        )
                        return
                elif box.clickedButton() != btn_continue:
                    return

            # Non-fatal warnings: show once so the user is aware.
            if v.warnings:
                self._show_msgbox_lines(
                    "Workspace validation",
                    list(v.warnings),
                    icon="warn",
                )
        except RunLayoutError as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid run folder",
                str(e),
            )
            return
        except Exception:
            pass

        self.edit_work_dir.setText(str(run_root))
        self._workdir_user_edited = True

        # Prefer config.yaml inside the run folder.
        cfg_p = run_root / "config.yaml"
        if cfg_p.is_file():
            self.edit_cfg_path.setText(str(cfg_p))
            self._do_reload_cfg()
        else:
            # No config yet; keep a reasonable default.
            self._cfg_path = cfg_p

    def _initialize_run(self) -> None:
        """Compute run_root from representative header + auto-increment."""

        try:
            from scorpio_pipe.run_root import get_or_create_run_root

            ws = Path((self.edit_workspace_root.text() or "").strip()).expanduser()
            if not str(ws):
                self._log_error("Workspace root is empty")
                return

            # representative header: prefer SCI/object frame, else ARC, else first.
            header = None
            try:
                if self._inspect is not None and getattr(self._inspect, "table", None) is not None:
                    df = self._inspect.table
                    if df is not None and not df.empty and "path" in df.columns:
                        def _pick(kind: str) -> str | None:
                            try:
                                sub = df[df.get("kind") == kind]
                                if not sub.empty:
                                    return str(sub.iloc[0]["path"])
                            except Exception:
                                return None
                            return None

                        p0 = _pick("obj") or _pick("arc") or str(df.iloc[0]["path"])
                        if p0:
                            from astropy.io import fits

                            header = fits.getheader(p0, memmap=False)
            except Exception:
                header = None

            run_id = int(self.spin_run_id.value()) if hasattr(self, "spin_run_id") else 0
            run_id_opt = None if run_id <= 0 else run_id

            run_root = get_or_create_run_root(
                ws,
                dataset_path=Path(self.edit_data_dir.text()) if hasattr(self, "edit_data_dir") else None,
                headers=header,
                run_id=run_id_opt,
                create=True,
            )

            from scorpio_pipe.work_layout import ensure_work_layout

            ensure_work_layout(run_root)
            self._open_run_root(run_root)
            self._refresh_recent_runs()
        except Exception as e:
            self._log_error(f"Initialize run failed: {e}")

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
                                prefixes.append(
                                    f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                                )
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
        """Suggest a run-folder path using the new Night/Run layout.

        This does not create any folders; it only fills the "Run folder" field.
        """

        try:
            from scorpio_pipe.run_root import get_or_create_run_root

            ws = Path((self.edit_workspace_root.text() or "").strip()).expanduser()
            if not str(ws):
                self._log_error("Workspace root is empty")
                return

            header = None
            try:
                if self._inspect is not None and getattr(self._inspect, "table", None) is not None:
                    df = self._inspect.table
                    if df is not None and not df.empty and "path" in df.columns:
                        # prefer science/obj, otherwise arc
                        p0 = None
                        try:
                            sub = df[df.get("kind") == "obj"]
                            if not sub.empty:
                                p0 = str(sub.iloc[0]["path"])
                        except Exception:
                            p0 = None
                        if not p0:
                            try:
                                sub = df[df.get("kind") == "arc"]
                                if not sub.empty:
                                    p0 = str(sub.iloc[0]["path"])
                            except Exception:
                                p0 = None
                        if not p0:
                            p0 = str(df.iloc[0]["path"])
                        if p0:
                            from astropy.io import fits

                            header = fits.getheader(p0, memmap=False)
            except Exception:
                header = None

            run_id = int(self.spin_run_id.value()) if hasattr(self, "spin_run_id") else 0
            run_id_opt = None if run_id <= 0 else run_id

            rr = get_or_create_run_root(
                ws,
                dataset_path=Path(self.edit_data_dir.text()) if hasattr(self, "edit_data_dir") else None,
                headers=header,
                run_id=run_id_opt,
                create=False,
            )
            self.edit_work_dir.setText(str(rr))
            self._workdir_user_edited = False
        except Exception:
            # fall back to the older heuristic
            dmy = self._infer_night_date_parts()
            if dmy is None:
                now = datetime.now()
                dd, mm, yyyy = int(now.day), int(now.month), int(now.year)
            else:
                dd, mm, yyyy = dmy
            night_dir = f"{dd:02d}_{mm:02d}_{yyyy:04d}"
            base = Path((self.edit_workspace_root.text() or "").strip()).expanduser() / night_dir
            self.edit_work_dir.setText(str(base))

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
            vals = sorted(
                [
                    v
                    for v in sci["disperser"].dropna().astype(str).unique().tolist()
                    if v.strip()
                ]
            )
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
            slits = sorted(
                [
                    v
                    for v in sci["slit"].dropna().astype(str).unique().tolist()
                    if v.strip()
                ]
            )
        if not slits:
            slits = [""]
        self.combo_slit.addItems(slits)

        # binning
        bins: list[str] = []
        if "binning" in sci.columns:
            bins = sorted(
                [
                    v
                    for v in sci["binning"].dropna().astype(str).unique().tolist()
                    if v.strip()
                ]
            )
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
        slit = (
            (self.combo_slit.currentText() or "").strip()
            if hasattr(self, "combo_slit")
            else ""
        )
        binning = (
            (self.combo_binning.currentText() or "").strip()
            if hasattr(self, "combo_binning")
            else ""
        )

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
            uniq = [
                v
                for v in sci["instrument"].dropna().astype(str).unique().tolist()
                if v.strip()
            ]
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
                hint_parts.append(
                    f"Range≈[{spec.range_A[0]:.0f}–{spec.range_A[1]:.0f}] Å"
                )
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
            if (not self.edit_work_dir.text().strip()) and (
                not getattr(self, "_workdir_user_edited", False)
            ):
                self._suggest_work_dir()
        except Exception:
            pass
        self._update_enables()

    def _on_disperser_changed(self, *_: object) -> None:
        self._update_slit_binning_from_inspect()
        self._update_setup_hint()
        try:
            if (not self.edit_work_dir.text().strip()) and (
                not getattr(self, "_workdir_user_edited", False)
            ):
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
        slit = (
            getattr(self, "combo_slit", None).currentText()
            if hasattr(self, "combo_slit")
            else ""
        )
        slit = (slit or "").strip() or None
        binning = (
            getattr(self, "combo_binning", None).currentText()
            if hasattr(self, "combo_binning")
            else ""
        )
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

        # also update object/disperser/workdir if available
        try:
            self.edit_work_dir.setText(
                str(Path(str(cfg.get("work_dir", ""))).expanduser())
            )
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
        self._sync_outputs_action_state()

        try:
            if hasattr(self, "stack") and int(self.stack.currentIndex()) == 7:
                self._update_wavesol_stepper()
        except Exception:
            pass

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
            if hasattr(self, "lbl_wavesol_dir") and self._cfg:
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

    def _show_prev_state_menu(self) -> None:
        """Show history snapshots and allow restoring a previous state."""

        try:
            rr_txt = (self.edit_work_dir.text() or "").strip()
            if not rr_txt:
                self._log_error("No run folder selected")
                return
            run_root = Path(rr_txt).expanduser().resolve()

            from scorpio_pipe.ui.session_store import list_snapshots, restore_snapshot

            items = list_snapshots(run_root, limit=15)
            if not items:
                self._show_msg(
                    "Previous state",
                    ["No snapshots yet.", "Save the config or run a stage to create snapshots."],
                    icon="info",
                )
                return

            menu = QtWidgets.QMenu(self)
            for it in items:
                ts = str(it.get("timestamp", ""))
                rs = str(it.get("reason", ""))
                rel = str(it.get("path", ""))
                if not rel:
                    continue
                txt = f"{ts} — {rs}" if rs else ts
                act = menu.addAction(txt)
                act.setData(rel)

            act = menu.exec(self.btn_prev_state.mapToGlobal(QtCore.QPoint(0, self.btn_prev_state.height())))
            if not act:
                return
            rel = act.data()
            if not rel:
                return
            ok = restore_snapshot(run_root, str(rel))
            if not ok:
                self._log_error("Restore failed")
                return
            # Reload config/session into UI.
            try:
                self.edit_cfg_path.setText(str(run_root / "config.yaml"))
            except Exception:
                pass
            self._do_reload_cfg()
            self._refresh_outputs_panels()
            self._update_enables()
            self._refresh_statusbar()
            self._log_info(f"Restored snapshot: {rel}")
        except Exception as e:
            self._log_exception(e)

    def _sync_cfg_from_editor(self) -> bool:
        cfg, err = _safe_parse_yaml(self.editor_yaml.toPlainText())
        if err:
            self._log_error(f"YAML invalid: {err}")
            return False
        # Parsing succeeded; keep internal cfg even if UI sync fails.
        self._cfg = cfg
        try:
            self._refresh_outputs_panels()
        except Exception as e:
            # UI-only; must not block saving/running.
            self._log_exception(e)
        try:
            self._sync_stage_controls_from_cfg()
        except Exception as e:
            # UI-only; must not block saving/running.
            self._log_exception(e)
        return True

    def _refresh_outputs_panels(self) -> None:
        # Keep the detached Outputs tool window in sync.
        self._update_outputs_tool_context()

    def _do_save_cfg(self) -> None:
        # Resolve target path.
        if not self._cfg_path:
            # prefer the explicit field on the Project page (users often edit it manually)
            ptxt = (
                getattr(self, "edit_cfg_path", None).text().strip()
                if hasattr(self, "edit_cfg_path")
                else ""
            )
            if ptxt:
                try:
                    self._cfg_path = Path(ptxt).expanduser().resolve()
                except Exception:
                    self._cfg_path = None

        if not self._cfg_path:
            # infer from work_dir
            wd_txt = (
                (self.edit_work_dir.text() or "").strip()
                if hasattr(self, "edit_work_dir")
                else ""
            )
            if wd_txt:
                wd = Path(wd_txt).expanduser()
                try:
                    wd.mkdir(parents=True, exist_ok=True)
                    self._cfg_path = (wd / "config.yaml").resolve()
                except Exception:
                    self._cfg_path = None

        if not self._cfg_path:
            # last resort: ask user
            try:
                fn, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Save config.yaml",
                    str(Path.home() / "config.yaml"),
                    "YAML (*.yaml *.yml)",
                )
                if fn:
                    self._cfg_path = Path(fn).expanduser().resolve()
            except Exception:
                self._cfg_path = None

        if not self._cfg_path:
            self._log_error("No config path")
            return

        if not self._sync_cfg_from_editor():
            return

        txt = self.editor_yaml.toPlainText()
        try:
            self._cfg_path.parent.mkdir(parents=True, exist_ok=True)
            self._cfg_path.write_text(txt, encoding="utf-8")
        except Exception as e:
            self._log_exception(e)
            self._show_msg(
                "Save failed", [f"Cannot write: {self._cfg_path}", str(e)], icon="error"
            )
            return

        self._yaml_saved_text = txt
        try:
            if hasattr(self, "edit_cfg_path"):
                self.edit_cfg_path.setText(str(self._cfg_path))
        except Exception:
            pass
        self._log_info(f"Saved: {self._cfg_path}")
        self.lbl_cfg_state.setText(f"Saved: {self._cfg_path.name}")
        try:
            self.statusBar().showMessage("Saved", 2000)
        except Exception:
            pass
        self._update_enables()
        self._refresh_statusbar()
        self._sync_outputs_action_state()

        # Persist GUI session + a rollback snapshot (ui/history).
        try:
            rr_txt = (self.edit_work_dir.text() or "").strip()
            if rr_txt:
                rr = Path(rr_txt).expanduser().resolve()
                from scorpio_pipe.run_passport import ensure_run_passport
                from scorpio_pipe.ui.session_store import snapshot, update_stage

                ensure_run_passport(rr)
                cfg_obj = self._cfg or {}
                if isinstance(cfg_obj, dict):
                    update_stage(
                        rr,
                        "sky",
                        cfg_section=cfg_obj.get("sky") if isinstance(cfg_obj.get("sky"), dict) else None,
                    )
                    update_stage(
                        rr,
                        "linearize",
                        cfg_section=cfg_obj.get("linearize") if isinstance(cfg_obj.get("linearize"), dict) else None,
                    )
                    update_stage(
                        rr,
                        "stack2d",
                        cfg_section=cfg_obj.get("stack2d") if isinstance(cfg_obj.get("stack2d"), dict) else None,
                    )
                    update_stage(
                        rr,
                        "extract1d",
                        cfg_section=cfg_obj.get("extract1d") if isinstance(cfg_obj.get("extract1d"), dict) else None,
                    )
                snapshot(rr, reason="config_save")
        except Exception:
            pass

        try:
            if hasattr(self, "stack") and int(self.stack.currentIndex()) == 7:
                self._update_wavesol_stepper()
        except Exception:
            pass

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

    # --- stage parameter apply/dirty state ---

    def _set_stage_dirty(self, stage: str, dirty: bool = True) -> None:
        self._stage_dirty[stage] = bool(dirty)
        btn = self._stage_apply_btns.get(stage)
        if btn is not None:
            btn.setEnabled(bool(dirty))
        lbl = self._stage_dirty_labels.get(stage)
        if lbl is not None:
            lbl.setText("Не применено" if dirty else "Применено")

    def _clear_stage_dirty_all(self) -> None:
        for st in list(getattr(self, "_stage_dirty", {}).keys()):
            self._set_stage_dirty(st, False)

    def _register_stage_apply_controls(
        self, stage: str, apply_btn: QtWidgets.QPushButton, state_lbl: QtWidgets.QLabel
    ) -> None:
        self._stage_apply_btns[stage] = apply_btn
        self._stage_dirty_labels[stage] = state_lbl
        apply_btn.setEnabled(False)
        state_lbl.setText("Применено")
        self._stage_dirty.setdefault(stage, False)
        self._stage_pending.setdefault(stage, {})

    def _stage_set_pending(self, stage: str, dotted: str, value: Any) -> None:
        self._stage_pending.setdefault(stage, {})[dotted] = value
        self._set_stage_dirty(stage, True)

    def _stage_apply(self, stage: str) -> None:
        # If a user is typing in a spinbox and immediately clicks Apply,
        # Qt might not have interpreted the text into the numeric value yet.
        # Force-commit the current editor widget to avoid "Apply does nothing".
        try:
            fw = QtWidgets.QApplication.focusWidget()
            if isinstance(fw, QtWidgets.QAbstractSpinBox):
                fw.interpretText()
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass

        pending = dict(self._stage_pending.get(stage, {}) or {})
        if not pending:
            self._set_stage_dirty(stage, False)
            return
        # Apply all changes into YAML/config.
        # Use per-key setters so that existing helper logic (apply_to ordering, etc.) stays consistent.
        for dotted, val in pending.items():
            # handle list selectors
            if str(dotted).endswith(".apply_to"):
                block = dotted.split(".")[0]
                self._cfg_set_apply_to(block, list(val or []))
            else:
                self._cfg_set_path(dotted.split("."), val)
        # Clear pending + mark clean
        self._stage_pending[stage] = {}
        self._set_stage_dirty(stage, False)
        try:
            self._sync_stage_controls_from_cfg()
        except Exception as e:
            # Do not block applying: this is UI-only.
            self._log_exception(e)
        try:
            self.statusBar().showMessage("Применено", 1500)
        except Exception:
            pass
        try:
            self._log_info(f"Applied: {stage}")
        except Exception:
            pass

    def _ensure_stage_applied(self, stage: str, title: str | None = None) -> bool:
        """If the stage has pending changes, ask user to apply them before running."""
        if title is None:
            # Human-friendly fallback
            title = stage
        if not self._stage_dirty.get(stage, False):
            return True
        m = QtWidgets.QMessageBox(self)
        m.setWindowTitle("Unapplied parameters")
        m.setIcon(QtWidgets.QMessageBox.Warning)
        m.setText(f"{title}: есть неприменённые изменения параметров.")
        m.setInformativeText(
            "Нажми Apply, чтобы применить параметры, и затем запусти этап."
        )
        btn_apply = m.addButton("Apply", QtWidgets.QMessageBox.AcceptRole)
        m.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        m.setDefaultButton(btn_apply)
        m.exec()
        if m.clickedButton() == btn_apply:
            self._stage_apply(stage)
            return True
        return False

    # --------------------------- page: calibrations ---------------------------

    def _cfg_get(
        self, cfg: dict[str, Any] | None, path: list[str], default: Any = None
    ) -> Any:
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
        # Keep the config status label in sync (signals were blocked).
        try:
            self._on_yaml_changed()
        except Exception:
            pass
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
        order = ["obj", "sky", "sunsky", "neon", "flat", "bias"]
        enabled_sorted = [k for k in order if k in enabled] + [
            k for k in enabled if k not in order
        ]
        self._cfg_set_path([block_key, "apply_to"], enabled_sorted)

    def _sync_stage_controls_from_cfg(self) -> None:
        cfg = getattr(self, "_cfg", None)
        if not isinstance(cfg, dict):
            return

        # NOTE: On Windows (and especially after rebuilding UI pages or switching
        # datasets) Qt widgets referenced as ``self.xxx`` may become stale.
        # Accessing them raises: "Internal C++ object ... already deleted".
        # This sync path must therefore be ultra-defensive.

        def _safe_block_call(w: Any, fn) -> None:
            """Run ``fn(widget)`` with signals blocked; ignore deleted widgets."""
            if w is None:
                return
            try:
                blocker = QtCore.QSignalBlocker(w)
            except RuntimeError:
                return
            try:
                fn(w)
            except RuntimeError:
                # Widget may get deleted between checks.
                return
            finally:
                try:
                    del blocker
                except Exception:
                    pass

        def _set_checked(w: Any, v: Any) -> None:
            _safe_block_call(w, lambda ww: ww.setChecked(bool(v)))

        def _set_value(w: Any, v: Any) -> None:
            _safe_block_call(w, lambda ww: ww.setValue(v))

        def _set_text(w: Any, s: str) -> None:
            _safe_block_call(w, lambda ww: ww.setText(s))

        def _set_combo_text(w: Any, text: str) -> None:
            def _fn(ww):
                idx = ww.findText(text)
                ww.setCurrentIndex(max(0, idx))

            _safe_block_call(w, _fn)

        def _set_label(w: Any, s: str) -> None:
            try:
                if w is not None:
                    w.setText(s)
            except RuntimeError:
                return

        # --- Calibrations ---
        if hasattr(self, "combo_bias_combine") and not self._stage_dirty.get(
            "calib", False
        ):
            calib = cfg.get("calib", {}) if isinstance(cfg.get("calib"), dict) else {}
            combine = str(calib.get("bias_combine", "median") or "median").lower()
            if combine not in ("median", "mean"):
                combine = "median"
            _set_combo_text(getattr(self, "combo_bias_combine", None), combine)
            if hasattr(self, "spin_bias_sigma_clip"):
                _set_value(
                    getattr(self, "spin_bias_sigma_clip", None),
                    _safe_float(calib.get("bias_sigma_clip", 0.0), 0.0),
                )

        # --- Cosmics ---
        if hasattr(self, "chk_cosmics_obj") and not self._stage_dirty.get(
            "cosmics", False
        ):
            apply_to = set(self._cfg_get(cfg, ["cosmics", "apply_to"], []) or [])
            for name, cb in [
                ("obj", getattr(self, "chk_cosmics_obj", None)),
                ("sky", getattr(self, "chk_cosmics_sky", None)),
                ("sunsky", getattr(self, "chk_cosmics_sunsky", None)),
                ("neon", getattr(self, "chk_cosmics_neon", None)),
            ]:
                _set_checked(cb, name in apply_to)
            if hasattr(self, "chk_cosmics_enabled"):
                _set_checked(
                    getattr(self, "chk_cosmics_enabled", None),
                    bool(self._cfg_get(cfg, ["cosmics", "enabled"], True)),
                )
            if hasattr(self, "combo_cosmics_method"):
                m = str(self._cfg_get(cfg, ["cosmics", "method"], "auto") or "auto")
                _set_combo_text(getattr(self, "combo_cosmics_method", None), m)
            if hasattr(self, "spin_cosmics_k"):
                _set_value(
                    getattr(self, "spin_cosmics_k", None),
                    _safe_float(self._cfg_get(cfg, ["cosmics", "k"], 9.0), 9.0),
                )
            if hasattr(self, "chk_cosmics_bias"):
                _set_checked(
                    getattr(self, "chk_cosmics_bias", None),
                    bool(self._cfg_get(cfg, ["cosmics", "bias_subtract"], True)),
                )
            if hasattr(self, "chk_cosmics_png"):
                _set_checked(
                    getattr(self, "chk_cosmics_png", None),
                    bool(self._cfg_get(cfg, ["cosmics", "save_png"], True)),
                )
            if hasattr(self, "chk_cosmics_mask_fits"):
                _set_checked(
                    getattr(self, "chk_cosmics_mask_fits", None),
                    bool(self._cfg_get(cfg, ["cosmics", "save_mask_fits"], True)),
                )
            if hasattr(self, "spin_cosmics_dilate"):
                _set_value(
                    getattr(self, "spin_cosmics_dilate", None),
                    _safe_int(self._cfg_get(cfg, ["cosmics", "dilate"], 1), 1),
                )
            if hasattr(self, "dspin_cosmics_mad_scale"):
                _set_value(
                    getattr(self, "dspin_cosmics_mad_scale", None),
                    _safe_float(self._cfg_get(cfg, ["cosmics", "mad_scale"], 1.0), 1.0),
                )
            if hasattr(self, "dspin_cosmics_min_mad"):
                _set_value(
                    getattr(self, "dspin_cosmics_min_mad", None),
                    _safe_float(self._cfg_get(cfg, ["cosmics", "min_mad"], 0.0), 0.0),
                )
            if hasattr(self, "dspin_cosmics_max_frac"):
                v = self._cfg_get(cfg, ["cosmics", "max_frac_per_frame"], None)
                vv = 0.0
                try:
                    if v not in (None, ""):
                        vv = float(v)
                except Exception:
                    vv = 0.0
                _set_value(getattr(self, "dspin_cosmics_max_frac", None), vv)
            if hasattr(self, "spin_cosmics_local_r"):
                _set_value(
                    getattr(self, "spin_cosmics_local_r", None),
                    _safe_int(self._cfg_get(cfg, ["cosmics", "local_r"], 2), 2),
                )
            if hasattr(self, "dspin_cosmics_k2_scale"):
                _set_value(
                    getattr(self, "dspin_cosmics_k2_scale", None),
                    _safe_float(
                        self._cfg_get(cfg, ["cosmics", "two_diff_k2_scale"], 0.8), 0.8
                    ),
                )
            if hasattr(self, "dspin_cosmics_k2_min"):
                _set_value(
                    getattr(self, "dspin_cosmics_k2_min", None),
                    _safe_float(
                        self._cfg_get(cfg, ["cosmics", "two_diff_k2_min"], 5.0), 5.0
                    ),
                )
            if hasattr(self, "dspin_cosmics_thr_a"):
                _set_value(
                    getattr(self, "dspin_cosmics_thr_a", None),
                    _safe_float(
                        self._cfg_get(cfg, ["cosmics", "two_diff_thr_local_a"], 4.0),
                        4.0,
                    ),
                )
            if hasattr(self, "dspin_cosmics_thr_b"):
                _set_value(
                    getattr(self, "dspin_cosmics_thr_b", None),
                    _safe_float(
                        self._cfg_get(cfg, ["cosmics", "two_diff_thr_local_b"], 2.5),
                        2.5,
                    ),
                )
            if hasattr(self, "dspin_cosmics_lap_k_scale"):
                _set_value(
                    getattr(self, "dspin_cosmics_lap_k_scale", None),
                    _safe_float(
                        self._cfg_get(cfg, ["cosmics", "lap_k_scale"], 0.8), 0.8
                    ),
                )
            if hasattr(self, "dspin_cosmics_lap_k_min"):
                _set_value(
                    getattr(self, "dspin_cosmics_lap_k_min", None),
                    _safe_float(self._cfg_get(cfg, ["cosmics", "lap_k_min"], 5.0), 5.0),
                )

        # --- Flatfield ---
        if hasattr(self, "chk_flat_enabled") and not self._stage_dirty.get(
            "flatfield", False
        ):
            _set_checked(
                getattr(self, "chk_flat_enabled", None),
                bool(self._cfg_get(cfg, ["flatfield", "enabled"], False)),
            )
            apply_to = set(self._cfg_get(cfg, ["flatfield", "apply_to"], []) or [])
            for name, cb in [
                ("obj", getattr(self, "chk_flat_obj", None)),
                ("sky", getattr(self, "chk_flat_sky", None)),
                ("sunsky", getattr(self, "chk_flat_sunsky", None)),
                ("neon", getattr(self, "chk_flat_neon", None)),
            ]:
                _set_checked(cb, name in apply_to)
            if hasattr(self, "chk_flat_bias"):
                _set_checked(
                    getattr(self, "chk_flat_bias", None),
                    bool(self._cfg_get(cfg, ["flatfield", "bias_subtract"], True)),
                )
            if hasattr(self, "chk_flat_png"):
                _set_checked(
                    getattr(self, "chk_flat_png", None),
                    bool(self._cfg_get(cfg, ["flatfield", "save_png"], True)),
                )

        # --- SuperNeon ---
        if hasattr(self, "spin_sn_y_half") and not self._stage_dirty.get(
            "superneon", False
        ):
            _set_value(
                getattr(self, "spin_sn_y_half", None),
                _safe_int(self._cfg_get(cfg, ["wavesol", "y_half"], 20), 20),
            )
            if hasattr(self, "spin_sn_xshift"):
                _set_value(
                    getattr(self, "spin_sn_xshift", None),
                    _safe_int(self._cfg_get(cfg, ["wavesol", "xshift_max_abs"], 2), 2),
                )
            if hasattr(self, "chk_sn_bias_sub"):
                _set_checked(
                    getattr(self, "chk_sn_bias_sub", None),
                    bool(self._cfg_get(cfg, ["superneon", "bias_sub"], True)),
                )
            # noise
            noise = self._cfg_get(cfg, ["wavesol", "noise"], {}) or {}
            if hasattr(self, "spin_sn_bl_bin"):
                _set_value(
                    getattr(self, "spin_sn_bl_bin", None),
                    _safe_int(noise.get("baseline_bin_size", 32), 32),
                )
            if hasattr(self, "dspin_sn_bl_q"):
                _set_value(
                    getattr(self, "dspin_sn_bl_q", None),
                    _safe_float(noise.get("baseline_quantile", 0.2), 0.2),
                )
            if hasattr(self, "spin_sn_bl_smooth"):
                _set_value(
                    getattr(self, "spin_sn_bl_smooth", None),
                    _safe_int(noise.get("baseline_smooth_bins", 5), 5),
                )
            if hasattr(self, "dspin_sn_empty_q"):
                _set_value(
                    getattr(self, "dspin_sn_empty_q", None),
                    _safe_float(noise.get("empty_quantile", 0.7), 0.7),
                )
            if hasattr(self, "dspin_sn_clip"):
                _set_value(
                    getattr(self, "dspin_sn_clip", None),
                    _safe_float(noise.get("clip", 3.5), 3.5),
                )
            if hasattr(self, "spin_sn_niter"):
                _set_value(
                    getattr(self, "spin_sn_niter", None),
                    _safe_int(noise.get("n_iter", 3), 3),
                )
            # peaks
            for key, attr, default in [
                ("peak_snr", "dspin_sn_peak_snr", 4.5),
                ("peak_prom_snr", "dspin_sn_peak_prom", 3.5),
                ("peak_floor_snr", "dspin_sn_peak_floor", 3.0),
            ]:
                if hasattr(self, attr):
                    w = getattr(self, attr, None)
                    _set_value(
                        w,
                        _safe_float(
                            self._cfg_get(cfg, ["wavesol", key], default),
                            float(default),
                        ),
                    )
            if hasattr(self, "spin_sn_peak_dist"):
                _set_value(
                    getattr(self, "spin_sn_peak_dist", None),
                    _safe_int(self._cfg_get(cfg, ["wavesol", "peak_distance"], 3), 3),
                )
            if hasattr(self, "chk_sn_autotune"):
                _set_checked(
                    getattr(self, "chk_sn_autotune", None),
                    bool(self._cfg_get(cfg, ["wavesol", "peak_autotune"], True)),
                )
            if hasattr(self, "spin_sn_target_min"):
                _set_value(
                    getattr(self, "spin_sn_target_min", None),
                    _safe_int(self._cfg_get(cfg, ["wavesol", "peak_target_min"], 0), 0),
                )
            if hasattr(self, "spin_sn_target_max"):
                _set_value(
                    getattr(self, "spin_sn_target_max", None),
                    _safe_int(self._cfg_get(cfg, ["wavesol", "peak_target_max"], 0), 0),
                )

        # --- LineID (GUI) ---
        if hasattr(self, "dspin_lineid_sigma_k") and not self._stage_dirty.get(
            "lineid", False
        ):
            _set_value(
                getattr(self, "dspin_lineid_sigma_k", None),
                _safe_float(
                    self._cfg_get(cfg, ["wavesol", "gui_min_amp_sigma_k"], 5.0), 5.0
                ),
            )
            if hasattr(self, "dspin_lineid_min_amp"):
                v = self._cfg_get(cfg, ["wavesol", "gui_min_amp"], None)
                vv = _safe_float(v, 0.0) if v not in (None, "") else 0.0
                _set_value(getattr(self, "dspin_lineid_min_amp", None), vv)
            if hasattr(self, "edit_lineid_lines_csv"):
                _set_text(
                    getattr(self, "edit_lineid_lines_csv", None),
                    str(self._cfg_get(cfg, ["wavesol", "neon_lines_csv"], "") or ""),
                )
            if hasattr(self, "edit_lineid_atlas_pdf"):
                _set_text(
                    getattr(self, "edit_lineid_atlas_pdf", None),
                    str(self._cfg_get(cfg, ["wavesol", "atlas_pdf"], "") or ""),
                )

        # --- Wavelength solution ---
        if hasattr(self, "spin_ws_poly_deg") and not self._stage_dirty.get(
            "wavesol", False
        ):
            _set_value(
                getattr(self, "spin_ws_poly_deg", None),
                _safe_int(self._cfg_get(cfg, ["wavesol", "poly_deg_1d"], 4), 4),
            )
            if hasattr(self, "dspin_ws_blend"):
                _set_value(
                    getattr(self, "dspin_ws_blend", None),
                    _safe_float(
                        self._cfg_get(cfg, ["wavesol", "blend_weight"], 0.35), 0.35
                    ),
                )
            if hasattr(self, "dspin_ws_poly_clip"):
                _set_value(
                    getattr(self, "dspin_ws_poly_clip", None),
                    _safe_float(
                        self._cfg_get(cfg, ["wavesol", "poly_sigma_clip"], 3.0), 3.0
                    ),
                )
            if hasattr(self, "spin_ws_poly_iter"):
                _set_value(
                    getattr(self, "spin_ws_poly_iter", None),
                    _safe_int(self._cfg_get(cfg, ["wavesol", "poly_maxiter"], 6), 6),
                )
            if hasattr(self, "combo_ws_model2d"):
                m = str(self._cfg_get(cfg, ["wavesol", "model2d"], "auto") or "auto")
                _set_combo_text(getattr(self, "combo_ws_model2d", None), m)
            for key, attr, default in [
                ("power_deg", "spin_ws_power_deg", 3),
                ("cheb_degx", "spin_ws_cheb_x", 4),
                ("cheb_degy", "spin_ws_cheb_y", 2),
                ("edge_crop_x", "spin_ws_crop_x", 0),
                ("edge_crop_y", "spin_ws_crop_y", 0),
            ]:
                if hasattr(self, attr):
                    w = getattr(self, attr, None)
                    _set_value(
                        w,
                        _safe_int(
                            self._cfg_get(cfg, ["wavesol", key], default), default
                        ),
                    )

        # --- Linearize ---
        if hasattr(self, "chk_lin_enabled") and not self._stage_dirty.get(
            "linearize", False
        ):
            lin = (
                cfg.get("linearize", {})
                if isinstance(cfg.get("linearize"), dict)
                else {}
            )
            _set_checked(
                getattr(self, "chk_lin_enabled", None), bool(lin.get("enabled", True))
            )
            for attr, key, default in [
                ("dspin_lin_dlambda", "dlambda_A", 0.0),
                ("dspin_lin_lmin", "lambda_min_A", 0.0),
                ("dspin_lin_lmax", "lambda_max_A", 0.0),
            ]:
                if hasattr(self, attr):
                    w = getattr(self, attr, None)
                    v = lin.get(key, None)
                    vv = _safe_float(v, 0.0) if v not in (None, "") else 0.0
                    _set_value(w, vv)
            for attr, key, default in [
                ("spin_lin_crop_top", "y_crop_top", 0),
                ("spin_lin_crop_bot", "y_crop_bottom", 0),
            ]:
                if hasattr(self, attr):
                    w = getattr(self, attr, None)
                    _set_value(w, _safe_int(lin.get(key, default), default))
            if hasattr(self, "chk_lin_png"):
                _set_checked(
                    getattr(self, "chk_lin_png", None), bool(lin.get("save_png", True))
                )
            if hasattr(self, "chk_lin_per_frame"):
                _set_checked(
                    getattr(self, "chk_lin_per_frame", None),
                    bool(lin.get("save_per_frame", False)),
                )

        # --- Sky subtraction ---
        if hasattr(self, "chk_sky_enabled") and not self._stage_dirty.get("sky", False):
            sky = cfg.get("sky", {}) if isinstance(cfg.get("sky"), dict) else {}
            _set_checked(
                getattr(self, "chk_sky_enabled", None), bool(sky.get("enabled", True))
            )


            # Primary method (observer-friendly). Support legacy keys (input_geometry/method).
            if hasattr(self, "combo_sky_primary_method"):
                pm_raw = sky.get("primary_method") or sky.get("method") or sky.get("primary")
                ig = str(sky.get("input_geometry", "") or "").strip().lower()
                pm = str(pm_raw or "kelson_raw").strip().lower()
                # Legacy RECTIFIED branch is not exposed in UI anymore; map to safe default.
                if ig.startswith("rect"):
                    pm = "kelson_raw"
                if pm.startswith("kelson"):
                    val = "kelson_raw"
                elif "scale" in pm:
                    val = "sky_scale_raw"
                else:
                    val = "kelson_raw"
                combo = getattr(self, "combo_sky_primary_method", None)
                try:
                    i = combo.findData(val)
                    if i >= 0:
                        combo.setCurrentIndex(i)
                except Exception:
                    pass

            # Post-rectification residual cleanup (executed in Linearization)
            if hasattr(self, "combo_sky_post_cleanup"):
                lin = cfg.get("linearize", {}) if isinstance(cfg.get("linearize"), dict) else {}
                v = lin.get("post_sky_cleanup", None)
                if v in (None, ""):
                    v = sky.get("post_cleanup", "auto")
                vv = str(v or "auto").strip().lower()
                if vv not in ("off", "auto", "on"):
                    vv = "auto"
                combo = getattr(self, "combo_sky_post_cleanup", None)
                try:
                    i = combo.findData(vv)
                    if i >= 0:
                        combo.setCurrentIndex(i)
                except Exception:
                    pass
            if hasattr(self, "chk_sky_per_exp"):
                _set_checked(
                    getattr(self, "chk_sky_per_exp", None),
                    bool(sky.get("per_exposure", True)),
                )
            if hasattr(self, "chk_sky_stack_after"):
                _set_checked(
                    getattr(self, "chk_sky_stack_after", None),
                    False,
                )
            if hasattr(self, "chk_sky_save_models"):
                _set_checked(
                    getattr(self, "chk_sky_save_models", None),
                    bool(sky.get("save_per_exp_model", False)),
                )
            if hasattr(self, "dspin_sky_step"):
                _set_value(
                    getattr(self, "dspin_sky_step", None),
                    _safe_float(sky.get("bsp_step_A", 2.0), 2.0),
                )
            if hasattr(self, "spin_sky_deg"):
                _set_value(
                    getattr(self, "spin_sky_deg", None),
                    _safe_int(sky.get("bsp_degree", 3), 3),
                )
            if hasattr(self, "dspin_sky_clip"):
                _set_value(
                    getattr(self, "dspin_sky_clip", None),
                    _safe_float(sky.get("sigma_clip", 3.0), 3.0),
                )
            if hasattr(self, "spin_sky_maxiter"):
                _set_value(
                    getattr(self, "spin_sky_maxiter", None),
                    _safe_int(sky.get("maxiter", 6), 6),
                )
            if hasattr(self, "chk_sky_spatial"):
                _set_checked(
                    getattr(self, "chk_sky_spatial", None),
                    bool(sky.get("use_spatial_scale", True)),
                )
            if hasattr(self, "spin_sky_poly"):
                _set_value(
                    getattr(self, "spin_sky_poly", None),
                    _safe_int(sky.get("spatial_poly_deg", 0), 0),
                )
            # ROI label
            if hasattr(self, "lbl_sky_roi"):
                roi = sky.get("roi", {}) if isinstance(sky.get("roi"), dict) else {}

                def _f(k: str) -> str:
                    v = roi.get(k, None)
                    return str(_safe_int(v, 0)) if v not in (None, "") else "—"

                _set_label(
                    getattr(self, "lbl_sky_roi", None),
                    f"Object: [{_f('obj_y0')}..{_f('obj_y1')}],  "
                    f"Sky(top): [{_f('sky_top_y0')}..{_f('sky_top_y1')}],  "
                    f"Sky(bot): [{_f('sky_bot_y0')}..{_f('sky_bot_y1')}]",
                )

            # Flexure UI (optional)
            flex = (
                sky.get("flexure", {}) if isinstance(sky.get("flexure"), dict) else {}
            )

            def _fmt_windows(unit: str, winA, winP) -> str:
                unit = str(unit or "auto")
                if unit.lower() in ("a", "angstrom", "å"):
                    if not winA:
                        return "<no windows>"
                    return (
                        "; ".join(
                            [
                                f"{_safe_float(a, 0.0):.1f}–{_safe_float(b, 0.0):.1f}"
                                for a, b in winA
                            ]
                        )
                        + " Å"
                    )
                if unit.lower() in ("pix", "pixel", "pixels"):
                    if not winP:
                        return "<no windows>"
                    return (
                        "; ".join(
                            [f"{_safe_int(a, 0)}–{_safe_int(b, 0)}" for a, b in winP]
                        )
                        + " pix"
                    )
                # auto: prefer A if present
                if winA:
                    return (
                        "; ".join(
                            [
                                f"{_safe_float(a, 0.0):.1f}–{_safe_float(b, 0.0):.1f}"
                                for a, b in winA
                            ]
                        )
                        + " Å"
                    )
                if winP:
                    return (
                        "; ".join(
                            [f"{_safe_int(a, 0)}–{_safe_int(b, 0)}" for a, b in winP]
                        )
                        + " pix"
                    )
                return "<no windows>"

            if hasattr(self, "chk_sky_flex_enabled"):
                _set_checked(
                    getattr(self, "chk_sky_flex_enabled", None),
                    bool(flex.get("enabled", False)),
                )
            if hasattr(self, "combo_sky_flex_mode"):
                m = str(flex.get("mode", "full") or "full")
                _set_combo_text(getattr(self, "combo_sky_flex_mode", None), m)
            if hasattr(self, "spin_sky_flex_max"):
                _set_value(
                    getattr(self, "spin_sky_flex_max", None),
                    _safe_int(flex.get("max_shift_pix", 6), 6),
                )
            if hasattr(self, "combo_sky_flex_windows_unit"):
                u = str(flex.get("windows_unit", "auto") or "auto")
                _set_combo_text(getattr(self, "combo_sky_flex_windows_unit", None), u)
            if hasattr(self, "chk_sky_flex_ydep"):
                _set_checked(
                    getattr(self, "chk_sky_flex_ydep", None),
                    bool(flex.get("y_dependent", False)),
                )
            if hasattr(self, "spin_sky_flex_y_poly"):
                _set_value(
                    getattr(self, "spin_sky_flex_y_poly", None),
                    _safe_int(flex.get("y_poly_deg", 1), 1),
                )
            if hasattr(self, "spin_sky_flex_y_smooth"):
                _set_value(
                    getattr(self, "spin_sky_flex_y_smooth", None),
                    _safe_int(flex.get("y_smooth_bins", 5), 5),
                )
            if hasattr(self, "dspin_sky_flex_min_score"):
                _set_value(
                    getattr(self, "dspin_sky_flex_min_score", None),
                    _safe_float(flex.get("min_score", 0.06), 0.06),
                )
            if hasattr(self, "lbl_sky_flex_windows"):
                u = str(flex.get("windows_unit", "auto") or "auto")
                winA = (
                    flex.get("windows_A")
                    or flex.get("windows")
                    or flex.get("windows_angstrom")
                    or []
                )
                winP = flex.get("windows_pix") or flex.get("windows_pixels") or []
                _set_label(
                    getattr(self, "lbl_sky_flex_windows", None),
                    _fmt_windows(u, winA, winP),
                )

            # Stack2D UI (optional)
            st = cfg.get("stack2d", {}) if isinstance(cfg.get("stack2d"), dict) else {}
            ya = st.get("y_align", {}) if isinstance(st.get("y_align"), dict) else {}

            if hasattr(self, "dspin_stack_sigma"):
                _set_value(
                    getattr(self, "dspin_stack_sigma", None),
                    _safe_float(st.get("sigma_clip", 3.0), 3.0),
                )
            if hasattr(self, "spin_stack_maxiter"):
                _set_value(
                    getattr(self, "spin_stack_maxiter", None),
                    _safe_int(st.get("maxiter", 6), 6),
                )
            if hasattr(self, "chk_stack_y_align"):
                _set_checked(
                    getattr(self, "chk_stack_y_align", None),
                    bool(ya.get("enabled", False)),
                )
            if hasattr(self, "spin_stack_y_align_max"):
                _set_value(
                    getattr(self, "spin_stack_y_align_max", None),
                    _safe_int(ya.get("max_shift_pix", 8), 8),
                )
            if hasattr(self, "combo_stack_y_align_mode"):
                m = str(ya.get("mode", "full") or "full")
                _set_combo_text(getattr(self, "combo_stack_y_align_mode", None), m)
            if hasattr(self, "combo_stack_y_align_windows_unit"):
                u = str(ya.get("windows_unit", "auto") or "auto")
                _set_combo_text(
                    getattr(self, "combo_stack_y_align_windows_unit", None), u
                )
            if hasattr(self, "chk_stack_y_align_pos"):
                _set_checked(
                    getattr(self, "chk_stack_y_align_pos", None),
                    bool(ya.get("use_positive_flux", True)),
                )
            if hasattr(self, "lbl_stack_y_align_windows"):
                u = str(ya.get("windows_unit", "auto") or "auto")
                winA = (
                    ya.get("windows_A")
                    or ya.get("windows")
                    or ya.get("windows_angstrom")
                    or []
                )
                winP = ya.get("windows_pix") or ya.get("windows_pixels") or []
                _set_label(
                    getattr(self, "lbl_stack_y_align_windows", None),
                    _fmt_windows(u, winA, winP),
                )

        # --- Extract 1D ---
        if hasattr(self, "chk_ex1d_enabled") and not self._stage_dirty.get(
            "extract1d", False
        ):
            ex = (
                cfg.get("extract1d", {})
                if isinstance(cfg.get("extract1d"), dict)
                else {}
            )
            _set_checked(
                getattr(self, "chk_ex1d_enabled", None), bool(ex.get("enabled", True))
            )

            # Input mode (stack2d vs single_frame)
            if hasattr(self, "combo_ex1d_input_mode"):
                mode = str(ex.get("input_mode", "stack2d") or "stack2d")
                if bool(ex.get("allow_sky_fallback", False)) and mode == "stack2d":
                    mode = "single_frame"
                _set_combo_text(getattr(self, "combo_ex1d_input_mode", None), mode)
            if hasattr(self, "edit_ex1d_single_stem"):
                stem = str(ex.get("single_frame_stem", "") or "")
                if not stem and ex.get("single_frame_path"):
                    try:
                        p = Path(str(ex.get("single_frame_path")))
                        stem = p.name.replace("_skysub.fits", "")
                    except Exception:
                        pass
                _set_text(getattr(self, "edit_ex1d_single_stem", None), stem)
            if hasattr(self, "combo_ex1d_method"):
                m = str(ex.get("method", "boxcar") or "boxcar")
                _set_combo_text(getattr(self, "combo_ex1d_method", None), m)
            if hasattr(self, "spin_ex1d_ap_hw"):
                ahw = ex.get("aperture_half_width", None)
                _set_value(getattr(self, "spin_ex1d_ap_hw", None), _safe_int(ahw, 0))
            if hasattr(self, "dspin_ex1d_trace_bin"):
                _set_value(
                    getattr(self, "dspin_ex1d_trace_bin", None),
                    _safe_float(ex.get("trace_bin_A", 60.0), 60.0),
                )
            if hasattr(self, "spin_ex1d_trace_deg"):
                _set_value(
                    getattr(self, "spin_ex1d_trace_deg", None),
                    _safe_int(ex.get("trace_smooth_deg", 3), 3),
                )
            if hasattr(self, "spin_ex1d_prof_hw"):
                _set_value(
                    getattr(self, "spin_ex1d_prof_hw", None),
                    _safe_int(ex.get("optimal_profile_half_width", 15), 15),
                )
            if hasattr(self, "dspin_ex1d_opt_clip"):
                _set_value(
                    getattr(self, "dspin_ex1d_opt_clip", None),
                    _safe_float(ex.get("optimal_sigma_clip", 5.0), 5.0),
                )
            if hasattr(self, "chk_ex1d_png"):
                _set_checked(
                    getattr(self, "chk_ex1d_png", None), bool(ex.get("save_png", True))
                )

        # keep derived labels fresh
        try:
            self._refresh_pair_sets_combo()
        except Exception:
            pass
        try:
            self._refresh_pairs_label()
        except Exception:
            pass
        try:
            self._update_wavesol_stepper()
        except Exception:
            pass

        try:
            self._update_wavesol_model2d_enables()
        except Exception:
            pass


        # Refresh 'default' icons: sync uses blocked signals for stability.
        for _upd in getattr(self, "_param_default_buttons", []):
            try:
                _upd()
            except Exception:
                pass

    def _build_page_calib(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setSpacing(12)

        # left: controls
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setSpacing(12)

        g = _box("Calibrations")
        left_layout.addWidget(g)
        gl = QtWidgets.QVBoxLayout(g)
        self.lbl_calib = QtWidgets.QLabel(
            "Build report/manifest.json and calib/superbias.fits.\n"
            "Tune parameters below and press Apply before running the stage."
        )
        self.lbl_calib.setWordWrap(True)
        gl.addWidget(self.lbl_calib)

        # --- parameters ---
        pbox = _box("Parameters")
        pl = QtWidgets.QVBoxLayout(pbox)
        pl.setSpacing(10)

        basic = QtWidgets.QWidget()
        bf = QtWidgets.QFormLayout(basic)
        bf.setLabelAlignment(QtCore.Qt.AlignLeft)
        bf.setHorizontalSpacing(12)

        self.combo_bias_combine = QtWidgets.QComboBox()
        self.combo_bias_combine.addItems(["median", "mean"])
        bf.addRow(
            self._param_label(
                "Bias combine",
                "Способ объединения bias-кадров.\n"
                "median — устойчив к выбросам, mean — чуть выше S/N при чистых данных.\n"
                "Типично: median.",
                cfg_path="calib.bias_combine",
            ),
            self._param_field(self.combo_bias_combine, cfg_path="calib.bias_combine"),
        )

        self.spin_bias_sigma_clip = QtWidgets.QDoubleSpinBox()
        self.spin_bias_sigma_clip.setRange(0.0, 10.0)
        self.spin_bias_sigma_clip.setDecimals(2)
        self.spin_bias_sigma_clip.setSingleStep(0.5)
        self.spin_bias_sigma_clip.setToolTip("0 = disable")
        bf.addRow(
            self._param_label(
                "Sigma-clip",
                "Сигма-клиппинг при построении superbias.\n"
                "0 = отключить.\n"
                "Типично: 0–3.",
                cfg_path="calib.bias_sigma_clip",
                units="σ",
            ),
            self._param_field(self.spin_bias_sigma_clip, cfg_path="calib.bias_sigma_clip"),
        )

        adv = QtWidgets.QWidget()
        adv_lay = QtWidgets.QVBoxLayout(adv)
        adv_lay.setContentsMargins(0, 0, 0, 0)
        adv_lay.setSpacing(8)
        note = QtWidgets.QLabel("(Пока без дополнительных параметров для этого этапа.)")
        note.setWordWrap(True)
        adv_lay.addWidget(note)

        # locale for doubles
        self._force_dot_locale(self.spin_bias_sigma_clip)

        pl.addWidget(self._mk_basic_advanced_tabs(basic, adv))
        pl.addWidget(self._mk_stage_apply_row("calib"))

        gl.addWidget(pbox)

        # wire (pending → Apply)
        self.combo_bias_combine.currentTextChanged.connect(
            lambda t: self._stage_set_pending("calib", "calib.bias_combine", str(t))
        )
        self.spin_bias_sigma_clip.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "calib", "calib.bias_sigma_clip", float(v)
            )
        )

        row = QtWidgets.QHBoxLayout()
        self.btn_run_calib = QtWidgets.QPushButton("Run: Manifest + Superbias")
        self.btn_run_calib.setProperty("primary", True)
        self.btn_qc_calib = QtWidgets.QPushButton("QC")
        self.btn_frames_calib = QtWidgets.QPushButton("Frames…")
        self.btn_frames_calib.setToolTip(
            "Open Frames Browser for the Calibrations stage"
        )
        row.addWidget(self.btn_run_calib)
        row.addWidget(self.btn_qc_calib)
        row.addWidget(self.btn_frames_calib)
        row.addStretch(1)
        gl.addLayout(row)
        left_layout.addStretch(1)

        # Outputs are shown in a detached tool window.
        lay.addWidget(left, 1)

        # Outputs are shown in a detached tool window (toolbar: Outputs).
        lay.addWidget(left, 1)


        self.btn_run_calib.clicked.connect(self._do_run_calib)
        self.btn_qc_calib.clicked.connect(self._open_qc_viewer)
        self.btn_frames_calib.clicked.connect(lambda: self._open_frames_window("calib"))

        try:
            self._refresh_pair_sets_combo()
        except Exception:
            pass
        self._sync_stage_controls_from_cfg()
        return w

    def _do_run_calib(self) -> None:
        if not self._ensure_stage_applied("calib", "Calibrations"):
            return
        if not self._ensure_cfg_saved():
            return
        self._set_step_status(self._stage_row_index("biascorr"), "running")
        try:
            ctx = load_context(self._cfg_path)
            self._run_sequence_with_qc_prompt(ctx, ["manifest", "superbias"])
            self._log_info("Calibrations done")
            self._set_step_status(self._stage_row_index("biascorr"), "ok")
            self._maybe_auto_qc()
        except Exception as e:
            self._set_step_status(self._stage_row_index("biascorr"), "fail")
            self._log_exception(e)

    # --------------------------- page: cosmics ---------------------------

    def _build_page_cosmics(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setSpacing(12)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setSpacing(12)

        g = _box("Clean Cosmics")
        left_layout.addWidget(g)
        gl = QtWidgets.QVBoxLayout(g)
        lbl = QtWidgets.QLabel(
            "Clean cosmic rays in object/sky frames.\n"
            "Methods:\n"
            "• auto — choose the best method by the number of frames\n"
            "• stack_mad — robust stack MAD masking (>=3 frames)\n"
            "• two_frame_diff — 2-frame diff-based masking (2 frames)\n"
            "• laplacian — single-frame Laplacian detector (1 frame)\n"
            "Outputs are written under work_dir/cosmics/."
        )
        lbl.setWordWrap(True)
        gl.addWidget(lbl)

        # --- per-stage params ---
        pbox = _box("Parameters")
        pl = QtWidgets.QVBoxLayout(pbox)
        pl.setSpacing(10)

        basic = QtWidgets.QWidget()
        bf = QtWidgets.QFormLayout(basic)
        bf.setLabelAlignment(QtCore.Qt.AlignLeft)
        bf.setHorizontalSpacing(12)

        # Keep checkbox text empty: the label column is the single source of truth.
        self.chk_cosmics_enabled = QtWidgets.QCheckBox("")
        bf.addRow(
            self._param_label(
                "Enabled",
                "Включает/отключает этап Clean Cosmics.\n"
                "Типично: включено для obj/sky (и sunsky при наличии).",
                cfg_path="cosmics.enabled",
            ),
            self._param_field(self.chk_cosmics_enabled, cfg_path="cosmics.enabled"),
        )

        self.combo_cosmics_method = QtWidgets.QComboBox()
        self.combo_cosmics_method.addItems(
            ["auto", "la_cosmic", "stack_mad", "two_frame_diff", "laplacian"]
        )
        bf.addRow(
            self._param_label(
                "Method",
                "Алгоритм подавления космиков.\n"
                "auto — выбрать метод автоматически по числу кадров.\n"
                "stack_mad — устойчивый метод по стеку кадров.\n"
                "two_frame_diff — разностный метод для 2 экспозиций.\n"
                "laplacian — одиночный кадр (fallback).\n"
                "Типично: auto (по умолчанию).",
                cfg_path="cosmics.method",
            ),
            self._param_field(self.combo_cosmics_method, cfg_path="cosmics.method"),
        )

        apply_row_w = QtWidgets.QWidget()
        apply_row = QtWidgets.QHBoxLayout(apply_row_w)
        apply_row.setContentsMargins(0, 0, 0, 0)
        self.chk_cosmics_obj = QtWidgets.QCheckBox("obj")
        self.chk_cosmics_sky = QtWidgets.QCheckBox("sky")
        for cb in (self.chk_cosmics_obj, self.chk_cosmics_sky):
            apply_row.addWidget(cb)
        apply_row.addStretch(1)
        bf.addRow(
            self._param_label(
                "Apply to",
                "К каким типам кадров применять очистку.\nТипично: obj + sky.",
                cfg_path="cosmics.apply_to",
            ),
            self._param_field(apply_row_w, cfg_path="cosmics.apply_to"),
        )

        self.spin_cosmics_k = QtWidgets.QDoubleSpinBox()
        self.spin_cosmics_k.setRange(1.0, 51.0)
        self.spin_cosmics_k.setDecimals(1)
        self.spin_cosmics_k.setSingleStep(0.5)
        self.spin_cosmics_k.setToolTip(
            "Порог (k) в единицах σ/MAD для детекции космиков. "
            "Меньше → агрессивнее (больше масок), больше → мягче. "
            "Ориентир: 3–8."
        )
        bf.addRow(
            self._param_label(
                "k (threshold)",
                "Порог (k) в единицах σ/MAD для детекции космиков.\n"
                "Меньше k → агрессивнее чистка (может резать сигнал).\n"
                "Больше k → мягче (может оставлять космики).\n"
                "Ориентир: 3–8. Советы: недочищает → уменьшить k; режет полезное → увеличить k.",
                cfg_path="cosmics.k",
            ),
            self._param_field(self.spin_cosmics_k, cfg_path="cosmics.k"),
        )

        self.spin_cosmics_dilate = QtWidgets.QSpinBox()
        self.spin_cosmics_dilate.setRange(0, 10)
        self.spin_cosmics_dilate.setSingleStep(1)
        self.spin_cosmics_dilate.setToolTip(
            "Радиус дилатации бинарной маски (пикс). 0 — без расширения. "
            "Полезно, чтобы захватывать 'хвосты' космиков. Ориентир: 0–2."
        )
        bf.addRow(
            self._param_label(
                "Dilate",
                "Радиус расширения маски космиков (в пикселях).\n"
                "0 — отключено; 1–2 — типично; больше — агрессивно.",
                cfg_path="cosmics.dilate",
                units="px",
            ),
            self._param_field(self.spin_cosmics_dilate, cfg_path="cosmics.dilate"),
        )
        adv = QtWidgets.QWidget()
        af = QtWidgets.QFormLayout(adv)
        af.setLabelAlignment(QtCore.Qt.AlignLeft)
        af.setHorizontalSpacing(12)

        self.chk_cosmics_bias = QtWidgets.QCheckBox("Yes")
        af.addRow(
            self._param_label(
                "Bias subtract",
                "Вычитать superbias перед поиском космиков.\nТипично: включено.",
                cfg_path="cosmics.bias_subtract",
            ),
            self._param_field(self.chk_cosmics_bias, cfg_path="cosmics.bias_subtract"),
        )

        self.chk_cosmics_png = QtWidgets.QCheckBox("Yes")
        af.addRow(
            self._param_label(
                "Save PNG",
                "Сохранять QC-картинки (coverage/sum).\n"
                "Типично: включено для диагностики.",
                cfg_path="cosmics.save_png",
            ),
            self._param_field(self.chk_cosmics_png, cfg_path="cosmics.save_png"),
        )

        self.chk_cosmics_mask_fits = QtWidgets.QCheckBox("Yes")
        af.addRow(
            self._param_label(
                "Save mask FITS",
                "Сохранять FITS-маски космиков (по кадрам) в work_dir/cosmics/.../masks_fits/.\n"
                "Полезно для диагностики и повторной обработки.",
                cfg_path="cosmics.save_mask_fits",
            ),
            self._param_field(self.chk_cosmics_mask_fits, cfg_path="cosmics.save_mask_fits"),
        )

        # --- method-specific advanced tuning ---
        af.addRow(QtWidgets.QLabel("<b>stack_mad</b>"))

        self.dspin_cosmics_mad_scale = QtWidgets.QDoubleSpinBox()
        self.dspin_cosmics_mad_scale.setRange(0.05, 10.0)
        self.dspin_cosmics_mad_scale.setDecimals(3)
        self.dspin_cosmics_mad_scale.setSingleStep(0.05)
        self.dspin_cosmics_mad_scale.setToolTip(
            "Масштаб MAD: используется в |x-med|/(mad_scale*MAD) > k. "
            "1.0 — как сейчас; 1.4826 ≈ перевод MAD→σ для нормального распределения."
        )
        af.addRow(
            self._param_label(
                "MAD scale",
                "Масштаб для MAD в stack_mad: |x-med|/(mad_scale*MAD) > k.\n"
                "1.0 — оставить поведение прежним; 1.4826 — приблизить к σ.",
                cfg_path="cosmics.mad_scale",
            ),
            self._param_field(self.dspin_cosmics_mad_scale, cfg_path="cosmics.mad_scale"),
        )

        self.dspin_cosmics_min_mad = QtWidgets.QDoubleSpinBox()
        self.dspin_cosmics_min_mad.setRange(0.0, 1e6)
        self.dspin_cosmics_min_mad.setDecimals(6)
        self.dspin_cosmics_min_mad.setSingleStep(1e-6)
        self.dspin_cosmics_min_mad.setToolTip(
            "Минимально допустимый MAD (floor). 0 — авто (eps). "
            "Помогает, если кадры слишком плоские и маска становится огромной."
        )
        af.addRow(
            self._param_label(
                "min MAD",
                "Нижняя граница для MAD (floor).\n"
                "0 — авто (eps); увеличить, если видишь неадекватно большую маску.",
                cfg_path="cosmics.min_mad",
            ),
            self._param_field(self.dspin_cosmics_min_mad, cfg_path="cosmics.min_mad"),
        )

        self.dspin_cosmics_max_frac = QtWidgets.QDoubleSpinBox()
        self.dspin_cosmics_max_frac.setRange(0.0, 1.0)
        self.dspin_cosmics_max_frac.setDecimals(3)
        self.dspin_cosmics_max_frac.setSingleStep(0.01)
        self.dspin_cosmics_max_frac.setToolTip(
            "Ограничение доли замаскированных пикселей на кадр (0..1). "
            "0 отключает. Типично 0.01–0.05, если нужно подстраховаться."
        )
        af.addRow(
            self._param_label(
                "max masked frac",
                "Лимит доли замаскированных пикселей (на кадр) для stack_mad.\n"
                "0 — отключено; например 0.02 = максимум 2% пикселей.",
                cfg_path="cosmics.max_frac_per_frame",
            ),
            self._param_field(self.dspin_cosmics_max_frac, cfg_path="cosmics.max_frac_per_frame"),
        )

        af.addRow(QtWidgets.QLabel("<b>two_frame_diff</b>"))

        self.spin_cosmics_local_r = QtWidgets.QSpinBox()
        self.spin_cosmics_local_r.setRange(0, 20)
        self.spin_cosmics_local_r.setSingleStep(1)
        self.spin_cosmics_local_r.setToolTip(
            "Радиус локального окна для |diff| (среднее). 2 => 5x5. Типично 1–3."
        )
        af.addRow(
            self._param_label(
                "local r",
                "Радиус локального окна для оценки локального масштаба |diff| (two_frame_diff / laplacian).\n"
                "2 ⇒ 5×5; типично 1–3.",
                cfg_path="cosmics.local_r",
                units="px",
            ),
            self._param_field(self.spin_cosmics_local_r, cfg_path="cosmics.local_r"),
        )

        self.dspin_cosmics_k2_scale = QtWidgets.QDoubleSpinBox()
        self.dspin_cosmics_k2_scale.setRange(0.0, 5.0)
        self.dspin_cosmics_k2_scale.setDecimals(3)
        self.dspin_cosmics_k2_scale.setSingleStep(0.05)
        af.addRow(
            self._param_label(
                "k2 scale",
                "two_frame_diff: k2 = max(k2_min, k2_scale*k).\nТипично: 0.8.",
                cfg_path="cosmics.two_diff_k2_scale",
            ),
            self._param_field(self.dspin_cosmics_k2_scale, cfg_path="cosmics.two_diff_k2_scale"),
        )

        self.dspin_cosmics_k2_min = QtWidgets.QDoubleSpinBox()
        self.dspin_cosmics_k2_min.setRange(0.0, 50.0)
        self.dspin_cosmics_k2_min.setDecimals(2)
        self.dspin_cosmics_k2_min.setSingleStep(0.5)
        af.addRow(
            self._param_label(
                "k2 min",
                "two_frame_diff: нижняя граница для k2.\nТипично: 5.",
                cfg_path="cosmics.two_diff_k2_min",
            ),
            self._param_field(self.dspin_cosmics_k2_min, cfg_path="cosmics.two_diff_k2_min"),
        )

        self.dspin_cosmics_thr_a = QtWidgets.QDoubleSpinBox()
        self.dspin_cosmics_thr_a.setRange(0.0, 20.0)
        self.dspin_cosmics_thr_a.setDecimals(2)
        self.dspin_cosmics_thr_a.setSingleStep(0.2)
        af.addRow(
            self._param_label(
                "local a",
                "two_frame_diff: thr_local = a*loc + b*sigma.\nТипично: a=4.",
                cfg_path="cosmics.two_diff_thr_local_a",
            ),
            self._param_field(self.dspin_cosmics_thr_a, cfg_path="cosmics.two_diff_thr_local_a"),
        )

        self.dspin_cosmics_thr_b = QtWidgets.QDoubleSpinBox()
        self.dspin_cosmics_thr_b.setRange(0.0, 20.0)
        self.dspin_cosmics_thr_b.setDecimals(2)
        self.dspin_cosmics_thr_b.setSingleStep(0.2)
        af.addRow(
            self._param_label(
                "local b",
                "two_frame_diff: thr_local = a*loc + b*sigma.\nТипично: b=2.5.",
                cfg_path="cosmics.two_diff_thr_local_b",
            ),
            self._param_field(self.dspin_cosmics_thr_b, cfg_path="cosmics.two_diff_thr_local_b"),
        )

        af.addRow(QtWidgets.QLabel("<b>laplacian</b>"))

        self.dspin_cosmics_lap_k_scale = QtWidgets.QDoubleSpinBox()
        self.dspin_cosmics_lap_k_scale.setRange(0.0, 5.0)
        self.dspin_cosmics_lap_k_scale.setDecimals(3)
        self.dspin_cosmics_lap_k_scale.setSingleStep(0.05)
        af.addRow(
            self._param_label(
                "lap k scale",
                "laplacian: thr = max(lap_k_min, lap_k_scale*k) * sigma(lap).\n"
                "Типично: 0.8.",
                cfg_path="cosmics.lap_k_scale",
            ),
            self._param_field(self.dspin_cosmics_lap_k_scale, cfg_path="cosmics.lap_k_scale"),
        )

        self.dspin_cosmics_lap_k_min = QtWidgets.QDoubleSpinBox()
        self.dspin_cosmics_lap_k_min.setRange(0.0, 50.0)
        self.dspin_cosmics_lap_k_min.setDecimals(2)
        self.dspin_cosmics_lap_k_min.setSingleStep(0.5)
        af.addRow(
            self._param_label(
                "lap k min",
                "laplacian: нижняя граница для k-терма.\nТипично: 5.",
                cfg_path="cosmics.lap_k_min",
            ),
            self._param_field(self.dspin_cosmics_lap_k_min, cfg_path="cosmics.lap_k_min"),
        )

        # locale for doubles
        self._force_dot_locale(self.spin_cosmics_k)
        for w in (
            self.dspin_cosmics_mad_scale,
            self.dspin_cosmics_min_mad,
            self.dspin_cosmics_max_frac,
            self.dspin_cosmics_k2_scale,
            self.dspin_cosmics_k2_min,
            self.dspin_cosmics_thr_a,
            self.dspin_cosmics_thr_b,
            self.dspin_cosmics_lap_k_scale,
            self.dspin_cosmics_lap_k_min,
        ):
            self._force_dot_locale(w)

        pl.addWidget(self._mk_basic_advanced_tabs(basic, adv))
        pl.addWidget(self._mk_stage_apply_row("cosmics"))

        gl.addWidget(pbox)

        row = QtWidgets.QHBoxLayout()
        self.btn_run_cosmics = QtWidgets.QPushButton("Run: Clean cosmics")
        self.btn_run_cosmics.setProperty("primary", True)
        self.btn_qc_cosmics = QtWidgets.QPushButton("QC")
        self.btn_manual_cosmics = QtWidgets.QPushButton("Manual…")
        self.btn_manual_cosmics.setToolTip(
            "Manual cleanup after automatic cosmics (rectangle → Enter, Ctrl+Z undo)"
        )
        self.btn_frames_cosmics = QtWidgets.QPushButton("Frames…")
        self.btn_frames_cosmics.setToolTip("Open Frames Browser for the Cosmics stage")
        row.addWidget(self.btn_run_cosmics)
        row.addWidget(self.btn_qc_cosmics)
        row.addWidget(self.btn_manual_cosmics)
        row.addWidget(self.btn_frames_cosmics)
        row.addStretch(1)
        gl.addLayout(row)
        left_layout.addStretch(1)

        # Outputs are shown in a detached tool window (toolbar: Outputs).
        lay.addWidget(left, 1)


        self.btn_run_cosmics.clicked.connect(self._do_run_cosmics)
        self.btn_qc_cosmics.clicked.connect(self._open_qc_viewer)
        self.btn_manual_cosmics.clicked.connect(self._do_manual_cosmics)
        self.btn_frames_cosmics.clicked.connect(
            lambda: self._open_frames_window("cosmics")
        )

        # wire per-stage controls (pending → Apply)
        def _apply_to_from_ui() -> list[str]:
            out: list[str] = []
            if self.chk_cosmics_obj.isChecked():
                out.append("obj")
            if self.chk_cosmics_sky.isChecked():
                out.append("sky")
            return out

        def _on_apply_to(*_):
            self._stage_set_pending("cosmics", "cosmics.apply_to", _apply_to_from_ui())

        for cb in (self.chk_cosmics_obj, self.chk_cosmics_sky):
            cb.toggled.connect(_on_apply_to)
        self.chk_cosmics_enabled.toggled.connect(
            lambda v: self._stage_set_pending("cosmics", "cosmics.enabled", bool(v))
        )
        self.combo_cosmics_method.currentTextChanged.connect(
            lambda t: self._stage_set_pending("cosmics", "cosmics.method", str(t))
        )
        self.spin_cosmics_k.valueChanged.connect(
            lambda v: self._stage_set_pending("cosmics", "cosmics.k", float(v))
        )
        self.chk_cosmics_bias.toggled.connect(
            lambda v: self._stage_set_pending(
                "cosmics", "cosmics.bias_subtract", bool(v)
            )
        )
        self.chk_cosmics_png.toggled.connect(
            lambda v: self._stage_set_pending("cosmics", "cosmics.save_png", bool(v))
        )

        self.spin_cosmics_dilate.valueChanged.connect(
            lambda v: self._stage_set_pending("cosmics", "cosmics.dilate", int(v))
        )
        self.chk_cosmics_mask_fits.toggled.connect(
            lambda v: self._stage_set_pending(
                "cosmics", "cosmics.save_mask_fits", bool(v)
            )
        )
        self.dspin_cosmics_mad_scale.valueChanged.connect(
            lambda v: self._stage_set_pending("cosmics", "cosmics.mad_scale", float(v))
        )
        self.dspin_cosmics_min_mad.valueChanged.connect(
            lambda v: self._stage_set_pending("cosmics", "cosmics.min_mad", float(v))
        )
        self.dspin_cosmics_max_frac.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "cosmics", "cosmics.max_frac_per_frame", float(v)
            )
        )
        self.spin_cosmics_local_r.valueChanged.connect(
            lambda v: self._stage_set_pending("cosmics", "cosmics.local_r", int(v))
        )
        self.dspin_cosmics_k2_scale.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "cosmics", "cosmics.two_diff_k2_scale", float(v)
            )
        )
        self.dspin_cosmics_k2_min.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "cosmics", "cosmics.two_diff_k2_min", float(v)
            )
        )
        self.dspin_cosmics_thr_a.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "cosmics", "cosmics.two_diff_thr_local_a", float(v)
            )
        )
        self.dspin_cosmics_thr_b.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "cosmics", "cosmics.two_diff_thr_local_b", float(v)
            )
        )
        self.dspin_cosmics_lap_k_scale.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "cosmics", "cosmics.lap_k_scale", float(v)
            )
        )
        self.dspin_cosmics_lap_k_min.valueChanged.connect(
            lambda v: self._stage_set_pending("cosmics", "cosmics.lap_k_min", float(v))
        )

        # initial sync from YAML
        self._sync_stage_controls_from_cfg()
        return w

    def _do_run_cosmics(self) -> None:
        if not self._ensure_stage_applied("cosmics", "Clean Cosmics"):
            return
        if not self._ensure_cfg_saved():
            return
        self._set_step_status(self._stage_row_index("cosmics"), "running")
        try:
            ctx = load_context(self._cfg_path)
            self._run_sequence_with_qc_prompt(ctx, ["cosmics"])
            self._log_info("Cosmics cleaning done")
            self._set_step_status(self._stage_row_index("cosmics"), "ok")
            self._maybe_auto_qc()
        except Exception as e:
            self._set_step_status(self._stage_row_index("cosmics"), "fail")
            self._log_exception(e)

    def _do_manual_cosmics(self) -> None:
        # Manual is only meaningful after automatic cosmics produced clean frames.
        try:
            if not self._ensure_cfg_saved():
                return
            dlg = CosmicsManualDialog(self._cfg_path, parent=self)
            dlg.exec()
            # Refresh outputs panel (masks/clean may have changed)
            self._maybe_auto_qc()
        except Exception as e:
            self._log_exception(e)

    # --------------------------- page: flatfield ---------------------------

    def _build_page_flatfield(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setSpacing(12)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setSpacing(12)

        g = _box("Flat-fielding (optional)")
        left_layout.addWidget(g)
        gl = QtWidgets.QVBoxLayout(g)

        lbl = QtWidgets.QLabel(
            "Divide frames by a superflat built from selected FLAT frames.\n"
            "Superbias can be subtracted (recommended).\n"
            "Tip: keep this stage disabled if you have no proper flats."
        )
        lbl.setWordWrap(True)
        gl.addWidget(lbl)

        gpar = _box("Parameters")
        left_layout.addWidget(gpar)
        pl = QtWidgets.QVBoxLayout(gpar)
        pl.setSpacing(8)

        # ---------------- BASIC ----------------
        basic = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(basic)
        form.setLabelAlignment(QtCore.Qt.AlignLeft)
        form.setFormAlignment(QtCore.Qt.AlignTop)

        self.chk_flat_enabled = QtWidgets.QCheckBox("Enable flat-fielding")
        self.chk_flat_enabled.setToolTip(
            "Включает/выключает этап. Обычно: выключено, если нет корректных flat-кадров."
        )
        form.addRow(
            self._param_label(
                "Enabled",
                "Типично: включать только если в ночи есть корректные flat.\n\nЗначения: True/False.",
                cfg_path="flatfield.enabled",
            ),
            self._param_field(self.chk_flat_enabled, cfg_path="flatfield.enabled"),
        )

        apply_row = QtWidgets.QHBoxLayout()
        self.chk_flat_obj = QtWidgets.QCheckBox("obj")
        self.chk_flat_sky = QtWidgets.QCheckBox("sky")
        self.chk_flat_sunsky = QtWidgets.QCheckBox("sunsky")
        self.chk_flat_neon = QtWidgets.QCheckBox("neon")
        for cb in (
            self.chk_flat_obj,
            self.chk_flat_sky,
            self.chk_flat_sunsky,
            self.chk_flat_neon,
        ):
            apply_row.addWidget(cb)
        apply_row.addStretch(1)
        apply_w = QtWidgets.QWidget()
        apply_w.setLayout(apply_row)
        form.addRow(
            self._param_label(
                "Apply to",
                "К каким типам кадров применять flat-fielding.\n"
                "Типично: obj+sky (+sunsky при наличии).\n"
                "neon — включать только если это нужно для вашей схемы обработки.",
                cfg_path="flatfield.apply_to",
            ),
            self._param_field(apply_w, cfg_path="flatfield.apply_to"),
        )

        self.chk_flat_bias = QtWidgets.QCheckBox("Subtract superbias")
        self.chk_flat_bias.setToolTip(
            "Рекомендуется: True. Отключайте только если bias уже вычтен ранее."
        )
        form.addRow(
            self._param_label(
                "Bias subtraction",
                "Вычитать superbias при построении и применении superflat.\n"
                "Типично: True.\n"
                "Если вы используете продукт, где bias уже вычтен, можно выключить.",
                cfg_path="flatfield.bias_subtract",
            ),
            self._param_field(self.chk_flat_bias, cfg_path="flatfield.bias_subtract"),
        )

        self.chk_flat_png = QtWidgets.QCheckBox("Save QC PNGs")
        form.addRow(
            self._param_label(
                "QC PNG",
                "Сохранять диагностические PNG.\n"
                "Типично: True (полезно для контроля качества).",
                cfg_path="flatfield.save_png",
            ),
            self._param_field(self.chk_flat_png, cfg_path="flatfield.save_png"),
        )

        adv = QtWidgets.QWidget()
        adv_form = QtWidgets.QFormLayout(adv)
        adv_form.setLabelAlignment(QtCore.Qt.AlignLeft)
        adv_lbl = QtWidgets.QLabel(
            "(Advanced reserved for future: per-kind flats, normalization, etc.)"
        )
        adv_lbl.setWordWrap(True)
        adv_form.addRow("", adv_lbl)

        pl.addWidget(self._mk_basic_advanced_tabs(basic, adv))
        pl.addWidget(self._mk_stage_apply_row("flatfield"))

        row = QtWidgets.QHBoxLayout()
        self.btn_run_flatfield = QtWidgets.QPushButton("Run: Flat-fielding")
        self.btn_run_flatfield.setProperty("primary", True)
        self.btn_qc_flatfield = QtWidgets.QPushButton("QC")
        self.btn_frames_flatfield = QtWidgets.QPushButton("Frames…")
        self.btn_frames_flatfield.setToolTip(
            "Open Frames Browser for the Flat-fielding stage"
        )
        row.addWidget(self.btn_run_flatfield)
        row.addWidget(self.btn_qc_flatfield)
        row.addWidget(self.btn_frames_flatfield)
        row.addStretch(1)
        gl.addLayout(row)

        left_layout.addStretch(1)

        # Outputs are shown in a detached tool window (toolbar: Outputs).
        lay.addWidget(left, 1)


        # actions
        self.btn_run_flatfield.clicked.connect(self._do_run_flatfield)
        self.btn_qc_flatfield.clicked.connect(self._open_qc_viewer)
        self.btn_frames_flatfield.clicked.connect(
            lambda: self._open_frames_window("flatfield")
        )

        # ---- pending wiring (Apply button governs persistence) ----
        def _apply_to_from_ui() -> list[str]:
            out: list[str] = []
            if self.chk_flat_obj.isChecked():
                out.append("obj")
            if self.chk_flat_sky.isChecked():
                out.append("sky")
            if self.chk_flat_sunsky.isChecked():
                out.append("sunsky")
            if self.chk_flat_neon.isChecked():
                out.append("neon")
            return out

        self.chk_flat_enabled.toggled.connect(
            lambda v: self._stage_set_pending("flatfield", "flatfield.enabled", bool(v))
        )
        for cb in (
            self.chk_flat_obj,
            self.chk_flat_sky,
            self.chk_flat_sunsky,
            self.chk_flat_neon,
        ):
            cb.toggled.connect(
                lambda *_: self._stage_set_pending(
                    "flatfield", "flatfield.apply_to", _apply_to_from_ui()
                )
            )
        self.chk_flat_bias.toggled.connect(
            lambda v: self._stage_set_pending(
                "flatfield", "flatfield.bias_subtract", bool(v)
            )
        )
        self.chk_flat_png.toggled.connect(
            lambda v: self._stage_set_pending(
                "flatfield", "flatfield.save_png", bool(v)
            )
        )

        self._sync_stage_controls_from_cfg()
        return w

    def _do_run_flatfield(self) -> None:
        if not self._ensure_stage_applied("flatfield"):
            return
        if not self._ensure_cfg_saved():
            return
        self._set_step_status(self._stage_row_index("flatfield"), "running")
        try:
            ctx = load_context(self._cfg_path)
            self._run_sequence_with_qc_prompt(ctx, ["flatfield"])
            self._log_info("Flat-fielding done")
            self._set_step_status(self._stage_row_index("flatfield"), "ok")
            self._maybe_auto_qc()
        except Exception as e:
            self._set_step_status(self._stage_row_index("flatfield"), "fail")
            self._log_exception(e)

    # --------------------------- page: superneon ---------------------------

    def _build_page_superneon(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setSpacing(12)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setSpacing(12)

        g = _box("SuperNeon")
        left_layout.addWidget(g)
        gl = QtWidgets.QVBoxLayout(g)
        lbl = QtWidgets.QLabel(
            "Stack all NEON frames into a single superneon image,\n"
            "build a robust 1D profile and auto-detect peaks for LineID."
        )
        lbl.setWordWrap(True)
        gl.addWidget(lbl)

        gpar = _box("Parameters")
        left_layout.addWidget(gpar)
        pl = QtWidgets.QVBoxLayout(gpar)
        pl.setSpacing(8)

        # ---------------- BASIC ----------------
        basic = QtWidgets.QWidget()
        bform = QtWidgets.QFormLayout(basic)
        bform.setLabelAlignment(QtCore.Qt.AlignLeft)
        bform.setFormAlignment(QtCore.Qt.AlignTop)

        self.chk_sn_bias_sub = QtWidgets.QCheckBox("Subtract superbias")
        bform.addRow(
            self._param_label(
                "Bias subtraction",
                "Вычитать superbias из NEON перед суммированием.\n"
                "Типично: True.\n\n"
                "Если у вас уже bias-вычтенные кадры — можно выключить.",
                cfg_path="superneon.bias_sub",
            ),
            self._param_field(self.chk_sn_bias_sub, cfg_path="superneon.bias_sub"),
        )

        self.spin_sn_y_half = QtWidgets.QSpinBox()
        self.spin_sn_y_half.setRange(5, 300)
        self.spin_sn_y_half.setSingleStep(1)
        bform.addRow(
            self._param_label(
                "Profile half-height",
                "Полувысота окна по Y для построения 1D профиля (суммирование по щели).\n"
                "Типично: 10–40 px (зависит от binning и ширины LSF).\n"
                "Слишком мало → шумно; слишком много → смешивание фоновых градиентов.",
                cfg_path="wavesol.y_half",
                units="px",
            ),
            self._param_field(self.spin_sn_y_half, cfg_path="wavesol.y_half"),
        )

        self.spin_sn_xshift = QtWidgets.QSpinBox()
        self.spin_sn_xshift.setRange(0, 50)
        self.spin_sn_xshift.setSingleStep(1)
        bform.addRow(
            self._param_label(
                "Max |x-shift|",
                "Ограничение на модуль сдвига по X при выравнивании NEON кадров.\n"
                "Типично: 2–8 px (если стабильная механика).\n"
                "Если кадры гуляют сильнее — увеличьте.",
                cfg_path="wavesol.xshift_max_abs",
                units="px",
            ),
            self._param_field(self.spin_sn_xshift, cfg_path="wavesol.xshift_max_abs"),
        )

        # peak thresholds
        self.dspin_sn_peak_snr = QtWidgets.QDoubleSpinBox()
        self.dspin_sn_peak_snr.setRange(0.5, 50.0)
        self.dspin_sn_peak_snr.setDecimals(2)
        self.dspin_sn_peak_snr.setSingleStep(0.25)
        bform.addRow(
            self._param_label(
                "Peak SNR",
                "Порог детекции пиков в единицах робастной сигмы.\n"
                "Типично: 4–8.\n"
                "Меньше → больше ложных линий; больше → можно потерять слабые линии.",
                cfg_path="wavesol.peak_snr",
                units="σ",
            ),
            self._param_field(self.dspin_sn_peak_snr, cfg_path="wavesol.peak_snr"),
        )

        self.dspin_sn_peak_prom = QtWidgets.QDoubleSpinBox()
        self.dspin_sn_peak_prom.setRange(0.0, 50.0)
        self.dspin_sn_peak_prom.setDecimals(2)
        self.dspin_sn_peak_prom.setSingleStep(0.25)
        bform.addRow(
            self._param_label(
                "Prominence",
                "Требуемая prominence для пика (в сигмах).\n"
                "Типично: 3–7.\n"
                "Полезно для подавления мелких шумовых бугорков.",
                cfg_path="wavesol.peak_prom_snr",
                units="σ",
            ),
            self._param_field(self.dspin_sn_peak_prom, cfg_path="wavesol.peak_prom_snr"),
        )

        self.dspin_sn_peak_floor = QtWidgets.QDoubleSpinBox()
        self.dspin_sn_peak_floor.setRange(0.0, 50.0)
        self.dspin_sn_peak_floor.setDecimals(2)
        self.dspin_sn_peak_floor.setSingleStep(0.25)
        bform.addRow(
            self._param_label(
                "Floor",
                "Минимальная высота над локальным фоном (в сигмах).\nТипично: 2–5.",
                cfg_path="wavesol.peak_floor_snr",
                units="σ",
            ),
            self._param_field(self.dspin_sn_peak_floor, cfg_path="wavesol.peak_floor_snr"),
        )

        self.spin_sn_peak_dist = QtWidgets.QSpinBox()
        self.spin_sn_peak_dist.setRange(1, 50)
        self.spin_sn_peak_dist.setSingleStep(1)
        bform.addRow(
            self._param_label(
                "Min distance",
                "Минимальная дистанция между пиками (в пикселях по X).\n"
                "Типично: 2–6 (зависит от дисперсии и ширины линий).",
                cfg_path="wavesol.peak_distance",
                units="px",
            ),
            self._param_field(self.spin_sn_peak_dist, cfg_path="wavesol.peak_distance"),
        )

        self.chk_sn_autotune = QtWidgets.QCheckBox("Auto-tune threshold")
        bform.addRow(
            self._param_label(
                "Auto-tune",
                "Автоматически подстраивать порог, если найдено слишком мало/слишком много пиков.\n"
                "Типично: True (удобно в реальных данных).",
                cfg_path="wavesol.peak_autotune",
            ),
            self._param_field(self.chk_sn_autotune, cfg_path="wavesol.peak_autotune"),
        )

        self.spin_sn_target_min = QtWidgets.QSpinBox()
        self.spin_sn_target_min.setRange(0, 5000)
        self.spin_sn_target_min.setSingleStep(10)
        self.spin_sn_target_max = QtWidgets.QSpinBox()
        self.spin_sn_target_max.setRange(0, 5000)
        self.spin_sn_target_max.setSingleStep(10)
        tgt_row = QtWidgets.QHBoxLayout()
        tgt_row.addWidget(QtWidgets.QLabel("min"))
        tgt_row.addWidget(
            self._param_field(self.spin_sn_target_min, cfg_path="wavesol.peak_target_min")
        )
        tgt_row.addSpacing(12)
        tgt_row.addWidget(QtWidgets.QLabel("max"))
        tgt_row.addWidget(
            self._param_field(self.spin_sn_target_max, cfg_path="wavesol.peak_target_max")
        )
        tgt_row.addStretch(1)
        tgt_w = QtWidgets.QWidget()
        tgt_w.setLayout(tgt_row)
        bform.addRow(
            self._param_label(
                "Target peaks",
                "",
                cfg_path="wavesol.peak_target_range",
            ),
            self._param_field(tgt_w, show_default_icon=False),
        )

        # ---------------- ADVANCED ----------------
        adv = QtWidgets.QWidget()
        aform = QtWidgets.QFormLayout(adv)
        aform.setLabelAlignment(QtCore.Qt.AlignLeft)

        # noise model
        self.spin_sn_bl_bin = QtWidgets.QSpinBox()
        self.spin_sn_bl_bin.setRange(8, 200)
        self.spin_sn_bl_bin.setSingleStep(4)
        aform.addRow(
            self._param_label(
                "Baseline bin",
                "Размер бина по X при оценке базовой линии.\nТипично: 20–80.",
                cfg_path="wavesol.noise.baseline_bin_size",
                units="px",
            ),
            self._param_field(self.spin_sn_bl_bin, cfg_path="wavesol.noise.baseline_bin_size"),
        )

        self.dspin_sn_bl_q = QtWidgets.QDoubleSpinBox()
        self.dspin_sn_bl_q.setRange(0.01, 0.99)
        self.dspin_sn_bl_q.setDecimals(2)
        self.dspin_sn_bl_q.setSingleStep(0.05)
        aform.addRow(
            self._param_label(
                "Baseline quantile",
                "Квантиль для базовой линии (0..1).\nТипично: 0.2–0.5.",
                cfg_path="wavesol.noise.baseline_quantile",
            ),
            self._param_field(self.dspin_sn_bl_q, cfg_path="wavesol.noise.baseline_quantile"),
        )

        self.spin_sn_bl_smooth = QtWidgets.QSpinBox()
        self.spin_sn_bl_smooth.setRange(0, 50)
        self.spin_sn_bl_smooth.setSingleStep(1)
        aform.addRow(
            self._param_label(
                "Baseline smooth",
                "Сглаживание базовой линии в бин-единицах.\nТипично: 2–8.",
                cfg_path="wavesol.noise.baseline_smooth_bins",
                units="bins",
            ),
            self._param_field(self.spin_sn_bl_smooth, cfg_path="wavesol.noise.baseline_smooth_bins"),
        )

        self.dspin_sn_empty_q = QtWidgets.QDoubleSpinBox()
        self.dspin_sn_empty_q.setRange(0.01, 0.99)
        self.dspin_sn_empty_q.setDecimals(2)
        self.dspin_sn_empty_q.setSingleStep(0.05)
        aform.addRow(
            self._param_label(
                "Empty quantile",
                "Квантиль для оценки 'пустого' уровня (0..1).\nТипично: 0.05–0.2.",
                cfg_path="wavesol.noise.empty_quantile",
            ),
            self._param_field(self.dspin_sn_empty_q, cfg_path="wavesol.noise.empty_quantile"),
        )

        self.dspin_sn_clip = QtWidgets.QDoubleSpinBox()
        self.dspin_sn_clip.setRange(0.0, 20.0)
        self.dspin_sn_clip.setDecimals(2)
        self.dspin_sn_clip.setSingleStep(0.5)
        aform.addRow(
            self._param_label(
                "Noise clip",
                "Сигма-клиппинг при оценке робастной сигмы.\n"
                "0 = выключено. Типично: 2–5.",
                cfg_path="wavesol.noise.clip",
                units="σ",
            ),
            self._param_field(self.dspin_sn_clip, cfg_path="wavesol.noise.clip"),
        )

        self.spin_sn_niter = QtWidgets.QSpinBox()
        self.spin_sn_niter.setRange(1, 50)
        self.spin_sn_niter.setSingleStep(1)
        aform.addRow(
            self._param_label(
                "Noise iters",
                "Число итераций сигма-клиппинга.\nТипично: 2–8.",
                cfg_path="wavesol.noise.n_iter",
            ),
            self._param_field(self.spin_sn_niter, cfg_path="wavesol.noise.n_iter"),
        )

        self.spin_sn_gauss_hw = QtWidgets.QSpinBox()
        self.spin_sn_gauss_hw.setRange(1, 50)
        self.spin_sn_gauss_hw.setSingleStep(1)
        aform.addRow(
            self._param_label(
                "Gaussian half-window",
                "Полуокно (в пикселях) для гауссовой аппроксимации пика.\n"
                "Типично: 3–8.",
                cfg_path="wavesol.gauss_half_win",
                units="px",
            ),
            self._param_field(self.spin_sn_gauss_hw, cfg_path="wavesol.gauss_half_win"),
        )

        # autotune bounds
        self.dspin_sn_snr_min = QtWidgets.QDoubleSpinBox()
        self.dspin_sn_snr_min.setRange(0.1, 50.0)
        self.dspin_sn_snr_min.setDecimals(2)
        self.dspin_sn_snr_min.setSingleStep(0.1)
        self.dspin_sn_snr_max = QtWidgets.QDoubleSpinBox()
        self.dspin_sn_snr_max.setRange(0.1, 100.0)
        self.dspin_sn_snr_max.setDecimals(2)
        self.dspin_sn_snr_max.setSingleStep(0.1)
        snr_row = QtWidgets.QHBoxLayout()
        snr_row.addWidget(QtWidgets.QLabel("min"))
        snr_row.addWidget(
            self._param_field(self.dspin_sn_snr_min, cfg_path="wavesol.peak_snr_min")
        )
        snr_row.addSpacing(12)
        snr_row.addWidget(QtWidgets.QLabel("max"))
        snr_row.addWidget(
            self._param_field(self.dspin_sn_snr_max, cfg_path="wavesol.peak_snr_max")
        )
        snr_row.addStretch(1)
        snr_w = QtWidgets.QWidget()
        snr_w.setLayout(snr_row)
        aform.addRow(
            self._param_label(
                "Auto-tune SNR bounds",
                "",
                cfg_path="wavesol.peak_snr_bounds",
            ),
            self._param_field(snr_w, show_default_icon=False),
        )

        self.dspin_sn_relax = QtWidgets.QDoubleSpinBox()
        self.dspin_sn_relax.setRange(0.1, 0.99)
        self.dspin_sn_relax.setDecimals(2)
        self.dspin_sn_relax.setSingleStep(0.01)
        self.dspin_sn_boost = QtWidgets.QDoubleSpinBox()
        self.dspin_sn_boost.setRange(1.01, 2.0)
        self.dspin_sn_boost.setDecimals(2)
        self.dspin_sn_boost.setSingleStep(0.01)
        rb_row = QtWidgets.QHBoxLayout()
        rb_row.addWidget(QtWidgets.QLabel("relax"))
        rb_row.addWidget(
            self._param_field(self.dspin_sn_relax, cfg_path="wavesol.peak_snr_relax")
        )
        rb_row.addSpacing(12)
        rb_row.addWidget(QtWidgets.QLabel("boost"))
        rb_row.addWidget(
            self._param_field(self.dspin_sn_boost, cfg_path="wavesol.peak_snr_boost")
        )
        rb_row.addStretch(1)
        rb_w = QtWidgets.QWidget()
        rb_w.setLayout(rb_row)
        aform.addRow(
            self._param_label(
                "Relax/Boost",
                "",
                cfg_path="wavesol.peak_snr_relax_boost",
            ),
            self._param_field(rb_w, show_default_icon=False),
        )

        self.spin_sn_max_tries = QtWidgets.QSpinBox()
        self.spin_sn_max_tries.setRange(1, 100)
        self.spin_sn_max_tries.setSingleStep(1)
        aform.addRow(
            self._param_label(
                "Max tries",
                "Максимум попыток авто-подстройки.\nТипично: 6–15.",
                cfg_path="wavesol.peak_autotune_max_tries",
            ),
            self._param_field(
                self.spin_sn_max_tries, cfg_path="wavesol.peak_autotune_max_tries"
            ),
        )

        # dot locale everywhere
        self._force_dot_locale(
            self.dspin_sn_peak_snr,
            self.dspin_sn_peak_prom,
            self.dspin_sn_peak_floor,
            self.dspin_sn_bl_q,
            self.dspin_sn_empty_q,
            self.dspin_sn_clip,
            self.dspin_sn_snr_min,
            self.dspin_sn_snr_max,
            self.dspin_sn_relax,
            self.dspin_sn_boost,
        )

        pl.addWidget(self._mk_basic_advanced_tabs(basic, adv))
        pl.addWidget(self._mk_stage_apply_row("superneon"))

        row = QtWidgets.QHBoxLayout()
        self.btn_run_superneon = QtWidgets.QPushButton("Run: SuperNeon")
        self.btn_run_superneon.setProperty("primary", True)
        self.btn_qc_superneon = QtWidgets.QPushButton("QC")
        self.btn_frames_superneon = QtWidgets.QPushButton("Frames…")
        self.btn_frames_superneon.setToolTip(
            "Open Frames Browser for the SuperNeon stage"
        )
        row.addWidget(self.btn_run_superneon)
        row.addWidget(self.btn_qc_superneon)
        row.addWidget(self.btn_frames_superneon)
        row.addStretch(1)
        gl.addLayout(row)

        left_layout.addStretch(1)

        # Outputs are shown in a detached tool window (toolbar: Outputs).
        lay.addWidget(left, 1)


        # actions
        self.btn_run_superneon.clicked.connect(self._do_run_superneon)
        self.btn_qc_superneon.clicked.connect(self._open_qc_viewer)
        self.btn_frames_superneon.clicked.connect(
            lambda: self._open_frames_window("superneon")
        )

        # pending wiring
        self.chk_sn_bias_sub.toggled.connect(
            lambda v: self._stage_set_pending(
                "superneon", "superneon.bias_sub", bool(v)
            )
        )
        self.spin_sn_y_half.valueChanged.connect(
            lambda v: self._stage_set_pending("superneon", "wavesol.y_half", int(v))
        )
        self.spin_sn_xshift.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "superneon", "wavesol.xshift_max_abs", int(v)
            )
        )
        self.dspin_sn_peak_snr.valueChanged.connect(
            lambda v: self._stage_set_pending("superneon", "wavesol.peak_snr", float(v))
        )
        self.dspin_sn_peak_prom.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "superneon", "wavesol.peak_prom_snr", float(v)
            )
        )
        self.dspin_sn_peak_floor.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "superneon", "wavesol.peak_floor_snr", float(v)
            )
        )
        self.spin_sn_peak_dist.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "superneon", "wavesol.peak_distance", int(v)
            )
        )
        self.chk_sn_autotune.toggled.connect(
            lambda v: self._stage_set_pending(
                "superneon", "wavesol.peak_autotune", bool(v)
            )
        )
        self.spin_sn_target_min.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "superneon", "wavesol.peak_target_min", int(v)
            )
        )
        self.spin_sn_target_max.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "superneon", "wavesol.peak_target_max", int(v)
            )
        )

        self.spin_sn_bl_bin.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "superneon", "wavesol.noise.baseline_bin_size", int(v)
            )
        )
        self.dspin_sn_bl_q.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "superneon", "wavesol.noise.baseline_quantile", float(v)
            )
        )
        self.spin_sn_bl_smooth.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "superneon", "wavesol.noise.baseline_smooth_bins", int(v)
            )
        )
        self.dspin_sn_empty_q.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "superneon", "wavesol.noise.empty_quantile", float(v)
            )
        )
        self.dspin_sn_clip.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "superneon", "wavesol.noise.clip", float(v)
            )
        )
        self.spin_sn_niter.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "superneon", "wavesol.noise.n_iter", int(v)
            )
        )
        self.spin_sn_gauss_hw.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "superneon", "wavesol.gauss_half_win", int(v)
            )
        )
        self.dspin_sn_snr_min.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "superneon", "wavesol.peak_snr_min", float(v)
            )
        )
        self.dspin_sn_snr_max.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "superneon", "wavesol.peak_snr_max", float(v)
            )
        )
        self.dspin_sn_relax.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "superneon", "wavesol.peak_snr_relax", float(v)
            )
        )
        self.dspin_sn_boost.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "superneon", "wavesol.peak_snr_boost", float(v)
            )
        )
        self.spin_sn_max_tries.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "superneon", "wavesol.peak_autotune_max_tries", int(v)
            )
        )

        self._sync_stage_controls_from_cfg()
        return w

    def _do_run_superneon(self) -> None:
        if not self._ensure_stage_applied("superneon"):
            return
        if not self._ensure_cfg_saved():
            return
        self._set_step_status(self._stage_row_index("superneon"), "running")
        try:
            ctx = load_context(self._cfg_path)
            self._run_sequence_with_qc_prompt(ctx, ["superneon"])
            self._log_info("SuperNeon done")
            self._set_step_status(self._stage_row_index("superneon"), "ok")
            self._maybe_auto_qc()
        except Exception as e:
            self._set_step_status(self._stage_row_index("superneon"), "fail")
            self._log_exception(e)

    # --------------------------- page: lineid ---------------------------

    def _build_page_lineid(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setSpacing(12)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setSpacing(12)

        g = _box("LineID (manual pairs)")
        left_layout.addWidget(g)
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

        # --- Parameters (Basic/Advanced) + Apply ---
        gpar = _box("Parameters")
        left_layout.addWidget(gpar)
        pl = QtWidgets.QVBoxLayout(gpar)
        pl.setContentsMargins(10, 10, 10, 10)

        basic = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(basic)
        form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.dspin_lineid_sigma_k = QtWidgets.QDoubleSpinBox()
        self.dspin_lineid_sigma_k.setRange(1.0, 20.0)
        self.dspin_lineid_sigma_k.setSingleStep(0.5)
        self.dspin_lineid_sigma_k.setDecimals(1)

        self.dspin_lineid_min_amp = QtWidgets.QDoubleSpinBox()
        self.dspin_lineid_min_amp.setRange(0.0, 1e9)
        self.dspin_lineid_min_amp.setSingleStep(50.0)
        self.dspin_lineid_min_amp.setDecimals(1)
        self.dspin_lineid_min_amp.setToolTip("0 = auto threshold from sigma_k")

        self.edit_lineid_lines_csv = QtWidgets.QLineEdit()
        self.edit_lineid_atlas_pdf = QtWidgets.QLineEdit()
        btn_browse_csv = QtWidgets.QToolButton()
        btn_browse_csv.setText("…")
        btn_browse_pdf = QtWidgets.QToolButton()
        btn_browse_pdf.setText("…")

        form.addRow(
            self._param_label(
                "Min amp (auto)",
                "Автопорог по шуму для показа/подбора линий.\n"
                "Типичный диапазон: 3–8. Больше — меньше ложных пиков, но можно пропустить слабые.",
                cfg_path="wavesol.gui_min_amp_sigma_k",
                units="σ",
            ),
            self._param_field(self.dspin_lineid_sigma_k, cfg_path="wavesol.gui_min_amp_sigma_k"),
        )
        form.addRow(
            self._param_label(
                "Min amp (absolute)",
                "Абсолютный порог амплитуды (ADU) для GUI. 0 = отключено (используется σ_k).\n"
                "Ориентир: 0, либо ~500–5000 (зависит от экспозиции/усиления).",
                cfg_path="wavesol.gui_min_amp",
                units="ADU",
            ),
            self._param_field(self.dspin_lineid_min_amp, cfg_path="wavesol.gui_min_amp"),
        )

        row_csv = QtWidgets.QHBoxLayout()
        row_csv.addWidget(self.edit_lineid_lines_csv, 1)
        row_csv.addWidget(btn_browse_csv)
        form.addRow(
            self._param_label(
                "Line list (CSV)",
                "CSV со справочными длинами волн неона (и др.).\n"
                "Обычно: neon_lines.csv. Можно указать абсолютный путь.",
                cfg_path="wavesol.neon_lines_csv",
            ),
            self._param_field(row_csv, cfg_path="wavesol.neon_lines_csv"),
        )

        row_pdf = QtWidgets.QHBoxLayout()
        row_pdf.addWidget(self.edit_lineid_atlas_pdf, 1)
        row_pdf.addWidget(btn_browse_pdf)
        form.addRow(
            self._param_label(
                "Atlas (PDF)",
                "PDF-атлас линий для справки в GUI.\n"
                "Обычно: HeNeAr_atlas.pdf. Можно указать абсолютный путь.",
                cfg_path="wavesol.atlas_pdf",
            ),
            self._param_field(row_pdf, cfg_path="wavesol.atlas_pdf"),
        )

        self._force_dot_locale(self.dspin_lineid_sigma_k, self.dspin_lineid_min_amp)

        # Advanced (reserved)
        adv = QtWidgets.QWidget()
        adv_l = QtWidgets.QVBoxLayout(adv)
        adv_l.setContentsMargins(0, 0, 0, 0)
        adv_l.addWidget(
            QtWidgets.QLabel(
                "Advanced: (пока пусто)\n"
                "Сюда можно будет добавить тонкие параметры GUI/подбора линий."
            )
        )

        pl.addWidget(self._mk_basic_advanced_tabs(basic, adv))
        pl.addWidget(self._mk_stage_apply_row("lineid"))

        # --- Actions ---
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
        self.combo_pair_sets.setToolTip(
            "Select a pair set (built-in or from your user library)."
        )
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
        self.act_export_current_pairs = mexp.addAction(
            "Export current pairs (workdir)…"
        )
        mexp.addSeparator()
        self.act_export_user_library_zip = mexp.addAction(
            "Export full user library (.zip)…"
        )
        self.btn_export_pairs.setMenu(mexp)
        row2.addWidget(self.combo_pair_sets, 1)
        row2.addWidget(self.btn_use_pair_set)
        row2.addWidget(self.btn_copy_pair_set)
        row2.addWidget(self.btn_save_workdir_pairs)
        row2.addWidget(self.btn_open_pairs_library)
        row2.addWidget(self.btn_export_pairs)
        row2.addStretch(1)
        gl.addLayout(row2)

        left_layout.addStretch(1)

        # Outputs are shown in a detached tool window (toolbar: Outputs).
        lay.addWidget(left, 1)


        # --- Actions wiring ---
        self.btn_open_lineid.clicked.connect(self._do_open_lineid)
        self.btn_qc_lineid.clicked.connect(self._open_qc_viewer)
        self.btn_frames_lineid.clicked.connect(
            lambda: self._open_frames_window("lineid")
        )
        self.btn_use_pair_set.clicked.connect(self._do_use_pair_set)
        self.btn_copy_pair_set.clicked.connect(self._do_copy_pair_set)
        self.btn_save_workdir_pairs.clicked.connect(self._do_save_workdir_pairs)
        self.btn_open_pairs_library.clicked.connect(self._do_open_pairs_library)
        self.act_export_selected_pair_set.triggered.connect(
            self._do_export_selected_pair_set
        )
        self.act_export_current_pairs.triggered.connect(self._do_export_current_pairs)
        self.act_export_user_library_zip.triggered.connect(
            self._do_export_user_library_zip
        )

        def _browse_csv() -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select line list CSV", "", "CSV (*.csv);;All files (*.*)"
            )
            if path:
                self.edit_lineid_lines_csv.setText(path)

        def _browse_pdf() -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select atlas PDF", "", "PDF (*.pdf);;All files (*.*)"
            )
            if path:
                self.edit_lineid_atlas_pdf.setText(path)

        btn_browse_csv.clicked.connect(_browse_csv)
        btn_browse_pdf.clicked.connect(_browse_pdf)

        # --- Pending wiring ---
        self.dspin_lineid_sigma_k.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "lineid", "wavesol.gui_min_amp_sigma_k", float(v)
            )
        )

        def _set_min_amp(v: float) -> None:
            vv = None if float(v) <= 0.0 else float(v)
            self._stage_set_pending("lineid", "wavesol.gui_min_amp", vv)

        self.dspin_lineid_min_amp.valueChanged.connect(_set_min_amp)
        self.edit_lineid_lines_csv.textChanged.connect(
            lambda t: self._stage_set_pending(
                "lineid", "wavesol.neon_lines_csv", str(t)
            )
        )
        self.edit_lineid_atlas_pdf.textChanged.connect(
            lambda t: self._stage_set_pending("lineid", "wavesol.atlas_pdf", str(t))
        )

        self._sync_stage_controls_from_cfg()
        return w

    def _do_open_lineid(self) -> None:
        if not self._ensure_stage_applied("lineid"):
            return
        if not self._ensure_cfg_saved():
            return

        try:
            # LineID preparation can take noticeable time on large frames.
            from PySide6.QtWidgets import QApplication
            from PySide6.QtCore import Qt

            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()

            ctx = load_context(self._cfg_path)

            # If config points to a built-in pairs file (absolute), avoid overwriting it:
            # switch back to the default workdir hand_pairs.
            pairs = self._current_pairs_path()
            if pairs and pairs.is_absolute():
                self._set_cfg_value("wavesol.hand_pairs_path", "")
                self.editor_yaml.blockSignals(True)
                self.editor_yaml.setPlainText(_yaml_dump(self._cfg or {}))
                self.editor_yaml.blockSignals(False)
                self._do_save_cfg()
                ctx = load_context(self._cfg_path)

            # 1) Generate auxiliary artifacts (auto/template/report) if needed.
            out = run_lineid_prepare(ctx)

            # Restore cursor BEFORE opening interactive GUI.
            try:
                QApplication.restoreOverrideCursor()
            except Exception:
                pass

            # 2) Open the interactive LineID GUI to create/update hand_pairs.txt.
            try:
                from scorpio_pipe.wavesol_paths import wavesol_dir
                from scorpio_pipe.stages.lineid_gui import prepare_lineid

                wdir = wavesol_dir(ctx.cfg)
                superneon_f = wdir / "superneon.fits"
                peaks_f = wdir / "peaks_candidates.csv"
                hand_f = self._current_pairs_path() or (wdir / "hand_pairs.txt")

                wcfg = (
                    (ctx.cfg.get("wavesol", {}) or {})
                    if isinstance(ctx.cfg, dict)
                    else {}
                )
                y_half = int(wcfg.get("y_half", 20))

                prepare_lineid(
                    ctx.cfg,
                    superneon_fits=superneon_f,
                    peaks_candidates_csv=peaks_f,
                    hand_file=hand_f,
                    y_half=y_half,
                    title="LineID",
                )
            except Exception as e:
                self._log_exception(e)

            # Pretty-print: show the key artifacts, not a raw Python object.
            try:
                msg = ", ".join(f"{k}={v}" for k, v in out.items())
            except Exception:
                msg = str(out)
            self._log_info(f"LineID wrote: {msg}")

            self._refresh_pairs_label()
            self._maybe_auto_qc()
        except Exception as e:
            self._log_exception(e)
        finally:
            try:
                from PySide6.QtWidgets import QApplication

                QApplication.restoreOverrideCursor()
            except Exception:
                pass
            try:
                self._update_wavesol_stepper()
            except Exception:
                pass

    # --------------------------- pair sets (wavesol hand pairs) ---------------------------

    def _show_msgbox_lines(
        self, title: str, lines: list[str], icon: str = "info"
    ) -> None:
        try:
            box = QtWidgets.QMessageBox(self)
            if icon == "warn":
                box.setIcon(QtWidgets.QMessageBox.Warning)
            elif icon == "error":
                box.setIcon(QtWidgets.QMessageBox.Critical)
            else:
                box.setIcon(QtWidgets.QMessageBox.Information)
            box.setWindowTitle(title)
            box.setText("\n".join(lines) if lines else "")
            box.exec()
        except Exception:
            # last resort: log
            try:
                self._log_error(f"{title}: " + "; ".join(lines))
            except Exception:
                pass


    def _show_linearize_missing_sky_dialog(self, details: str) -> None:
        """Show a user-friendly dialog when Linearization prerequisites are missing.

        Typical case: user runs Step 10 (Linearization) before Step 09 (Sky Subtraction),
        so 09_sky/*_skysub_raw.fits is missing.
        """
        try:
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Warning)
            box.setWindowTitle("Linearization: missing Sky products")
            box.setText(
                "Linearization (Step 10) requires sky-subtracted RAW products from Step 09 (Sky Subtraction).\n\n"
                "Please run Sky Subtraction first, then retry Linearization."
            )
            if details:
                box.setInformativeText(details)
            box.addButton("Close", QtWidgets.QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() == btn_go:
                try:
                    self.steps.setCurrentRow(self._stage_row_index("sky"))
                except Exception:
                    pass
        except Exception:
            # last resort: show a simple info box
            try:
                self._show_msgbox_lines(
                    "Linearization: missing Sky products",
                    [
                        "Linearization requires Sky Subtraction products (Step 09).",
                        details or "",
                    ],
                    icon="warn",
                )
            except Exception:
                pass


    def _setup_key_for_pairs(self) -> dict[str, str]:
        """Return a *stable* setup key used by the pairs library.

        Note: some datasets store binning as a string like "1x2" (SCORPIO often
        does). The UI must *not* cast it to int.
        """

        cfg = self._cfg or {}
        setup = cfg.get("setup", {}) if isinstance(cfg.get("setup", {}), dict) else {}

        def _norm(v: Any) -> str:
            return str(v or "").strip()

        def _norm_binning(v: Any) -> str:
            if v is None:
                return ""
            if isinstance(v, (tuple, list)) and len(v) >= 2:
                try:
                    return f"{int(v[0])}x{int(v[1])}"
                except Exception:
                    return "x".join(str(x) for x in v[:2])
            s = str(v).strip()
            # common forms: "1x2", "1×2", "1 x 2"
            import re

            m = re.match(r"^\s*(\d+)\s*[x×]\s*(\d+)\s*$", s)
            if m:
                return f"{int(m.group(1))}x{int(m.group(2))}"
            return s

        # Only keep fields that define pair-set identity
        return {
            "instrument": _norm(setup.get("instrument", "")),
            "disperser": _norm(setup.get("disperser", "")),
            "slit": _norm(setup.get("slit", "")),
            "binning": _norm_binning(setup.get("binning", "")),
        }

    def _refresh_pair_sets_combo(self) -> None:
        """Populate pair-set combos (LineID page and Wavesol stepper)."""
        combos: list[tuple[str, tuple[str, ...]]] = []
        if hasattr(self, "combo_pair_sets"):
            combos.append(
                ("combo_pair_sets", ("btn_use_pair_set", "btn_copy_pair_set"))
            )
        if hasattr(self, "combo_pair_sets_ws"):
            combos.append(("combo_pair_sets_ws", ("btn_use_pair_set_ws",)))
        if not combos:
            return

        try:
            setup = self._setup_key_for_pairs()
            items = list_pair_sets(setup)
        except Exception as e:
            self._log_exception(e)
            items = []

        for combo_name, btn_names in combos:
            combo = getattr(self, combo_name)

            # preserve current selection if possible
            try:
                prev = combo.currentData()
            except Exception:
                prev = None

            combo.blockSignals(True)
            combo.clear()
            if not items:
                combo.addItem("(нет наборов)", None)
                combo.setEnabled(False)
            else:
                combo.setEnabled(True)
                for it in items:
                    tag = "user" if str(it.origin) == "user" else "built-in"
                    combo.addItem(
                        f"{it.label} [{tag}]",
                        {
                            "origin": str(it.origin),
                            "label": str(it.label),
                            "path": str(it.path),
                        },
                    )

            # try restore
            if prev is not None and items:
                for i in range(combo.count()):
                    if combo.itemData(i) == prev:
                        combo.setCurrentIndex(i)
                        break
            combo.blockSignals(False)

            # update buttons/actions
            try:
                has_sel = isinstance(combo.currentData(), dict)
                for btn_name in btn_names:
                    if hasattr(self, btn_name):
                        getattr(self, btn_name).setEnabled(bool(has_sel))
                if combo_name == "combo_pair_sets" and hasattr(
                    self, "act_export_selected_pair_set"
                ):
                    self.act_export_selected_pair_set.setEnabled(bool(has_sel))
            except Exception:
                pass

    def _selected_pair_set_id(self) -> dict | None:
        if not hasattr(self, "combo_pair_sets"):
            return None
        data = self.combo_pair_sets.currentData()
        if not isinstance(data, dict):
            return None
        if "path" not in data:
            return None
        return {
            "path": str(data["path"]),
            "label": str(data.get("label", "")),
            "origin": str(data.get("origin", "")),
        }

    def _current_pairs_path(self):
        """Resolve current hand pairs path from config (may be empty)."""
        cfg = dict(self._cfg or {})
        if self._cfg_path:
            # ensure relative work_dir is resolved against config folder
            cfg.setdefault("config_dir", str(self._cfg_path.parent))
        hp = str(self._get_cfg_value("wavesol.hand_pairs_path", "") or "").strip()
        if hp:
            p = Path(hp).expanduser()
            if not p.is_absolute() and self._cfg_path:
                # hand_pairs_path is relative to work_dir or config_dir; prefer work_dir
                wd = resolve_work_dir(cfg)
                cand = wd / p
                if cand.exists():
                    return cand
                return (self._cfg_path.parent / p).resolve()
            return p

        if not self._cfg_path:
            return None
        wd = resolve_work_dir(cfg)
        # default location (disperser-specific)
        try:
            return wavesol_dir(cfg) / "hand_pairs.txt"
        except Exception:
            return wd / "wavesol" / "hand_pairs.txt"

    def _refresh_pairs_label(self) -> None:
        p = self._current_pairs_path()
        txt = "hand pairs: —" if p is None else f"hand pairs: {p}"
        for name in ("lbl_pairs_file", "lbl_pairs_file_ws"):
            if hasattr(self, name):
                try:
                    getattr(self, name).setText(txt)
                except Exception:
                    pass

    def _do_use_pair_set(self) -> None:
        """Copy selected pair set into work_dir and point config to it."""
        if not self._ensure_cfg_saved():
            return
        if not self._sync_cfg_from_editor():
            return
        ps = self._selected_pair_set_id()
        if ps is None:
            self._show_msgbox_lines("Pairs", ["Не выбран набор пар."], icon="warn")
            return
        self._do_use_pair_set_impl(ps)

    def _do_use_pair_set_ws(self) -> None:
        """Same as 'Use pair set', but triggered from the Wavesolution stepper."""
        if not self._ensure_cfg_saved():
            return
        if not self._sync_cfg_from_editor():
            return
        if not hasattr(self, "combo_pair_sets_ws"):
            return
        data = self.combo_pair_sets_ws.currentData()
        if not isinstance(data, dict) or "path" not in data:
            self._show_msgbox_lines("Pairs", ["Не выбран набор пар."], icon="warn")
            return
        ps = {
            "path": str(data["path"]),
            "label": str(data.get("label", "")),
            "origin": str(data.get("origin", "")),
        }
        self._do_use_pair_set_impl(ps)

    def _do_use_pair_set_impl(self, ps: dict) -> None:
        """Implementation: copy pair set into work_dir and update config."""
        cfg = self._cfg or {}
        if not self._cfg_path:
            self._show_msgbox_lines(
                "Pairs", ["Сначала сохраните config.yaml (Apply)."], icon="warn"
            )
            return
        try:
            from pathlib import Path

            cfg2 = dict(cfg)
            cfg2.setdefault("config_dir", str(self._cfg_path.parent))
            wd = resolve_work_dir(cfg2)
            disperser = str(cfg.get("setup", {}).get("disperser", "") or "")
            src = Path(str(ps.get("path", ""))).expanduser()
            dst = copy_pair_set_to_workdir(disperser, wd, src)
            rel = _rel_to_workdir(wd, dst)
            self._set_cfg_value("wavesol.hand_pairs_path", rel)
            self.editor_yaml.blockSignals(True)
            self.editor_yaml.setPlainText(_yaml_dump(self._cfg or {}))
            self.editor_yaml.blockSignals(False)
            self._do_save_cfg()
            self._refresh_pairs_label()
            self._log_info(f"Using pair set '{ps.get('label', '')}' → {dst}")
            try:
                self._update_wavesol_stepper()
            except Exception:
                pass
        except Exception as e:
            self._log_exception(e)
            self._show_msgbox_lines(
                "Pairs",
                ["Не удалось скопировать набор пар в work_dir.", str(e)],
                icon="error",
            )

    def _do_copy_pair_set(self) -> None:
        """Copy selected pair set into work_dir under a custom filename (does not change config)."""
        if not self._ensure_cfg_saved():
            return
        if not self._sync_cfg_from_editor():
            return
        ps = self._selected_pair_set_id()
        if ps is None:
            self._show_msgbox_lines("Pairs", ["Не выбран набор пар."], icon="warn")
            return
        if not self._cfg_path:
            self._show_msgbox_lines(
                "Pairs", ["Сначала сохраните config.yaml (Apply)."], icon="warn"
            )
            return
        cfg = self._cfg or {}

        def _safe_name(s: str) -> str:
            import re

            s = re.sub(r"[^0-9A-Za-z._-]+", "_", s.strip())
            s = re.sub(r"_+", "_", s).strip("_")
            return s or "pair_set"

        default = f"pairset_{_safe_name(ps.get('label', 'selected'))}.txt"
        new_name, ok = QtWidgets.QInputDialog.getText(
            self, "Copy pair set", "Filename in workdir:", text=default
        )
        if not ok or not str(new_name).strip():
            return
        new_name = str(new_name).strip()
        try:
            from pathlib import Path

            cfg2 = dict(cfg)
            cfg2.setdefault("config_dir", str(self._cfg_path.parent))
            wd = resolve_work_dir(cfg2)
            disperser = str(cfg.get("setup", {}).get("disperser", "") or "")
            src = Path(ps["path"]).expanduser()
            dst = copy_pair_set_to_workdir(disperser, wd, src, filename=new_name)
            self._log_info(f"Copied selected pair set into work_dir: {dst}")
            self._refresh_pair_sets_combo()
        except Exception as e:
            self._log_exception(e)
            self._show_msgbox_lines(
                "Pairs",
                ["Не удалось скопировать набор в work_dir.", str(e)],
                icon="error",
            )

    def _do_save_workdir_pairs(self) -> None:
        """Save current workdir hand-pairs into the *user library* (reusable)."""
        if not self._sync_cfg_from_editor():
            return
        p = self._current_pairs_path()
        if p is None or not p.exists():
            self._show_msgbox_lines(
                "Pairs", ["Текущий файл пар не найден."], icon="warn"
            )
            return
        try:
            setup = self._setup_key_for_pairs()
            default = (
                f"{setup.get('disperser', '')}_{setup.get('slit', '')}_{setup.get('binning', '')}".strip(
                    "_"
                )
                or "pairs"
            )
            label, ok = QtWidgets.QInputDialog.getText(
                self, "Save to user library", "Label:", text=default
            )
            if not ok or not str(label).strip():
                return
            out = save_user_pair_set(setup, p, label=str(label).strip())
            self._log_info(f"Saved to user library: {out}")
            self._refresh_pair_sets_combo()
        except Exception as e:
            self._log_exception(e)
            self._show_msgbox_lines(
                "Pairs",
                ["Не удалось сохранить пары в user library.", str(e)],
                icon="error",
            )

    def _do_open_pairs_library(self) -> None:
        try:
            root = user_pairs_root()
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(root)))
        except Exception as e:
            self._log_exception(e)
            self._show_msgbox_lines(
                "Pairs", ["Не удалось открыть папку библиотеки.", str(e)], icon="error"
            )

    def _do_export_selected_pair_set(self) -> None:
        ps = self._selected_pair_set_id()
        if ps is None:
            self._show_msgbox_lines("Pairs", ["Не выбран набор пар."], icon="warn")
            return
        from pathlib import Path

        src = Path(ps["path"]).expanduser()
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export pair set",
            f"{ps.get('label', 'pair_set')}.txt",
            "Text files (*.txt);;All files (*)",
        )
        if not fn:
            return
        try:
            export_pair_set(src, Path(fn))
            self._log_info(f"Exported: {fn}")
        except Exception as e:
            self._log_exception(e)
            self._show_msgbox_lines(
                "Pairs", ["Экспорт не удался.", str(e)], icon="error"
            )

    def _do_export_current_pairs(self) -> None:
        if not self._sync_cfg_from_editor():
            return
        p = self._current_pairs_path()
        if p is None or not p.exists():
            self._show_msgbox_lines(
                "Pairs", ["Текущий файл пар не найден."], icon="warn"
            )
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export current pairs",
            "hand_pairs.txt",
            "Text files (*.txt);;All files (*)",
        )
        if not fn:
            return
        try:
            from shutil import copy2

            copy2(str(p), str(fn))
            self._log_info(f"Exported: {fn}")
        except Exception as e:
            self._log_exception(e)
            self._show_msgbox_lines(
                "Pairs", ["Экспорт не удался.", str(e)], icon="error"
            )

    def _do_export_user_library_zip(self) -> None:
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export user pairs library",
            "pairs_library.zip",
            "Zip (*.zip);;All files (*)",
        )
        if not fn:
            return
        try:
            from pathlib import Path

            export_user_library_zip(Path(fn))
            self._log_info(f"Exported user library: {fn}")
        except Exception as e:
            self._log_exception(e)
            self._show_msgbox_lines(
                "Pairs", ["Экспорт не удался.", str(e)], icon="error"
            )

    # --------------------------- page: wavesolution ---------------------------

    def _build_page_wavesol(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setSpacing(12)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setSpacing(12)

        g = _box("Wavelength solution (1D + 2D)")
        left_layout.addWidget(g)
        gl = QtWidgets.QVBoxLayout(g)

        lbl = QtWidgets.QLabel(
            "Build 1D λ(x) from hand pairs and a 2D λ(x,y) map from traced lines."
            "Настройки ниже влияют на точность (RMS) и устойчивость решения."
            "Изменения применяются только после кнопки Apply."
        )
        lbl.setWordWrap(True)
        gl.addWidget(lbl)

        self.lbl_wavesol_dir = QtWidgets.QLabel("wavesol: —")
        self.lbl_wavesol_dir.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        gl.addWidget(self.lbl_wavesol_dir)

        # ---------------- director workflow (director's cut) ----------------
        gflow = _box("Workflow 3.1–3.6 (director's cut)")
        left_layout.addWidget(gflow)
        fl = QtWidgets.QGridLayout(gflow)
        fl.setHorizontalSpacing(10)
        fl.setVerticalSpacing(8)

        note = QtWidgets.QLabel(
            "Режиссёрская сборка: жёсткий маршрут к стабильной λ-карте. "
            "Кнопки блокируются, пока нет нужных артефактов (superneon, пары, λ-map и т.д.). "
            "Подсказка: библиотека line-ID/hand pairs доступна прямо здесь."
        )
        note.setWordWrap(True)
        fl.addWidget(note, 0, 0, 1, 3)

        def _mk_step_frame(
            title: str,
        ) -> tuple[QtWidgets.QFrame, QtWidgets.QVBoxLayout]:
            fr = QtWidgets.QFrame()
            fr.setFrameShape(QtWidgets.QFrame.StyledPanel)
            fr.setFrameShadow(QtWidgets.QFrame.Raised)
            fr.setMinimumWidth(240)
            v = QtWidgets.QVBoxLayout(fr)
            v.setContentsMargins(8, 8, 8, 8)
            v.setSpacing(6)
            t = QtWidgets.QLabel(f"<b>{title}</b>")
            t.setTextFormat(QtCore.Qt.RichText)
            v.addWidget(t)
            return fr, v

        def _mk_reason_label() -> QtWidgets.QLabel:
            r = QtWidgets.QLabel("")
            r.setWordWrap(True)
            r.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            try:
                r.setStyleSheet("color: #a33;")
            except Exception:
                pass
            return r

        # 3.1 SuperNeon
        fr31, v31 = _mk_step_frame("3.1 SuperNeon")
        self.lbl_ws31_status = QtWidgets.QLabel("—")
        self.lbl_ws31_status.setWordWrap(True)
        v31.addWidget(self.lbl_ws31_status)
        self.lbl_ws31_reason = _mk_reason_label()
        v31.addWidget(self.lbl_ws31_reason)
        fl.addWidget(fr31, 1, 0)

        # 3.2 Line ID + pairs library (mini)
        fr32, v32 = _mk_step_frame("3.2 Line ID / hand pairs")
        self.lbl_ws32_status = QtWidgets.QLabel("—")
        self.lbl_ws32_status.setWordWrap(True)
        v32.addWidget(self.lbl_ws32_status)
        self.lbl_pairs_file_ws = QtWidgets.QLabel("hand pairs: —")
        self.lbl_pairs_file_ws.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        v32.addWidget(self.lbl_pairs_file_ws)

        self.btn_ws_open_lineid = QtWidgets.QPushButton("Open LineID GUI")
        self.btn_ws_open_lineid.setCursor(QtCore.Qt.PointingHandCursor)
        v32.addWidget(self.btn_ws_open_lineid)
        self.lbl_ws32_reason = _mk_reason_label()
        v32.addWidget(self.lbl_ws32_reason)

        row32 = QtWidgets.QHBoxLayout()
        self.combo_pair_sets_ws = QtWidgets.QComboBox()
        self.combo_pair_sets_ws.setMinimumWidth(220)
        self.combo_pair_sets_ws.setToolTip(
            "Библиотека наборов пар: встроенные + пользовательские"
        )
        row32.addWidget(self.combo_pair_sets_ws, 1)
        self.btn_use_pair_set_ws = QtWidgets.QPushButton("Use")
        self.btn_use_pair_set_ws.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_use_pair_set_ws.setToolTip(
            "Скопировать выбранный набор пар в work_dir и использовать его"
        )
        row32.addWidget(self.btn_use_pair_set_ws)
        self.btn_open_pairs_library_ws = QtWidgets.QPushButton("Library")
        self.btn_open_pairs_library_ws.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_open_pairs_library_ws.setToolTip(
            "Открыть папку пользовательской библиотеки"
        )
        row32.addWidget(self.btn_open_pairs_library_ws)
        v32.addLayout(row32)
        fl.addWidget(fr32, 1, 1)

        # 3.3 Solve λ-map
        fr33, v33 = _mk_step_frame("3.3 Solve λ(x,y)")
        self.lbl_ws33_status = QtWidgets.QLabel("—")
        self.lbl_ws33_status.setWordWrap(True)
        v33.addWidget(self.lbl_ws33_status)
        self.btn_ws_run_wavesol = QtWidgets.QPushButton("Run: Wavesolution")
        self.btn_ws_run_wavesol.setCursor(QtCore.Qt.PointingHandCursor)
        v33.addWidget(self.btn_ws_run_wavesol)
        self.lbl_ws33_reason = _mk_reason_label()
        v33.addWidget(self.lbl_ws33_reason)
        fl.addWidget(fr33, 1, 2)

        # 3.4 Clean pairs
        fr34, v34 = _mk_step_frame("3.4 Clean pairs")
        self.lbl_ws34_status = QtWidgets.QLabel("—")
        self.lbl_ws34_status.setWordWrap(True)
        v34.addWidget(self.lbl_ws34_status)
        self.btn_ws_clean_pairs = QtWidgets.QPushButton("Open: Pairs cleaner…")
        self.btn_ws_clean_pairs.setCursor(QtCore.Qt.PointingHandCursor)
        v34.addWidget(self.btn_ws_clean_pairs)
        self.lbl_ws34_reason = _mk_reason_label()
        v34.addWidget(self.lbl_ws34_reason)
        fl.addWidget(fr34, 2, 0)

        # 3.5 Clean 2D lines
        fr35, v35 = _mk_step_frame("3.5 Clean 2D lines")
        self.lbl_ws35_status = QtWidgets.QLabel("—")
        self.lbl_ws35_status.setWordWrap(True)
        v35.addWidget(self.lbl_ws35_status)
        self.btn_ws_clean_2d = QtWidgets.QPushButton("Open: 2D line cleaner…")
        self.btn_ws_clean_2d.setCursor(QtCore.Qt.PointingHandCursor)
        v35.addWidget(self.btn_ws_clean_2d)
        self.lbl_ws35_reason = _mk_reason_label()
        v35.addWidget(self.lbl_ws35_reason)
        fl.addWidget(fr35, 2, 1)

        # 3.6 QC / Frames
        fr36, v36 = _mk_step_frame("3.6 QC / Frames")
        self.lbl_ws36_status = QtWidgets.QLabel("—")
        self.lbl_ws36_status.setWordWrap(True)
        v36.addWidget(self.lbl_ws36_status)
        row36 = QtWidgets.QHBoxLayout()
        self.btn_ws_open_qc = QtWidgets.QPushButton("QC")
        self.btn_ws_open_qc.setCursor(QtCore.Qt.PointingHandCursor)
        row36.addWidget(self.btn_ws_open_qc)
        self.btn_ws_open_frames = QtWidgets.QPushButton("Frames")
        self.btn_ws_open_frames.setCursor(QtCore.Qt.PointingHandCursor)
        row36.addWidget(self.btn_ws_open_frames)
        v36.addLayout(row36)
        self.lbl_ws36_reason = _mk_reason_label()
        v36.addWidget(self.lbl_ws36_reason)
        fl.addWidget(fr36, 2, 2)

        # Backward-compatible attribute aliases (older patches may refer to these names)
        self.lbl_ws_step_31 = self.lbl_ws31_status
        self.lbl_ws_step_32 = self.lbl_ws32_status
        self.lbl_ws_step_33 = self.lbl_ws33_status
        self.lbl_ws_step_34 = self.lbl_ws34_status
        self.lbl_ws_step_35 = self.lbl_ws35_status
        self.lbl_ws_step_36 = self.lbl_ws36_status

        # ---------------- parameters ----------------
        gpar = _box("Parameters")
        left_layout.addWidget(gpar)
        pl = QtWidgets.QVBoxLayout(gpar)
        pl.setSpacing(8)

        # -------- BASIC: Core / 2D model / Edge crop --------
        basic = QtWidgets.QWidget()
        bvl = QtWidgets.QVBoxLayout(basic)
        bvl.setContentsMargins(0, 0, 0, 0)
        bvl.setSpacing(10)

        grp_core = _box("Core")
        install_delayed_tooltip(
            grp_core,
            "High-level solution controls.\nMost users only adjust these.",
        )
        core = QtWidgets.QFormLayout(grp_core)
        core.setLabelAlignment(QtCore.Qt.AlignLeft)
        core.setHorizontalSpacing(12)

        self.spin_ws_poly_deg = QtWidgets.QSpinBox()
        self.spin_ws_poly_deg.setRange(1, 12)
        core.addRow(
            self._param_label("1D poly degree", "", cfg_path="wavesol.poly_deg_1d"),
            self._param_field(self.spin_ws_poly_deg, cfg_path="wavesol.poly_deg_1d"),
        )

        self.dspin_ws_blend = QtWidgets.QDoubleSpinBox()
        self.dspin_ws_blend.setRange(0.0, 1.0)
        self.dspin_ws_blend.setSingleStep(0.05)
        self.dspin_ws_blend.setDecimals(2)
        core.addRow(
            self._param_label("Blend weight", "", cfg_path="wavesol.blend_weight"),
            self._param_field(self.dspin_ws_blend, cfg_path="wavesol.blend_weight"),
        )

        self.dspin_ws_poly_clip = QtWidgets.QDoubleSpinBox()
        self.dspin_ws_poly_clip.setRange(0.0, 20.0)
        self.dspin_ws_poly_clip.setSingleStep(0.2)
        self.dspin_ws_poly_clip.setDecimals(2)
        core.addRow(
            self._param_label("1D σ-clip", "", cfg_path="wavesol.poly_sigma_clip"),
            self._param_field(self.dspin_ws_poly_clip, cfg_path="wavesol.poly_sigma_clip"),
        )

        self.spin_ws_poly_iter = QtWidgets.QSpinBox()
        self.spin_ws_poly_iter.setRange(1, 100)
        core.addRow(
            self._param_label("1D maxiter", "", cfg_path="wavesol.poly_maxiter"),
            self._param_field(self.spin_ws_poly_iter, cfg_path="wavesol.poly_maxiter"),
        )

        self.combo_ws_model2d = QtWidgets.QComboBox()
        self.combo_ws_model2d.addItem("auto", "auto")
        self.combo_ws_model2d.addItem("power", "power")
        self.combo_ws_model2d.addItem("cheb", "cheb")
        core.addRow(
            self._param_label("2D model", "", cfg_path="wavesol.model2d"),
            self._param_field(self.combo_ws_model2d, cfg_path="wavesol.model2d"),
        )

        bvl.addWidget(grp_core)

        grp_edge = _box("Edge crop")
        install_delayed_tooltip(
            grp_edge,
            "Exclude slit edges from the 2D fit.\nHelps when edges are noisy/vignetted.",
        )
        edge = QtWidgets.QFormLayout(grp_edge)
        edge.setLabelAlignment(QtCore.Qt.AlignLeft)
        edge.setHorizontalSpacing(12)

        self.spin_ws_crop_x = QtWidgets.QSpinBox()
        self.spin_ws_crop_x.setRange(0, 200)
        edge.addRow(
            self._param_label("Edge crop X", "", cfg_path="wavesol.edge_crop_x"),
            self._param_field(self.spin_ws_crop_x, cfg_path="wavesol.edge_crop_x"),
        )

        self.spin_ws_crop_y = QtWidgets.QSpinBox()
        self.spin_ws_crop_y.setRange(0, 200)
        edge.addRow(
            self._param_label("Edge crop Y", "", cfg_path="wavesol.edge_crop_y"),
            self._param_field(self.spin_ws_crop_y, cfg_path="wavesol.edge_crop_y"),
        )

        bvl.addWidget(grp_edge)

        self.grp_ws_power_model = _box("Power model")
        install_delayed_tooltip(
            self.grp_ws_power_model,
            "Parameters for the Power 2D model.\nEnabled when 2D model = power/auto.",
        )
        pmod = QtWidgets.QFormLayout(self.grp_ws_power_model)
        pmod.setLabelAlignment(QtCore.Qt.AlignLeft)
        pmod.setHorizontalSpacing(12)

        self.spin_ws_power_deg = QtWidgets.QSpinBox()
        self.spin_ws_power_deg.setRange(1, 12)
        pmod.addRow(
            self._param_label("Power degree", "", cfg_path="wavesol.power_deg"),
            self._param_field(self.spin_ws_power_deg, cfg_path="wavesol.power_deg"),
        )

        bvl.addWidget(self.grp_ws_power_model)

        self.grp_ws_cheb_model = _box("Chebyshev model")
        install_delayed_tooltip(
            self.grp_ws_cheb_model,
            "Parameters for the Chebyshev 2D model.\nEnabled when 2D model = cheb/auto.",
        )
        cmod = QtWidgets.QFormLayout(self.grp_ws_cheb_model)
        cmod.setLabelAlignment(QtCore.Qt.AlignLeft)
        cmod.setHorizontalSpacing(12)

        self.spin_ws_cheb_x = QtWidgets.QSpinBox()
        self.spin_ws_cheb_x.setRange(1, 12)
        cmod.addRow(
            self._param_label("Cheb degX", "", cfg_path="wavesol.cheb_degx"),
            self._param_field(self.spin_ws_cheb_x, cfg_path="wavesol.cheb_degx"),
        )
        self.spin_ws_cheb_y = QtWidgets.QSpinBox()
        self.spin_ws_cheb_y.setRange(1, 12)
        cmod.addRow(
            self._param_label("Cheb degY", "", cfg_path="wavesol.cheb_degy"),
            self._param_field(self.spin_ws_cheb_y, cfg_path="wavesol.cheb_degy"),
        )

        bvl.addWidget(self.grp_ws_cheb_model)
        bvl.addStretch(1)

        # -------- ADVANCED: Trace / Robust fit --------
        adv = QtWidgets.QWidget()
        avl = QtWidgets.QVBoxLayout(adv)
        avl.setContentsMargins(0, 0, 0, 0)
        avl.setSpacing(10)

        grp_trace = _box("Trace")
        install_delayed_tooltip(
            grp_trace,
            "How 2D lines are traced across Y before fitting λ(x,y).\nUsually safe to keep defaults.",
        )
        af = QtWidgets.QFormLayout(grp_trace)
        af.setLabelAlignment(QtCore.Qt.AlignLeft)
        af.setHorizontalSpacing(12)

        self.spin_ws_trace_template_hw = QtWidgets.QSpinBox()
        self.spin_ws_trace_template_hw.setRange(1, 64)
        af.addRow(
            self._param_label(
                "Trace template HW",
                "",
                cfg_path="wavesol.trace_template_hw",
            ),
            self._param_field(self.spin_ws_trace_template_hw, cfg_path="wavesol.trace_template_hw"),
        )

        self.spin_ws_trace_avg_half = QtWidgets.QSpinBox()
        self.spin_ws_trace_avg_half.setRange(0, 64)
        af.addRow(
            self._param_label(
                "Trace avg HW",
                "",
                cfg_path="wavesol.trace_avg_half",
            ),
            self._param_field(self.spin_ws_trace_avg_half, cfg_path="wavesol.trace_avg_half"),
        )

        self.spin_ws_trace_search_rad = QtWidgets.QSpinBox()
        self.spin_ws_trace_search_rad.setRange(1, 128)
        af.addRow(
            self._param_label(
                "Trace search",
                "",
                cfg_path="wavesol.trace_search_rad",
            ),
            self._param_field(self.spin_ws_trace_search_rad, cfg_path="wavesol.trace_search_rad"),
        )

        self.spin_ws_trace_y_step = QtWidgets.QSpinBox()
        self.spin_ws_trace_y_step.setRange(1, 20)
        af.addRow(
            self._param_label(
                "Trace Y step",
                "",
                cfg_path="wavesol.trace_y_step",
            ),
            self._param_field(self.spin_ws_trace_y_step, cfg_path="wavesol.trace_y_step"),
        )

        self.dspin_ws_trace_amp_thresh = QtWidgets.QDoubleSpinBox()
        self.dspin_ws_trace_amp_thresh.setRange(0.0, 1e9)
        self.dspin_ws_trace_amp_thresh.setSingleStep(10.0)
        self.dspin_ws_trace_amp_thresh.setDecimals(1)
        af.addRow(
            self._param_label(
                "Trace amp thresh",
                "",
                cfg_path="wavesol.trace_amp_thresh",
            ),
            self._param_field(
                self.dspin_ws_trace_amp_thresh, cfg_path="wavesol.trace_amp_thresh"
            ),
        )

        self.spin_ws_trace_min_pts = QtWidgets.QSpinBox()
        self.spin_ws_trace_min_pts.setRange(10, 10000)
        af.addRow(
            self._param_label(
                "Trace min points",
                "",
                cfg_path="wavesol.trace_min_pts",
            ),
            self._param_field(self.spin_ws_trace_min_pts, cfg_path="wavesol.trace_min_pts"),
        )

        self.spin_ws_trace_y0 = QtWidgets.QSpinBox()
        self.spin_ws_trace_y0.setRange(-1, 100000)
        self.spin_ws_trace_y0.setSpecialValueText("auto")
        af.addRow(
            self._param_label(
                "Trace y0",
                "",
                cfg_path="wavesol.trace_y0",
            ),
            self._param_field(self.spin_ws_trace_y0, cfg_path="wavesol.trace_y0"),
        )

        avl.addWidget(grp_trace)

        self.grp_ws_power_fit = _box("Power fit")
        install_delayed_tooltip(
            self.grp_ws_power_fit,
            "Robust fitting controls for the Power 2D model.\nEnabled when 2D model = power/auto.",
        )
        pf = QtWidgets.QFormLayout(self.grp_ws_power_fit)
        pf.setLabelAlignment(QtCore.Qt.AlignLeft)
        pf.setHorizontalSpacing(12)

        self.dspin_ws_power_clip = QtWidgets.QDoubleSpinBox()
        self.dspin_ws_power_clip.setRange(0.0, 20.0)
        self.dspin_ws_power_clip.setSingleStep(0.2)
        self.dspin_ws_power_clip.setDecimals(2)
        pf.addRow(
            self._param_label("2D σ-clip (power)", "", cfg_path="wavesol.power_sigma_clip"),
            self._param_field(self.dspin_ws_power_clip, cfg_path="wavesol.power_sigma_clip"),
        )

        self.spin_ws_power_iter = QtWidgets.QSpinBox()
        self.spin_ws_power_iter.setRange(1, 100)
        pf.addRow(
            self._param_label("2D maxiter (power)", "", cfg_path="wavesol.power_maxiter"),
            self._param_field(self.spin_ws_power_iter, cfg_path="wavesol.power_maxiter"),
        )

        avl.addWidget(self.grp_ws_power_fit)

        self.grp_ws_cheb_fit = _box("Chebyshev fit")
        install_delayed_tooltip(
            self.grp_ws_cheb_fit,
            "Robust fitting controls for the Chebyshev 2D model.\nEnabled when 2D model = cheb/auto.",
        )
        cf = QtWidgets.QFormLayout(self.grp_ws_cheb_fit)
        cf.setLabelAlignment(QtCore.Qt.AlignLeft)
        cf.setHorizontalSpacing(12)

        self.dspin_ws_cheb_clip = QtWidgets.QDoubleSpinBox()
        self.dspin_ws_cheb_clip.setRange(0.0, 20.0)
        self.dspin_ws_cheb_clip.setSingleStep(0.2)
        self.dspin_ws_cheb_clip.setDecimals(2)
        cf.addRow(
            self._param_label("2D σ-clip (cheb)", "", cfg_path="wavesol.cheb_sigma_clip"),
            self._param_field(self.dspin_ws_cheb_clip, cfg_path="wavesol.cheb_sigma_clip"),
        )

        self.spin_ws_cheb_iter = QtWidgets.QSpinBox()
        self.spin_ws_cheb_iter.setRange(1, 100)
        cf.addRow(
            self._param_label("2D maxiter (cheb)", "", cfg_path="wavesol.cheb_maxiter"),
            self._param_field(self.spin_ws_cheb_iter, cfg_path="wavesol.cheb_maxiter"),
        )

        avl.addWidget(self.grp_ws_cheb_fit)
        avl.addStretch(1)

        # locale for doubles
        self._force_dot_locale(
            self.dspin_ws_blend,
            self.dspin_ws_poly_clip,
            self.dspin_ws_trace_amp_thresh,
            self.dspin_ws_power_clip,
            self.dspin_ws_cheb_clip,
        )

        pl.addWidget(self._mk_basic_advanced_tabs(basic, adv))
        pl.addWidget(self._mk_stage_apply_row("wavesol"))

        # wiring (pending)
        self.spin_ws_poly_deg.valueChanged.connect(
            lambda v: self._stage_set_pending("wavesol", "wavesol.poly_deg_1d", int(v))
        )
        self.dspin_ws_blend.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "wavesol", "wavesol.blend_weight", float(v)
            )
        )
        self.dspin_ws_poly_clip.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "wavesol", "wavesol.poly_sigma_clip", float(v)
            )
        )
        self.spin_ws_poly_iter.valueChanged.connect(
            lambda v: self._stage_set_pending("wavesol", "wavesol.poly_maxiter", int(v))
        )

        def _model_changed(*_):
            v = self.combo_ws_model2d.currentData()
            if v is None:
                v = self.combo_ws_model2d.currentText()
            self._stage_set_pending("wavesol", "wavesol.model2d", str(v))
            self._update_wavesol_model2d_enables()

        self.combo_ws_model2d.currentIndexChanged.connect(_model_changed)

        self.spin_ws_crop_x.valueChanged.connect(
            lambda v: self._stage_set_pending("wavesol", "wavesol.edge_crop_x", int(v))
        )
        self.spin_ws_crop_y.valueChanged.connect(
            lambda v: self._stage_set_pending("wavesol", "wavesol.edge_crop_y", int(v))
        )

        self.spin_ws_power_deg.valueChanged.connect(
            lambda v: self._stage_set_pending("wavesol", "wavesol.power_deg", int(v))
        )
        self.spin_ws_cheb_x.valueChanged.connect(
            lambda v: self._stage_set_pending("wavesol", "wavesol.cheb_degx", int(v))
        )
        self.spin_ws_cheb_y.valueChanged.connect(
            lambda v: self._stage_set_pending("wavesol", "wavesol.cheb_degy", int(v))
        )

        self.spin_ws_trace_template_hw.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "wavesol", "wavesol.trace_template_hw", int(v)
            )
        )
        self.spin_ws_trace_avg_half.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "wavesol", "wavesol.trace_avg_half", int(v)
            )
        )
        self.spin_ws_trace_search_rad.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "wavesol", "wavesol.trace_search_rad", int(v)
            )
        )
        self.spin_ws_trace_y_step.valueChanged.connect(
            lambda v: self._stage_set_pending("wavesol", "wavesol.trace_y_step", int(v))
        )
        self.dspin_ws_trace_amp_thresh.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "wavesol", "wavesol.trace_amp_thresh", float(v)
            )
        )
        self.spin_ws_trace_min_pts.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "wavesol", "wavesol.trace_min_pts", int(v)
            )
        )
        self.spin_ws_trace_y0.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "wavesol", "wavesol.trace_y0", None if int(v) == -1 else int(v)
            )
        )

        self.dspin_ws_power_clip.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "wavesol", "wavesol.power_sigma_clip", float(v)
            )
        )
        self.spin_ws_power_iter.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "wavesol", "wavesol.power_maxiter", int(v)
            )
        )
        self.dspin_ws_cheb_clip.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "wavesol", "wavesol.cheb_sigma_clip", float(v)
            )
        )
        self.spin_ws_cheb_iter.valueChanged.connect(
            lambda v: self._stage_set_pending("wavesol", "wavesol.cheb_maxiter", int(v))
        )

        # --------------- actions row ---------------
        row = QtWidgets.QHBoxLayout()
        self.btn_clean_pairs = QtWidgets.QPushButton("Clean pairs…")
        self.btn_clean_wavesol2d = QtWidgets.QPushButton("Clean 2D lines…")
        self.btn_run_wavesol = QtWidgets.QPushButton("Run: Wavelength solution")
        self.btn_run_wavesol.setProperty("primary", True)
        self.btn_qc_wavesol = QtWidgets.QPushButton("QC")
        self.btn_frames_wavesol = QtWidgets.QPushButton("Frames…")
        self.btn_frames_wavesol.setToolTip(
            "Open Frames Browser for the Wavelength solution stage"
        )
        row.addWidget(self.btn_clean_pairs)
        row.addWidget(self.btn_clean_wavesol2d)
        row.addWidget(self.btn_run_wavesol)
        row.addWidget(self.btn_qc_wavesol)
        row.addWidget(self.btn_frames_wavesol)
        row.addStretch(1)
        gl.addLayout(row)


        left_layout.addStretch(1)

        # Outputs are shown in a detached tool window (toolbar: Outputs).
        lay.addWidget(left, 1)

        self.btn_clean_pairs.clicked.connect(self._do_clean_pairs)
        self.btn_clean_wavesol2d.clicked.connect(self._do_clean_wavesol2d)
        self.btn_run_wavesol.clicked.connect(self._do_wavesolution)
        self.btn_qc_wavesol.clicked.connect(self._open_qc_viewer)
        self.btn_frames_wavesol.clicked.connect(
            lambda: self._open_frames_window("wavesol")
        )

        # stepper wiring (director's cut)
        try:
            self.btn_ws_open_lineid.clicked.connect(lambda: self.steps.setCurrentRow(self._stage_row_index("arclineid")))
            self.btn_ws_run_wavesol.clicked.connect(self._do_wavesolution)
            self.btn_ws_clean_pairs.clicked.connect(self._do_clean_pairs)
            self.btn_ws_clean_2d.clicked.connect(self._do_clean_wavesol2d)
            self.btn_ws_open_qc.clicked.connect(self._open_qc_viewer)
            self.btn_ws_open_frames.clicked.connect(
                lambda: self._open_frames_window("wavesol")
            )
            self.btn_use_pair_set_ws.clicked.connect(self._do_use_pair_set_ws)
            self.btn_open_pairs_library_ws.clicked.connect(self._do_open_pairs_library)
        except Exception:
            pass

        # initial sync
        self._sync_stage_controls_from_cfg()
        self._update_wavesol_model2d_enables()
        return w

    def _update_wavesol_model2d_enables(self) -> None:
        """Enable/disable model-specific Wavesol sections without changing layout."""
        if not hasattr(self, "combo_ws_model2d"):
            return

        try:
            v = self.combo_ws_model2d.currentData()
            if v is None:
                v = self.combo_ws_model2d.currentText()
            m = str(v).strip().lower()
        except Exception:
            m = "auto"

        power_on = m in {"power", "auto"}
        cheb_on = m in {"cheb", "auto"}

        for attr in ("grp_ws_power_model", "grp_ws_power_fit"):
            try:
                if hasattr(self, attr):
                    getattr(self, attr).setEnabled(power_on)
            except Exception:
                pass

        for attr in ("grp_ws_cheb_model", "grp_ws_cheb_fit"):
            try:
                if hasattr(self, attr):
                    getattr(self, attr).setEnabled(cheb_on)
            except Exception:
                pass

    def _update_wavesol_stepper(self) -> None:
        """Update director's-cut workflow widgets on the Wavelength solution page."""
        if not hasattr(self, "lbl_ws_step_31"):
            return

        cfg = dict(self._cfg or {})
        if self._cfg_path:
            cfg.setdefault("config_dir", str(self._cfg_path.parent))

        # Resolve wavesol dir (best-effort)
        wsdir = None
        try:
            wsdir = wavesol_dir(cfg)
        except Exception:
            try:
                wd = resolve_work_dir(cfg)
                wsdir = wd / "wavesol"
            except Exception:
                wsdir = None

        def _set(lbl, kind: str, msg: str) -> None:
            if lbl is None:
                return
            icon = {
                "ok": "✅",
                "warn": "⚠️",
                "lock": "🔒",
                "partial": "🟡",
                "run": "⏳",
            }.get(kind, "•")
            try:
                lbl.setText(f"{icon} {msg}")
            except Exception:
                pass

        # directory label
        try:
            if wsdir is not None and hasattr(self, "lbl_wavesol_dir"):
                self.lbl_wavesol_dir.setText(f"wavesol: {wsdir}")
        except Exception:
            pass

        # Files
        def _p(name: str):
            return (wsdir / name) if wsdir is not None else None

        superneon = _p("superneon.fits")
        peaks = _p("peaks_candidates.csv")
        pairs = self._current_pairs_path()
        w1d = _p("wavesolution_1d.json")
        w2d = _p("wavesolution_2d.json")
        lmap = _p("lambda_map.fits")
        cpts = _p("control_points_2d.csv")
        resid = _p("residuals_2d.png")

        def _set_reason(lbl, missing: list[str]) -> None:
            if lbl is None:
                return
            try:
                if missing:
                    lbl.setText("\n".join(["Причина:"] + [f"• {m}" for m in missing]))
                else:
                    lbl.setText("")
            except Exception:
                pass

        # Step 3.1
        sn_ready = bool(superneon and superneon.exists())
        if sn_ready:
            _set(self.lbl_ws_step_31, "ok", "SuperNeon готов")
            _set_reason(getattr(self, "lbl_ws31_reason", None), [])
        else:
            _set(self.lbl_ws_step_31, "lock", "Нужен SuperNeon")
            _set_reason(
                getattr(self, "lbl_ws31_reason", None),
                [
                    f"не найден {superneon.name}"
                    if superneon is not None
                    else "не найден superneon.fits"
                ],
            )

        # Step 3.2
        pairs_ready = bool(pairs and pairs.exists())
        if pairs_ready:
            _set(self.lbl_ws_step_32, "ok", "Пары линий готовы")
            _set_reason(getattr(self, "lbl_ws32_reason", None), [])
        else:
            _set(self.lbl_ws_step_32, "lock", "Нужен LineID (hand_pairs)")
            _set_reason(
                getattr(self, "lbl_ws32_reason", None),
                ["не задан или не найден hand_pairs (CSV)"],
            )
        try:
            if hasattr(self, "lbl_pairs_file_ws"):
                self.lbl_pairs_file_ws.setText(
                    "hand pairs: —" if not pairs else f"hand pairs: {pairs}"
                )
        except Exception:
            pass
        try:
            self.btn_ws_clean_pairs.setEnabled(pairs_ready)
        except Exception:
            pass

        # Step 3.3
        has_1d = bool(w1d and w1d.exists())
        has_2d = bool(w2d and w2d.exists())
        has_map = bool(lmap and lmap.exists())
        if has_map and has_2d:
            _set(self.lbl_ws_step_33, "ok", "λ-map построен")
        elif has_1d:
            _set(self.lbl_ws_step_33, "partial", "Есть 1D, нужен 2D λ-map")
        else:
            _set(self.lbl_ws_step_33, "warn", "Решение ещё не построено")

        # Enable/disable main 'Run' button + show concrete missing artefacts
        try:
            miss = []
            if not sn_ready:
                miss.append("нет superneon.fits")
            if not pairs_ready:
                miss.append("нет hand_pairs (CSV)")
            self.btn_ws_run_wavesol.setEnabled(sn_ready and pairs_ready)
            _set_reason(
                getattr(self, "lbl_ws33_reason", None),
                miss if not (sn_ready and pairs_ready) else [],
            )
        except Exception:
            pass

        # Step 3.4
        if pairs_ready:
            _set(self.lbl_ws_step_34, "ok", "Можно чистить пары")
            _set_reason(getattr(self, "lbl_ws34_reason", None), [])
        else:
            _set(self.lbl_ws_step_34, "lock", "Нечего чистить без hand_pairs")
            _set_reason(
                getattr(self, "lbl_ws34_reason", None), ["нет hand_pairs (CSV)"]
            )

        # Step 3.5
        cpts_ready = bool(cpts and cpts.exists())
        if cpts_ready:
            _set(self.lbl_ws_step_35, "ok", "2D контрольные точки готовы")
            _set_reason(getattr(self, "lbl_ws35_reason", None), [])
        else:
            _set(self.lbl_ws_step_35, "lock", "Нужны control_points_2d.csv")
            _set_reason(
                getattr(self, "lbl_ws35_reason", None),
                ["нет control_points_2d.csv (сначала Run: Wavesolution)"],
            )
        try:
            self.btn_ws_clean_2d.setEnabled(cpts_ready)
        except Exception:
            pass

        # Step 3.6
        qc_ready = bool(resid and resid.exists())
        if qc_ready:
            _set(self.lbl_ws_step_36, "ok", "QC графики готовы")
            _set_reason(getattr(self, "lbl_ws36_reason", None), [])
        elif has_map:
            _set(self.lbl_ws_step_36, "warn", "λ-map есть, QC ещё не найден")
            _set_reason(
                getattr(self, "lbl_ws36_reason", None),
                ["нет residuals_2d.png (перезапустите Wavesolution/QC)"],
            )
        else:
            _set(self.lbl_ws_step_36, "lock", "Сначала постройте λ-map")
            _set_reason(getattr(self, "lbl_ws36_reason", None), ["нет lambda_map.fits"])

        # Minor nicety: show small hint if peaks file exists
        try:
            if (
                hasattr(self, "lbl_ws_step_31")
                and peaks
                and peaks.exists()
                and not sn_ready
            ):
                # ignore
                pass
        except Exception:
            pass

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
            self._update_wavesol_stepper()
        except Exception:
            pass


        # Refresh 'default' icons: sync uses blocked signals for stability.
        for _upd in getattr(self, "_param_default_buttons", []):
            try:
                _upd()
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
            from scorpio_pipe.ui.wavesol_2d_cleaner import (
                Wave2DCleanConfig,
                Wave2DLineCleanerDialog,
            )
        except Exception as e:
            self._log_exception(e)
            return

        cfg = self._cfg or {}
        outdir = Path(wavesol_dir(cfg))
        cp_csv = outdir / "control_points_2d.csv"
        if not cp_csv.exists():
            self._log_error(
                "control_points_2d.csv not found (run Wavelength solution once first)"
            )
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
        wcfg = cfg.get("wavesol", {}) if isinstance(cfg.get("wavesol"), dict) else {}
        dlg_cfg = Wave2DCleanConfig(
            model2d=str(wcfg.get("model2d", "auto")),
            power_deg=int(
                wcfg.get(
                    "power_deg",
                    max(int(wcfg.get("cheb_degx", 5)), int(wcfg.get("cheb_degy", 3))),
                )
            ),
            cheb_degx=int(wcfg.get("cheb_degx", 5)),
            cheb_degy=int(wcfg.get("cheb_degy", 3)),
            power_sigma_clip=float(
                wcfg.get("power_sigma_clip", wcfg.get("cheb_sigma_clip", 3.0))
            ),
            power_maxiter=int(wcfg.get("power_maxiter", wcfg.get("cheb_maxiter", 10))),
            cheb_sigma_clip=float(wcfg.get("cheb_sigma_clip", 3.0)),
            cheb_maxiter=int(wcfg.get("cheb_maxiter", 10)),
        )

        dlg = Wave2DLineCleanerDialog(
            cp_csv, cfg=dlg_cfg, rejected_lines_A=rejected, parent=self
        )
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return

        new_rej = dlg.rejected_lines()
        self._set_cfg_value("wavesol.rejected_lines_A", new_rej)
        self.editor_yaml.blockSignals(True)
        self.editor_yaml.setPlainText(_yaml_dump(self._cfg or {}))
        self.editor_yaml.blockSignals(False)
        self._do_save_cfg()
        self._log_info(f"Updated wavesol.rejected_lines_A (N={len(new_rej)})")

        # save a couple of diagnostic plots for reports/audit
        try:
            saved = dlg.save_plots(outdir, stem="wavesol2d_clean")
            if saved:
                self._log_info(
                    "Saved 2D-clean plots:\n" + "\n".join(f"  {s}" for s in saved)
                )
        except Exception as e:
            self._log_exception(e)

        try:
            self._update_wavesol_stepper()
        except Exception:
            pass


        # Refresh 'default' icons: sync uses blocked signals for stability.
        for _upd in getattr(self, "_param_default_buttons", []):
            try:
                _upd()
            except Exception:
                pass

    def _do_wavesolution(self) -> None:
        if not self._ensure_stage_applied("wavesol", "Wavelength solution"):
            return
        if not self._ensure_cfg_saved():
            return
        self._set_step_status(self._stage_row_index("wavesol"), "running")
        try:
            ctx = load_context(self._cfg_path)
            out = run_wavesolution(ctx)
            self._log_info("Wavelength solution done")
            self._set_step_status(self._stage_row_index("wavesol"), "ok")
            self._log_info(
                "Outputs:\n" + "\n".join(f"  {k}: {v}" for k, v in out.items())
            )
            self._maybe_auto_qc()
        except Exception as e:
            self._set_step_status(self._stage_row_index("wavesol"), "fail")
            self._log_exception(e)

        finally:
            try:
                self._update_wavesol_stepper()
            except Exception:
                pass

    # --------------------------- linearize / sky / extract1d (v5) ---------------------------

    def _build_page_linearize(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setSpacing(12)

        # left: controls
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setSpacing(12)

        g = _box("Linearize (2D dispersion solution)")
        left_layout.addWidget(g)
        gl = QtWidgets.QVBoxLayout(g)

        lbl = QtWidgets.QLabel(
            "Resample SKY-subtracted RAW frames (09_sky/*_skysub_raw.fits) onto a linear wavelength grid using lambda_map(y,x).\n"
            "Output: 10_linearize/lin_preview.fits (+ per_exp/ for individual frames).\n"
            "Option: post-rectification sky residual cleanup can be toggled on the Sky page.\n"
            "Tip: keep dlambda/lambda_min/lambda_max = 0 for auto."
        )
        lbl.setWordWrap(True)
        gl.addWidget(lbl)

        gpar = _box("Parameters")
        pl = QtWidgets.QVBoxLayout(gpar)
        pl.setSpacing(10)

        # -------- BASIC: Core / Geometry / Diagnostics --------
        basic = QtWidgets.QWidget()
        bvl = QtWidgets.QVBoxLayout(basic)
        bvl.setContentsMargins(0, 0, 0, 0)
        bvl.setSpacing(10)

        g_core = _box("Core")
        install_delayed_tooltip(
            g_core,
            "Resampling grid and stage enable/disable.\nTune only if you need a fixed wavelength grid.",
        )
        core = QtWidgets.QFormLayout(g_core)
        core.setLabelAlignment(QtCore.Qt.AlignLeft)
        core.setHorizontalSpacing(12)

        self.chk_lin_enabled = QtWidgets.QCheckBox("")
        core.addRow(
            self._param_label("Enable", "", cfg_path="linearize.enabled"),
            self._param_field(self.chk_lin_enabled, cfg_path="linearize.enabled"),
        )

        self.dspin_lin_dlambda = QtWidgets.QDoubleSpinBox()
        self.dspin_lin_dlambda.setRange(0.0, 50.0)
        self.dspin_lin_dlambda.setDecimals(4)
        self.dspin_lin_dlambda.setSingleStep(0.05)
        self.dspin_lin_dlambda.setSpecialValueText("auto")
        core.addRow(
            self._param_label("Δλ", "", cfg_path="linearize.dlambda_A"),
            self._param_field(self.dspin_lin_dlambda, cfg_path="linearize.dlambda_A"),
        )

        self.dspin_lin_lmin = QtWidgets.QDoubleSpinBox()
        self.dspin_lin_lmin.setRange(0.0, 1_000_000.0)
        self.dspin_lin_lmin.setDecimals(2)
        self.dspin_lin_lmin.setSingleStep(50.0)
        self.dspin_lin_lmin.setSpecialValueText("auto")
        core.addRow(
            self._param_label("λ min", "", cfg_path="linearize.lambda_min_A"),
            self._param_field(self.dspin_lin_lmin, cfg_path="linearize.lambda_min_A"),
        )

        self.dspin_lin_lmax = QtWidgets.QDoubleSpinBox()
        self.dspin_lin_lmax.setRange(0.0, 1_000_000.0)
        self.dspin_lin_lmax.setDecimals(2)
        self.dspin_lin_lmax.setSingleStep(50.0)
        self.dspin_lin_lmax.setSpecialValueText("auto")
        core.addRow(
            self._param_label("λ max", "", cfg_path="linearize.lambda_max_A"),
            self._param_field(self.dspin_lin_lmax, cfg_path="linearize.lambda_max_A"),
        )

        bvl.addWidget(g_core)

        g_geom = _box("Geometry")
        install_delayed_tooltip(
            g_geom,
            "Spatial (Y) cropping of the rectified frame.\nUse to exclude vignetted/noisy slit edges.",
        )
        geom = QtWidgets.QFormLayout(g_geom)
        geom.setLabelAlignment(QtCore.Qt.AlignLeft)
        geom.setHorizontalSpacing(12)

        self.spin_lin_crop_top = QtWidgets.QSpinBox()
        self.spin_lin_crop_top.setRange(0, 10000)
        self.spin_lin_crop_top.setSingleStep(10)
        geom.addRow(
            self._param_label("Crop top", "", cfg_path="linearize.y_crop_top"),
            self._param_field(self.spin_lin_crop_top, cfg_path="linearize.y_crop_top"),
        )

        self.spin_lin_crop_bot = QtWidgets.QSpinBox()
        self.spin_lin_crop_bot.setRange(0, 10000)
        self.spin_lin_crop_bot.setSingleStep(10)
        geom.addRow(
            self._param_label("Crop bottom", "", cfg_path="linearize.y_crop_bottom"),
            self._param_field(self.spin_lin_crop_bot, cfg_path="linearize.y_crop_bottom"),
        )

        bvl.addWidget(g_geom)

        g_diag = _box("Diagnostics")
        install_delayed_tooltip(
            g_diag,
            "Quicklook/QA outputs.\nThese do not affect the science result.",
        )
        diag = QtWidgets.QFormLayout(g_diag)
        diag.setLabelAlignment(QtCore.Qt.AlignLeft)
        diag.setHorizontalSpacing(12)

        self.chk_lin_png = QtWidgets.QCheckBox("")
        diag.addRow(
            self._param_label("Save PNG", "", cfg_path="linearize.save_png"),
            self._param_field(self.chk_lin_png, cfg_path="linearize.save_png"),
        )

        bvl.addWidget(g_diag)
        bvl.addStretch(1)

        # -------- ADVANCED: Outputs --------
        adv = QtWidgets.QWidget()
        avl = QtWidgets.QVBoxLayout(adv)
        avl.setContentsMargins(0, 0, 0, 0)
        avl.setSpacing(10)

        g_out = _box("Outputs")
        install_delayed_tooltip(
            g_out,
            "Debug outputs for each input exposure.\nUseful for debugging, but increases disk and time.",
        )
        out = QtWidgets.QFormLayout(g_out)
        out.setLabelAlignment(QtCore.Qt.AlignLeft)
        out.setHorizontalSpacing(12)

        self.chk_lin_per_frame = QtWidgets.QCheckBox("")
        out.addRow(
            self._param_label("Save per frame", "", cfg_path="linearize.save_per_frame"),
            self._param_field(self.chk_lin_per_frame, cfg_path="linearize.save_per_frame"),
        )

        avl.addWidget(g_out)
        avl.addStretch(1)

        # locale for doubles
        self._force_dot_locale(
            self.dspin_lin_dlambda, self.dspin_lin_lmin, self.dspin_lin_lmax
        )

        pl.addWidget(self._mk_basic_advanced_tabs(basic, adv))
        pl.addWidget(self._mk_stage_apply_row("linearize"))
        gl.addWidget(gpar)

        row = QtWidgets.QHBoxLayout()
        self.btn_run_linearize = QtWidgets.QPushButton("Run: Linearize")
        self.btn_run_linearize.setProperty("primary", True)
        self.btn_qc_linearize = QtWidgets.QPushButton("QC")
        self.btn_frames_linearize = QtWidgets.QPushButton("Frames…")
        row.addWidget(self.btn_run_linearize)
        row.addWidget(self.btn_qc_linearize)
        row.addWidget(self.btn_frames_linearize)
        row.addStretch(1)
        gl.addLayout(row)

        left_layout.addStretch(1)
        # Outputs are shown in a detached tool window.
        lay.addWidget(left, 1)


        # pending wiring
        self.chk_lin_enabled.stateChanged.connect(
            lambda _: self._stage_set_pending(
                "linearize", "linearize.enabled", bool(self.chk_lin_enabled.isChecked())
            )
        )
        self.dspin_lin_dlambda.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "linearize", "linearize.dlambda_A", None if float(v) <= 0 else float(v)
            )
        )
        self.dspin_lin_lmin.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "linearize",
                "linearize.lambda_min_A",
                None if float(v) <= 0 else float(v),
            )
        )
        self.dspin_lin_lmax.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "linearize",
                "linearize.lambda_max_A",
                None if float(v) <= 0 else float(v),
            )
        )
        self.spin_lin_crop_top.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "linearize", "linearize.y_crop_top", int(v)
            )
        )
        self.spin_lin_crop_bot.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "linearize", "linearize.y_crop_bottom", int(v)
            )
        )
        self.chk_lin_png.stateChanged.connect(
            lambda _: self._stage_set_pending(
                "linearize", "linearize.save_png", bool(self.chk_lin_png.isChecked())
            )
        )
        self.chk_lin_per_frame.stateChanged.connect(
            lambda _: self._stage_set_pending(
                "linearize",
                "linearize.save_per_frame",
                bool(self.chk_lin_per_frame.isChecked()),
            )
        )

        self.btn_run_linearize.clicked.connect(self._do_run_linearize)
        self.btn_qc_linearize.clicked.connect(self._open_qc_viewer)
        self.btn_frames_linearize.clicked.connect(
            lambda: self._open_frames_window("lin")
        )

        self._sync_stage_controls_from_cfg()
        return w

    def _do_run_linearize(self) -> None:
        if not self._ensure_stage_applied("linearize", "Linearize"):
            return
        if not self._ensure_cfg_saved():
            return
        self._set_step_status(self._stage_row_index("linearize"), "running")
        try:
            ctx = load_context(self._cfg_path)
            self._run_sequence_with_qc_prompt(ctx, ["linearize"])
            self._log_info("Linearize done")
            self._set_step_status(self._stage_row_index("linearize"), "ok")
            self._maybe_auto_qc()
        except FileNotFoundError as e:
            self._set_step_status(self._stage_row_index("linearize"), "fail")
            msg = str(e)
            # Friendly UX: common mistake is running Linearize without Sky products.
            if (
                "Sky outputs not found" in msg
                or "Missing sky product" in msg
                or "expected 09_sky" in msg
                or "/09_sky" in msg
            ):
                self._show_linearize_missing_sky_dialog(msg)
                try:
                    self._log_error("Linearize prerequisites missing: " + msg)
                except Exception:
                    pass
                return
            self._log_exception(e)
        except Exception as e:
            self._set_step_status(self._stage_row_index("linearize"), "fail")
            self._log_exception(e)

    def _build_page_sky(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setSpacing(12)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setSpacing(12)

        g = _box("Sky Subtraction")
        left_layout.addWidget(g)
        gl = QtWidgets.QVBoxLayout(g)

        lbl = QtWidgets.QLabel(
            "Primary sky subtraction in detector geometry (RAW), guided by λ(x,y) from Wavelength solution.\n"
            "Workflow: select OBJ/SKY regions on a cosmics-clean preview → run Step 09 Sky.\n"
            "Linearization (Step 10) resamples <stem>_skysub_raw.fits → <stem>_skysub.fits and can optionally clean residual sky."
        )
        lbl.setWordWrap(True)
        gl.addWidget(lbl)

        # ROI
        groi = _box("Regions (OBJ / SKY)")
        rlay = QtWidgets.QVBoxLayout(groi)
        self.lbl_sky_roi = QtWidgets.QLabel("ROI: <not set>")
        self.lbl_sky_roi.setWordWrap(True)
        rlay.addWidget(self.lbl_sky_roi)
        row_roi = QtWidgets.QHBoxLayout()
        self.btn_sky_select_roi = QtWidgets.QPushButton("Select regions…")
        self.btn_sky_select_roi.setProperty("primary", True)
        row_roi.addWidget(self.btn_sky_select_roi)
        row_roi.addStretch(1)
        rlay.addLayout(row_roi)
        gl.addWidget(groi)

        gpar = _box("Parameters")
        pl = QtWidgets.QVBoxLayout(gpar)
        pl.setSpacing(10)

        basic = QtWidgets.QWidget()
        bf = QtWidgets.QFormLayout(basic)
        bf.setLabelAlignment(QtCore.Qt.AlignLeft)
        bf.setHorizontalSpacing(12)

        self.chk_sky_enabled = QtWidgets.QCheckBox("Enabled")
        bf.addRow(
            self._param_label(
                "Enabled",
                "Включить этап вычитания ночного неба.\nТипично: включено.",
            ),
            self._param_field(self.chk_sky_enabled),
        )


        # Primary method (observer-friendly): how do we subtract sky?
        self.combo_sky_primary_method = QtWidgets.QComboBox()
        self.combo_sky_primary_method.addItem("Kelson RAW", "kelson_raw")
        self.combo_sky_primary_method.addItem("Sky-Scale RAW", "sky_scale_raw")
        bf.addRow(
            self._param_label(
                "Primary method",
                "Основной метод вычитания неба (в детекторной геометрии).\n"
                "Kelson RAW — универсальный режим по одному объектному кадру.\n"
                "Sky-Scale RAW — масштабирование по отдельным sky-кадрам (если они есть).",
            ),
            self._param_field(self.combo_sky_primary_method),
        )

        # Post-rectification residual cleanup (executed in Linearization)
        self.combo_sky_post_cleanup = QtWidgets.QComboBox()
        self.combo_sky_post_cleanup.addItem("Off", "off")
        self.combo_sky_post_cleanup.addItem("Auto", "auto")
        self.combo_sky_post_cleanup.addItem("On", "on")
        bf.addRow(
            self._param_label(
                "Post-cleanup",
                "Дочистка остатка неба после переноса в линейную λ-сетку.\n"
                "Выполняется на Step 10 (Linearization) — уже на rectified кадре.\n"
                "Off/Auto/On. Типично: Auto.",
            ),
            self._param_field(self.combo_sky_post_cleanup),
        )
        self.chk_sky_per_exp = QtWidgets.QCheckBox("Per exposure (recommended)")
        bf.addRow(
            self._param_label(
                "Per exposure",
                "Вычитать небо для каждого экспозиционного кадра отдельно (best practice).\n"
                "Типично: включено.",
            ),
            self._param_field(self.chk_sky_per_exp),
        )

        self.chk_sky_stack_after = QtWidgets.QCheckBox("Stack2D is a separate step (see Step 11)")
        self.chk_sky_stack_after.setEnabled(False)
        bf.addRow(
            self._param_label(
                "Stack2D",
                "В v5.39.1 Stack2D вынесен в отдельную стадию.\n"
                "Небо (Step 09) больше не запускает stacking автоматически.\n"
                "Запустите Step 11 Stack2D отдельно.",
            ),
            self._param_field(self.chk_sky_stack_after),
        )

        self.chk_sky_save_models = QtWidgets.QCheckBox("Save per-exp sky model")
        bf.addRow(
            self._param_label(
                "Save models",
                "Сохранять модель неба для каждой экспозиции (FITS).\n"
                "Полезно для отладки и QC, но занимает место.\n"
                "Типично: выключено.",
            ),
            self._param_field(self.chk_sky_save_models),
        )

        self.dspin_sky_step = QtWidgets.QDoubleSpinBox()
        self.dspin_sky_step.setRange(0.1, 50.0)
        self.dspin_sky_step.setDecimals(2)
        self.dspin_sky_step.setSingleStep(0.25)
        bf.addRow(
            self._param_label(
                "B-spline step [Å]",
                "Шаг узлов B-сплайна по λ.\n"
                "Меньше шаг → точнее линии, но выше риск переобучения.\n"
                "Типично: 1–5 Å.",
            ),
            self._param_field(self.dspin_sky_step),
        )

        self.spin_sky_deg = QtWidgets.QSpinBox()
        self.spin_sky_deg.setRange(1, 5)
        self.spin_sky_deg.setSingleStep(1)
        bf.addRow(
            self._param_label(
                "B-spline degree",
                "Степень B-сплайна (обычно 3).\nТипично: 3.",
            ),
            self._param_field(self.spin_sky_deg),
        )

        self.dspin_sky_clip = QtWidgets.QDoubleSpinBox()
        self.dspin_sky_clip.setRange(0.5, 10.0)
        self.dspin_sky_clip.setDecimals(2)
        self.dspin_sky_clip.setSingleStep(0.25)
        bf.addRow(
            self._param_label(
                "Sigma clip",
                "Сигма-клиппинг при подгонке сплайна к S(λ).\nТипично: 2.5–4.",
            ),
            self._param_field(self.dspin_sky_clip),
        )

        self.spin_sky_maxiter = QtWidgets.QSpinBox()
        self.spin_sky_maxiter.setRange(1, 20)
        self.spin_sky_maxiter.setSingleStep(1)
        bf.addRow(
            self._param_label(
                "Max iters",
                "Максимум итераций клиппинга.\nТипично: 3–8.",
            ),
            self._param_field(self.spin_sky_maxiter),
        )

        self.chk_sky_spatial = QtWidgets.QCheckBox("Spatial polynomial")
        bf.addRow(
            self._param_label(
                "Spatial",
                "Моделировать изменение амплитуды неба вдоль щели: a(y)*S(λ)+b(y).\n"
                "Типично: включено.",
            ),
            self._param_field(self.chk_sky_spatial),
        )

        self.spin_sky_poly = QtWidgets.QSpinBox()
        self.spin_sky_poly.setRange(0, 3)
        self.spin_sky_poly.setSingleStep(1)
        bf.addRow(
            self._param_label(
                "Poly deg",
                "Степень полинома для a(y), b(y).\n"
                "0 = константа, 1 = линейно, 2 = квадратично.\n"
                "Типично: 0–1.",
            ),
            self._param_field(self.spin_sky_poly),
        )

        adv = QtWidgets.QWidget()
        adv_l = QtWidgets.QVBoxLayout(adv)
        adv_l.setContentsMargins(0, 0, 0, 0)
        adv_l.setSpacing(10)

        # Flexure correction (Δλ)
        gflex = _box("Flexure correction (Δλ)")
        fl = QtWidgets.QFormLayout(gflex)
        fl.setLabelAlignment(QtCore.Qt.AlignLeft)
        fl.setHorizontalSpacing(12)

        # Keep checkbox text empty: the label column is the single source of truth.
        self.chk_sky_flex_enabled = QtWidgets.QCheckBox("")
        fl.addRow(
            self._param_label(
                "Flexure",
                "Коррекция сдвига Δλ по sky-линиям (кросс-корреляция).\n"
                "Рекомендовано, если есть флексия/дрейф.\n"
                "Типично: включено.\n",
            ),
            self._param_field(self.chk_sky_flex_enabled),
        )

        self.combo_sky_flex_mode = QtWidgets.QComboBox()
        self.combo_sky_flex_mode.addItems(["full", "windows"])
        fl.addRow(
            self._param_label(
                "Mode",
                "full: использовать весь спектр.\n"
                "windows: только выбранные окна (макс. S/N) — стабильнее.\n"
                "Типично: windows.\n",
            ),
            self._param_field(self.combo_sky_flex_mode),
        )

        self.spin_sky_flex_max = QtWidgets.QSpinBox()
        self.spin_sky_flex_max.setRange(0, 50)
        self.spin_sky_flex_max.setSingleStep(1)
        fl.addRow(
            self._param_label(
                "Max shift [pix]",
                "Максимальный допустимый сдвиг в пикселях по λ (субпиксельно внутри).\n"
                "Типично: 2–10 pix.\n",
            ),
            self._param_field(self.spin_sky_flex_max),
        )

        self.combo_sky_flex_windows_unit = QtWidgets.QComboBox()
        self.combo_sky_flex_windows_unit.addItems(["auto", "A", "pix"])
        fl.addRow(
            self._param_label(
                "Windows units",
                "Единицы окон: auto = Å если кадр уже в длинах волн, иначе пиксели.\n"
                "Типично: auto.\n",
            ),
            self._param_field(self.combo_sky_flex_windows_unit),
        )

        row_w = QtWidgets.QWidget()
        row_w_l = QtWidgets.QHBoxLayout(row_w)
        row_w_l.setContentsMargins(0, 0, 0, 0)
        self.lbl_sky_flex_windows = QtWidgets.QLabel("<no windows>")
        self.lbl_sky_flex_windows.setWordWrap(True)
        self.btn_sky_flex_pick_windows = QtWidgets.QPushButton("Pick…")
        row_w_l.addWidget(self.lbl_sky_flex_windows, 1)
        row_w_l.addWidget(self.btn_sky_flex_pick_windows)
        fl.addRow(
            self._param_label(
                "λ windows",
                "Диапазоны по X для кросс-корреляции.\n"
                "Выбирайте окна с яркими и узкими линиями (макс. S/N).\n"
                "Типично: 2–6 окон.\n",
            ),
            self._param_field(row_w),
        )

        self.chk_sky_flex_ydep = QtWidgets.QCheckBox("Δλ(y)")
        fl.addRow(
            self._param_label(
                "y-dependent",
                "Разрешить зависимость сдвига от y вдоль щели.\n"
                "Сглаженный низкий порядок полинома (макс. стабильность).\n"
                "Типично: выключено (включать при явном градиенте).\n",
            ),
            self._param_field(self.chk_sky_flex_ydep),
        )

        self.spin_sky_flex_y_poly = QtWidgets.QSpinBox()
        self.spin_sky_flex_y_poly.setRange(0, 3)
        self.spin_sky_flex_y_poly.setSingleStep(1)
        fl.addRow(
            self._param_label(
                "Poly deg",
                "Порядок полинома для Δλ(y).\n"
                "0 = константа, 1 = линейно.\n"
                "Типично: 1.\n",
            ),
            self._param_field(self.spin_sky_flex_y_poly),
        )

        self.spin_sky_flex_y_smooth = QtWidgets.QSpinBox()
        self.spin_sky_flex_y_smooth.setRange(1, 21)
        self.spin_sky_flex_y_smooth.setSingleStep(2)
        fl.addRow(
            self._param_label(
                "Smooth bins",
                "Медианная гладкость для измеренных точек Δλ(y) перед фиттом.\n"
                "1 = без сглаживания.\n"
                "Типично: 3–7.\n",
            ),
            self._param_field(self.spin_sky_flex_y_smooth),
        )

        self.dspin_sky_flex_min_score = QtWidgets.QDoubleSpinBox()
        self.dspin_sky_flex_min_score.setRange(0.0, 1.0)
        self.dspin_sky_flex_min_score.setDecimals(3)
        self.dspin_sky_flex_min_score.setSingleStep(0.05)
        fl.addRow(
            self._param_label(
                "Min score",
                "Минимальный score кросс-корреляции; ниже — сдвиг игнорируется.\n"
                "Типично: 0.02–0.2.\n",
            ),
            self._param_field(self.dspin_sky_flex_min_score),
        )

        adv_l.addWidget(gflex)

        # Stack2D tuning (runs after sky if enabled)
        gstack = _box("Stack2D (after sky)")
        sl = QtWidgets.QFormLayout(gstack)
        sl.setLabelAlignment(QtCore.Qt.AlignLeft)
        sl.setHorizontalSpacing(12)

        self.dspin_stack_sigma = QtWidgets.QDoubleSpinBox()
        self.dspin_stack_sigma.setRange(0.0, 10.0)
        self.dspin_stack_sigma.setDecimals(2)
        self.dspin_stack_sigma.setSingleStep(0.25)
        self.dspin_stack_sigma.setToolTip("0 = disable")
        sl.addRow(
            self._param_label(
                "Sigma clip",
                "Сигма-клиппинг при объединении (stack2d). 0 = отключить.\n"
                "Типично: 2–4.\n",
            ),
            self._param_field(self.dspin_stack_sigma),
        )

        self.spin_stack_maxiter = QtWidgets.QSpinBox()
        self.spin_stack_maxiter.setRange(1, 20)
        self.spin_stack_maxiter.setSingleStep(1)
        sl.addRow(
            self._param_label(
                "Max iters",
                "Итерации клиппинга.\nТипично: 3–8.\n",
            ),
            self._param_field(self.spin_stack_maxiter),
        )

        # Keep checkbox text empty: the label column is the single source of truth.
        self.chk_stack_y_align = QtWidgets.QCheckBox("")
        sl.addRow(
            self._param_label(
                "y-align",
                "Выравнивание по y перед stacking (субпиксельно).\n"
                "Полезно при дрейфе/дизеринге.\n"
                "Типично: выключено, включать если заметен сдвиг.\n",
            ),
            self._param_field(self.chk_stack_y_align),
        )

        self.spin_stack_y_align_max = QtWidgets.QSpinBox()
        self.spin_stack_y_align_max.setRange(0, 50)
        self.spin_stack_y_align_max.setSingleStep(1)
        sl.addRow(
            self._param_label(
                "Max y-shift [pix]",
                "Ограничение для y-shift.\nТипично: 2–15.\n",
            ),
            self._param_field(self.spin_stack_y_align_max),
        )

        self.combo_stack_y_align_mode = QtWidgets.QComboBox()
        self.combo_stack_y_align_mode.addItems(["full", "windows"])
        sl.addRow(
            self._param_label(
                "Mode",
                "full: весь спектр. windows: только выбранные окна (макс. S/N).\n"
                "Типично: windows.\n",
            ),
            self._param_field(self.combo_stack_y_align_mode),
        )

        self.combo_stack_y_align_windows_unit = QtWidgets.QComboBox()
        self.combo_stack_y_align_windows_unit.addItems(["auto", "A", "pix"])
        sl.addRow(
            self._param_label(
                "Windows units",
                "auto = Å если есть WCS по λ, иначе пиксели.\nТипично: auto.\n",
            ),
            self._param_field(self.combo_stack_y_align_windows_unit),
        )

        row_sw = QtWidgets.QWidget()
        row_sw_l = QtWidgets.QHBoxLayout(row_sw)
        row_sw_l.setContentsMargins(0, 0, 0, 0)
        self.lbl_stack_y_align_windows = QtWidgets.QLabel("<no windows>")
        self.lbl_stack_y_align_windows.setWordWrap(True)
        self.btn_stack_pick_windows = QtWidgets.QPushButton("Pick…")
        row_sw_l.addWidget(self.lbl_stack_y_align_windows, 1)
        row_sw_l.addWidget(self.btn_stack_pick_windows)
        sl.addRow(
            self._param_label(
                "λ windows",
                "Окна по X для построения профиля и y-xcorr.\n"
                "Типично: те же, что и для flexure.\n",
            ),
            self._param_field(row_sw),
        )

        self.chk_stack_y_align_pos = QtWidgets.QCheckBox("Use positive flux")
        sl.addRow(
            self._param_label(
                "Positive only",
                "В профиле учитывать только положительный поток (стабильнее при шуме).\n"
                "Типично: включено.\n",
            ),
            self._param_field(self.chk_stack_y_align_pos),
        )

        adv_l.addWidget(gstack)

        # locale for doubles
        self._force_dot_locale(
            self.dspin_sky_step,
            self.dspin_sky_clip,
            self.dspin_sky_flex_min_score,
            self.dspin_stack_sigma,
        )

        pl.addWidget(self._mk_basic_advanced_tabs(basic, adv))

        pl.addWidget(self._mk_stage_apply_row("sky"))
        gl.addWidget(gpar)

        row = QtWidgets.QHBoxLayout()
        self.btn_run_sky = QtWidgets.QPushButton("Run: Sky subtraction")
        self.btn_run_sky.setProperty("primary", True)
        self.btn_qc_sky = QtWidgets.QPushButton("QC")
        self.btn_frames_sky = QtWidgets.QPushButton("Frames…")
        row.addWidget(self.btn_run_sky)
        row.addWidget(self.btn_qc_sky)
        row.addWidget(self.btn_frames_sky)
        row.addStretch(1)
        gl.addLayout(row)
        left_layout.addStretch(1)
        lay.addWidget(left, 1)


        # pending wiring
        self.chk_sky_enabled.stateChanged.connect(
            lambda _: self._stage_set_pending(
                "sky", "sky.enabled", bool(self.chk_sky_enabled.isChecked())
            )
        )
        self.dspin_sky_step.valueChanged.connect(
            lambda v: self._stage_set_pending("sky", "sky.bsp_step_A", float(v))
        )
        self.spin_sky_deg.valueChanged.connect(
            lambda v: self._stage_set_pending("sky", "sky.bsp_degree", int(v))
        )
        self.dspin_sky_clip.valueChanged.connect(
            lambda v: self._stage_set_pending("sky", "sky.sigma_clip", float(v))
        )
        self.spin_sky_maxiter.valueChanged.connect(
            lambda v: self._stage_set_pending("sky", "sky.maxiter", int(v))
        )
        self.chk_sky_spatial.stateChanged.connect(
            lambda _: self._stage_set_pending(
                "sky", "sky.use_spatial_scale", bool(self.chk_sky_spatial.isChecked())
            )
        )
        self.spin_sky_poly.valueChanged.connect(
            lambda v: self._stage_set_pending("sky", "sky.spatial_poly_deg", int(v))
        )


        # Primary sky method
        self.combo_sky_primary_method.currentIndexChanged.connect(
            lambda _: self._stage_set_pending(
                "sky",
                "sky.primary_method",
                str(self.combo_sky_primary_method.currentData() or "kelson_raw"),
            )
        )

        # Post-rectification residual cleanup (belongs to Linearization, but controlled here)
        def _on_sky_post_cleanup_changed() -> None:
            v = str(self.combo_sky_post_cleanup.currentData() or "auto").strip().lower()
            if v not in {"off", "auto", "on"}:
                v = "auto"
            # preferred key (Linearization)
            self._stage_set_pending("sky", "linearize.post_sky_cleanup", v)
            # legacy key (Sky)
            self._stage_set_pending("sky", "sky.post_cleanup", v)

        self.combo_sky_post_cleanup.currentIndexChanged.connect(lambda _: _on_sky_post_cleanup_changed())
        self.chk_sky_per_exp.stateChanged.connect(
            lambda _: self._stage_set_pending(
                "sky", "sky.per_exposure", bool(self.chk_sky_per_exp.isChecked())
            )
        )
        self.chk_sky_stack_after.stateChanged.connect(
            lambda _: self._stage_set_pending(
                "sky", "sky.stack_after", bool(self.chk_sky_stack_after.isChecked())
            )
        )
        self.chk_sky_save_models.stateChanged.connect(
            lambda _: self._stage_set_pending(
                "sky",
                "sky.save_per_exp_model",
                bool(self.chk_sky_save_models.isChecked()),
            )
        )

        # --- Advanced: flexure (Δλ) ---
        self.chk_sky_flex_enabled.stateChanged.connect(
            lambda _: self._stage_set_pending(
                "sky",
                "sky.flexure.enabled",
                bool(self.chk_sky_flex_enabled.isChecked()),
            )
        )
        self.combo_sky_flex_mode.currentTextChanged.connect(
            lambda t: self._stage_set_pending("sky", "sky.flexure.mode", str(t))
        )
        self.spin_sky_flex_max.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "sky", "sky.flexure.max_shift_pix", int(v)
            )
        )
        self.combo_sky_flex_windows_unit.currentTextChanged.connect(
            lambda t: self._stage_set_pending("sky", "sky.flexure.windows_unit", str(t))
        )
        self.chk_sky_flex_ydep.stateChanged.connect(
            lambda _: self._stage_set_pending(
                "sky",
                "sky.flexure.y_dependent",
                bool(self.chk_sky_flex_ydep.isChecked()),
            )
        )
        self.spin_sky_flex_y_poly.valueChanged.connect(
            lambda v: self._stage_set_pending("sky", "sky.flexure.y_poly_deg", int(v))
        )
        self.spin_sky_flex_y_smooth.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "sky", "sky.flexure.y_smooth_bins", int(v)
            )
        )
        self.dspin_sky_flex_min_score.valueChanged.connect(
            lambda v: self._stage_set_pending("sky", "sky.flexure.min_score", float(v))
        )
        self.btn_sky_flex_pick_windows.clicked.connect(
            lambda: self._do_pick_lambda_windows(cfg_prefix="sky.flexure")
        )

        # --- Advanced: stack2d tuning / y-align ---
        self.dspin_stack_sigma.valueChanged.connect(
            lambda v: self._stage_set_pending("sky", "stack2d.sigma_clip", float(v))
        )
        self.spin_stack_maxiter.valueChanged.connect(
            lambda v: self._stage_set_pending("sky", "stack2d.maxiter", int(v))
        )
        self.chk_stack_y_align.stateChanged.connect(
            lambda _: self._stage_set_pending(
                "sky",
                "stack2d.y_align.enabled",
                bool(self.chk_stack_y_align.isChecked()),
            )
        )
        self.spin_stack_y_align_max.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "sky", "stack2d.y_align.max_shift_pix", int(v)
            )
        )
        self.combo_stack_y_align_mode.currentTextChanged.connect(
            lambda t: self._stage_set_pending("sky", "stack2d.y_align.mode", str(t))
        )
        self.combo_stack_y_align_windows_unit.currentTextChanged.connect(
            lambda t: self._stage_set_pending(
                "sky", "stack2d.y_align.windows_unit", str(t)
            )
        )
        self.chk_stack_y_align_pos.stateChanged.connect(
            lambda _: self._stage_set_pending(
                "sky",
                "stack2d.y_align.use_positive_flux",
                bool(self.chk_stack_y_align_pos.isChecked()),
            )
        )
        self.btn_stack_pick_windows.clicked.connect(
            lambda: self._do_pick_lambda_windows(cfg_prefix="stack2d.y_align")
        )

        self.btn_sky_select_roi.clicked.connect(self._do_select_sky_rois)
        self.btn_run_sky.clicked.connect(self._do_run_sky)
        self.btn_qc_sky.clicked.connect(self._open_qc_viewer)
        self.btn_frames_sky.clicked.connect(lambda: self._open_frames_window("sky"))

        self._sync_stage_controls_from_cfg()
        return w

    def _do_select_sky_rois(self) -> None:
        from scorpio_pipe.ui.sky_roi_dialog import SkyRoiDialog

        # We select regions on a cosmics-cleaned RAW frame (detector geometry).
        if not self._ensure_cfg_saved():
            return
        try:
            ctx = load_context(self._cfg_path)
            work = resolve_work_dir(ctx.cfg)
            # Sky is performed in RAW detector geometry. We select sky/object windows on a cosmic-cleaned RAW frame.
            from scorpio_pipe.workspace_paths import stage_dir

            cos_root = stage_dir(work, "cosmics") / "clean"
            cand = sorted(cos_root.glob("*_clean.fits")) if cos_root.exists() else []
            if not cand:
                # fallback: any clean product under the cosmics stage
                cand = sorted(stage_dir(work, "cosmics").rglob("*_clean.fits"))
            if not cand:
                self._log_warn("No cosmic-cleaned frames found. Run Cosmics first.")
                return
            fits_path = cand[0]
            roi = (self._cfg.get("sky", {}) or {}).get("roi", {}) or {}
            dlg = SkyRoiDialog(fits_path, roi, parent=self)
            if dlg.exec() == QtWidgets.QDialog.Accepted:
                r = dlg.roi.to_dict()
                for k, v in r.items():
                    self._stage_set_pending("sky", f"sky.roi.{k}", int(v))
                self._sync_stage_controls_from_cfg()
        except Exception as e:
            self._log_exception(e)

    def _do_pick_lambda_windows(self, *, cfg_prefix: str) -> None:
        """Interactive selection of λ/pixel windows and store into config.

        cfg_prefix examples:
        - "sky.flexure"
        - "stack2d.y_align"
        """
        from scorpio_pipe.ui.lambda_windows_dialog import LambdaWindowsDialog
        from scorpio_pipe.workspace_paths import stage_dir

        # We select on a linearized preview if possible; otherwise on the first available rectified frame.
        if not self._ensure_cfg_saved():
            return
        try:
            ctx = load_context(self._cfg_path)
            work = resolve_work_dir(ctx.cfg)
            fits_path = stage_dir(work, "linearize") / "lin_preview.fits"
            if not fits_path.exists():
                fits_path = work / "lin" / "obj_sum_lin.fits"
            if not fits_path.exists():
                # try any per-exp sky frame
                # Fallback: any sky product in canonical stage tree.
                sky_root = stage_dir(work, "sky")
                cand = list(sky_root.rglob("*.fits"))
                if cand:
                    fits_path = cand[0]
            if not fits_path.exists():
                self._log_warn("No rectified frame found. Run Linearize first.")
                return

            roi = (self._cfg.get("sky", {}) or {}).get("roi", {}) or {}
            dlg = LambdaWindowsDialog(fits_path, roi, parent=self)
            if dlg.exec() == QtWidgets.QDialog.Accepted:
                d = dlg.windows.to_dict()
                unit = str(d.get("unit", "auto")).strip()
                winA = d.get("windows_A", []) or []
                winP = d.get("windows_pix", []) or []
                self._stage_set_pending("sky", f"{cfg_prefix}.windows_unit", unit)
                self._stage_set_pending("sky", f"{cfg_prefix}.windows_A", winA)
                self._stage_set_pending("sky", f"{cfg_prefix}.windows_pix", winP)
                self._sync_stage_controls_from_cfg()
        except Exception as e:
            self._log_exception(e)

    def _do_run_sky(self) -> None:
        if not self._ensure_stage_applied("sky", "Sky subtraction"):
            return
        if not self._ensure_cfg_saved():
            return
        # quick ROI sanity
        roi = (self._cfg.get("sky", {}) or {}).get("roi", {}) or {}
        need = [
            "obj_y0",
            "obj_y1",
            "sky_top_y0",
            "sky_top_y1",
            "sky_bot_y0",
            "sky_bot_y1",
        ]
        if any(k not in roi for k in need):
            self._log_warn("ROI is not set. Use 'Select regions…' first.")
            return
        self._set_step_status(self._stage_row_index("sky"), "running")
        try:
            ctx = load_context(self._cfg_path)
            sky_cfg = (
                (ctx.cfg.get("sky") or {})
                if isinstance(ctx.cfg.get("sky"), dict)
                else {}
            )
            # Sky runs as a standalone stage. Stack2D is executed explicitly
            # in its own step.
            tasks = ["sky"]
            self._run_sequence_with_qc_prompt(ctx, tasks)
            self._log_info("Sky subtraction done")
            self._set_step_status(self._stage_row_index("sky"), "ok")
            self._maybe_auto_qc()
        except Exception as e:
            self._set_step_status(self._stage_row_index("sky"), "fail")
            self._log_exception(e)

    def _do_run_stack2d(self) -> None:
        from scorpio_pipe.workspace_paths import stage_dir

        if not self._ensure_stage_applied("stack2d", "Stack2D"):
            return
        if not self._ensure_cfg_saved():
            return

        self._set_step_status(self._stage_row_index("stack2d"), "running")
        try:
            ctx = load_context(self._cfg_path)
            work = resolve_work_dir(ctx.cfg)

            # Sanity (P1-E): Stack2D consumes ONLY rectified sky-subtracted frames from Linearization:
            #   10_linearize/<stem>_skysub.fits
            lin_root = stage_dir(work, "linearize")
            frames = ctx.cfg.get("frames") if isinstance(ctx.cfg.get("frames"), dict) else {}
            obj_list = frames.get("obj") if isinstance(frames.get("obj"), list) else []
            stems = [Path(x).stem for x in obj_list if isinstance(x, str) and x.strip()]

            if stems:
                missing: list[str] = []
                for stem in stems:
                    p = lin_root / f"{stem}_skysub.fits"
                    if not p.exists():
                        missing.append(stem)
                if missing:
                    raise FileNotFoundError(
                        "Missing linearized sky-subtracted inputs for stems: "
                        + ", \n".join(missing)
                        + ". Expected 10_linearize/<stem>_skysub.fits. Run 'Linearization' first."
                    )
            else:
                inputs = list(lin_root.glob("*_skysub.fits"))
                if not inputs:
                    raise FileNotFoundError(
                        "No linearized sky-subtracted frames found in 10_linearize/. Run 'Linearization' first."
                    )

            self._run_sequence_with_qc_prompt(ctx, ["stack2d"])
            self._log_info("Stack2D done")
            self._set_step_status(self._stage_row_index("stack2d"), "ok")
            self._maybe_auto_qc()
        except Exception as e:
            self._set_step_status(self._stage_row_index("stack2d"), "fail")
            self._log_exception(e)

    def _build_page_stack2d(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setSpacing(12)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setSpacing(12)

        g = _box("Stack2D (combine)")
        left_layout.addWidget(g)
        gl = QtWidgets.QVBoxLayout(g)

        lbl = QtWidgets.QLabel(
            "Robust 2D stacking of rectified sky-subtracted frames in (λ, y).\n"
            "Input: 10_linearize/<stem>_skysub.fits\n"
            "Output: 11_stack/stack2d.fits"
        )
        lbl.setWordWrap(True)
        gl.addWidget(lbl)

        gpar = _box("Parameters")
        pl = QtWidgets.QVBoxLayout(gpar)
        pl.setSpacing(10)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignLeft)
        form.setHorizontalSpacing(12)

        self.dspin_stack2d_clip = QtWidgets.QDoubleSpinBox()
        self.dspin_stack2d_clip.setRange(0.5, 10.0)
        self.dspin_stack2d_clip.setDecimals(2)
        self.dspin_stack2d_clip.setSingleStep(0.25)
        form.addRow(
            self._param_label(
                "Sigma clip",
                "Sigma-clipping during stack. Typical: 2.5–4.",
            ),
            self._param_field(self.dspin_stack2d_clip),
        )

        self.spin_stack2d_maxiter = QtWidgets.QSpinBox()
        self.spin_stack2d_maxiter.setRange(1, 20)
        self.spin_stack2d_maxiter.setSingleStep(1)
        form.addRow(
            self._param_label(
                "Max iters",
                "Max clipping iterations. Typical: 3–8.",
            ),
            self._param_field(self.spin_stack2d_maxiter),
        )

        self.chk_stack2d_png = QtWidgets.QCheckBox("Save PNG")
        form.addRow(
            self._param_label(
                "Save PNG",
                "Save diagnostic PNG of the stacked frame. Typical: enabled.",
            ),
            self._param_field(self.chk_stack2d_png),
        )

        pl.addLayout(form)
        gl.addWidget(gpar)

        # Apply / Run row
        row = QtWidgets.QHBoxLayout()
        self.btn_stack2d_apply = QtWidgets.QPushButton("Apply")
        self.btn_stack2d_run = QtWidgets.QPushButton("Run Stack2D")
        self.btn_stack2d_run.setProperty("primary", True)
        row.addWidget(self.btn_stack2d_apply)
        row.addStretch(1)
        row.addWidget(self.btn_stack2d_run)
        gl.addLayout(row)
        lay.addWidget(left, 1)

        # Wire
        self.dspin_stack2d_clip.valueChanged.connect(
            lambda v: self._stage_set_pending("stack2d", "stack2d.sigma_clip", float(v))
        )
        self.spin_stack2d_maxiter.valueChanged.connect(
            lambda v: self._stage_set_pending("stack2d", "stack2d.max_iter", int(v))
        )
        self.chk_stack2d_png.toggled.connect(
            lambda b: self._stage_set_pending("stack2d", "stack2d.save_png", bool(b))
        )
        self.btn_stack2d_apply.clicked.connect(
            lambda: self._stage_apply("stack2d", "Stack2D")
        )
        self.btn_stack2d_run.clicked.connect(self._do_run_stack2d)

        return w

    def _build_page_extract1d(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setSpacing(12)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setSpacing(12)

        g = _box("Extract 1D spectrum")
        left_layout.addWidget(g)
        gl = QtWidgets.QVBoxLayout(g)

        lbl = QtWidgets.QLabel(
            "Extract 1D spectra F(λ) from a rectified 2D product (y, λ).\n"
            "Default input: 11_stack/stack2d.fits.\n"
            "Expert input: 10_linearize/<stem>_skysub.fits (when Input mode = single_frame).\n"
            "Outputs: 12_extract/spec1d.fits (+ trace.json and spec1d.png)."
        )
        lbl.setWordWrap(True)
        gl.addWidget(lbl)

        gpar = _box("Parameters")
        pl = QtWidgets.QVBoxLayout(gpar)
        pl.setSpacing(10)

        basic = QtWidgets.QWidget()
        bf = QtWidgets.QFormLayout(basic)
        bf.setLabelAlignment(QtCore.Qt.AlignLeft)
        bf.setHorizontalSpacing(12)

        self.chk_ex1d_enabled = QtWidgets.QCheckBox("Enabled")
        bf.addRow(
            self._param_label(
                "Enabled",
                "Включить этап извлечения 1D-спектра.\nТипично: включено.",
            ),
            self._param_field(self.chk_ex1d_enabled),
        )

        self.combo_ex1d_input_mode = QtWidgets.QComboBox()
        self.combo_ex1d_input_mode.addItems(["stack2d", "single_frame"])
        bf.addRow(
            self._param_label(
                "Input mode",
                "Выбор входного 2D продукта для извлечения.\n"
                "stack2d: 11_stack/stack2d.fits (рекомендуется, по умолчанию).\n"
                "single_frame: 10_linearize/<stem>_skysub.fits (экспертный режим; задайте stem).",
            ),
            self._param_field(self.combo_ex1d_input_mode),
        )

        self.edit_ex1d_single_stem = QtWidgets.QLineEdit()
        self.edit_ex1d_single_stem.setPlaceholderText("<stem>")
        bf.addRow(
            self._param_label(
                "Single-frame stem",
                "Используется только при Input mode = single_frame.\n"
                "Файл: 10_linearize/<stem>_skysub.fits.\n"
                "Важно: режим не включён по умолчанию.",
            ),
            self._param_field(self.edit_ex1d_single_stem),
        )

        self.combo_ex1d_method = QtWidgets.QComboBox()
        # Keep legacy options for compatibility, but surface best-practice choices.
        self.combo_ex1d_method.addItems(["boxcar", "optimal", "mean", "sum"])
        bf.addRow(
            self._param_label(
                "Method",
                "Метод извлечения 1D-спектра.\n"
                "boxcar = суммирование в апертуре вокруг trace (Basic).\n"
                "optimal = Horne-style optimal extraction (Advanced).\n"
                "Типично: boxcar.",
            ),
            self._param_field(self.combo_ex1d_method),
        )

        self.spin_ex1d_ap_hw = QtWidgets.QSpinBox()
        # 0 = auto-estimate aperture half-width.
        self.spin_ex1d_ap_hw.setRange(0, 10_000)
        self.spin_ex1d_ap_hw.setSingleStep(1)
        bf.addRow(
            self._param_label(
                "Aperture half-width [px]",
                "Половина ширины апертуры вокруг trace (в пикселях по Y).\n"
                "0 = auto (оценка из ROI/профиля).\n"
                "Типично: 4–10 px (зависит от seeing и биннинга).",
            ),
            self._param_field(self.spin_ex1d_ap_hw),
        )

        self.dspin_ex1d_trace_bin = QtWidgets.QDoubleSpinBox()
        self.dspin_ex1d_trace_bin.setRange(5.0, 5_000.0)
        self.dspin_ex1d_trace_bin.setDecimals(1)
        self.dspin_ex1d_trace_bin.setSingleStep(5.0)
        bf.addRow(
            self._param_label(
                "Trace bin [Å]",
                "Ширина бина по λ для оценки центроидов trace y(λ).\n"
                "Больше → стабильнее в шуме, меньше → точнее локальные изгибы.\n"
                "Типично: 30–100 Å.",
            ),
            self._param_field(self.dspin_ex1d_trace_bin),
        )

        self.chk_ex1d_png = QtWidgets.QCheckBox("Save plot PNG")
        bf.addRow(
            self._param_label(
                "PNG",
                "Сохранить quicklook график products/spec/spec1d.png.\nТипично: включено.",
            ),
            self._param_field(self.chk_ex1d_png),
        )

        adv = QtWidgets.QWidget()
        af = QtWidgets.QFormLayout(adv)
        af.setLabelAlignment(QtCore.Qt.AlignLeft)
        af.setHorizontalSpacing(12)

        self.spin_ex1d_trace_deg = QtWidgets.QSpinBox()
        self.spin_ex1d_trace_deg.setRange(0, 8)
        self.spin_ex1d_trace_deg.setSingleStep(1)
        af.addRow(
            self._param_label(
                "Trace smooth deg",
                "Степень полинома для сглаживания trace y(λ).\n"
                "0 = константа, 1 = линейно, 2–4 обычно достаточно.\n"
                "Типично: 3.",
            ),
            self._param_field(self.spin_ex1d_trace_deg),
        )

        self.spin_ex1d_prof_hw = QtWidgets.QSpinBox()
        self.spin_ex1d_prof_hw.setRange(2, 10_000)
        self.spin_ex1d_prof_hw.setSingleStep(1)
        af.addRow(
            self._param_label(
                "Optimal profile half-width [px]",
                "Для optimal extraction: половина окна по Y для построения профиля.\n"
                "Типично: 10–20 px.",
            ),
            self._param_field(self.spin_ex1d_prof_hw),
        )

        self.dspin_ex1d_opt_clip = QtWidgets.QDoubleSpinBox()
        self.dspin_ex1d_opt_clip.setRange(1.0, 20.0)
        self.dspin_ex1d_opt_clip.setDecimals(1)
        self.dspin_ex1d_opt_clip.setSingleStep(0.5)
        af.addRow(
            self._param_label(
                "Optimal sigma clip",
                "Sigma-clip при построении профиля (optimal).\nТипично: 4–6.",
            ),
            self._param_field(self.dspin_ex1d_opt_clip),
        )

        # locale for doubles
        self._force_dot_locale(self.dspin_ex1d_trace_bin, self.dspin_ex1d_opt_clip)

        pl.addWidget(self._mk_basic_advanced_tabs(basic, adv))

        pl.addWidget(self._mk_stage_apply_row("extract1d"))
        gl.addWidget(gpar)

        row = QtWidgets.QHBoxLayout()
        self.btn_run_ex1d = QtWidgets.QPushButton("Run: Extract 1D")
        self.btn_run_ex1d.setProperty("primary", True)
        self.btn_qc_ex1d = QtWidgets.QPushButton("QC")
        self.btn_frames_ex1d = QtWidgets.QPushButton("Frames…")
        row.addWidget(self.btn_run_ex1d)
        row.addWidget(self.btn_qc_ex1d)
        row.addWidget(self.btn_frames_ex1d)
        row.addStretch(1)
        gl.addLayout(row)
        left_layout.addStretch(1)
        lay.addWidget(left, 1)


        self.chk_ex1d_enabled.stateChanged.connect(
            lambda _: self._stage_set_pending(
                "extract1d",
                "extract1d.enabled",
                bool(self.chk_ex1d_enabled.isChecked()),
            )
        )
        if hasattr(self, "combo_ex1d_input_mode"):
            self.combo_ex1d_input_mode.currentTextChanged.connect(
                lambda t: self._stage_set_pending(
                    "extract1d", "extract1d.input_mode", str(t)
                )
            )
        if hasattr(self, "edit_ex1d_single_stem"):
            self.edit_ex1d_single_stem.textChanged.connect(
                lambda t: self._stage_set_pending(
                    "extract1d", "extract1d.single_frame_stem", str(t).strip()
                )
            )
        self.combo_ex1d_method.currentTextChanged.connect(
            lambda t: self._stage_set_pending("extract1d", "extract1d.method", str(t))
        )
        self.spin_ex1d_ap_hw.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "extract1d", "extract1d.aperture_half_width", int(v)
            )
        )
        self.dspin_ex1d_trace_bin.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "extract1d", "extract1d.trace_bin_A", float(v)
            )
        )
        self.spin_ex1d_trace_deg.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "extract1d", "extract1d.trace_smooth_deg", int(v)
            )
        )
        self.spin_ex1d_prof_hw.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "extract1d", "extract1d.optimal_profile_half_width", int(v)
            )
        )
        self.dspin_ex1d_opt_clip.valueChanged.connect(
            lambda v: self._stage_set_pending(
                "extract1d", "extract1d.optimal_sigma_clip", float(v)
            )
        )
        self.chk_ex1d_png.stateChanged.connect(
            lambda _: self._stage_set_pending(
                "extract1d", "extract1d.save_png", bool(self.chk_ex1d_png.isChecked())
            )
        )

        self.btn_run_ex1d.clicked.connect(self._do_run_extract1d)
        self.btn_qc_ex1d.clicked.connect(self._open_qc_viewer)
        self.btn_frames_ex1d.clicked.connect(lambda: self._open_frames_window("spec"))

        self._sync_stage_controls_from_cfg()
        return w

    def _do_run_extract1d(self) -> None:
        if not self._ensure_stage_applied("extract1d", "Extract 1D"):
            return
        if not self._ensure_cfg_saved():
            return
        from scorpio_pipe.workspace_paths import stage_dir

        # Default: Extract1D runs on top of Stack2D output.
        # Expert: the user can explicitly choose a single_frame input.
        try:
            wd = resolve_work_dir(self._cfg)
            ex = (self._cfg or {}).get("extract1d", {}) if isinstance((self._cfg or {}).get("extract1d"), dict) else {}
            mode = str(ex.get("input_mode", "stack2d") or "stack2d")
            if bool(ex.get("allow_sky_fallback", False)) and mode == "stack2d":
                mode = "single_frame"
            if mode == "stack2d" and not (stage_dir(wd, "stack2d") / "stack2d_done.json").exists():
                self._show_msg(
                    "Stack2D not complete",
                    [
                        "Run Step 11: Stack2D first (it produces stack2d.fits).",
                        "Or switch Extract1D Input mode to single_frame and specify <stem>_skysub.fits.",
                    ],
                    icon="warn",
                )
                self._set_step_status(self._stage_row_index("extract1d"), "fail")
                return
        except Exception:
            # If work_dir cannot be resolved, runner will report a clearer error.
            pass

        self._set_step_status(self._stage_row_index("extract1d"), "running")
        try:
            ctx = load_context(self._cfg_path)
            self._run_sequence_with_qc_prompt(ctx, ["extract1d"])
            self._log_info("Extract 1D done")
            self._set_step_status(self._stage_row_index("extract1d"), "ok")
            self._maybe_auto_qc()
        except Exception as e:
            self._set_step_status(self._stage_row_index("extract1d"), "fail")
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
            self._show_msg(
                "Config is missing",
                ["Create (Create new config) or open a config.yaml first."],
                icon="warn",
            )
            return False
        if not self._sync_cfg_from_editor():
            self._show_msg(
                "Config YAML invalid",
                ["Fix YAML (Validate) and try again."],
                icon="warn",
            )
            return False
        # ensure cfg is on disk for runner
        try:
            self._cfg_path.parent.mkdir(parents=True, exist_ok=True)
            self._cfg_path.write_text(self.editor_yaml.toPlainText(), encoding="utf-8")
        except Exception as e:
            self._log_exception(e)
            self._show_msg(
                "Cannot save config", [f"Path: {self._cfg_path}", str(e)], icon="error"
            )
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
        try:
            self._qc.showMaximized()
        except Exception:
            self._qc.show()
        self._qc.raise_()
        self._qc.activateWindow()

    # --------------------------- navigation glue ---------------------------

    def _current_work_dir_resolved(self) -> Path | None:
        """Return absolute work_dir from current config/cfg_path (if possible)."""
        cfg = self._cfg or {}
        wd_raw = str(cfg.get("work_dir", "") or "").strip()
        if not wd_raw:
            return None
        wd = Path(wd_raw).expanduser()
        if (not wd.is_absolute()) and self._cfg_path is not None:
            wd = (self._cfg_path.parent / wd).resolve()
        return wd

    def _on_step_changed(self, idx: int) -> None:
        try:
            i = int(idx)
            page_i = self._stage_page_index[i] if hasattr(self, "_stage_page_index") else i
            self.stack.setCurrentIndex(int(page_i))
        except Exception:
            pass
        self._refresh_statusbar()
        self._sync_outputs_action_state()
        self._update_outputs_tool_context()
        # If Outputs tool window is open, keep it pinned to current stage.
        self._update_outputs_tool_context()

        try:
            if hasattr(self, "stack") and int(self.stack.currentIndex()) == 7:
                self._update_wavesol_stepper()
        except Exception:
            pass

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

        self._sb_btn_data = _mk_btn("Data: —", self._open_data_folder)
        self._sb_btn_work = _mk_btn("Work: —", self._open_work_folder)
        self._sb_btn_cfg = _mk_btn("Cfg: —", self._open_cfg_folder)
        self._sb_btn_report = _mk_btn("Report", self._open_report_html)

        sb.addPermanentWidget(self._sb_btn_data)
        sb.addPermanentWidget(self._sb_btn_work)
        sb.addPermanentWidget(self._sb_btn_cfg)
        sb.addPermanentWidget(self._sb_btn_report)

        try:
            self.edit_data_dir.textChanged.connect(lambda *_: self._refresh_statusbar())
            self.edit_cfg_path.textChanged.connect(lambda *_: self._refresh_statusbar())
            if hasattr(self, "edit_work_dir"):
                self.edit_work_dir.textChanged.connect(
                    lambda *_: self._refresh_statusbar()
                )
        except Exception:
            pass

        self._refresh_statusbar()
        self._sync_outputs_action_state()

        try:
            if hasattr(self, "stack") and int(self.stack.currentIndex()) == 7:
                self._update_wavesol_stepper()
        except Exception:
            pass

    def _short_path(self, p: Path | None) -> str:
        if not p:
            return "—"
        s = str(p)
        # keep it readable in a status bar
        return p.name or s

    def _refresh_statusbar(self) -> None:
        try:
            d = (
                Path(self.edit_data_dir.text()).expanduser()
                if self.edit_data_dir.text().strip()
                else None
            )
        except Exception:
            d = None
        wd = self._current_work_dir_resolved()
        cfgp = None
        try:
            cfgp = (
                Path(self.edit_cfg_path.text()).expanduser()
                if self.edit_cfg_path.text().strip()
                else None
            )
        except Exception:
            cfgp = None

        try:
            self._sb_btn_data.setText(f"Data: {self._short_path(d)}")
            self._sb_btn_data.setToolTip(str(d) if d else "")
            self._sb_btn_work.setText(f"Work: {self._short_path(wd)}")
            self._sb_btn_work.setToolTip(str(wd) if wd else "")
            self._sb_btn_cfg.setText(f"Cfg: {self._short_path(cfgp)}")
            self._sb_btn_cfg.setToolTip(str(cfgp) if cfgp else "")
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
        wd = self._get_work_dir()
        p = wd / "index.html"
        if not p.exists():
            p = wd / "manifest" / "index.html"
        if p.exists():
            self._open_in_browser(p)

    def _install_shortcuts(self) -> None:
        # Keep the UI fast: shortcuts call existing handlers.
        def _sc(seq: str, cb) -> None:
            s = QtGui.QShortcut(QtGui.QKeySequence(seq), self)
            s.activated.connect(cb)

        _sc("Ctrl+I", self._do_inspect)
        _sc("Ctrl+S", self._do_save_cfg)
        _sc("Ctrl+R", self._run_all_steps)
        _sc("Ctrl+P", self._open_run_plan)
        _sc("Ctrl+Q", self._open_qc_viewer)
        _sc("Ctrl+O", self._open_data_folder)
        _sc("Ctrl+W", self._open_work_folder)

    def _maybe_auto_qc(self) -> None:
        if (
            getattr(self, "act_auto_qc", None) is not None
            and self.act_auto_qc.isChecked()
        ):
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
        self.act_toggle_log.triggered.connect(
            lambda checked: self.dock_log.setVisible(checked)
        )
        self.dock_log.visibilityChanged.connect(
            lambda v: self.act_toggle_log.setChecked(v)
        )

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
        self.act_auto_qc.triggered.connect(
            lambda v: self._settings.setValue("ui/auto_qc", bool(v))
        )

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

        self.act_outputs = QtGui.QAction("Outputs", self)
        self.act_outputs.setCheckable(True)
        self.act_outputs.setChecked(False)
        self.act_outputs.triggered.connect(self._toggle_outputs)
        tb.addAction(self.act_outputs)

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
            idx = int(self.steps.currentRow()) if hasattr(self, "steps") else 0
            mapping = {
                0: "project",
                2: "calib",
                3: "cosmics",
                4: "flatfield",
                5: "superneon",
                6: "lineid",
                7: "wavesol",
            }
            return mapping.get(idx, "project")

        frames_btn.clicked.connect(
            lambda: self._open_frames_window(_stage_for_current_step())
        )
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
            d = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Open data folder", str(Path.home())
            )
            if d:
                self.edit_data_dir.setText(d)
                self._open_in_explorer(Path(d))
        except Exception as e:
            self._log_exception(e)

    def _menu_open_cfg(self) -> None:
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open config", str(Path.home()), "YAML (*.yaml *.yml)"
        )
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
            TextViewerDialog(
                "Scorpio Pipe — Quick manual",
                manual.read_text(encoding="utf-8"),
                parent=self,
            ).exec()
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
        self.btn_run_calib.setEnabled(
            self._cfg_path is not None or bool(self.edit_cfg_path.text().strip())
        )
        self.btn_run_cosmics.setEnabled(self.btn_run_calib.isEnabled())
        if hasattr(self, "btn_run_flatfield"):
            self.btn_run_flatfield.setEnabled(self.btn_run_calib.isEnabled())
        self.btn_run_superneon.setEnabled(self.btn_run_calib.isEnabled())
        self.btn_open_lineid.setEnabled(self.btn_run_calib.isEnabled())
        self.btn_run_wavesol.setEnabled(self.btn_run_calib.isEnabled())

        # "Previous state" is available only when there is an opened run and at
        # least one snapshot in ui/history.
        try:
            rr_txt = (self.edit_work_dir.text() or "").strip()
            from scorpio_pipe.ui.session_store import list_snapshots

            has_hist = bool(rr_txt) and bool(list_snapshots(Path(rr_txt), limit=1))
            if hasattr(self, "btn_prev_state"):
                self.btn_prev_state.setEnabled(bool(has_hist))
        except Exception:
            try:
                if hasattr(self, "btn_prev_state"):
                    self.btn_prev_state.setEnabled(False)
            except Exception:
                pass

    def _run_sequence_with_qc_prompt(
        self,
        ctx_or_path: dict[str, Any] | Path,
        tasks: list[str],
        *,
        resume: bool = True,
        force: bool = False,
    ) -> dict[str, Any]:
        """Run tasks with a QC gate dialog.

        The runner enforces a fail-fast QC gate based on upstream ``done.json`` flags.
        Here we provide an explicit UX: cancel or proceed with an override.
        """
        try:
            return run_sequence(ctx_or_path, tasks, resume=resume, force=force, qc_override=False)
        except QCGateError as ge:
            sev = str(getattr(ge, "upstream_max_severity", "ERROR") or "ERROR").upper()
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Warning)
            box.setWindowTitle("QC gate остановил запуск")
            if sev == "FATAL":
                box.setText(
                    "Запуск остановлен: предыдущие стадии помечены QC-флагами FATAL.\n\n"
                    "Override недоступен: FATAL означает повреждённые данные/нарушение контракта.\n"
                    "Исправьте входы или перегенерируйте продукты upstream и запустите заново."
                )
            else:
                box.setText(
                    "Запуск остановлен: предыдущие стадии помечены QC-флагами ERROR/FATAL.\n\n"
                    "Рекомендуется исправить причину (входы/параметры) и запустить заново.\n"
                    "Если вы уверены — можно продолжить с override (это будет записано в manifest)."
                )

            try:
                box.setDetailedText(ge.summary(max_items=50))
            except Exception:
                box.setDetailedText(str(ge))

            btn_override = None
            if sev != "FATAL":
                btn_override = box.addButton(
                    "Запустить с override", QtWidgets.QMessageBox.DestructiveRole
                )
            box.addButton(QtWidgets.QMessageBox.Cancel)
            box.exec()

            if btn_override is not None and box.clickedButton() is btn_override:
                self._log_warn("QC override enabled by user; proceeding.")
                try:
                    return run_sequence(
                        ctx_or_path, tasks, resume=resume, force=force, qc_override=True
                    )
                except QCGateError as ge2:
                    # Most likely: upstream has FATAL blockers.
                    self._log_error("QC override refused (FATAL blockers).")
                    QtWidgets.QMessageBox.critical(
                        self,
                        "QC gate: FATAL",
                        "Override невозможен: обнаружены FATAL-флаги upstream.\n\n"
                        + (ge2.summary(max_items=50) if hasattr(ge2, "summary") else str(ge2)),
                    )
                    return {}

            self._log_warn("Run cancelled by QC gate.")
            return {}

    # --------------------------- logging helpers ---------------------------

    def _log_info(self, msg: str) -> None:
        self.log_view.appendPlainText(msg)

    def _log_warn(self, msg: str) -> None:
        # Keep the log format consistent with errors, but do not spam dialogs.
        self.log_view.appendPlainText("[WARN] " + msg)

    def _log_error(self, msg: str) -> None:
        self.log_view.appendPlainText("[ERROR] " + msg)

    def _log_exception(self, e: BaseException) -> None:
        self._log_error(f"{type(e).__name__}: {e}")
        tb = traceback.format_exc(limit=12)
        self.log_view.appendPlainText(tb)
