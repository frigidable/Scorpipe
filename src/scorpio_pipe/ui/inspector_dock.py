from __future__ import annotations

"""Always-visible right-side inspector panel.

Keeps key context on-screen:
- Outputs: expected products for the current step
- Plan: RUN/SKIP decisions in resume/force mode
- QC alerts: compact warnings/errors summary

No heavy computations are performed here.
"""

import json
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from scorpio_pipe.products import products_for_task, task_is_complete
from scorpio_pipe.ui.pipeline_runner import TASKS
from scorpio_pipe.ui.outputs_panel import OutputsPanel


DEFAULT_TASK_ORDER = [
    "manifest",
    "superbias",
    "cosmics",
    "superneon",
    "lineid_prepare",
    "wavesolution",
    "qc_report",
]


class RunPlanWidget(QtWidgets.QWidget):
    """Embedded run plan table (same logic as RunPlanDialog)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._cfg: dict[str, Any] | None = None

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        bar = QtWidgets.QHBoxLayout()
        lay.addLayout(bar)
        self.chk_resume = QtWidgets.QCheckBox("Resume (skip finished)")
        self.chk_resume.setChecked(True)
        self.chk_force = QtWidgets.QCheckBox("Force")
        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        bar.addWidget(self.chk_resume)
        bar.addWidget(self.chk_force)
        bar.addStretch(1)
        bar.addWidget(self.btn_refresh)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Task", "Decision", "Reason", "Key outputs"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        lay.addWidget(self.table, 1)

        self.btn_refresh.clicked.connect(self.refresh)
        self.chk_resume.toggled.connect(self.refresh)
        self.chk_force.toggled.connect(self.refresh)

    def set_context(self, cfg: dict[str, Any] | None) -> None:
        self._cfg = cfg
        self.refresh()

    def refresh(self) -> None:
        self.table.setRowCount(0)
        if not self._cfg:
            return

        resume = bool(self.chk_resume.isChecked())
        force = bool(self.chk_force.isChecked())

        ok_icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogApplyButton)
        skip_icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowRight)
        warn_icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning)

        for name in DEFAULT_TASK_ORDER:
            if name not in TASKS:
                continue

            complete = task_is_complete(self._cfg, name)
            will_skip = bool(resume and (not force) and complete)

            ps = products_for_task(self._cfg, name)
            outs = ", ".join([p.key for p in ps]) if ps else "—"

            reason = "products exist" if will_skip else ("forced" if force and complete else "")
            decision = "SKIP" if will_skip else "RUN"

            r = self.table.rowCount()
            self.table.insertRow(r)

            it0 = QtWidgets.QTableWidgetItem(name)
            it1 = QtWidgets.QTableWidgetItem(decision)
            it2 = QtWidgets.QTableWidgetItem(reason or "—")
            it3 = QtWidgets.QTableWidgetItem(outs)

            if will_skip:
                it1.setIcon(skip_icon)
            else:
                it1.setIcon(ok_icon if not complete else warn_icon)

            self.table.setItem(r, 0, it0)
            self.table.setItem(r, 1, it1)
            self.table.setItem(r, 2, it2)
            self.table.setItem(r, 3, it3)

        self.table.resizeColumnsToContents()


class QCAlertsWidget(QtWidgets.QWidget):
    """Compact QC alerts viewer (reads report/qc_report.json)."""

    openRequested = QtCore.Signal(Path)  # should open QC viewer/report

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._work_dir: Path | None = None

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        bar = QtWidgets.QHBoxLayout()
        lay.addLayout(bar)
        self.lbl_title = QtWidgets.QLabel("QC alerts")
        self.lbl_title.setStyleSheet("font-weight: 600;")
        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_open = QtWidgets.QPushButton("Open QC")
        bar.addWidget(self.lbl_title)
        bar.addStretch(1)
        bar.addWidget(self.btn_open)
        bar.addWidget(self.btn_refresh)

        self.lbl_counts = QtWidgets.QLabel("—")
        self.lbl_counts.setWordWrap(True)
        lay.addWidget(self.lbl_counts)

        self.list = QtWidgets.QListWidget()
        self.list.setAlternatingRowColors(True)
        lay.addWidget(self.list, 1)

        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_open.clicked.connect(self._open)
        self.list.itemDoubleClicked.connect(lambda *_: self._open())

    def set_work_dir(self, work_dir: Path | None) -> None:
        self._work_dir = Path(work_dir) if work_dir else None
        self.refresh()

    def refresh(self) -> None:
        self.list.clear()
        if not self._work_dir:
            self.lbl_counts.setText("—")
            return

        qc_json = self._work_dir / "report" / "qc_report.json"
        if not qc_json.exists():
            self.lbl_counts.setText("QC report not built yet")
            return

        try:
            payload = json.loads(qc_json.read_text(encoding="utf-8", errors="replace"))
        except Exception as e:
            self.lbl_counts.setText(f"Failed to read qc_report.json: {type(e).__name__}")
            return

        qc = payload.get("qc", {}) if isinstance(payload, dict) else {}
        counts = qc.get("alert_counts", {}) if isinstance(qc, dict) else {}
        alerts = qc.get("alerts", []) if isinstance(qc, dict) else []

        def _i(k: str) -> int:
            try:
                return int(counts.get(k, 0))
            except Exception:
                return 0

        self.lbl_counts.setText(
            f"bad: {_i('bad')}  |  warn: {_i('warn')}  |  info: {_i('info')}  |  total: {_i('total')}"
        )

        if not isinstance(alerts, list) or not alerts:
            self.list.addItem("(No alerts)")
            return

        # sort by severity, then by message
        sev_rank = {"bad": 0, "warn": 1, "info": 2, "ok": 3}

        def _key(a: dict) -> tuple[int, str]:
            sev = str(a.get("severity", ""))
            return (sev_rank.get(sev.lower(), 99), str(a.get("message", "")))

        for a in sorted([x for x in alerts if isinstance(x, dict)], key=_key):
            sev = str(a.get("severity", ""))
            msg = str(a.get("message", ""))
            stage = str(a.get("stage", ""))
            it = QtWidgets.QListWidgetItem(f"[{sev}] {stage}: {msg}" if stage else f"[{sev}] {msg}")
            it.setData(QtCore.Qt.ItemDataRole.UserRole, sev)
            self.list.addItem(it)

    def _open(self) -> None:
        if self._work_dir:
            self.openRequested.emit(self._work_dir)


class InspectorPanel(QtWidgets.QWidget):
    """Tabbed inspector panel: Outputs / Plan / QC."""

    openQCRequested = QtCore.Signal(Path)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._cfg: dict[str, Any] | None = None
        self._stage: str | None = None

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)

        self.tabs = QtWidgets.QTabWidget()
        lay.addWidget(self.tabs, 1)

        self.outputs = OutputsPanel()
        self.plan = RunPlanWidget()
        self.qc = QCAlertsWidget()
        self.qc.openRequested.connect(self.openQCRequested.emit)

        self.tabs.addTab(self.outputs, "Outputs")
        self.tabs.addTab(self.plan, "Plan")
        self.tabs.addTab(self.qc, "QC")

    def set_context(self, cfg: dict[str, Any] | None, *, stage: str | None = None, work_dir: Path | None = None) -> None:
        self._cfg = cfg
        self._stage = stage
        self.outputs.set_context(cfg, stage=stage)
        self.plan.set_context(cfg)
        self.qc.set_work_dir(work_dir)


class InspectorDock(QtWidgets.QDockWidget):
    """Dock wrapper for InspectorPanel."""

    def __init__(self, parent: QtWidgets.QMainWindow):
        super().__init__("Inspector", parent)
        self.setObjectName("dock_inspector")
        self.setAllowedAreas(QtCore.Qt.RightDockWidgetArea | QtCore.Qt.LeftDockWidgetArea)
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        self.panel = InspectorPanel()
        self.setWidget(self.panel)
