from __future__ import annotations

"""Run plan dialog.

Shows what tasks will run and which will be skipped in resume mode.
"""

from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtWidgets

from scorpio_pipe.products import products_for_task, task_is_complete
from scorpio_pipe.ui.pipeline_runner import TASKS


DEFAULT_TASK_ORDER = [
    "manifest",
    "superbias",
    "cosmics",
    "superneon",
    "lineid_prepare",
    "wavesolution",
    "qc_report",
]


class RunPlanDialog(QtWidgets.QDialog):
    def __init__(self, cfg: dict[str, Any], parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Scorpio Pipe — Run plan")
        self.resize(980, 560)
        self._cfg = cfg

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        bar = QtWidgets.QHBoxLayout()
        lay.addLayout(bar)
        self.chk_resume = QtWidgets.QCheckBox("Resume (skip finished)")
        self.chk_resume.setChecked(True)
        self.chk_force = QtWidgets.QCheckBox("Force (never skip)")
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

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        lay.addWidget(btns)
        btns.rejected.connect(self.reject)

        self.btn_refresh.clicked.connect(self.refresh)
        self.chk_resume.toggled.connect(self.refresh)
        self.chk_force.toggled.connect(self.refresh)

        self.refresh()

    def refresh(self) -> None:
        self.table.setRowCount(0)
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
