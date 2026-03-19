"""Run plan dialog.

Shows what tasks will run and which will be skipped in resume mode.
"""

from __future__ import annotations


from typing import Any
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from scorpio_pipe.products import products_for_task
from scorpio_pipe.pipeline.public_api import TASKS
from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.ui.slow_action_profiler import profile_ui_action
from scorpio_pipe.ui.session_store import get_ui_run_cache


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
    def __init__(self, cfg: dict[str, Any], parent: QtWidgets.QWidget | None = None, *, config_path: Path | None = None):
        super().__init__(parent)
        self.setWindowTitle("Scorpio Pipe — Run plan")
        self.resize(980, 560)
        # User preference: open windows maximized by default.
        try:
            self.setWindowState(self.windowState() | QtCore.Qt.WindowMaximized)
        except Exception:
            pass
        self._cfg = cfg
        self._config_path = Path(config_path).expanduser().resolve() if config_path else None

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
        self.table.setHorizontalHeaderLabels(
            ["Task", "Decision", "Reason", "Key outputs"]
        )
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
        work_dir = None
        try:
            work_dir = resolve_work_dir(self._cfg)
        except Exception:
            work_dir = None

        with profile_ui_action(
            "ui.run_plan.refresh",
            work_dir=work_dir,
            extra={
                "resume": bool(self.chk_resume.isChecked()),
                "force": bool(self.chk_force.isChecked()),
            },
        ):
            self.table.setRowCount(0)
            resume = bool(self.chk_resume.isChecked())
            force = bool(self.chk_force.isChecked())

            ok_icon = self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogApplyButton
            )
            skip_icon = self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_ArrowRight
            )
            warn_icon = self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning
            )

            tasks = [t for t in DEFAULT_TASK_ORDER if t in TASKS]

            cache = None
            plan_items = []
            status_info: dict[str, Any] = {}

            if work_dir is not None:
                try:
                    cache = get_ui_run_cache(work_dir)
                except Exception:
                    cache = None

            if cache is not None:
                try:
                    plan_items = list(
                        cache.plan_sequence_cached(
                            self._cfg,
                            tasks,
                            resume=resume,
                            force=force,
                            config_path=self._config_path,
                        )
                    )
                    status_info = cache.get_stage_statuses()
                except Exception:
                    plan_items = []
                    status_info = {}

            if not plan_items:
                # Fallback (no cache / cache failed): compute directly.
                try:
                    from scorpio_pipe.pipeline.public_api import plan_sequence as _plan_sequence  # noqa: PLC0415

                    plan_items = list(
                        _plan_sequence(
                            self._cfg,
                            tasks,
                            resume=resume,
                            force=force,
                            config_path=self._config_path,
                        )
                    )
                except Exception:
                    plan_items = []

            for pi in plan_items:
                name = str(getattr(pi, "task", "") or "")
                action = str(getattr(pi, "action", "run") or "run").strip().lower()
                reason = str(getattr(pi, "reason", "") or "")

                ps = products_for_task(self._cfg, name)
                outs = ", ".join([p.key for p in ps]) if ps else "—"

                info = status_info.get(name) if isinstance(status_info, dict) else None
                st = None
                try:
                    if isinstance(info, dict):
                        st = str(info.get("status") or "").strip().lower()
                except Exception:
                    st = None

                decision = "SKIP" if action == "skip" else "RUN"

                r = self.table.rowCount()
                self.table.insertRow(r)

                it0 = QtWidgets.QTableWidgetItem(name)
                it1 = QtWidgets.QTableWidgetItem(decision)
                it2 = QtWidgets.QTableWidgetItem(reason or "—")
                it3 = QtWidgets.QTableWidgetItem(outs)

                if action == "skip":
                    it1.setIcon(skip_icon)
                else:
                    # If stage previously succeeded but needs rerun (dirty/forced),
                    # hint that it's not a "fresh" run.
                    if st in {"ok", "warn"} and ("dirty" in reason.lower() or "forced" in reason.lower()):
                        it1.setIcon(warn_icon)
                    elif st in {"fail", "blocked", "cancelled"}:
                        it1.setIcon(warn_icon)
                    else:
                        it1.setIcon(ok_icon)

                self.table.setItem(r, 0, it0)
                self.table.setItem(r, 1, it1)
                self.table.setItem(r, 2, it2)
                self.table.setItem(r, 3, it3)

            self.table.resizeColumnsToContents()
