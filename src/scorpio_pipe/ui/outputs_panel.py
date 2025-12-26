from __future__ import annotations

"""UI widget to show expected products and their existence."""

from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from scorpio_pipe.products import group_by_stage, list_products


class OutputsPanel(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._cfg: dict[str, Any] | None = None
        self._stage: str | None = None

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        bar = QtWidgets.QHBoxLayout()
        lay.addLayout(bar)
        self.lbl_title = QtWidgets.QLabel("Outputs")
        self.lbl_title.setStyleSheet("font-weight: 600;")
        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_open_folder = QtWidgets.QPushButton("Open folder")
        bar.addWidget(self.lbl_title)
        bar.addStretch(1)
        bar.addWidget(self.btn_open_folder)
        bar.addWidget(self.btn_refresh)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["", "Product", "Path", "Size"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(False)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents
        )
        self.tree.header().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeToContents
        )
        self.tree.header().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(
            3, QtWidgets.QHeaderView.ResizeToContents
        )
        lay.addWidget(self.tree, 1)

        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_open_folder.clicked.connect(self._open_folder)
        self.tree.itemDoubleClicked.connect(self._open_selected)

    def set_context(self, cfg: dict[str, Any] | None, stage: str | None = None) -> None:
        self._cfg = cfg
        self._stage = stage
        self.lbl_title.setText(f"Outputs — {stage}" if stage else "Outputs")
        self.refresh()

    def refresh(self) -> None:
        self.tree.clear()
        if not self._cfg:
            return

        prods = list_products(self._cfg)
        if self._stage:
            prods = [p for p in prods if p.stage == self._stage]

        ok_icon = self.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_DialogApplyButton
        )
        miss_icon = self.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_DialogCancelButton
        )
        opt_icon = self.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning
        )

        for p in prods:
            it = QtWidgets.QTreeWidgetItem()
            it.setData(0, QtCore.Qt.ItemDataRole.UserRole, str(p.path))
            it.setText(1, p.key)
            it.setText(2, str(p.path))
            sz = p.size()
            it.setText(
                3,
                ""
                if sz is None
                else f"{sz/1024:.1f} KB"
                if sz < 1024**2
                else f"{sz/1024**2:.2f} MB",
            )
            if p.exists():
                it.setIcon(0, ok_icon)
            else:
                it.setIcon(0, opt_icon if p.optional else miss_icon)
            if p.description:
                it.setToolTip(1, p.description)
                it.setToolTip(2, p.description)
            self.tree.addTopLevelItem(it)

    def _open_folder(self) -> None:
        if not self._cfg:
            return
        try:
            wd = Path(str(self._cfg.get("work_dir", ""))).expanduser()
            if wd.exists():
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(wd)))
        except Exception:
            pass

    def _open_selected(self, item: QtWidgets.QTreeWidgetItem) -> None:
        try:
            p = Path(str(item.data(0, QtCore.Qt.ItemDataRole.UserRole) or ""))
            if p.exists():
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(p)))
            else:
                # open parent folder when file missing but parent exists
                if p.parent.exists():
                    QtGui.QDesktopServices.openUrl(
                        QtCore.QUrl.fromLocalFile(str(p.parent))
                    )
        except Exception:
            pass
