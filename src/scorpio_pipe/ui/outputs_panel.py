"""UI widget to show expected products and their existence."""

from __future__ import annotations


from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from scorpio_pipe.products import list_products


class OutputsPanel(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._cfg: dict[str, Any] | None = None
        self._stage: str | None = None

        lay = QtWidgets.QVBoxLayout()
        self.setLayout(lay)
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
                else f"{sz / 1024:.1f} KB"
                if sz < 1024**2
                else f"{sz / 1024**2:.2f} MB",
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


class OutputsToolDialog(QtWidgets.QDialog):
    """Non-modal Outputs viewer that does not affect the main window layout."""

    visibilityChanged = QtCore.Signal(bool)  # type: ignore[attr-defined]

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Outputs")
        # Tool-like window: stays on top of the main window, but does not steal focus.
        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.Tool
            | QtCore.Qt.WindowCloseButtonHint
            | QtCore.Qt.WindowMinimizeButtonHint
        )
        self.setModal(False)
        self.setMinimumSize(420, 520)

        lay = QtWidgets.QVBoxLayout()
        self.setLayout(lay)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        self.panel = OutputsPanel()
        lay.addWidget(self.panel, 1)

    def set_context(self, cfg: dict, stage: str | None) -> None:
        self.panel.set_context(cfg, stage=stage)

    def showEvent(self, e: QtGui.QShowEvent) -> None:  # noqa: N802
        super().showEvent(e)
        try:
            self.visibilityChanged.emit(True)
        except Exception:
            pass

    def hideEvent(self, e: QtGui.QHideEvent) -> None:  # noqa: N802
        super().hideEvent(e)
        try:
            self.visibilityChanged.emit(False)
        except Exception:
            pass

    def closeEvent(self, e: QtGui.QCloseEvent) -> None:  # noqa: N802
        super().closeEvent(e)
        try:
            self.visibilityChanged.emit(False)
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


class OutputsDrawer(QtWidgets.QWidget):
    """A compact, collapsible wrapper around :class:`OutputsPanel`.

    The drawer keeps its width stable to avoid "jumping" layouts when the user
    shows/hides outputs. Outputs are secondary, so the default state is folded.
    """

    def __init__(
        self,
        inner: OutputsPanel | None = None,
        *,
        title: str = "Outputs",
        folded: bool = True,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._folded = bool(folded)

        lay = QtWidgets.QVBoxLayout()
        self.setLayout(lay)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        header = QtWidgets.QHBoxLayout()
        lay.addLayout(header)
        self.btn_toggle = QtWidgets.QToolButton()
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(not self._folded)
        self.btn_toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow
            if not self._folded
            else QtCore.Qt.ArrowType.RightArrow
        )
        self.btn_toggle.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.lbl_title = QtWidgets.QLabel(title)
        self.lbl_title.setStyleSheet("font-weight: 600;")
        header.addWidget(self.btn_toggle)
        header.addWidget(self.lbl_title)
        header.addStretch(1)

        self.inner = inner or OutputsPanel()
        lay.addWidget(self.inner, 1)
        self.inner.setVisible(not self._folded)

        # Keep the panel width stable: fold only the content.
        self.setMinimumWidth(420)
        self.setMaximumWidth(520)

        self.btn_toggle.toggled.connect(self.setFolded)

    def setFolded(self, folded: bool) -> None:  # noqa: N802
        # The toggle is "expanded" when checked.
        _ = folded  # compatibility with Qt slot signature
        self._folded = not self.btn_toggle.isChecked()
        self.inner.setVisible(not self._folded)
        self.btn_toggle.setArrowType(
            QtCore.Qt.ArrowType.RightArrow if self._folded else QtCore.Qt.ArrowType.DownArrow
        )
