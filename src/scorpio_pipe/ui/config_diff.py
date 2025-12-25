from __future__ import annotations

import difflib

from PySide6 import QtCore, QtGui, QtWidgets


class ConfigDiffDialog(QtWidgets.QDialog):
    def __init__(self, title: str, old_text: str, new_text: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(980, 640)
        # User preference: open windows maximized by default.
        try:
            self.setWindowState(self.windowState() | QtCore.Qt.WindowMaximized)
        except Exception:
            pass

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        self.txt = QtWidgets.QPlainTextEdit()
        self.txt.setReadOnly(True)
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.txt.setFont(mono)
        lay.addWidget(self.txt, 1)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        lay.addWidget(btns)
        btns.rejected.connect(self.reject)

        diff = difflib.unified_diff(
            old_text.splitlines(),
            new_text.splitlines(),
            fromfile="saved config",
            tofile="current editor",
            lineterm="",
        )
        self.txt.setPlainText("\n".join(diff) or "(no changes)")
