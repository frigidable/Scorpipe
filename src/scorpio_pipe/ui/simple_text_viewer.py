from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class TextViewerDialog(QtWidgets.QDialog):
    def __init__(
        self, title: str, text: str, *, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 650)
        # User preference: open windows maximized by default.
        try:
            self.setWindowState(self.windowState() | QtCore.Qt.WindowMaximized)
        except Exception:
            pass

        lay = QtWidgets.QVBoxLayout()
        self.setLayout(lay)
        lay.setContentsMargins(12, 12, 12, 12)

        self.editor = QtWidgets.QPlainTextEdit()
        self.editor.setReadOnly(True)
        self.editor.setPlainText(text)
        lay.addWidget(self.editor, 1)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)