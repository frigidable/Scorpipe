from __future__ import annotations

import re

from PySide6 import QtGui


class LogHighlighter(QtGui.QSyntaxHighlighter):
    """Simple colorizer for the log view (QPlainTextEdit)."""

    _re = re.compile(r"^(?P<lvl>DEBUG|INFO|WARNING|ERROR|CRITICAL):")

    def __init__(self, document: QtGui.QTextDocument):
        super().__init__(document)

        def _fmt(color_name: str, bold: bool = False) -> QtGui.QTextCharFormat:
            fmt = QtGui.QTextCharFormat()
            fmt.setForeground(QtGui.QColor(color_name))
            if bold:
                fmt.setFontWeight(QtGui.QFont.Bold)
            return fmt

        self._fmts = {
            "DEBUG": _fmt("#9aa0a6"),
            "INFO": _fmt("#e8eaed"),
            "WARNING": _fmt("#fbbc04", True),
            "ERROR": _fmt("#ea4335", True),
            "CRITICAL": _fmt("#ff0000", True),
        }

    def highlightBlock(self, text: str) -> None:
        m = self._re.match(text)
        if not m:
            return
        lvl = m.group("lvl")
        fmt = self._fmts.get(lvl)
        if fmt is None:
            return
        self.setFormat(0, len(text), fmt)
