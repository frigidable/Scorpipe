"""Label widget with stable single-line ellipsis.

QLabel does not elide long text automatically. For a compact scientific GUI we
need predictable row heights and no layout jumps when labels are long.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


class ElidedLabel(QtWidgets.QLabel):
    """Single-line label that elides with "…" when it does not fit."""

    def __init__(self, text: str = "", parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._full_text = str(text or "")
        self.setText(self._full_text)
        self.setWordWrap(False)
        # Keyboard focus for tooltip access (P1-06).
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def setFullText(self, text: str) -> None:  # noqa: N802 (Qt style)
        self._full_text = str(text or "")
        self.update()

    def fullText(self) -> str:  # noqa: N802
        return self._full_text

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
        fm = QtGui.QFontMetrics(self.font())

        # Leave small breathing room for the focus rectangle.
        rect = self.rect().adjusted(2, 0, -2, 0)
        elided = fm.elidedText(
            self._full_text, QtCore.Qt.TextElideMode.ElideRight, rect.width()
        )
        p.setPen(self.palette().color(QtGui.QPalette.ColorRole.WindowText))
        p.drawText(
            rect,
            int(
                QtCore.Qt.AlignmentFlag.AlignVCenter
                | QtCore.Qt.AlignmentFlag.AlignLeft
            ),
            elided,
        )

        if self.hasFocus():
            opt = QtWidgets.QStyleOptionFocusRect()
            opt.initFrom(self)
            opt.rect = self.rect().adjusted(1, 1, -1, -1)
            opt.backgroundColor = self.palette().color(QtGui.QPalette.ColorRole.Window)
            self.style().drawPrimitive(QtWidgets.QStyle.PrimitiveElement.PE_FrameFocusRect, opt, p, self)

        p.end()
