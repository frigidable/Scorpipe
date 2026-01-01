"""Delayed tooltips for dense scientific UI.

Qt shows tooltips immediately on hover by default, which creates visual noise
in parameter-heavy panels. Scorpipe uses a short, predictable delay so the UI
feels stable and does not "flicker" when the pointer moves across labels.

The filter implemented here:

* shows the tooltip after a configurable *show* delay (default: 200 ms),
* hides it after a configurable *hide* delay (default: 200 ms),
* supports both mouse hover and keyboard focus.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


class DelayedTooltipFilter(QtCore.QObject):
    """Event filter that shows a tooltip with a delay.

    Attach it to a widget by calling :func:`install_delayed_tooltip`.
    """

    def __init__(
        self,
        widget: QtWidgets.QWidget,
        text: str,
        *,
        show_delay_ms: int = 200,
        hide_delay_ms: int = 200,
    ) -> None:
        super().__init__(widget)
        self._w = widget
        self._text = text or ""
        self._show_ms = int(show_delay_ms)
        self._hide_ms = int(hide_delay_ms)

        self._t_show = QtCore.QTimer(self)
        self._t_show.setSingleShot(True)
        self._t_show.timeout.connect(self._do_show)

        self._t_hide = QtCore.QTimer(self)
        self._t_hide.setSingleShot(True)
        self._t_hide.timeout.connect(self._do_hide)

    def setText(self, text: str) -> None:  # noqa: N802 (Qt style)
        self._text = text or ""

    def eventFilter(self, obj: QtCore.QObject, ev: QtCore.QEvent) -> bool:
        if obj is not self._w:
            return False

        t = ev.type()
        if t in (QtCore.QEvent.Type.Enter, QtCore.QEvent.Type.HoverEnter, QtCore.QEvent.Type.FocusIn):
            self._t_hide.stop()
            self._t_show.start(self._show_ms)
        elif t in (
            QtCore.QEvent.Type.Leave,
            QtCore.QEvent.Type.HoverLeave,
            QtCore.QEvent.Type.FocusOut,
        ):
            self._t_show.stop()
            self._t_hide.start(self._hide_ms)
        elif t in (QtCore.QEvent.Type.MouseButtonPress,):
            # Never keep a tooltip open while interacting.
            self._t_show.stop()
            self._do_hide()
        return False

    def _do_show(self) -> None:
        if not self._text:
            return
        try:
            # Use the current cursor position for mouse hover, otherwise anchor
            # to the widget for keyboard focus.
            pos = QtGui.QCursor.pos()
            if self._w.hasFocus():
                try:
                    pos = self._w.mapToGlobal(self._w.rect().bottomLeft())
                except Exception:
                    pass
            QtWidgets.QToolTip.showText(pos, self._text, self._w)
        except Exception:
            pass

    def _do_hide(self) -> None:
        try:
            QtWidgets.QToolTip.hideText()
        except Exception:
            pass


def install_delayed_tooltip(
    widget: QtWidgets.QWidget,
    text: str,
    *,
    show_delay_ms: int = 200,
    hide_delay_ms: int = 200,
) -> DelayedTooltipFilter:
    """Install delayed tooltip behaviour on the widget.

    Returns the created filter instance (kept alive by QObject parenting).
    """

    w = widget
    w.setToolTip("")  # avoid immediate default Qt tooltip behaviour
    filt = DelayedTooltipFilter(
        w, text, show_delay_ms=show_delay_ms, hide_delay_ms=hide_delay_ms
    )
    try:
        w.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
    except Exception:
        pass
    w.installEventFilter(filt)
    return filt
