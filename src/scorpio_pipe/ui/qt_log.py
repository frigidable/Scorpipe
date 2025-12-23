from __future__ import annotations

import logging

from PySide6 import QtCore


class QtLogEmitter(QtCore.QObject):
    """Thread-safe bridge from logging -> Qt UI."""

    message = QtCore.Signal(str)


class QtLogHandler(logging.Handler):
    """A logging handler that emits formatted records via Qt signals."""

    def __init__(self, emitter: QtLogEmitter):
        super().__init__()
        self._emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        # Qt will queue cross-thread signal emissions automatically.
        self._emitter.message.emit(msg)


def install(
    text_edit,
    *,
    logger_name: str = "scorpio",
    level: int = logging.INFO,
) -> QtLogEmitter:
    """Attach a Qt log bridge to a QPlainTextEdit.

    The pipeline uses the standard :mod:`logging` module. This helper routes
    records into the UI in a thread-safe way.
    """
    emitter = QtLogEmitter()

    def _append(msg: str) -> None:
        # keep UI responsive
        text_edit.appendPlainText(msg)

    emitter.message.connect(_append)

    h = QtLogHandler(emitter)
    # Mark the handler so we can de-duplicate cleanly on re-install.
    setattr(h, "_scorpio_qt_log", True)
    fmt = logging.Formatter("%(levelname)s: %(message)s")
    h.setFormatter(fmt)
    h.setLevel(level)

    # De-dup: if the UI is re-created, avoid stacking multiple Qt handlers.
    root = logging.getLogger()
    for hh in list(root.handlers):
        if isinstance(hh, QtLogHandler) or bool(getattr(hh, "_scorpio_qt_log", False)):
            root.removeHandler(hh)

    # In the UI build we don't necessarily call `setup_logging()`, so route
    # everything through root. Keep the named logger propagating to root.
    log = logging.getLogger(logger_name)
    log.setLevel(level)
    log.propagate = True
    for hh in list(log.handlers):
        if isinstance(hh, QtLogHandler) or bool(getattr(hh, "_scorpio_qt_log", False)):
            log.removeHandler(hh)

    # Ensure root level is permissive enough for the UI.
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)
    root.addHandler(h)

    return emitter
