from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

from scorpio_pipe.ui.log_highlighter import LogHighlighter

from PySide6 import QtCore, QtGui


class QtLogEmitter(QtCore.QObject):
    """Thread-safe bridge from logging -> Qt UI."""

    # Payload is either a plain string (legacy) or a tuple: (levelno, formatted_msg)
    message = QtCore.Signal(object)


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
        try:
            lvl = int(getattr(record, "levelno", 0) or 0)
        except Exception:
            lvl = 0
        self._emitter.message.emit((lvl, msg))


# A small, thread-safe backlog counter used by the UI freeze watchdog.
_BACKLOG_LOCK = threading.Lock()
_BACKLOG_LEN = 0
_BACKLOG_CHARS = 0


def get_backlog_len() -> int:
    """Return how many log lines are currently queued for UI flush."""

    with _BACKLOG_LOCK:
        return int(_BACKLOG_LEN)


def get_backlog_snapshot() -> dict[str, int]:
    """Return lightweight queued-log diagnostics for freeze dumps."""

    with _BACKLOG_LOCK:
        return {"lines": int(_BACKLOG_LEN), "chars": int(_BACKLOG_CHARS)}


def install(
    text_edit,
    *,
    logger_name: str = "scorpio",
    level: int = logging.INFO,
) -> QtLogEmitter:
    """Attach a Qt log bridge to a QPlainTextEdit.

    The pipeline uses the standard :mod:`logging` module. This helper routes
    records into the UI in a thread-safe way.

    PERF-10.5: includes basic backpressure (drop/collapse) and a UI level filter.
    """

    emitter = QtLogEmitter()

    # Batch log updates to avoid UI stalls on high-volume logging.
    try:
        # Keep the last N lines to prevent QPlainTextEdit slowdowns.
        text_edit.setMaximumBlockCount(5000)
    except Exception:
        pass

    # Backpressure knobs (lines, not characters).
    drop_threshold = int(os.getenv("SCORPIPE_UI_LOG_BACKLOG_DROP", "10000"))
    hard_threshold = int(os.getenv("SCORPIPE_UI_LOG_BACKLOG_HARD", "25000"))
    notice_cooldown_s = float(os.getenv("SCORPIPE_UI_LOG_BACKLOG_NOTICE_S", "0.5"))

    # UI filter: show messages with level >= min_level.
    # Default: INFO (hides DEBUG by default).
    try:
        setattr(text_edit, "_scorpipe_log_min_level", int(max(logging.INFO, int(level))))
    except Exception:
        pass

    _queue: list[str] = []
    _timer = QtCore.QTimer(text_edit)
    _timer.setSingleShot(True)

    skipped_debug = 0
    skipped_info = 0
    last_notice_t = 0.0
    _notice_timer = QtCore.QTimer(text_edit)
    _notice_timer.setSingleShot(True)

    def _emit_skip_notice() -> None:
        nonlocal skipped_debug, skipped_info, last_notice_t
        if skipped_debug <= 0 and skipped_info <= 0:
            return

        parts: list[str] = []
        if skipped_debug:
            parts.append(f"{skipped_debug} DEBUG")
        if skipped_info:
            parts.append(f"{skipped_info} INFO")

        skipped_debug = 0
        skipped_info = 0
        last_notice_t = time.monotonic()

        # Keep it in the same "INFO:" prefix format so the highlighter is happy.
        notice = "INFO: [ui-log] backpressure: " + ", ".join(parts) + " lines skipped"
        _queue.append(notice)

        global _BACKLOG_LEN, _BACKLOG_CHARS
        with _BACKLOG_LOCK:
            _BACKLOG_LEN = len(_queue)
            _BACKLOG_CHARS = int(_BACKLOG_CHARS) + len(notice)
        if not _timer.isActive():
            _timer.start(40)

    def _note_skip(levelno: int) -> None:
        nonlocal skipped_debug, skipped_info, last_notice_t
        if levelno < logging.INFO:
            skipped_debug += 1
        else:
            skipped_info += 1

        now = time.monotonic()
        if (now - last_notice_t) >= notice_cooldown_s and not _notice_timer.isActive():
            _notice_timer.start(int(max(1.0, notice_cooldown_s * 1000.0)))

    def _flush() -> None:
        global _BACKLOG_LEN, _BACKLOG_CHARS
        try:
            if not _queue:
                return
            # Flush in chunks to keep the event loop responsive.
            chunk = _queue[:200]
            del _queue[:200]
            chunk_chars = int(sum(len(str(x)) for x in chunk))
            with _BACKLOG_LOCK:
                _BACKLOG_LEN = len(_queue)
                _BACKLOG_CHARS = max(0, int(_BACKLOG_CHARS) - chunk_chars)
            text_edit.appendPlainText("\n".join(chunk))
            text_edit.ensureCursorVisible()
        finally:
            if _queue:
                _timer.start(40)

    def _enqueue(payload: Any) -> None:
        global _BACKLOG_LEN, _BACKLOG_CHARS

        # Payload can be legacy string or (levelno, msg)
        levelno: int
        msg: str
        if isinstance(payload, tuple) and len(payload) == 2:
            try:
                levelno = int(payload[0])
            except Exception:
                levelno = int(level)
            msg = str(payload[1])
        else:
            levelno = int(level)
            msg = str(payload)

        # UI level filter (min-level).
        try:
            min_level = int(getattr(text_edit, "_scorpipe_log_min_level", level))
        except Exception:
            min_level = int(level)
        if min_level > 0 and levelno < min_level:
            return

        # Backpressure:
        # - if backlog is large: drop DEBUG first.
        # - if backlog is huge: drop INFO as well.
        qlen = len(_queue)
        if qlen >= hard_threshold:
            if levelno < logging.WARNING:
                _note_skip(levelno)
                return
        elif qlen >= drop_threshold:
            if levelno < logging.INFO:
                _note_skip(levelno)
                return

        _queue.append(msg)
        with _BACKLOG_LOCK:
            _BACKLOG_LEN = len(_queue)
            _BACKLOG_CHARS = int(_BACKLOG_CHARS) + len(msg)
        if not _timer.isActive():
            _timer.start(40)

    _timer.timeout.connect(_flush)
    emitter.message.connect(_enqueue)
    _notice_timer.timeout.connect(_emit_skip_notice)

    # Colorize log levels (line-based)
    try:
        text_edit._scorpipe_log_highlighter = LogHighlighter(text_edit.document())
    except Exception:
        pass

    # Context menu: lightweight UI level filter.
    # Applies to new lines (we don't keep a full structured log history).
    try:
        text_edit.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

        def _show_menu(pos) -> None:
            menu = text_edit.createStandardContextMenu()
            menu.addSeparator()
            sub = menu.addMenu("Log level filter")
            group = QtGui.QActionGroup(sub)
            group.setExclusive(True)

            def _add(label: str, lvl: int) -> None:
                act = sub.addAction(label)
                act.setCheckable(True)
                act.setData(int(lvl))
                group.addAction(act)
                cur = int(getattr(text_edit, "_scorpipe_log_min_level", level))
                act.setChecked(cur == int(lvl))

            _add("INFO", logging.INFO)
            _add("WARN", logging.WARNING)
            _add("ERROR", logging.ERROR)

            def _on_pick(action) -> None:
                try:
                    new_level = int(action.data())
                except Exception:
                    new_level = int(max(logging.INFO, int(level)))
                try:
                    setattr(text_edit, "_scorpipe_log_min_level", int(new_level))
                except Exception:
                    pass
                # Also set handler level so low-level records stop even emitting into Qt.
                try:
                    hh = getattr(text_edit, "_scorpipe_qt_log_handler", None)
                    if hh is not None:
                        hh.setLevel(int(new_level))
                except Exception:
                    pass

            group.triggered.connect(_on_pick)
            menu.exec(text_edit.mapToGlobal(pos))

        text_edit.customContextMenuRequested.connect(_show_menu)
    except Exception:
        pass

    h = QtLogHandler(emitter)
    # Mark the handler so we can de-duplicate cleanly on re-install.
    setattr(h, "_scorpio_qt_log", True)
    fmt = logging.Formatter("%(levelname)s: %(message)s")
    h.setFormatter(fmt)
    # Start with current UI filter level.
    try:
        h.setLevel(int(getattr(text_edit, "_scorpipe_log_min_level", level)))
    except Exception:
        h.setLevel(level)
    try:
        setattr(text_edit, "_scorpipe_qt_log_handler", h)
    except Exception:
        pass

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
