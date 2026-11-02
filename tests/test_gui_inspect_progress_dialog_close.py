"""Regression: programmatic QProgressDialog closure is not a user cancellation."""
from __future__ import annotations

import os
import threading

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtCore = pytest.importorskip("PySide6.QtCore")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")


def _window_with_open_progress_dialog():
    from scorpio_pipe.ui.launcher_window import LauncherWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = LauncherWindow()
    win._ensure_page_built(0)
    dlg = QtWidgets.QProgressDialog("Inspecting…", "Cancel", 0, 0, win)
    dlg.setMinimumDuration(0)
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)
    dlg.canceled.connect(win._cancel_inspect)
    dlg.show()
    app.processEvents()
    token = threading.Event()
    win._inspect_cancel_event = token
    win._inspect_progress_dialog = dlg
    win.btn_inspect.setText("Cancel Inspect")
    win.btn_inspect.setEnabled(True)
    return app, win, dlg, token


def test_success_closure_does_not_spuriously_enter_cancelling_state():
    app, win, dlg, token = _window_with_open_progress_dialog()
    try:
        win._finish_inspect_ui()
        app.processEvents()
        assert not token.is_set(), (
            "Closing a completed QProgressDialog must not cancel its finished job"
        )
        assert win._inspect_cancel_event is None
        assert win._inspect_progress_dialog is None
        assert win._inspect_job is None
        assert win.btn_inspect.text() == "Inspect dataset"
        assert win.btn_inspect.isEnabled()
        assert not dlg.isVisible()
    finally:
        win.close()


def test_real_cancel_signal_remains_connected_until_finish():
    app, win, dlg, token = _window_with_open_progress_dialog()
    try:
        # Simulates the user choosing the dialog's Cancel action while
        # the scan is running; the signal must still stop the worker.
        dlg.canceled.emit()
        app.processEvents()
        assert token.is_set()
        assert win.btn_inspect.text() == "Cancelling…"
        assert not win.btn_inspect.isEnabled()
        win._finish_inspect_ui()
        app.processEvents()
        assert win.btn_inspect.text() == "Inspect dataset"
        assert win._inspect_cancel_event is None
    finally:
        win.close()
