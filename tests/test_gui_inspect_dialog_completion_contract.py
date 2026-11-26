"""Headless contract for inspect progress completion (no Qt dependency).

Qt QProgressDialog.closeEvent() emits canceled(), including when the
application itself calls close(). These tests run the actual production
LauncherWindow methods against minimal signal/dialog doubles in Linux CI,
which intentionally does not install the optional PySide6 extras.
"""
from __future__ import annotations

import ast
from pathlib import Path
from threading import Event


def _method_from_production(name: str):
    src = Path(__file__).resolve().parents[1] / "src" / "scorpio_pipe" / "ui" / "launcher_window.py"
    module = ast.parse(src.read_text(encoding="utf-8"))
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "LauncherWindow")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    isolated = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = {}
    exec(compile(isolated, str(src), "exec"), namespace)
    return namespace[name]


class _Signal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def disconnect(self, callback):
        self.callbacks.remove(callback)

    def emit(self):
        for cb in list(self.callbacks):
            cb()


class _Dialog:
    def __init__(self):
        self.canceled = _Signal()
        self.closed = False
        self.cancel_button_removed = False

    def close(self):
        self.closed = True
        self.canceled.emit()  # Qt QProgressDialog.closeEvent behavior

    def setLabelText(self, value):
        self.label = value

    def setCancelButton(self, value):
        self.cancel_button_removed = value is None


class _Button:
    def __init__(self):
        self.label = "Cancel Inspect"
        self.enabled = True

    def setText(self, label):
        self.label = label

    def setEnabled(self, enabled):
        self.enabled = enabled


class _Label:
    def setText(self, text):
        self.value = text


class _FakeLauncher:
    _finish_inspect_ui = _method_from_production("_finish_inspect_ui")
    _cancel_inspect = _method_from_production("_cancel_inspect")

    def __init__(self):
        self.btn_inspect = _Button()
        self.lbl_inspect = _Label()
        self._inspect_cancel_event = Event()
        self._inspect_progress_dialog = _Dialog()
        self._inspect_job = object()
        self._inspect_progress_dialog.canceled.connect(self._cancel_inspect)
        self.refresh_count = 0

    def _update_enables(self):
        self.refresh_count += 1


def test_successful_qt_dialog_close_does_not_trigger_user_cancel():
    win = _FakeLauncher()
    token = win._inspect_cancel_event
    dlg = win._inspect_progress_dialog
    win._finish_inspect_ui()
    assert dlg.closed
    assert not token.is_set()
    assert win.btn_inspect.label == "Inspect dataset"
    assert win.btn_inspect.enabled
    assert win._inspect_progress_dialog is None
    assert win._inspect_cancel_event is None
    assert win._inspect_job is None
    assert win.refresh_count == 1


def test_real_cancel_signal_still_stops_job_while_active():
    win = _FakeLauncher()
    token = win._inspect_cancel_event
    dlg = win._inspect_progress_dialog
    dlg.canceled.emit()
    assert token.is_set()
    assert win.btn_inspect.label == "Cancelling…"
    assert not win.btn_inspect.enabled
    win._finish_inspect_ui()
    assert win.btn_inspect.label == "Inspect dataset"
    assert win.btn_inspect.enabled
    assert win._inspect_cancel_event is None
