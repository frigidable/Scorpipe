"""GUI smoke tests for lazy page selection and Qt stylesheet contract."""

from __future__ import annotations

import os

import pytest

# Must be set before the first QApplication is created on headless CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6 import QtWidgets  # noqa: E402

from scorpio_pipe.ui.launcher_window import LauncherWindow  # noqa: E402
from scorpio_pipe.ui.theme import apply_theme  # noqa: E402


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_initial_lazy_project_page_replaces_selected_placeholder() -> None:
    app = _app()
    win = LauncherWindow()
    try:
        # The actual application queues this operation on the event loop.
        win._ensure_page_built(0)
        assert 0 in win._pages_built
        assert win.steps.currentRow() == 0
        assert win.stack.currentWidget() is win.page_project
        assert win.stack.currentWidget() is not win.page_config

        # Replacing an explicitly selected placeholder must keep that page visible.
        win.steps.setCurrentRow(1)
        assert 1 in win._pages_built
        assert win.stack.currentWidget() is win.page_config
        assert win.steps.currentRow() == 1

        win.steps.setCurrentRow(0)
        assert win.stack.currentWidget() is win.page_project
        app.processEvents()
    finally:
        win.close()


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_qss_uses_css_colors_and_not_inline_svg_data_urls(mode: str) -> None:
    app = _app()
    apply_theme(app, mode=mode)
    qss = app.styleSheet()
    assert "color: #" in qss
    assert "color: <" not in qss
    assert "QtGui.QColor" not in qss
    assert "data:image/svg+xml" not in qss
