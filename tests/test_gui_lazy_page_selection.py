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
        # The selected Project page must already be real BEFORE the event
        # loop starts (the previous queued build could leave Setup visible).
        assert 0 in win._pages_built
        assert win.steps.currentRow() == 0
        assert win.stack.currentWidget() is win.page_project
        assert win.stack.currentWidget() is not win.page_config
        assert "Loading" not in win.stack.currentWidget().findChild(
            QtWidgets.QLabel
        ).text() if win.stack.currentWidget().findChild(QtWidgets.QLabel) else True

        win.show()
        app.processEvents()
        assert win.stack.currentWidget() is win.page_project
        assert win.steps.currentRow() == 0

        # Replacing an explicitly selected placeholder must keep that page visible.
        win.steps.setCurrentRow(1)
        assert 1 in win._pages_built
        assert win.stack.currentWidget() is win.page_config
        assert win.steps.currentRow() == 1

        win.steps.setCurrentRow(0)
        assert win.stack.currentWidget() is win.page_project
        app.processEvents()

        # Test the non-sequential navigation path, including swapped stage
        # indices (stage "Flat-Fielding" uses page 4 rather than page 3).
        for row in (3, 2, 4, 0, 1, 0):
            win.steps.setCurrentRow(row)
            app.processEvents()
            page_idx = win._stage_page_index[row]
            assert win.stack.currentWidget() is win._page_widgets[page_idx]
            assert page_idx in win._pages_built
        assert win.stack.currentWidget() is win.page_project
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
