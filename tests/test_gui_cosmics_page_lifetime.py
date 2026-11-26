"""Regression: Cosmics lazy page must retain the owning QWidget and layout."""
from __future__ import annotations

import gc
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6 import QtWidgets  # noqa: E402
from shiboken6 import isValid  # noqa: E402

from scorpio_pipe.ui.launcher_window import LauncherWindow  # noqa: E402


def test_cosmics_lazy_page_retains_live_root_widget_and_layout() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = LauncherWindow()
    try:
        assert 3 not in win._pages_built
        win.steps.setCurrentRow(win._stage_row["cosmics"])
        app.processEvents()

        page = win.page_cosmics
        assert win._page_widgets[3] is page
        assert win.stack.currentWidget() is page
        assert 3 in win._pages_built

        # The former local 'for w in (...)' shadowed the page QWidget and
        # deleted its C++ QVBoxLayout before widget construction completed.
        gc.collect()
        assert isValid(page)
        layout = page.layout()
        assert layout is not None
        assert isValid(layout)
        assert layout.count() > 0

        # Navigating away and back must retain a usable page and layout.
        win.steps.setCurrentRow(win._stage_row["project"])
        win.steps.setCurrentRow(win._stage_row["cosmics"])
        app.processEvents()
        assert win.stack.currentWidget() is page
        assert isValid(page.layout())
    finally:
        win.close()
