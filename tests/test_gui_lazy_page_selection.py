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
        assert win.page_project.objectName() == win.stack.currentWidget().objectName()
        assert win.page_project.layout() is not None

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


def test_use_setup_from_frame_materializes_setup_without_prior_navigation() -> None:
    from types import SimpleNamespace

    app = _app()
    win = LauncherWindow()
    try:
        assert 1 not in win._pages_built
        assert not hasattr(win, "combo_object")
        win._use_setup_from_frame(SimpleNamespace(
            object="N2146sat", disperser="VPHG1200B",
            slit="1.2", binning="1x2",
        ))
        app.processEvents()
        assert 1 in win._pages_built
        assert win.steps.currentRow() == 1
        assert win.stack.currentWidget() is win.page_config
        assert win.combo_object.currentText() == "N2146sat"
        assert win.combo_disperser.currentText() == "VPHG1200B"
        assert win.combo_slit.currentText() == "1.2"
        assert win.combo_binning.currentText() == "1x2"
    finally:
        win.close()


def test_overview_doubleclick_initializes_setup_before_setting_object() -> None:
    app = _app()
    win = LauncherWindow()
    try:
        assert 1 not in win._pages_built
        item = QtWidgets.QListWidgetItem("N2146sat  (11)")
        item.setData(256, "N2146sat")
        win._jump_to_object_from_overview(item)
        app.processEvents()
        assert win.steps.currentRow() == 1
        assert win.stack.currentWidget() is win.page_config
        assert win.combo_object.currentText() == "N2146sat"
    finally:
        win.close()


def test_night_inspect_before_setup_is_deferred_and_populates_on_first_visit() -> None:
    """Reproduce BTA field case: 306 FITS inspected with only Project built."""
    from types import SimpleNamespace
    import pandas as pd

    app = _app()
    win = LauncherWindow()
    try:
        assert 1 not in win._pages_built
        assert not hasattr(win, "combo_object")
        win._inspect = SimpleNamespace(
            objects=["N2146sat"],
            table=pd.DataFrame([
                {
                    "kind": "obj", "object": "N2146sat",
                    "object_norm": "N2146SAT", "disperser": "VPHG1200B",
                    "slit": "1.2", "binning": "1x2",
                    "instrument": "SCORPIO-1", "exptime": 1200.,
                },
                {
                    "kind": "neon", "object": "N2146sat",
                    "object_norm": "N2146SAT", "disperser": "VPHG1200B",
                    "slit": "1.2", "binning": "1x2",
                    "instrument": "SCORPIO-1", "exptime": 60.,
                },
            ]),
        )

        # Previously raised AttributeError: no combo_object, causing the
        # GUI to mark a successfully finished Inspect job as failed.
        win._populate_objects_from_inspect()
        assert 1 not in win._pages_built
        assert not hasattr(win, "combo_object")
        assert win.stack.currentWidget() is win.page_project

        # The same InspectResult must become available when Setup is built.
        win.steps.setCurrentRow(1)
        app.processEvents()
        assert win.stack.currentWidget() is win.page_config
        assert win.combo_object.currentText() == "N2146sat"
        assert win.combo_disperser.currentText() == "VPHG1200B"
        assert win.combo_slit.currentText() == "1.2"
        assert win.combo_binning.currentText() == "1x2"
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
