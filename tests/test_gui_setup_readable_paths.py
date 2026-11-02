"""Setup layout regression: long BTA workspaces must be visible and selectable."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QtCore = pytest.importorskip("PySide6.QtCore")

from scorpio_pipe.ui.launcher_window import LauncherWindow


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_setup_path_fields_are_full_width_and_have_wrapped_readback():
    app = _app()
    win = LauncherWindow()
    try:
        win.resize(1680, 900)
        win._ensure_page_built(1)
        win.steps.setCurrentRow(1)
        win.show()
        app.processEvents()
        workspace = r"C:\Users\frigi\Documents\PhD\ScorpipeWork"
        run = workspace + r"\06_12_2024\N2146sat_VPHG1200B_01"
        win.edit_workspace_root.setText(workspace)
        win.edit_work_dir.setText(run)
        app.processEvents()

        assert win.edit_workspace_root.width() >= 300
        assert win.edit_work_dir.width() >= 300
        assert win.lbl_workspace_root_full.text() == workspace
        assert win.lbl_run_folder_full.text() == run
        assert win.lbl_run_folder_full.wordWrap()
        assert win.lbl_workspace_root_full.wordWrap()
        assert (win.lbl_run_folder_full.textInteractionFlags()
                & QtCore.Qt.TextSelectableByMouse)
        assert win.edit_work_dir.toolTip() == run
        assert win.lbl_run_folder_full.toolTip() == run
        assert win.btn_init_run.width() >= 142
        assert win.btn_switch_run.width() >= 76
        assert win.btn_open_run.width() >= 74

        # A later auto-suggestion must refresh visible provenance.
        win.edit_work_dir.setText(run.replace("_01", "_02"))
        assert win.lbl_run_folder_full.text().endswith("_02")
    finally:
        win.close()


def test_setup_paths_readback_does_not_change_user_override_state():
    app = _app()
    win = LauncherWindow()
    try:
        win._ensure_page_built(1)
        assert win._workdir_user_edited is False
        win.edit_work_dir.setText(r"C:\scratch\new-run")
        app.processEvents()
        # setText is a programmatic update, never a user edit.
        assert win._workdir_user_edited is False
        assert win.lbl_run_folder_full.text() == r"C:\scratch\new-run"
    finally:
        win.close()
