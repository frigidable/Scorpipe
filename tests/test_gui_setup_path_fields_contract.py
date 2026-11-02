"""Headless structure checks for the real BTA Setup path layout.

Default Linux CI omits the optional PySide6 GUI dependency. The
separate Qt offscreen test tests actual widget sizes on Windows.
"""
from __future__ import annotations

import ast
from pathlib import Path


def _setup_helper() -> ast.FunctionDef:
    p = (Path(__file__).resolve().parents[1]
         / "src" / "scorpio_pipe" / "ui" / "launcher_window.py")
    tree = ast.parse(p.read_text(encoding="utf-8"))
    win = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "LauncherWindow")
    method = next(n for n in win.body
                  if isinstance(n, ast.FunctionDef) and n.name == "_build_page_config")
    return next(n for n in method.body
                if isinstance(n, ast.FunctionDef) and n.name == "_add_full_width_path")


def test_setup_path_is_single_36px_editor_with_matching_browse_button():
    body = ast.unparse(_setup_helper())
    assert "editor.setFixedHeight(36)" in body
    assert "browse.setFixedSize(36, 36)" in body
    assert "editor.setToolTip(editor.text())" in body
    assert "editor.textChanged.connect(editor.setToolTip)" in body
    assert "row.addWidget(editor, 1)" in body
    assert "row.addWidget(browse, 0)" in body


def test_setup_path_does_not_duplicate_readout_label():
    body = ast.unparse(_setup_helper())
    assert "full = QtWidgets.QLabel" not in body
    assert "column.addWidget(full)" not in body
    assert "QLabel" not in body
