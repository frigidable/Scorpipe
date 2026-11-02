"""GUI must never create a role manifest in immutable raw BTA FITS folders."""
from __future__ import annotations

import os
import pytest
import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from scorpio_pipe.ui.project_manifest_dialog import ProjectManifestDialog


def test_manifest_editor_defaults_to_selected_run_not_raw_night(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    raw = tmp_path / "raw" / "s241206"
    run = tmp_path / "runs" / "N2146sat_VPHG1200B_01"
    raw.mkdir(parents=True)
    run.mkdir(parents=True)
    (raw / "s22800607.fts").write_bytes(b"raw-immutable")

    cfg = {"work_dir": str(run), "data_dir": str(raw),
           "config_dir": str(run)}
    dialog = ProjectManifestDialog(cfg=cfg, inspect_df=pd.DataFrame())
    try:
        assert dialog._manifest_path == run / "project_manifest.yaml"
        assert dialog._manifest_path.exists()
        assert not (raw / "project_manifest.yaml").exists()
        assert (raw / "s22800607.fts").read_bytes() == b"raw-immutable"
    finally:
        dialog.close()


def test_explicit_shared_manifest_path_is_respected(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    raw = tmp_path / "raw"
    run = tmp_path / "run"
    shared = tmp_path / "shared" / "night_roles.yaml"
    raw.mkdir()
    run.mkdir()
    cfg = {
        "work_dir": str(run), "data_dir": str(raw),
        "config_dir": str(run),
        "project_manifest_path": str(shared),
    }
    dialog = ProjectManifestDialog(cfg=cfg, inspect_df=pd.DataFrame())
    try:
        assert dialog._manifest_path == shared
        assert not (raw / "project_manifest.yaml").exists()
        assert shared.is_file()
    finally:
        dialog.close()
