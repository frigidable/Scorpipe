"""Project manifest editor (project_manifest.yaml).

This dialog is intentionally simple:
  - left side: inspected frames browser (filters + preview)
  - right side: role lists (OBJECT/SKY/ARCS/FLATS/BIAS/SUNSKY)

The manifest is the *source of truth* for frame roles. Header-based inference is
helpful, but never mandatory for the user.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

try:
    from scorpio_pipe.project_manifest import (
        ProjectManifestModel,
        read_project_manifest,
        resolve_project_manifest_path,
        write_project_manifest,
    )
except ImportError:  # pragma: no cover
    # Backward-compat / frozen builds may expose ProjectManifest alias
    from scorpio_pipe.project_manifest import (
        ProjectManifest as ProjectManifestModel,
        read_project_manifest,
        resolve_project_manifest_path,
        write_project_manifest,
    )
from scorpio_pipe.ui.frame_browser import FrameBrowser


class ProjectManifestDialog(QtWidgets.QDialog):
    """Edit ``project_manifest.yaml`` located in the work directory."""

    manifestSaved = QtCore.Signal(object)  # Path

    def __init__(
        self,
        *,
        cfg: dict[str, Any],
        inspect_df: pd.DataFrame,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Project manifest")
        self.setModal(True)
        self.resize(1180, 720)

        self._cfg = cfg
        self._inspect_df = inspect_df.copy() if inspect_df is not None else pd.DataFrame()

        # Resolve manifest path.
        work_dir = Path(str(cfg.get("work_dir", "work"))).expanduser()
        base_dir = Path(str(cfg.get("config_dir", "."))).expanduser().resolve()
        if not work_dir.is_absolute():
            work_dir = (base_dir / work_dir).resolve()
        self._manifest_path = resolve_project_manifest_path(work_dir)

        self._pm = read_project_manifest(self._manifest_path)
        self._yaml_dirty = False

        root = QtWidgets.QVBoxLayout(self)
        # --- top info bar ---
        info = QtWidgets.QHBoxLayout()
        root.addLayout(info)
        self.lbl_path = QtWidgets.QLabel(str(self._manifest_path))
        self.lbl_path.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        info.addWidget(QtWidgets.QLabel("File:"))
        info.addWidget(self.lbl_path, 1)
        self.btn_open_folder = QtWidgets.QPushButton("Open folder")
        info.addWidget(self.btn_open_folder)

        # --- main split ---
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        # Left: inspected frames
        self.browser = FrameBrowser()
        self.browser.set_frames_df(self._inspect_df)
        # Dialog expects multi-select.
        self.browser.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        splitter.addWidget(self.browser)

        # Right: roles + YAML
        right = QtWidgets.QWidget()
        splitter.addWidget(right)
        rlay = QtWidgets.QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.setSpacing(8)

        self.tabs = QtWidgets.QTabWidget()
        rlay.addWidget(self.tabs, 1)

        # Role tabs
        self._role_lists: dict[str, QtWidgets.QListWidget] = {}
        self._add_role_tab("OBJECT_FRAMES", "Object")
        self._add_role_tab("SKY_FRAMES", "Sky")
        self._add_role_tab("SUNSKY_FRAMES", "Sunsky")
        self._add_role_tab("ARCS", "Arcs (Ne/Ar/He)")
        self._add_role_tab("FLATS", "Flats")
        self._add_role_tab("BIAS", "Bias")

        # YAML tab (advanced)
        tab_yaml = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab_yaml)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)
        self.edit_yaml = QtWidgets.QPlainTextEdit()
        self.edit_yaml.setPlaceholderText("YAML (advanced)\n\nYou may define globs and groups here.\n")
        self.edit_yaml.textChanged.connect(self._on_yaml_edited)
        v.addWidget(self.edit_yaml, 1)
        self.btn_reload_yaml = QtWidgets.QPushButton("Reload from disk")
        v.addWidget(self.btn_reload_yaml)
        self.tabs.addTab(tab_yaml, "YAML")

        # --- action buttons ---
        actions = QtWidgets.QHBoxLayout()
        root.addLayout(actions)
        self.btn_autofill = QtWidgets.QPushButton("Auto-fill by kind")
        self.btn_save = QtWidgets.QPushButton("Save")
        self.btn_save.setProperty("primary", True)
        self.btn_close = QtWidgets.QPushButton("Close")
        actions.addWidget(self.btn_autofill)
        actions.addStretch(1)
        actions.addWidget(self.btn_save)
        actions.addWidget(self.btn_close)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([760, 420])

        # signals
        self.btn_close.clicked.connect(self.close)
        self.btn_save.clicked.connect(self._save)
        self.btn_autofill.clicked.connect(self._autofill)
        self.btn_open_folder.clicked.connect(self._open_folder)
        self.btn_reload_yaml.clicked.connect(self._reload_yaml_from_disk)

        self._load_into_widgets()

    # ---------------------------- UI builders ----------------------------

    def _add_role_tab(self, role_key: str, title: str) -> None:
        tab = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(tab)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        lst = QtWidgets.QListWidget()
        lst.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        lst.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        lst.setDefaultDropAction(QtCore.Qt.MoveAction)
        lst.setAlternatingRowColors(True)
        self._role_lists[role_key] = lst
        lay.addWidget(lst, 1)

        bar = QtWidgets.QHBoxLayout()
        lay.addLayout(bar)
        btn_add = QtWidgets.QPushButton("Add selected ←")
        btn_remove = QtWidgets.QPushButton("Remove")
        btn_clear = QtWidgets.QPushButton("Clear")
        bar.addWidget(btn_add)
        bar.addWidget(btn_remove)
        bar.addWidget(btn_clear)
        bar.addStretch(1)

        btn_add.clicked.connect(lambda *_: self._add_selected_to_role(role_key))
        btn_remove.clicked.connect(lambda *_: self._remove_selected_from_role(role_key))
        btn_clear.clicked.connect(lambda *_: self._clear_role(role_key))

        self.tabs.addTab(tab, title)

    # ---------------------------- data sync ----------------------------

    def _load_into_widgets(self) -> None:
        # fill role lists from manifest
        pm = self._pm
        mapping = {
            "OBJECT_FRAMES": pm.object_frames,
            "SKY_FRAMES": pm.sky_frames,
            "SUNSKY_FRAMES": pm.sunsky_frames,
            "ARCS": pm.arcs,
            "FLATS": pm.flats,
            "BIAS": pm.bias,
        }
        for k, items in mapping.items():
            lst = self._role_lists.get(k)
            if lst is None:
                continue
            lst.clear()
            for p in items:
                lst.addItem(str(p))

        # YAML view
        try:
            if self._manifest_path.exists():
                txt = self._manifest_path.read_text(encoding="utf-8", errors="ignore")
            else:
                # Render current model as YAML
                txt = yaml.safe_dump(self._pm.dict(), sort_keys=False, allow_unicode=True)
            self.edit_yaml.blockSignals(True)
            self.edit_yaml.setPlainText(txt)
            self.edit_yaml.blockSignals(False)
            self._yaml_dirty = False
        except Exception:
            pass

    def _collect_from_widgets(self) -> ProjectManifestModel:
        def _list_items(lst: QtWidgets.QListWidget) -> list[str]:
            return [lst.item(i).text().strip() for i in range(lst.count()) if lst.item(i).text().strip()]

        pm = ProjectManifestModel(
            object_frames=_list_items(self._role_lists["OBJECT_FRAMES"]),
            sky_frames=_list_items(self._role_lists["SKY_FRAMES"]),
            sunsky_frames=_list_items(self._role_lists["SUNSKY_FRAMES"]),
            arcs=_list_items(self._role_lists["ARCS"]),
            flats=_list_items(self._role_lists["FLATS"]),
            bias=_list_items(self._role_lists["BIAS"]),
        )
        return pm

    # ---------------------------- role editing ----------------------------

    def _add_selected_to_role(self, role_key: str) -> None:
        lst = self._role_lists.get(role_key)
        if lst is None:
            return
        sel = self.browser.selected_frames()
        if not sel:
            return
        existing = {lst.item(i).text().strip() for i in range(lst.count())}
        for fr in sel:
            s = str(fr.path)
            if s not in existing:
                lst.addItem(s)
                existing.add(s)

    def _remove_selected_from_role(self, role_key: str) -> None:
        lst = self._role_lists.get(role_key)
        if lst is None:
            return
        for it in list(lst.selectedItems()):
            row = lst.row(it)
            lst.takeItem(row)

    def _clear_role(self, role_key: str) -> None:
        lst = self._role_lists.get(role_key)
        if lst is None:
            return
        lst.clear()

    # ---------------------------- actions ----------------------------

    def _autofill(self) -> None:
        """Populate lists based on the inspection ``kind`` column.

        This never guesses SKY frames – they remain a user decision.
        """
        df = self._inspect_df
        if df is None or df.empty or "kind" not in df.columns or "path" not in df.columns:
            return

        # Keep SKY as-is.
        cur_sky = [
            self._role_lists["SKY_FRAMES"].item(i).text().strip()
            for i in range(self._role_lists["SKY_FRAMES"].count())
        ]

        pm = ProjectManifestModel(
            object_frames=df.loc[df["kind"].astype(str) == "obj", "path"].astype(str).tolist(),
            sky_frames=cur_sky,
            sunsky_frames=df.loc[df["kind"].astype(str) == "sunsky", "path"].astype(str).tolist(),
            arcs=df.loc[df["kind"].astype(str) == "neon", "path"].astype(str).tolist(),
            flats=df.loc[df["kind"].astype(str) == "flat", "path"].astype(str).tolist(),
            bias=df.loc[df["kind"].astype(str) == "bias", "path"].astype(str).tolist(),
        )
        self._pm = pm
        self._load_into_widgets()

    def _on_yaml_edited(self) -> None:
        self._yaml_dirty = True

    def _reload_yaml_from_disk(self) -> None:
        self._pm = read_project_manifest(self._manifest_path)
        self._load_into_widgets()

    def _open_folder(self) -> None:
        try:
            QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(self._manifest_path.parent.resolve()))
            )
        except Exception:
            pass

    def _save(self) -> None:
        # If YAML was edited, parse it as the source of truth.
        if self._yaml_dirty:
            try:
                raw = self.edit_yaml.toPlainText()
                obj = yaml.safe_load(raw) or {}
                self._pm = ProjectManifestModel.model_validate(obj)
                self._yaml_dirty = False
                # also refresh role tabs
                self._load_into_widgets()
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Invalid YAML",
                    f"Cannot parse YAML into ProjectManifestModel:\n\n{e}",
                )
                return
        else:
            self._pm = self._collect_from_widgets()

        try:
            write_project_manifest(self._pm, self._manifest_path)
            self.manifestSaved.emit(self._manifest_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Save failed",
                f"Cannot write manifest:\n\n{e}",
            )
            return

        QtWidgets.QMessageBox.information(
            self,
            "Saved",
            f"Manifest saved:\n{self._manifest_path}",
        )
