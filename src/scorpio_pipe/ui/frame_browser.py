"""Frame browser widget (filters + table + FITS preview).

This is used on the "Project & data" page after Inspect.
"""

from __future__ import annotations


from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

from scorpio_pipe.ui.fits_preview import FitsPreviewWidget
from scorpio_pipe.ui.pandas_table_model import PandasTableModel


@dataclass(frozen=True)
class SelectedFrame:
    path: Path
    kind: str
    object: str
    disperser: str
    slit: str
    binning: str


class FrameBrowser(QtWidgets.QWidget):
    """Browse inspected frames with quick filters and preview."""

    selectedChanged = QtCore.Signal(object)  # SelectedFrame | None
    useSetupRequested = QtCore.Signal(object)  # SelectedFrame

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._df_all = pd.DataFrame()
        self._df = pd.DataFrame()

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # Filters
        filt = QtWidgets.QHBoxLayout()
        lay.addLayout(filt)
        self.edit_search = QtWidgets.QLineEdit()
        self.edit_search.setPlaceholderText("Search object/path…")
        self.combo_kind = QtWidgets.QComboBox()
        self.combo_kind.addItems(
            ["all", "obj", "sky", "sunsky", "neon", "flat", "bias"]
        )
        self.combo_disperser = QtWidgets.QComboBox()
        self.combo_disperser.addItem("all")
        self.combo_slit = QtWidgets.QComboBox()
        self.combo_slit.addItem("all")
        self.combo_binning = QtWidgets.QComboBox()
        self.combo_binning.addItem("all")
        self.btn_reset = QtWidgets.QPushButton("Reset")

        filt.addWidget(QtWidgets.QLabel("Filter:"))
        filt.addWidget(self.edit_search, 2)
        filt.addWidget(QtWidgets.QLabel("Kind"))
        filt.addWidget(self.combo_kind)
        filt.addWidget(QtWidgets.QLabel("Disp"))
        filt.addWidget(self.combo_disperser)
        filt.addWidget(QtWidgets.QLabel("Slit"))
        filt.addWidget(self.combo_slit)
        filt.addWidget(QtWidgets.QLabel("Bin"))
        filt.addWidget(self.combo_binning)
        filt.addWidget(self.btn_reset)

        # Split: table | preview
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        lay.addWidget(splitter, 1)

        left = QtWidgets.QWidget()
        vbox_left = QtWidgets.QVBoxLayout(left)
        vbox_left.setContentsMargins(0, 0, 0, 0)
        vbox_left.setSpacing(6)

        self.lbl_counts = QtWidgets.QLabel("—")
        self.lbl_counts.setStyleSheet("color: #A0A0A0;")
        vbox_left.addWidget(self.lbl_counts)

        self.model = PandasTableModel(pd.DataFrame())
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        vbox_left.addWidget(self.table, 1)

        splitter.addWidget(left)

        right = QtWidgets.QWidget()
        r = QtWidgets.QVBoxLayout(right)
        r.setContentsMargins(0, 0, 0, 0)
        r.setSpacing(6)

        self.preview = FitsPreviewWidget()
        r.addWidget(self.preview, 1)

        bar = QtWidgets.QHBoxLayout()
        r.addLayout(bar)
        self.btn_open_file = QtWidgets.QPushButton("Open file")
        self.btn_use_setup = QtWidgets.QPushButton("Use this setup →")
        self.btn_use_setup.setProperty("primary", True)
        self.btn_use_setup.setEnabled(False)
        bar.addWidget(self.btn_open_file)
        bar.addStretch(1)
        bar.addWidget(self.btn_use_setup)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([720, 520])

        # signals
        self.edit_search.textChanged.connect(self._apply_filters)
        self.combo_kind.currentTextChanged.connect(self._apply_filters)
        self.combo_disperser.currentTextChanged.connect(self._apply_filters)
        self.combo_slit.currentTextChanged.connect(self._apply_filters)
        self.combo_binning.currentTextChanged.connect(self._apply_filters)
        self.btn_reset.clicked.connect(self._reset_filters)
        self.table.selectionModel().selectionChanged.connect(
            lambda *_: self._on_selection_changed()
        )
        self.btn_open_file.clicked.connect(self._open_selected)
        self.btn_use_setup.clicked.connect(self._emit_use_setup)

    # ---------------------------- public API ----------------------------

    def set_frames_df(self, df: pd.DataFrame) -> None:
        """Load the inspected frame table."""
        self._df_all = df.copy() if df is not None else pd.DataFrame()
        self._populate_filter_values()
        self._reset_filters()

    # Backward-compatible alias used by some older UI components.
    def set_frames(self, df: pd.DataFrame, base_dir: Path | None = None) -> None:  # noqa: ARG002
        """Alias for :meth:`set_frames_df`.

        The inspection table already stores absolute paths, so *base_dir* is ignored.
        """
        self.set_frames_df(df)

    def selected_frames(self) -> list[SelectedFrame]:
        """Return all currently selected frames.

        The widget is usually used in single-selection mode, but dialogs may switch
        the table selection mode to ExtendedSelection. This helper keeps the logic
        in one place.
        """
        out: list[SelectedFrame] = []
        try:
            idxs = self.table.selectionModel().selectedRows()
            if not idxs:
                return out
            for idx in idxs:
                r = int(idx.row())
                row = self._df.iloc[r]
                p = Path(str(row.get("path", "") or "")).expanduser()
                if not p:
                    continue
                out.append(
                    SelectedFrame(
                        path=p,
                        kind=str(row.get("kind", "") or ""),
                        object=str(row.get("object", "") or ""),
                        disperser=str(row.get("disperser", "") or ""),
                        slit=str(row.get("slit", "") or ""),
                        binning=str(row.get("binning", "") or ""),
                    )
                )
        except Exception:
            return []
        # stable ordering
        out = sorted(out, key=lambda s: str(s.path))
        return out

    # ---------------------------- internals ----------------------------

    def _populate_filter_values(self) -> None:
        def _fill(combo: QtWidgets.QComboBox, col: str) -> None:
            combo.blockSignals(True)
            cur = combo.currentText()
            combo.clear()
            combo.addItem("all")
            if col in self._df_all.columns and not self._df_all.empty:
                vals = (
                    self._df_all[col]
                    .dropna()
                    .astype(str)
                    .replace("", pd.NA)
                    .dropna()
                    .unique()
                    .tolist()
                )
                for v in sorted(vals):
                    combo.addItem(str(v))
            if cur:
                idx = combo.findText(cur)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.blockSignals(False)

        _fill(self.combo_disperser, "disperser")
        _fill(self.combo_slit, "slit")
        _fill(self.combo_binning, "binning")

    def _reset_filters(self) -> None:
        self.edit_search.setText("")
        self.combo_kind.setCurrentText("all")
        self.combo_disperser.setCurrentText("all")
        self.combo_slit.setCurrentText("all")
        self.combo_binning.setCurrentText("all")
        self._apply_filters()

    def _apply_filters(self) -> None:
        df = self._df_all
        if df is None or df.empty:
            self._df = pd.DataFrame()
            self.model.set_df(self._df)
            self.lbl_counts.setText("No frames")
            self.preview.clear()
            self.btn_use_setup.setEnabled(False)
            return

        s = self.edit_search.text().strip().lower()
        kind = self.combo_kind.currentText().strip()
        disp = self.combo_disperser.currentText().strip()
        slit = self.combo_slit.currentText().strip()
        binning = self.combo_binning.currentText().strip()

        mask = pd.Series([True] * len(df))

        if kind and kind != "all" and "kind" in df.columns:
            mask &= df["kind"].astype(str) == kind
        if disp and disp != "all" and "disperser" in df.columns:
            mask &= df["disperser"].astype(str) == disp
        if slit and slit != "all" and "slit" in df.columns:
            mask &= df["slit"].astype(str) == slit
        if binning and binning != "all" and "binning" in df.columns:
            mask &= df["binning"].astype(str) == binning
        if s:
            cols = []
            for c in ("object", "path", "fid"):
                if c in df.columns:
                    cols.append(df[c].astype(str).str.lower().fillna(""))
            if cols:
                m2 = cols[0].str.contains(s)
                for extra in cols[1:]:
                    m2 |= extra.str.contains(s)
                mask &= m2

        out = df.loc[mask].copy()

        # choose a compact column set for UI
        preferred = [
            "kind",
            "object",
            "exptime",
            "disperser",
            "slit",
            "binning",
            "shape",
            "fid",
            "path",
        ]
        cols = [c for c in preferred if c in out.columns] + [
            c for c in out.columns if c not in preferred
        ]
        out = out[cols]

        self._df = out
        self.model.set_df(self._df)
        self.table.resizeColumnsToContents()
        self.lbl_counts.setText(f"Frames: {len(out)} / {len(df)}")

        # Auto-select first row.
        # NOTE: on some platforms selectionChanged is not emitted reliably right after a model reset,
        # so we also schedule an explicit preview update.
        if len(out) > 0:
            self.table.selectRow(0)
            QtCore.QTimer.singleShot(0, self._on_selection_changed)
        else:
            self.preview.clear()
            self.btn_use_setup.setEnabled(False)

    def _selected_frame(self) -> SelectedFrame | None:
        try:
            idxs = self.table.selectionModel().selectedRows()
            if not idxs:
                return None
            r = idxs[0].row()
            row = self._df.iloc[int(r)]
            p = Path(str(row.get("path", "") or "")).expanduser()
            if not p:
                return None
            return SelectedFrame(
                path=p,
                kind=str(row.get("kind", "") or ""),
                object=str(row.get("object", "") or ""),
                disperser=str(row.get("disperser", "") or ""),
                slit=str(row.get("slit", "") or ""),
                binning=str(row.get("binning", "") or ""),
            )
        except Exception:
            return None

    def _on_selection_changed(self) -> None:
        sel = self._selected_frame()
        if sel is None:
            self.preview.clear()
            self.btn_use_setup.setEnabled(False)
            self.selectedChanged.emit(None)
            return

        if sel.path.exists():
            self.preview.set_path(sel.path)
        else:
            self.preview.clear()
        self.btn_use_setup.setEnabled(bool(sel.object))
        self.selectedChanged.emit(sel)

    def _open_selected(self) -> None:
        sel = self._selected_frame()
        if sel is None:
            return
        try:
            if sel.path.exists():
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(sel.path)))
        except Exception:
            # parent may still catch selectionChanged and open folder
            pass

    def _emit_use_setup(self) -> None:
        sel = self._selected_frame()
        if sel is None:
            return
        self.useSetupRequested.emit(sel)