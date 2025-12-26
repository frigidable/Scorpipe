from __future__ import annotations

from typing import Any

import pandas as pd
from PySide6 import QtCore


class PandasTableModel(QtCore.QAbstractTableModel):
    """A tiny QAbstractTableModel wrapper for a pandas DataFrame."""

    def __init__(
        self, df: pd.DataFrame | None = None, parent: QtCore.QObject | None = None
    ):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()

    def set_df(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = df if df is not None else pd.DataFrame()
        self.endResetModel()

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else int(len(self._df))

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else int(self._df.shape[1])

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.DisplayRole,
    ):  # noqa: N802
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Horizontal:
            try:
                return str(self._df.columns[section])
            except Exception:
                return None
        return str(section)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole):  # noqa: N802
        if not index.isValid():
            return None
        r = index.row()
        c = index.column()
        if r < 0 or c < 0 or r >= len(self._df) or c >= self._df.shape[1]:
            return None

        if role in (QtCore.Qt.DisplayRole, QtCore.Qt.ToolTipRole):
            v: Any = self._df.iat[r, c]
            if v is None:
                return ""
            if isinstance(v, float):
                # compact numeric formatting for FITS headers
                return f"{v:.6g}"
            s = str(v)
            return s

        return None
