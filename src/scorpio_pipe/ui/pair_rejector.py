from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets


def _read_pairs(path: Path) -> list[tuple[float, float, bool]]:
    if not path.exists():
        return []
    out: list[tuple[float, float, bool]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        blend = ("blend" in s.lower())
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
        if len(nums) >= 2:
            out.append((float(nums[0]), float(nums[1]), blend))
    return out


def _save_pairs(path: Path, pairs: list[tuple[float, float, bool]], *, header_note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# manual pairs: x  lambda   (# blend if marked)\n")
        if header_note:
            f.write(f"# {header_note}\n")
        for x0, lam, blend in sorted(pairs, key=lambda t: t[0]):
            f.write(f"{x0:.6f}  {lam:.4f}{'   # blend' if blend else ''}\n")


def _polyfit(x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < deg + 1:
        raise RuntimeError(f"Need at least {deg+1} points for deg={deg}, got {x.size}")
    return np.polyfit(x, y, deg)


class PairRejectorDialog(QtWidgets.QDialog):
    """Interactively reject bad LineID pairs before building the solution."""

    def __init__(
        self,
        pairs_path: Path,
        *,
        poly_deg: int = 4,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Clean LineID pairs")
        self.resize(1100, 720)
        # User preference: open windows maximized by default.
        try:
            self.setWindowState(self.windowState() | QtCore.Qt.WindowMaximized)
        except Exception:
            pass

        self.pairs_path = Path(pairs_path)
        self.poly_deg = int(poly_deg)

        self._pairs = _read_pairs(self.pairs_path)
        self._active = [True] * len(self._pairs)

        # --- layout ---
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)

        head = QtWidgets.QHBoxLayout()
        lay.addLayout(head)
        self.lbl_path = QtWidgets.QLabel(str(self.pairs_path))
        self.lbl_path.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        head.addWidget(QtWidgets.QLabel("Pairs file:"))
        head.addWidget(self.lbl_path, 1)

        self.spin_deg = QtWidgets.QSpinBox()
        self.spin_deg.setRange(2, 10)
        self.spin_deg.setValue(self.poly_deg)
        self.spin_deg.setToolTip("Polynomial degree for quick 1D fit used in this QC tool")
        head.addWidget(QtWidgets.QLabel("deg:"))
        head.addWidget(self.spin_deg)

        self.btn_recalc = QtWidgets.QPushButton("Recalculate")
        self.btn_auto3 = QtWidgets.QPushButton("Auto-reject >3σ")
        self.btn_restore = QtWidgets.QPushButton("Restore all")
        head.addWidget(self.btn_recalc)
        head.addWidget(self.btn_auto3)
        head.addWidget(self.btn_restore)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        lay.addWidget(splitter, 1)

        # left: table
        left = QtWidgets.QWidget()
        llay = QtWidgets.QVBoxLayout(left)
        llay.setContentsMargins(0, 0, 0, 0)
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Use", "x", "λ", "Δλ", "blend", "note"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        llay.addWidget(self.table, 1)
        splitter.addWidget(left)

        # right: plot (matplotlib)
        right = QtWidgets.QWidget()
        rlay = QtWidgets.QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)
        self.lbl_stats = QtWidgets.QLabel("—")
        self.lbl_stats.setStyleSheet("font-weight:600;")
        rlay.addWidget(self.lbl_stats)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        import matplotlib as mpl
        from scorpio_pipe.plot_style import STYLE

        mpl.rcParams.update(STYLE)

        self.fig = Figure(figsize=(6.2, 4.5), dpi=120)
        self.canvas = FigureCanvas(self.fig)
        rlay.addWidget(self.canvas, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # footer
        foot = QtWidgets.QHBoxLayout()
        lay.addLayout(foot)
        self.btn_save_clean = QtWidgets.QPushButton("Save cleaned as…")
        self.btn_overwrite = QtWidgets.QPushButton("Overwrite pairs file")
        self.btn_cancel = QtWidgets.QPushButton("Close")
        self.btn_save_clean.setProperty("primary", True)
        self.btn_overwrite.setProperty("primary", True)
        foot.addWidget(self.btn_save_clean)
        foot.addWidget(self.btn_overwrite)
        foot.addStretch(1)
        foot.addWidget(self.btn_cancel)

        # signals
        self.btn_recalc.clicked.connect(self.recalculate)
        self.btn_restore.clicked.connect(self._restore_all)
        self.btn_auto3.clicked.connect(self._auto_reject_3sigma)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save_clean.clicked.connect(self._save_as)
        self.btn_overwrite.clicked.connect(self._overwrite)
        self.spin_deg.valueChanged.connect(lambda *_: self.recalculate())
        self.table.cellChanged.connect(self._on_cell_changed)

        self._populate_table()
        self.recalculate()

    # -------------- table / state --------------

    def _populate_table(self) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for i, (x0, lam, blend) in enumerate(self._pairs):
            self.table.insertRow(i)

            chk = QtWidgets.QTableWidgetItem("")
            chk.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            chk.setCheckState(QtCore.Qt.Checked if self._active[i] else QtCore.Qt.Unchecked)
            self.table.setItem(i, 0, chk)

            def _it(v: str) -> QtWidgets.QTableWidgetItem:
                it = QtWidgets.QTableWidgetItem(v)
                it.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                return it

            self.table.setItem(i, 1, _it(f"{x0:.3f}"))
            self.table.setItem(i, 2, _it(f"{lam:.3f}"))
            self.table.setItem(i, 3, _it("—"))
            self.table.setItem(i, 4, _it("yes" if blend else "no"))
            self.table.setItem(i, 5, _it(""))

        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)

    def _on_cell_changed(self, row: int, col: int) -> None:
        if col != 0:
            return
        it = self.table.item(row, 0)
        if it is None:
            return
        self._active[row] = (it.checkState() == QtCore.Qt.Checked)
        self.recalculate()

    def _restore_all(self) -> None:
        self._active = [True] * len(self._pairs)
        self._populate_table()
        self.recalculate()

    # -------------- fit / plotting --------------

    def _get_active_xy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = [i for i, a in enumerate(self._active) if a]
        if not idx:
            return np.array([]), np.array([]), np.array([], bool)
        x = np.array([self._pairs[i][0] for i in idx], float)
        lam = np.array([self._pairs[i][1] for i in idx], float)
        blend = np.array([self._pairs[i][2] for i in idx], bool)
        return x, lam, blend

    def recalculate(self) -> None:
        deg = int(self.spin_deg.value())
        x_all = np.array([p[0] for p in self._pairs], float) if self._pairs else np.array([])
        lam_all = np.array([p[1] for p in self._pairs], float) if self._pairs else np.array([])
        active = np.array(self._active, bool) if self._pairs else np.array([], bool)

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_xlabel("Pixel X")
        ax.set_ylabel("Δλ [Å]")
        ax.axhline(0, linewidth=1.0)

        if x_all.size == 0:
            self.lbl_stats.setText("No pairs loaded")
            self.canvas.draw()
            return

        x_use = x_all[active]
        lam_use = lam_all[active]
        if x_use.size < deg + 1:
            self.lbl_stats.setText(f"Need ≥{deg+1} active pairs for deg={deg} (now {x_use.size})")
            # still show which points are active
            ax.scatter(x_use, np.zeros_like(x_use), s=28)
            self.canvas.draw()
            return

        coeffs = _polyfit(x_use, lam_use, deg)
        resid_all = lam_all - np.polyval(coeffs, x_all)

        # update table residual column
        self.table.blockSignals(True)
        for i in range(len(self._pairs)):
            it = self.table.item(i, 3)
            if it is not None:
                it.setText(f"{resid_all[i]:+.3f}")
                # visually dim rejected
                for c in range(1, 6):
                    cell = self.table.item(i, c)
                    if cell is not None:
                        cell.setForeground(QtGui.QBrush(QtGui.QColor(200, 200, 200) if self._active[i] else QtGui.QColor(140, 140, 140)))
        self.table.blockSignals(False)

        rms = float(np.sqrt(np.mean(resid_all[active] ** 2)))
        self.lbl_stats.setText(f"Active: {active.sum()}/{len(active)}   deg={deg}   RMS={rms:.3f} Å")

        ax.scatter(x_all[active], resid_all[active], s=34, label="used")
        if (~active).any():
            ax.scatter(x_all[~active], resid_all[~active], s=28, marker="x", label="rejected")
        ax.legend(frameon=False, loc="best")
        self.fig.tight_layout()
        self.canvas.draw()

    def _auto_reject_3sigma(self) -> None:
        deg = int(self.spin_deg.value())
        x_all = np.array([p[0] for p in self._pairs], float)
        lam_all = np.array([p[1] for p in self._pairs], float)
        active = np.array(self._active, bool)
        if active.sum() < deg + 1:
            return
        coeffs = _polyfit(x_all[active], lam_all[active], deg)
        resid = lam_all - np.polyval(coeffs, x_all)
        s = float(np.std(resid[active]))
        if not np.isfinite(s) or s <= 0:
            return
        new_active = np.abs(resid) <= 3.0 * s
        self._active = new_active.tolist()
        self._populate_table()
        self.recalculate()

    # -------------- save --------------

    def _current_clean_pairs(self) -> list[tuple[float, float, bool]]:
        return [p for p, a in zip(self._pairs, self._active) if a]

    def _save_as(self) -> None:
        if not self._pairs:
            return
        default = self.pairs_path.with_name(self.pairs_path.stem + ".cleaned.txt")
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save cleaned pairs", str(default), "Text files (*.txt)")
        if not fn:
            return
        out = Path(fn)
        clean = self._current_clean_pairs()
        _save_pairs(out, clean, header_note=f"cleaned from {self.pairs_path.name}, kept {len(clean)}/{len(self._pairs)}")
        self.pairs_path = out
        self.lbl_path.setText(str(self.pairs_path))

    def _overwrite(self) -> None:
        if not self._pairs:
            return
        clean = self._current_clean_pairs()
        _save_pairs(self.pairs_path, clean, header_note=f"cleaned in-place, kept {len(clean)}/{len(self._pairs)}")
        self.accept()


def clean_pairs_interactively(
    pairs_path: str | Path,
    *,
    poly_deg: int = 4,
    parent: QtWidgets.QWidget | None = None,
) -> Path | None:
    """Run the dialog and return the (possibly new) path to the cleaned file."""
    dlg = PairRejectorDialog(Path(pairs_path), poly_deg=poly_deg, parent=parent)
    if dlg.exec() == QtWidgets.QDialog.Accepted:
        return dlg.pairs_path
    return None
