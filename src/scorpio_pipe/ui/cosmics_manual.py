from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from astropy.io import fits

from PySide6 import QtCore, QtWidgets

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.widgets import RectangleSelector

from scorpio_pipe.config import load_config_any
from scorpio_pipe.fits_utils import read_image_smart
from scorpio_pipe.plot_style import mpl_style
from scorpio_pipe.stages.cosmics import _boxcar_mean2d_masked

log = logging.getLogger(__name__)


def _safe_int_bounds(a: float, b: float, *, lo: int, hi: int) -> tuple[int, int]:
    i0 = int(np.floor(min(a, b)))
    i1 = int(np.ceil(max(a, b)))
    i0 = max(lo, min(hi, i0))
    i1 = max(lo, min(hi, i1))
    if i1 < i0:
        i0, i1 = i1, i0
    return i0, i1


def _work_dir_from_cfg(cfg: dict[str, Any]) -> Path:
    base_dir = Path(str(cfg.get("config_dir", "."))).resolve()
    wd = Path(str(cfg.get("work_dir", "work"))).expanduser()
    return wd if wd.is_absolute() else (base_dir / wd).resolve()


def _cosmics_kind_dirs(work_dir: Path, kind: str) -> dict[str, Path]:
    root = work_dir / "cosmics" / kind
    return {
        "root": root,
        "clean": root / "clean",
        "masks": root / "masks_fits",
        "auto": root / "auto_masks_fits",
        "manual": root / "manual_masks_fits",
        "backup": root / "auto_backup",
    }


def _mask_paths(dirs: dict[str, Path], base: str) -> dict[str, Path]:
    return {
        "auto": dirs["auto"] / f"{base}_auto_mask.fits",
        "manual": dirs["manual"] / f"{base}_manual_mask.fits",
        "final": dirs["masks"] / f"{base}_mask.fits",
    }


def _load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    try:
        m = fits.getdata(path, memmap=False)
        m = np.asarray(m)
        if m.ndim != 2:
            return np.zeros(shape, dtype=bool)
        ny = min(shape[0], m.shape[0])
        nx = min(shape[1], m.shape[1])
        out = np.zeros(shape, dtype=bool)
        out[:ny, :nx] = m[:ny, :nx] != 0
        return out
    except Exception:
        return np.zeros(shape, dtype=bool)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(path, np.asarray(mask, dtype=np.uint8), overwrite=True)


def _write_clean(
    path: Path,
    data: np.ndarray,
    *,
    hdr: Optional[fits.Header] = None,
    history: Optional[str] = None,
) -> None:
    h = hdr.copy() if hdr is not None else fits.Header()
    if history:
        h["HISTORY"] = history
    # float32 for science, avoids scaling keywords
    fits.writeto(path, np.asarray(data, dtype=np.float32), header=h, overwrite=True)


@dataclass
class _UndoItem:
    kind: str
    frame_idx: int
    y0: int
    y1: int
    x0: int
    x1: int
    prev_mask: np.ndarray
    prev_data: np.ndarray


class CosmicsManualDialog(QtWidgets.QDialog):
    """Manual cosmic cleaning, applied on top of auto-cleaned frames.

    Workflow (as requested): run automatic Cosmics first -> open this dialog ->
    mark rectangles -> Enter to apply, Ctrl+Z to undo. All edits persist as
    manual masks and updated clean FITS under work_dir/cosmics/<kind>/.
    """

    def __init__(self, cfg: Any, *, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Manual Cosmics Cleanup")
        self.setModal(True)
        try:
            self.setWindowState(self.windowState() | QtCore.Qt.WindowMaximized)
        except Exception:
            pass

        self.cfg = load_config_any(cfg)
        self.work_dir = _work_dir_from_cfg(self.cfg)

        c = (
            self.cfg.get("cosmics", {})
            if isinstance(self.cfg.get("cosmics"), dict)
            else {}
        )
        self.replace_r = int(c.get("manual_replace_r", c.get("la_replace_r", 2)))
        self.replace_r = max(1, min(25, self.replace_r))

        self.kind = "obj"
        self.frame_idx = 0
        self._paths: list[Path] = []

        self._img: Optional[np.ndarray] = None
        self._hdr: Optional[fits.Header] = None
        self._auto_mask: Optional[np.ndarray] = None
        self._manual_mask: Optional[np.ndarray] = None

        self._last_rect: Optional[tuple[int, int, int, int]] = None
        self._undo: list[_UndoItem] = []

        # --- Layout
        root = QtWidgets.QVBoxLayout()
        self.setLayout(root)
        root.setSpacing(10)

        top = QtWidgets.QHBoxLayout()
        root.addLayout(top)

        self.combo_kind = QtWidgets.QComboBox()
        self.combo_kind.addItems(["obj", "sky"])
        self.combo_kind.currentTextChanged.connect(self._on_kind_changed)
        top.addWidget(QtWidgets.QLabel("Frames:"))
        top.addWidget(self.combo_kind)

        self.btn_prev = QtWidgets.QPushButton("◀ Prev")
        self.btn_next = QtWidgets.QPushButton("Next ▶")
        self.btn_prev.clicked.connect(lambda: self._step_frame(-1))
        self.btn_next.clicked.connect(lambda: self._step_frame(+1))
        top.addWidget(self.btn_prev)
        top.addWidget(self.btn_next)

        top.addSpacing(20)
        self.spin_r = QtWidgets.QSpinBox()
        self.spin_r.setRange(1, 25)
        self.spin_r.setValue(self.replace_r)
        self.spin_r.setToolTip("Replacement boxcar radius (px)")
        self.spin_r.valueChanged.connect(self._on_r_changed)
        top.addWidget(QtWidgets.QLabel("Replace r:"))
        top.addWidget(self.spin_r)

        top.addStretch(1)

        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        root.addWidget(self.lbl_info)

        self.fig = Figure(figsize=(8.0, 4.8), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        root.addWidget(self.canvas, 1)

        help_txt = (
            "Draw a rectangle around a cosmic spike/cluster. "
            "Press Enter to apply replacement and write it to disk. "
            "Press Ctrl+Z to undo the last applied rectangle."
        )
        self.lbl_help = QtWidgets.QLabel(help_txt)
        self.lbl_help.setWordWrap(True)
        root.addWidget(self.lbl_help)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        bb.accepted.connect(self.accept)
        root.addWidget(bb)

        # mpl hooks
        self._selector: Optional[RectangleSelector] = None
        self._cid_key = self.canvas.mpl_connect("key_press_event", self._on_key)

        self._reload_paths()
        self._load_frame(0)
        self._install_selector()

    # ---- internal

    def _reload_paths(self) -> None:
        dirs = _cosmics_kind_dirs(self.work_dir, self.kind)
        clean_dir = dirs["clean"]
        if not clean_dir.exists():
            self._paths = []
            return
        self._paths = sorted(clean_dir.glob("*_clean.fits"))

    def _require_auto_done(self) -> bool:
        if not self._paths:
            QtWidgets.QMessageBox.warning(
                self,
                "Manual Cosmics",
                f"No auto-cleaned frames found for kind='{self.kind}'.\n\n"
                f"Expected: {(_cosmics_kind_dirs(self.work_dir, self.kind)['clean']).as_posix()}\n\n"
                "Run automatic Clean Cosmics first.",
            )
            return False
        return True

    def _on_kind_changed(self, t: str) -> None:
        self.kind = str(t)
        self.frame_idx = 0
        self._undo.clear()
        self._reload_paths()
        if not self._require_auto_done():
            return
        self._load_frame(0)

    def _on_r_changed(self, v: int) -> None:
        self.replace_r = int(v)

    def _step_frame(self, d: int) -> None:
        if not self._require_auto_done():
            return
        i = int(self.frame_idx) + int(d)
        i = max(0, min(len(self._paths) - 1, i))
        self._load_frame(i)

    def _load_frame(self, idx: int) -> None:
        if not self._require_auto_done():
            return
        idx = int(idx)
        idx = max(0, min(len(self._paths) - 1, idx))
        self.frame_idx = idx
        p = self._paths[self.frame_idx]

        # Load image and header
        try:
            with fits.open(p, memmap=False) as hdul:
                self._hdr = hdul[0].header.copy()
            self._img = np.asarray(read_image_smart(p), dtype=np.float32)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Manual Cosmics", f"Failed to read FITS: {p}\n\n{e}"
            )
            self._img = None
            self._hdr = None
            return

        if self._img.ndim != 2:
            QtWidgets.QMessageBox.critical(
                self,
                "Manual Cosmics",
                f"Expected 2D image, got shape={self._img.shape}",
            )
            self._img = None
            return

        base = p.stem
        if base.endswith("_clean"):
            base = base[: -len("_clean")]

        dirs = _cosmics_kind_dirs(self.work_dir, self.kind)
        for k in ("masks", "auto", "manual", "backup"):
            dirs[k].mkdir(parents=True, exist_ok=True)
        mpaths = _mask_paths(dirs, base)

        # auto baseline: if auto_mask exists, use it; else snapshot current final mask as auto
        if mpaths["auto"].exists():
            auto = _load_mask(mpaths["auto"], self._img.shape)
        else:
            auto = _load_mask(mpaths["final"], self._img.shape)
            # If final mask exists, treat it as auto baseline and snapshot it.
            if mpaths["final"].exists():
                _write_mask(mpaths["auto"], auto)

        manual = _load_mask(mpaths["manual"], self._img.shape)
        self._auto_mask = auto
        self._manual_mask = manual

        self._last_rect = None
        self._draw()
        self._update_info()

    def _update_info(self) -> None:
        if self._img is None:
            self.lbl_info.setText("—")
            return
        p = self._paths[self.frame_idx]
        base = p.name
        auto_frac = (
            float(self._auto_mask.mean()) if self._auto_mask is not None else 0.0
        )
        man_frac = (
            float(self._manual_mask.mean()) if self._manual_mask is not None else 0.0
        )
        fin = float(
            ((self._auto_mask | self._manual_mask).mean())
            if (self._auto_mask is not None and self._manual_mask is not None)
            else 0.0
        )
        self.lbl_info.setText(
            f"{self.kind} | frame {self.frame_idx + 1}/{len(self._paths)} | {base} | "
            f"auto={auto_frac:.4f}, manual={man_frac:.4f}, final={fin:.4f}"
        )

    def _draw(self) -> None:
        if self._img is None:
            return
        self.ax.clear()
        with mpl_style():
            finite = self._img[np.isfinite(self._img)]
            if finite.size:
                vmin, vmax = np.nanpercentile(finite, [5, 99])
            else:
                vmin, vmax = None, None
            self.ax.imshow(
                self._img,
                origin="lower",
                aspect="auto",
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
            )
            self.ax.set_title(
                "Manual cosmics: draw rectangle → Enter apply, Ctrl+Z undo"
            )
            self.ax.set_xlabel("X (pix)")
            self.ax.set_ylabel("Y (pix)")

            # overlay masks
            if self._auto_mask is not None:
                m = np.ma.masked_where(~self._auto_mask, self._auto_mask)
                self.ax.imshow(
                    m, origin="lower", aspect="auto", cmap="Reds", alpha=0.22
                )
            if self._manual_mask is not None:
                m = np.ma.masked_where(~self._manual_mask, self._manual_mask)
                self.ax.imshow(
                    m, origin="lower", aspect="auto", cmap="Oranges", alpha=0.30
                )

        self.canvas.draw_idle()

    def _install_selector(self) -> None:
        if self._selector is not None:
            try:
                self._selector.set_active(False)
            except Exception:
                pass
            self._selector = None

        def _on_select(eclick, erelease):
            if self._img is None:
                return
            x0, x1 = _safe_int_bounds(
                eclick.xdata or 0, erelease.xdata or 0, lo=0, hi=self._img.shape[1] - 1
            )
            y0, y1 = _safe_int_bounds(
                eclick.ydata or 0, erelease.ydata or 0, lo=0, hi=self._img.shape[0] - 1
            )
            self._last_rect = (y0, y1, x0, x1)

        self._selector = RectangleSelector(
            self.ax,
            _on_select,
            useblit=True,
            button=[1],
            minspanx=2,
            minspany=2,
            spancoords="pixels",
            interactive=True,
        )

    def _on_key(self, event) -> None:
        key = str(getattr(event, "key", "") or "").lower()
        if key == "enter":
            self._apply_last_rect()
        elif key in ("ctrl+z", "control+z"):
            self._undo_last()

    def _apply_last_rect(self) -> None:
        if self._img is None or self._auto_mask is None or self._manual_mask is None:
            return
        if self._last_rect is None:
            return
        y0, y1, x0, x1 = self._last_rect
        if y1 <= y0 or x1 <= x0:
            return

        rect = (slice(y0, y1 + 1), slice(x0, x1 + 1))
        prev_m = self._manual_mask[rect].copy()
        prev_d = self._img[rect].copy()

        # update manual mask
        self._manual_mask[rect] = True
        m_final = self._auto_mask | self._manual_mask

        # inpaint pixels in the rectangle that are newly manual
        try:
            mean_map = _boxcar_mean2d_masked(self._img, m_final, r=int(self.replace_r))
            self._img[rect][self._manual_mask[rect]] = mean_map[rect][
                self._manual_mask[rect]
            ]
        except Exception:
            # fallback: replace by median of unmasked pixels in rect
            rr = self._img[rect]
            good = ~m_final[rect]
            if np.any(good):
                med = float(np.nanmedian(rr[good]))
                rr2 = rr.copy()
                rr2[self._manual_mask[rect]] = med
                self._img[rect] = rr2

        self._undo.append(
            _UndoItem(
                kind=self.kind,
                frame_idx=int(self.frame_idx),
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                prev_mask=prev_m,
                prev_data=prev_d,
            )
        )

        self._persist_current()
        self._draw()
        self._update_info()

    def _undo_last(self) -> None:
        if not self._undo:
            return
        u = self._undo.pop()
        if u.kind != self.kind:
            # different kind, ignore (should not happen)
            return
        if u.frame_idx != self.frame_idx:
            self._load_frame(u.frame_idx)
        if self._img is None or self._manual_mask is None:
            return

        rect = (slice(u.y0, u.y1 + 1), slice(u.x0, u.x1 + 1))
        self._manual_mask[rect] = u.prev_mask
        self._img[rect] = u.prev_data

        self._persist_current()
        self._draw()
        self._update_info()

    def _persist_current(self) -> None:
        """Persist manual mask, final mask and clean FITS for the current frame."""
        if self._img is None or self._auto_mask is None or self._manual_mask is None:
            return
        p = self._paths[self.frame_idx]
        base = p.stem
        if base.endswith("_clean"):
            base = base[: -len("_clean")]

        dirs = _cosmics_kind_dirs(self.work_dir, self.kind)
        for k in ("masks", "auto", "manual", "backup"):
            dirs[k].mkdir(parents=True, exist_ok=True)
        mpaths = _mask_paths(dirs, base)

        # ensure we keep the auto baseline snapshot
        if not mpaths["auto"].exists() and mpaths["final"].exists():
            try:
                auto = _load_mask(mpaths["final"], self._img.shape)
                _write_mask(mpaths["auto"], auto)
            except Exception:
                pass

        # write manual and final
        _write_mask(mpaths["manual"], self._manual_mask)
        _write_mask(mpaths["final"], (self._auto_mask | self._manual_mask))

        # backup original clean once
        backup_p = dirs["backup"] / f"{p.stem}_auto.fits"
        if not backup_p.exists():
            try:
                backup_p.write_bytes(p.read_bytes())
            except Exception:
                pass

        # write updated clean
        h = self._hdr.copy() if self._hdr is not None else fits.Header()
        h["MANCR"] = (True, "Manual cosmics edits applied")
        h["MANCRR"] = (int(self.replace_r), "Manual replace radius")
        h["MANCRPX"] = (int(self._manual_mask.sum()), "Manual masked pixels")
        _write_clean(
            p,
            self._img,
            hdr=h,
            history="scorpio_pipe manual cosmics: replaced pixels in user rectangles",
        )