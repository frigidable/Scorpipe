from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from astropy.io import fits
from scorpio_pipe.io.atomic_fits import atomic_write_fits

from PySide6 import QtCore, QtWidgets

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.widgets import RectangleSelector

from scorpio_pipe.config import load_config_any
from scorpio_pipe.fits_utils import read_image_smart
from scorpio_pipe.plot_style import mpl_style
from scorpio_pipe.pipeline.public_api import boxcar_mean2d_masked as _boxcar_mean2d_masked
from scorpio_pipe.ui.async_worker import RunHandle, run_in_thread
from scorpio_pipe.ui.mpl_render import RenderThrottle
from scorpio_pipe.workspace_paths import stage_dir

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


def _cosmics_root(work_dir: Path) -> Path:
    """Locate the cosmics stage root (canonical + legacy fallback)."""

    canonical = stage_dir(work_dir, "cosmics")
    legacy = work_dir / "cosmics"
    if canonical.exists():
        return canonical
    if legacy.exists():
        return legacy
    # Prefer canonical for new workspaces.
    return canonical


def _cosmics_kind_dirs(work_dir: Path, kind: str) -> dict[str, Path]:
    root = _cosmics_root(work_dir) / kind
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
    atomic_write_fits(path, np.asarray(mask, dtype=np.uint8), overwrite=True)


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
    atomic_write_fits(path, np.asarray(data, dtype=np.float32), header=h, overwrite=True)


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


@dataclass
class _DisplayPayload:
    image: np.ndarray
    auto_mask: np.ndarray
    manual_mask: np.ndarray
    extent: tuple[float, float, float, float]
    vmin: float | None
    vmax: float | None
    step_x: int
    step_y: int


@dataclass
class _FramePayload:
    path: Path
    key: tuple[object, ...] | None
    image: np.ndarray
    header: fits.Header
    auto_mask: np.ndarray
    manual_mask: np.ndarray
    display: _DisplayPayload


_LAST_FRAME_CACHE_LOCK = threading.Lock()
_LAST_FRAME_CACHE_KEY: tuple[object, ...] | None = None
_LAST_FRAME_CACHE_PAYLOAD: _FramePayload | None = None


_MANUAL_VIEW_MAX_DIM = max(512, int(os.environ.get("SCORPIPE_UI_MANUAL_COSMICS_MAX_DIM", "1600")))


def _file_cache_key(path: Path) -> tuple[object, ...] | None:
    try:
        st = path.stat()
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        return (str(path.resolve()), mtime_ns, int(st.st_size))
    except Exception:
        return None


def _get_last_frame_cache(key: tuple[object, ...] | None) -> _FramePayload | None:
    if key is None:
        return None
    with _LAST_FRAME_CACHE_LOCK:
        if _LAST_FRAME_CACHE_KEY != key:
            return None
        return _LAST_FRAME_CACHE_PAYLOAD


def _put_last_frame_cache(payload: _FramePayload) -> None:
    with _LAST_FRAME_CACHE_LOCK:
        global _LAST_FRAME_CACHE_KEY, _LAST_FRAME_CACHE_PAYLOAD
        _LAST_FRAME_CACHE_KEY = payload.key
        _LAST_FRAME_CACHE_PAYLOAD = payload


def _prepare_display_payload(
    image: np.ndarray,
    auto_mask: np.ndarray,
    manual_mask: np.ndarray,
    *,
    max_dim: int = _MANUAL_VIEW_MAX_DIM,
) -> _DisplayPayload:
    img = np.asarray(image, dtype=np.float32)
    ny, nx = int(img.shape[0]), int(img.shape[1])
    step = max(1, int(np.ceil(max(ny, nx) / float(max(64, int(max_dim))))))
    img_disp = np.ascontiguousarray(img[::step, ::step], dtype=np.float32)
    auto_disp = np.ascontiguousarray(np.asarray(auto_mask[::step, ::step], dtype=bool))
    manual_disp = np.ascontiguousarray(np.asarray(manual_mask[::step, ::step], dtype=bool))

    finite = img_disp[np.isfinite(img_disp)]
    vmin: float | None = None
    vmax: float | None = None
    if finite.size:
        try:
            lo, hi = np.nanpercentile(finite, [5.0, 99.0])
            vmin = float(lo)
            vmax = float(hi)
        except Exception:
            pass
    if vmin is not None and vmax is not None and not np.isfinite(vmin - vmax):
        vmin = None
        vmax = None

    return _DisplayPayload(
        image=img_disp,
        auto_mask=auto_disp,
        manual_mask=manual_disp,
        extent=(-0.5, float(nx) - 0.5, -0.5, float(ny) - 0.5),
        vmin=vmin,
        vmax=vmax,
        step_x=int(step),
        step_y=int(step),
    )


def _load_frame_payload_sync(path: Path, *, work_dir: Path, kind: str) -> _FramePayload:
    p = Path(path)
    with fits.open(p, memmap=False) as hdul:
        header = hdul[0].header.copy()
    image = np.asarray(read_image_smart(p), dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={tuple(image.shape)}")

    base = p.stem
    if base.endswith("_clean"):
        base = base[: -len("_clean")]

    dirs = _cosmics_kind_dirs(work_dir, kind)
    mpaths = _mask_paths(dirs, base)

    if mpaths["auto"].exists():
        auto = _load_mask(mpaths["auto"], image.shape)
    else:
        auto = _load_mask(mpaths["final"], image.shape)

    manual = _load_mask(mpaths["manual"], image.shape)
    display = _prepare_display_payload(image, auto, manual)
    return _FramePayload(
        path=p,
        key=_file_cache_key(p),
        image=image,
        header=header,
        auto_mask=auto,
        manual_mask=manual,
        display=display,
    )


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
        self._display: Optional[_DisplayPayload] = None
        self._frame_job: Optional[RunHandle[_FramePayload]] = None
        self._frame_request: tuple[str, int, str] | None = None
        self._frame_busy: bool = False

        self._last_rect: Optional[tuple[int, int, int, int]] = None
        self._undo: list[_UndoItem] = []
        self._render_overlay_message: str | None = None
        self._render_throttle = RenderThrottle(self, self._draw_now, max_fps=10.0, debounce_ms=180)

        # --- Layout
        root = QtWidgets.QVBoxLayout(self)
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
        self._cancel_load_frame(mark_only=False)
        self._reload_paths()
        if not self._require_auto_done():
            return
        self._load_frame(0)

    def _on_r_changed(self, v: int) -> None:
        self.replace_r = int(v)

    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        self._frame_busy = bool(busy)
        self.combo_kind.setEnabled(not busy)
        self.btn_prev.setEnabled((not busy) and self.frame_idx > 0)
        self.btn_next.setEnabled((not busy) and self.frame_idx < max(0, len(self._paths) - 1))
        self.spin_r.setEnabled(not busy)
        if message:
            self.lbl_help.setText(message)
        elif busy:
            self.lbl_help.setText("Loading frame…")
        else:
            self.lbl_help.setText(
                "Draw a rectangle around a cosmic spike/cluster. "
                "Press Enter to apply replacement and write it to disk. "
                "Press Ctrl+Z to undo the last applied rectangle."
            )
        if self._selector is not None:
            try:
                self._selector.set_active(not busy)
            except Exception:
                pass

    def _cancel_load_frame(self, *, mark_only: bool = False) -> None:
        job = getattr(self, "_frame_job", None)
        if job is None:
            return
        try:
            job.mark_stale() if mark_only else job.cancel()
        except Exception:
            pass
        self._frame_job = None

    def _step_frame(self, d: int) -> None:
        if self._frame_busy:
            return
        if not self._require_auto_done():
            return
        i = int(self.frame_idx) + int(d)
        i = max(0, min(len(self._paths) - 1, i))
        self._load_frame(i)

    def _apply_frame_payload(self, payload: _FramePayload) -> None:
        self._img = np.asarray(payload.image, dtype=np.float32)
        self._hdr = payload.header.copy()
        self._auto_mask = np.asarray(payload.auto_mask, dtype=bool)
        self._manual_mask = np.asarray(payload.manual_mask, dtype=bool)
        self._display = payload.display
        self._last_rect = None
        self._set_busy(False)
        self._request_draw()
        self._update_info()

    def _load_frame(self, idx: int) -> None:
        if not self._require_auto_done():
            return
        idx = int(idx)
        idx = max(0, min(len(self._paths) - 1, idx))
        self.frame_idx = idx
        p = self._paths[self.frame_idx]

        self._cancel_load_frame(mark_only=False)
        self._last_rect = None
        self._img = None
        self._hdr = None
        self._auto_mask = None
        self._manual_mask = None
        self._display = None
        self._request_draw(overlay_message=f"Loading frame {p.name}…")
        self._update_info(extra=f"loading {p.name}")

        key = _file_cache_key(p)
        cached = _get_last_frame_cache(key)
        if cached is not None:
            self._apply_frame_payload(cached)
            return

        self._set_busy(True, f"Loading {p.name} in background…")
        request_kind = str(self.kind)
        request_idx = int(self.frame_idx)
        self._frame_request = (request_kind, request_idx, str(p))

        def _job() -> _FramePayload:
            return _load_frame_payload_sync(p, work_dir=self.work_dir, kind=request_kind)

        h: RunHandle[_FramePayload] = run_in_thread(
            _job,
            name="manual_cosmics_load_frame",
            dedupe_key=f"manual_cosmics:{p.resolve()}",
        )
        self._frame_job = h

        def _on_finished(payload: _FramePayload) -> None:
            if self._frame_job is not h:
                return
            self._frame_job = None
            if self._frame_request != (request_kind, request_idx, str(p)):
                return
            _put_last_frame_cache(payload)
            self._apply_frame_payload(payload)

        def _on_failed(tb: str) -> None:
            if self._frame_job is not h:
                return
            self._frame_job = None
            self._set_busy(False, "Failed to load frame")
            self._request_draw(overlay_message="Failed to load frame")
            self._update_info(extra="failed")
            QtWidgets.QMessageBox.critical(
                self,
                "Manual Cosmics",
                f"Failed to read FITS: {p}\n\n{tb.splitlines()[-1] if tb else tb}",
            )

        h.worker.finished.connect(_on_finished)
        h.worker.failed.connect(_on_failed)

    def _update_info(self, *, extra: str | None = None) -> None:
        if not self._paths:
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
        txt = (
            f"{self.kind} | frame {self.frame_idx + 1}/{len(self._paths)} | {base} | "
            f"auto={auto_frac:.4f}, manual={man_frac:.4f}, final={fin:.4f}"
        )
        if self._display is not None and (self._display.step_x > 1 or self._display.step_y > 1):
            txt += f" | preview 1/{self._display.step_x}"
        if extra:
            txt += f" | {extra}"
        self.lbl_info.setText(txt)

    def _request_draw(self, *, overlay_message: str | None = None, debounce: bool = False) -> None:
        self._render_overlay_message = overlay_message
        if debounce:
            self._render_throttle.request()
        else:
            self._render_throttle.request_immediate()

    def _draw_now(self) -> None:
        overlay_message = self._render_overlay_message
        self._render_overlay_message = None
        self.ax.clear()
        if self._display is None:
            with mpl_style():
                self.ax.set_title("Manual cosmics: draw rectangle → Enter apply, Ctrl+Z undo")
                self.ax.set_xlabel("X (pix)")
                self.ax.set_ylabel("Y (pix)")
                self.ax.text(
                    0.5,
                    0.5,
                    overlay_message or "Loading frame…",
                    transform=self.ax.transAxes,
                    ha="center",
                    va="center",
                )
            self.canvas.draw_idle()
            return

        disp = self._display
        with mpl_style():
            self.ax.imshow(
                disp.image,
                origin="lower",
                aspect="auto",
                cmap="gray",
                vmin=disp.vmin,
                vmax=disp.vmax,
                extent=disp.extent,
                interpolation="nearest",
            )
            self.ax.set_title(
                "Manual cosmics: draw rectangle → Enter apply, Ctrl+Z undo"
            )
            self.ax.set_xlabel("X (pix)")
            self.ax.set_ylabel("Y (pix)")

            if self._auto_mask is not None:
                m = np.ma.masked_where(~disp.auto_mask, disp.auto_mask)
                self.ax.imshow(
                    m,
                    origin="lower",
                    aspect="auto",
                    cmap="Reds",
                    alpha=0.22,
                    extent=disp.extent,
                    interpolation="nearest",
                )
            if self._manual_mask is not None:
                m = np.ma.masked_where(~disp.manual_mask, disp.manual_mask)
                self.ax.imshow(
                    m,
                    origin="lower",
                    aspect="auto",
                    cmap="Oranges",
                    alpha=0.30,
                    extent=disp.extent,
                    interpolation="nearest",
                )
            if overlay_message:
                self.ax.text(
                    0.02,
                    0.98,
                    overlay_message,
                    transform=self.ax.transAxes,
                    ha="left",
                    va="top",
                    bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none"},
                )

        self.canvas.draw_idle()

    def _refresh_display_from_current(self) -> None:
        if self._img is None or self._auto_mask is None or self._manual_mask is None:
            self._display = None
            return
        self._display = _prepare_display_payload(self._img, self._auto_mask, self._manual_mask)

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
        if self._frame_busy:
            return
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

        self._refresh_display_from_current()
        self._persist_current()
        self._request_draw()
        self._update_info()

    def _undo_last(self) -> None:
        if self._frame_busy:
            return
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

        self._refresh_display_from_current()
        self._persist_current()
        self._request_draw()
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
        self._refresh_display_from_current()
        try:
            _put_last_frame_cache(
                _FramePayload(
                    path=p,
                    key=_file_cache_key(p),
                    image=np.asarray(self._img, dtype=np.float32),
                    header=h.copy(),
                    auto_mask=np.asarray(self._auto_mask, dtype=bool),
                    manual_mask=np.asarray(self._manual_mask, dtype=bool),
                    display=self._display if self._display is not None else _prepare_display_payload(self._img, self._auto_mask, self._manual_mask),
                )
            )
        except Exception:
            pass

    def closeEvent(self, event) -> None:
        try:
            self._cancel_load_frame(mark_only=False)
        except Exception:
            pass
        super().closeEvent(event)
