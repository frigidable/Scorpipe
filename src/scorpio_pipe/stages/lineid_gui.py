from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional
import re
import numpy as np
from astropy.io import fits


def _project_root() -> Path:
    """Project root in source layout and PyInstaller (onefile) builds."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(str(meipass)).resolve()
    return Path(__file__).resolve().parents[3]


def _resolve_from_root(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (_project_root() / p).resolve()


def _profile_1d(img2d: np.ndarray, y_half: int = 20) -> np.ndarray:
    """
    Устойчивый 1D-профиль по X.
    1) пробуем nanmedian по полосе вокруг центра
    2) если там все NaN — ищем ближайшую "живую" полосу
    3) если и это не получилось — nanmedian по всему кадру
    4) если вообще все NaN — возвращаем нули (чтобы GUI не падал)
    """
    ny, nx = img2d.shape

    # быстрый выход: весь кадр NaN
    if not np.isfinite(img2d).any():
        return np.zeros(nx, dtype=float)

    yc = ny // 2
    y_half = int(max(1, y_half))

    def band(yc_try: int) -> np.ndarray:
        y0 = max(0, yc_try - y_half)
        y1 = min(ny, yc_try + y_half + 1)
        return np.nanmedian(img2d[y0:y1, :], axis=0)

    prof = band(yc)
    if np.isfinite(prof).any():
        return np.asarray(prof, dtype=float)

    # если центральная полоса "мертва" — ищем ближайшую живую
    for dy in range(1, ny // 2):
        for yc_try in (yc - dy, yc + dy):
            if 0 <= yc_try < ny:
                prof = band(yc_try)
                if np.isfinite(prof).any():
                    return np.asarray(prof, dtype=float)

    # fallback: по всему кадру
    prof = np.nanmedian(img2d, axis=0)
    if np.isfinite(prof).any():
        return np.asarray(prof, dtype=float)

    return np.zeros(nx, dtype=float)


def _read_peaks_candidates(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (x_pix, amp). If amp column missing -> amp = NaN.
    ожидаем заголовок: x_pix,amp,...
    """
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    names = set(arr.dtype.names or [])

    # x column
    x_name = None
    for n in ("x_pix", "x", "x0", "px"):
        if n in names:
            x_name = n
            break
    if x_name is None:
        x_name = arr.dtype.names[0]

    x = np.asarray(arr[x_name], float)

    # amp column
    amp_name = None
    for n in ("amp", "ampl", "height", "peak", "I", "profile_I"):
        if n in names:
            amp_name = n
            break
    if amp_name is None:
        amp = np.full_like(x, np.nan, dtype=float)
    else:
        amp = np.asarray(arr[amp_name], float)

    m = np.isfinite(x)
    x = x[m]
    amp = amp[m]

    # sort by x
    idx = np.argsort(x)
    return x[idx], amp[idx]


def _read_linelist_meta(path: Path) -> dict[str, str]:
    """Parse metadata comment lines from a linelist CSV.

    We accept simple comment headers like:
      # waveref: air
      # unit: angstrom

    Returns a dict with lower-case keys/values.
    """

    meta: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return meta

    for line in lines[:80]:
        s = line.strip()
        if not s.startswith("#"):
            continue
        s = s[1:].strip()
        if not s:
            continue
        if ":" in s:
            k, v = s.split(":", 1)
        elif "=" in s:
            k, v = s.split("=", 1)
        else:
            continue
        k = k.strip().lower()
        v = v.strip().lower()
        if k in {"waveref", "wave_ref", "ref"}:
            if v in {"air", "vacuum"}:
                meta["waveref"] = v
        elif k in {"unit", "units"}:
            meta["unit"] = v

    return meta


def _read_neon_lines_csv(path: Path) -> list[float]:
    # максимально терпимо к формату
    # берём все числа в разумном диапазоне
    out: list[float] = []
    txt = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in txt:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
        for n in nums:
            v = float(n)
            # nm -> Å эвристика
            if v < 1000:
                v *= 10.0
            if 1000.0 < v < 20000.0:
                out.append(v)
    out = sorted(set(out))
    if not out:
        raise RuntimeError(f"No wavelengths parsed from {path}")
    return out


def _auto_min_amp(
    pk_amp: np.ndarray, sigma_k: float = 5.0, q_noise: float = 0.40
) -> float:
    """Pick a sane default amplitude threshold.

    The peak list often contains a mix of real lines and local maxima from
    background/noise. We estimate the statistics of the "noise peak"
    population by taking the lower quantile of `pk_amp` and using a MAD-based
    sigma. Threshold = median(noise) + sigma_k * sigma(noise).

    Returns a non-negative float.
    """
    a = np.asarray(pk_amp, float)
    a = a[np.isfinite(a)]
    if a.size < 10:
        return 0.0

    q_noise = float(np.clip(q_noise, 0.10, 0.80))
    cut = np.quantile(a, q_noise)
    noise = a[a <= cut]
    if noise.size < 10:
        noise = a

    med = float(np.median(noise))
    mad = float(np.median(np.abs(noise - med)))
    sigma = 1.4826 * mad
    if sigma <= 0:
        return max(0.0, med)
    return max(0.0, med + float(sigma_k) * sigma)


def _read_hand_pairs(hand_file: Path) -> list[tuple[float, float, bool]]:
    if not hand_file.exists():
        return []
    pairs: list[tuple[float, float, bool]] = []
    for line in hand_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        blend = "blend" in s.lower()
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
        if len(nums) >= 2:
            pairs.append((float(nums[0]), float(nums[1]), blend))
    return pairs


def _save_hand_pairs(hand_file: Path, pairs: list[tuple[float, float, bool]]) -> None:
    hand_file.parent.mkdir(parents=True, exist_ok=True)
    with hand_file.open("w", encoding="utf-8") as f:
        f.write("# manual pairs: x  lambda   (# blend if marked)\n")
        for x0, lam, blend in sorted(pairs, key=lambda t: t[0]):
            f.write(f"{x0:.6f}  {lam:.4f}{'   # blend' if blend else ''}\n")


# NOTE: AtlasImageViewer was removed.
# We now use a single robust PdfViewer widget (QtPdfWidgets → PyMuPDF fallback)
# that supports mouse+keyboard navigation similar to a typical PDF reader.


@dataclass
class LineIdInputs:
    x: np.ndarray
    prof: np.ndarray
    pk_x: np.ndarray
    pk_amp: np.ndarray
    ref_lams: list[float]
    hand_file: Path
    title: str
    atlas_pdf: Optional[Path] = None
    atlas_page0: Optional[int] = None  # 0-indexed
    disperser: str | None = None
    lam_min_A: float | None = None
    lam_max_A: float | None = None
    min_amp_default: float | None = None
    min_amp_sigma_k: float = 5.0


def run_lineid_gui(inp: LineIdInputs) -> None:
    from PySide6 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg

    from scorpio_pipe.ui import PdfViewer

    # 1) QApplication ДО всего
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    # 3) Принудительно светлая тема (pyqtgraph) — ДО создания PlotWidget
    pg.setConfigOption("background", (255, 255, 255))
    pg.setConfigOption("foreground", (0, 0, 0))
    pg.setConfigOptions(antialias=False)

    class Dialog(QtWidgets.QDialog):
        def __init__(self):
            super().__init__()
            # делаем поведение как у обычного окна Windows (resize/snap/maximize)
            self.setWindowFlags(self.windowFlags() | QtCore.Qt.Window)
            self.setWindowFlag(QtCore.Qt.WindowType.WindowMaximizeButtonHint, True)
            self.setWindowFlag(QtCore.Qt.WindowType.WindowMinimizeButtonHint, True)
            self.setSizeGripEnabled(True)
            self.setWindowTitle(inp.title)

            self.lam_min_A = inp.lam_min_A
            self.lam_max_A = inp.lam_max_A

            self.x = inp.x
            self.prof = inp.prof
            self.pk_x = inp.pk_x
            self._peak_lines: list = []  # синие линии кандидатов
            self._pair_items: list = []  # красные линии + подписи
            self.pk_amp = inp.pk_amp
            if inp.min_amp_default is not None:
                self.pk_min_amp = float(inp.min_amp_default)
            else:
                self.pk_min_amp = _auto_min_amp(
                    self.pk_amp, sigma_k=float(inp.min_amp_sigma_k)
                )
            self._pk_active = self.pk_x.copy()
            self.ref_lams = inp.ref_lams
            self.hand_file = inp.hand_file

            # Atlas (optional)
            self.atlas_pdf = inp.atlas_pdf
            self.atlas_page0 = inp.atlas_page0
            self._atlas_inited = False

            self.pairs: list[tuple[float, float, bool]] = _read_hand_pairs(
                self.hand_file
            )
            self.used = {lam for _, lam, _ in self.pairs}

            self._build()
            self._plot()
            self._apply_peak_filter()
            self._refresh_lams()
            self._refresh_table()
            self._restore_marks()

        def _build(self):
            lay = QtWidgets.QVBoxLayout(self)

            # Main splitter:
            #   [Atlas panel] [Spectrum plot] [Controls]
            self.split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
            lay.addWidget(self.split, 1)

            # LEFT: atlas panel (hidden by default)
            self.atlas_panel = QtWidgets.QWidget()
            self.atlas_panel.setVisible(False)
            ap = QtWidgets.QVBoxLayout(self.atlas_panel)
            ap.setContentsMargins(6, 6, 6, 6)

            self.atlas_header = QtWidgets.QLabel("Атлас He–Ne–Ar")
            font = self.atlas_header.font()
            font.setBold(True)
            self.atlas_header.setFont(font)
            ap.addWidget(self.atlas_header)

            # Container where the actual PDF viewer (QtPdf) will be created lazily
            self.atlas_container = QtWidgets.QWidget()
            self.atlas_container_lay = QtWidgets.QVBoxLayout(self.atlas_container)
            self.atlas_container_lay.setContentsMargins(0, 0, 0, 0)
            ap.addWidget(self.atlas_container, 1)

            self.split.addWidget(self.atlas_panel)

            # MIDDLE: plot
            plotw = QtWidgets.QWidget()
            pl = QtWidgets.QVBoxLayout(plotw)
            self.wplot = pg.PlotWidget()
            self.pitem = self.wplot.getPlotItem()
            self.pitem.showGrid(x=True, y=True, alpha=0.2)
            pl.addWidget(self.wplot, 1)
            self.split.addWidget(plotw)

            # RIGHT: controls
            right = QtWidgets.QWidget()
            rlay = QtWidgets.QVBoxLayout(right)

            rowx = QtWidgets.QHBoxLayout()
            rowx.addWidget(QtWidgets.QLabel("x₀:"))
            self.edit_x = QtWidgets.QLineEdit()
            rowx.addWidget(self.edit_x, 1)
            self.btn_prev = QtWidgets.QPushButton("◀ пик")
            self.btn_next = QtWidgets.QPushButton("пик ▶")
            rowx.addWidget(self.btn_prev)
            rowx.addWidget(self.btn_next)
            rlay.addLayout(rowx)

            rowp = QtWidgets.QHBoxLayout()
            rowp.addWidget(QtWidgets.QLabel("Min amp:"))
            self.spin_amp = QtWidgets.QDoubleSpinBox()
            self.spin_amp.setDecimals(1)
            self.spin_amp.setRange(-1e12, 1e12)  # не мешаем
            self.spin_amp.setValue(float(self.pk_min_amp))
            self.spin_amp.setSingleStep(50.0)  # шаг удобный для ADU, потом подстроишь
            rowp.addWidget(self.spin_amp, 1)
            self.btn_apply_peaks = QtWidgets.QPushButton("Apply")
            rowp.addWidget(self.btn_apply_peaks)
            rlay.addLayout(rowp)

            rowf = QtWidgets.QHBoxLayout()
            rowf.addWidget(QtWidgets.QLabel("Фильтр λ:"))
            self.edit_filter = QtWidgets.QLineEdit()
            self.edit_filter.setPlaceholderText("5400-5700 или 5852.5 или 585")
            if self.lam_min_A is not None and self.lam_max_A is not None:
                lo = float(self.lam_min_A)
                hi = float(self.lam_max_A)
                if hi > lo:
                    self.edit_filter.setPlaceholderText(
                        f"Авто-диапазон: {lo:.0f}–{hi:.0f} Å (можно искать по числу)"
                    )
            rowf.addWidget(self.edit_filter, 1)
            rlay.addLayout(rowf)

            self.list_lams = QtWidgets.QListWidget()
            rlay.addWidget(self.list_lams, 1)

            rowl = QtWidgets.QHBoxLayout()
            rowl.addWidget(QtWidgets.QLabel("λ вручную:"))
            self.edit_lam = QtWidgets.QLineEdit()
            rowl.addWidget(self.edit_lam, 1)
            rlay.addLayout(rowl)

            rowb = QtWidgets.QHBoxLayout()
            self.chk_blend = QtWidgets.QCheckBox("blend")
            rowb.addWidget(self.chk_blend)
            self.lbl = QtWidgets.QLabel("")
            rowb.addWidget(self.lbl)
            rowb.addStretch(1)
            rlay.addLayout(rowb)

            rowbtn = QtWidgets.QHBoxLayout()
            self.btn_atlas = QtWidgets.QPushButton("Атлас")
            self.btn_atlas.setCheckable(True)
            self.btn_link = QtWidgets.QPushButton("Связать (Enter)")
            self.btn_del = QtWidgets.QPushButton("Удалить (Del)")
            self.btn_zoom = QtWidgets.QPushButton("Zoom")
            self.btn_reset = QtWidgets.QPushButton("Reset")
            rowbtn.addWidget(self.btn_atlas)
            rowbtn.addWidget(self.btn_link)
            rowbtn.addWidget(self.btn_del)
            rowbtn.addWidget(self.btn_zoom)
            rowbtn.addWidget(self.btn_reset)
            rlay.addLayout(rowbtn)

            self.table = QtWidgets.QTableWidget()
            self.table.setColumnCount(3)
            self.table.setHorizontalHeaderLabels(["x", "λ", "flag"])
            self.table.horizontalHeader().setStretchLastSection(True)
            self.table.setSelectionBehavior(
                QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
            )
            rlay.addWidget(self.table, 1)

            self.status = QtWidgets.QLabel("")
            rlay.addWidget(self.status)

            self.split.addWidget(right)
            # give plot more space than controls; atlas will be toggled
            self.split.setStretchFactor(0, 1)
            self.split.setStretchFactor(1, 3)
            self.split.setStretchFactor(2, 2)

            box = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.StandardButton.Ok
                | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            )
            lay.addWidget(box)

            box.accepted.connect(self._accept)
            box.rejected.connect(self.reject)

            # signals + hotkeys (как в твоей программе)
            self.wplot.scene().sigMouseClicked.connect(self._on_click)
            self.edit_filter.textChanged.connect(self._refresh_lams)
            self.list_lams.itemDoubleClicked.connect(lambda *_: self._link())
            self.btn_link.clicked.connect(self._link)
            self.btn_del.clicked.connect(self._delete)
            self.btn_zoom.clicked.connect(self._zoom)
            self.btn_reset.clicked.connect(lambda: self.pitem.enableAutoRange())
            self.btn_atlas.toggled.connect(self._toggle_atlas)
            self.table.itemSelectionChanged.connect(self._on_select)
            self.btn_prev.clicked.connect(lambda: self._step_peak(-1))
            self.btn_next.clicked.connect(lambda: self._step_peak(+1))
            self.btn_apply_peaks.clicked.connect(self._apply_peak_filter)
            self.spin_amp.valueChanged.connect(lambda *_: self._apply_peak_filter())

            QtGui.QShortcut(QtGui.QKeySequence("Return"), self, activated=self._link)
            QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self, activated=self._undo)
            QtGui.QShortcut(QtGui.QKeySequence("Delete"), self, activated=self._delete)
            QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self, activated=self._accept)
            QtGui.QShortcut(QtGui.QKeySequence("Escape"), self, activated=self.reject)
            QtGui.QShortcut(QtGui.QKeySequence("F11"), self, activated=self._toggle_max)

        def _plot(self):
            # профиль — строго чёрный
            self.pitem.plot(self.x, self.prof, pen=pg.mkPen(color=(0, 0, 0), width=1.0))

            # отрисовать линии пиков и пары
            self._redraw_markers()

            self.pitem.enableAutoRange()

        def _clear_items(self, items: list):
            for it in items:
                try:
                    self.pitem.removeItem(it)
                except Exception:
                    pass
            items.clear()

        def _peak_local_max(self, x0: float, halfwin: int = 8) -> float:
            """Оценка максимума пика по профилю в окрестности x0."""
            i0 = int(round(x0))
            lo = max(0, i0 - halfwin)
            hi = min(len(self.prof), i0 + halfwin + 1)
            if lo >= hi:
                return (
                    float(np.nanmax(self.prof)) if np.isfinite(self.prof).any() else 0.0
                )
            slab = self.prof[lo:hi]
            m = np.nanmax(slab)
            if not np.isfinite(m):
                m = float(np.nanmax(self.prof)) if np.isfinite(self.prof).any() else 0.0
            return float(m)

        def _redraw_markers(self):
            # 1) очистить старые линии/подписи
            self._clear_items(self._peak_lines)
            self._clear_items(self._pair_items)

            # 2) какие пики считаем "непривязанными"?
            # Мы считаем привязанным тот пик, возле которого стоит x0 из pairs.
            paired_x = (
                np.array([x0 for x0, _, _ in self.pairs], dtype=float)
                if self.pairs
                else np.array([], dtype=float)
            )

            def is_paired(px: float, tol: float = 0.5) -> bool:
                if paired_x.size == 0:
                    return False
                return bool(np.any(np.abs(paired_x - px) <= tol))

            # 3) синие линии для непривязанных кандидатов
            blue_pen = pg.mkPen(color=(0, 90, 255), width=0.8)  # насыщенный синий
            for px in self._pk_active:
                if is_paired(float(px)):
                    continue
                ln = pg.InfiniteLine(pos=float(px), angle=90, pen=blue_pen)
                self.pitem.addItem(ln)
                self._peak_lines.append(ln)

            # 4) красные линии + подписи λ для привязанных
            red_pen = pg.mkPen(color=(220, 0, 0), width=1.4)

            for x0, lam, blend in self.pairs:
                # линия
                ln = pg.InfiniteLine(pos=float(x0), angle=90, pen=red_pen)
                self.pitem.addItem(ln)
                self._pair_items.append(ln)

                # подпись: вертикально чуть выше максимума пика
                ymax = self._peak_local_max(float(x0), halfwin=10)
                y = ymax * 1.05 if np.isfinite(ymax) else 0.0

                txt = pg.TextItem(
                    text=f"{lam:.2f}", color=(220, 0, 0), anchor=(0.5, 0.0)
                )
                # повернуть текст вертикально
                txt.setAngle(90)

                txt.setPos(float(x0), float(y))
                self.pitem.addItem(txt)
                self._pair_items.append(txt)

        def _draw_peak_lines(self, peaks: np.ndarray):
            # больше не используем отдельную отрисовку, всё через единый механизм
            self._redraw_markers()

        def _apply_peak_filter(self):
            thr = float(self.spin_amp.value())
            self.pk_min_amp = thr

            if self.pk_amp.size == self.pk_x.size and np.isfinite(self.pk_amp).any():
                m = np.isfinite(self.pk_amp) & (self.pk_amp >= thr)
                self._pk_active = self.pk_x[m]
            else:
                # если амплитуд нет — ничего не фильтруем
                self._pk_active = self.pk_x.copy()

            self._redraw_markers()
            self._set_status(
                f"Peaks: {len(self._pk_active)} / {len(self.pk_x)} (min amp={thr:g})"
            )

        def _replot_peaks(self):
            self._redraw_markers()

        def _toggle_max(self):
            if self.isMaximized():
                self.showNormal()
            else:
                self.showMaximized()

        def _init_atlas_view(self) -> None:
            """Create the PDF viewer lazily (only when user asks for it)."""
            if self._atlas_inited:
                return
            self._atlas_inited = True

            # If atlas path is not provided or missing, show a short hint.
            if self.atlas_pdf is None or not Path(str(self.atlas_pdf)).is_file():
                msg = QtWidgets.QLabel(
                    "Атлас не найден.\n\n"
                    "Укажи путь в конфиге: wavesol.atlas_pdf: HeNeAr_atlas.pdf"
                )
                msg.setWordWrap(True)
                self.atlas_container_lay.addWidget(msg)
                return

            atlas_path = Path(str(self.atlas_pdf)).resolve()

            # Use a single robust embedded PDF viewer.
            # It prefers QtPdfWidgets (vector), and falls back to PyMuPDF with
            # full mouse+keyboard navigation.
            state_page0 = (
                int(self.atlas_page0) if isinstance(self.atlas_page0, int) else 0
            )
            viewer = PdfViewer(atlas_path, state=None, parent=self)
            # Jump to the relevant page right away.
            try:
                viewer.go_page(state_page0)
                viewer.fit_width()
            except Exception:
                pass
            self.atlas_container_lay.addWidget(viewer, 1)
            # Keep a reference to prevent GC.
            self._atlas_viewer = viewer
            return

        def _toggle_atlas(self, checked: bool):
            """Show/hide the atlas panel on the left."""
            self.atlas_panel.setVisible(bool(checked))
            if checked:
                self._init_atlas_view()
                # reasonably sized default proportions
                try:
                    self.split.setSizes([420, 900, 420])
                except Exception:
                    pass
            else:
                try:
                    self.split.setSizes([0, 900, 420])
                except Exception:
                    pass

        def _restore_marks(self):
            self._redraw_markers()

        def _set_status(self, s: str):
            self.status.setText(s)

        def _count(self):
            self.lbl.setText(f"Пар: {len(self.pairs)}")

        def _nearest_peak(self, x0: float, tol: float = 7.0) -> float:
            pk = self._pk_active
            if pk.size == 0:
                return x0
            i = int(np.argmin(np.abs(pk - x0)))
            return float(pk[i]) if abs(pk[i] - x0) <= tol else x0

        def _on_click(self, ev):
            from PySide6 import QtCore

            if ev.button() != QtCore.Qt.MouseButton.LeftButton:
                return
            pos = ev.scenePos()
            if not self.pitem.sceneBoundingRect().contains(pos):
                return
            mp = self.pitem.vb.mapSceneToView(pos)
            x0 = float(mp.x())
            xsel = self._nearest_peak(x0)
            self.edit_x.setText(f"{xsel:.3f}")
            self._set_status(f"Выбран x≈{xsel:.3f}. Теперь выбери λ и Enter.")

        def _filter_lams(self) -> list[float]:
            txt = self.edit_filter.text().strip()
            vals = [v for v in self.ref_lams if v not in self.used]

            # Auto range (per grism) — limit reference lines to the disperser window.
            if self.lam_min_A is not None and self.lam_max_A is not None:
                try:
                    lo = float(self.lam_min_A)
                    hi = float(self.lam_max_A)
                    if hi < lo:
                        lo, hi = hi, lo
                    vals = [v for v in vals if (lo <= float(v) <= hi)]
                except Exception:
                    pass

            if not txt:
                return vals

            m = re.match(r"^\s*(\d+(\.\d+)?)\s*-\s*(\d+(\.\d+)?)\s*$", txt)
            if m:
                a = float(m.group(1))
                b = float(m.group(3))
                lo, hi = min(a, b), max(a, b)
                return [v for v in vals if lo <= v <= hi]

            try:
                f = float(txt)
                if f < 1000:
                    # "585" → всё, что начинается на 585*
                    return [
                        v for v in vals if str(int(round(v))).startswith(str(int(f)))
                    ]
                return [v for v in vals if abs(v - f) <= 1.0]
            except Exception:
                return [v for v in vals if txt in f"{v:.2f}"]

        def _refresh_lams(self):
            self.list_lams.clear()
            for v in self._filter_lams():
                self.list_lams.addItem(f"{v:.2f}")

        def _get_x(self) -> Optional[float]:
            t = self.edit_x.text().strip()
            if not t:
                return None
            try:
                return float(t)
            except Exception:
                return None

        def _get_lam(self) -> Optional[float]:
            item = self.list_lams.currentItem()
            if item is not None:
                try:
                    return float(item.text())
                except Exception:
                    pass
            t = self.edit_lam.text().strip()
            if t:
                try:
                    return float(t)
                except Exception:
                    pass
            return None

        def _refresh_table(self):
            from PySide6 import QtCore, QtWidgets

            self.table.setRowCount(len(self.pairs))
            for r, (x0, lam, blend) in enumerate(self.pairs):
                itx = QtWidgets.QTableWidgetItem(f"{x0:.3f}")
                itl = QtWidgets.QTableWidgetItem(f"{lam:.2f}")
                itf = QtWidgets.QTableWidgetItem("blend" if blend else "")
                itx.setTextAlignment(
                    QtCore.Qt.AlignmentFlag.AlignRight
                    | QtCore.Qt.AlignmentFlag.AlignVCenter
                )
                itl.setTextAlignment(
                    QtCore.Qt.AlignmentFlag.AlignRight
                    | QtCore.Qt.AlignmentFlag.AlignVCenter
                )
                self.table.setItem(r, 0, itx)
                self.table.setItem(r, 1, itl)
                self.table.setItem(r, 2, itf)
            self._count()
            self._refresh_lams()

        def _link(self):
            x0 = self._get_x()
            if x0 is None:
                self._set_status("Нет x: кликни по профилю.")
                return
            lam = self._get_lam()
            if lam is None:
                self._set_status("Нет λ: выбери в списке или введи вручную.")
                return
            if lam in self.used:
                self._set_status(f"λ={lam:.2f} уже использована.")
                return
            blend = bool(self.chk_blend.isChecked())
            self.pairs.append((float(x0), float(lam), blend))
            self.used.add(float(lam))
            self._refresh_table()
            self._set_status(f"Добавлено: x={x0:.3f} ↔ λ={lam:.2f}")
            self._redraw_markers()
            # UX: после привязки очистить поле ручного ввода λ
            self.edit_lam.clear()

        def _delete(self):
            row = self.table.currentRow()
            if row < 0 or row >= len(self.pairs):
                self._set_status("Не выбрана строка.")
                return
            x0, lam, blend = self.pairs.pop(row)
            if lam in self.used:
                self.used.remove(lam)
            self._refresh_table()
            self._set_status(f"Удалено λ={lam:.2f}")
            self._redraw_markers()

        def _undo(self):
            if not self.pairs:
                return
            x0, lam, blend = self.pairs.pop()
            if lam in self.used:
                self.used.remove(lam)
            self._refresh_table()
            self._set_status("Undo.")
            self._redraw_markers()

        def _zoom(self):
            x0 = self._get_x()
            if x0 is None:
                return
            self.pitem.setXRange(x0 - 80, x0 + 80, padding=0.02)

        def _on_select(self):
            row = self.table.currentRow()
            if row < 0 or row >= len(self.pairs):
                return
            x0, lam, blend = self.pairs[row]
            self.edit_x.setText(f"{x0:.3f}")
            self.pitem.setXRange(x0 - 80, x0 + 80, padding=0.02)

        def _step_peak(self, step: int):
            if self._pk_active.size == 0:
                return
            x0 = self._get_x()
            if x0 is None:
                idx = 0 if step > 0 else len(self._pk_active) - 1
            else:
                idx = int(np.argmin(np.abs(self._pk_active - x0)))
                idx = max(0, min(len(self._pk_active) - 1, idx + step))
            xsel = float(self._pk_active[idx])
            self.edit_x.setText(f"{xsel:.3f}")
            self.pitem.setXRange(xsel - 80, xsel + 80, padding=0.02)

        def _accept(self):
            if self.pairs:
                _save_hand_pairs(self.hand_file, self.pairs)
            self.accept()

        def reject(self):
            # autosave on close
            if self.pairs:
                try:
                    _save_hand_pairs(self.hand_file, self.pairs)
                except Exception:
                    pass
            super().reject()

    dlg = Dialog()

    # обычное поведение окна Windows (resize + maximize)
    dlg.setWindowFlag(QtCore.Qt.WindowType.WindowMaximizeButtonHint, True)
    dlg.setSizeGripEnabled(True)

    # старт на весь экран (если хочешь)
    dlg.showMaximized()

    dlg.exec()


def prepare_lineid(
    cfg: dict,
    *,
    superneon_fits: str | Path,
    peaks_candidates_csv: str | Path,
    hand_file: str | Path,
    neon_lines_csv: str | Path | None = None,
    y_half: int = 20,
    title: Optional[str] = None,
) -> None:
    base = Path(str(cfg.get("config_dir", "."))).resolve()
    work_dir = Path(str(cfg["work_dir"]))
    if not work_dir.is_absolute():
        work_dir = (base / work_dir).resolve()

    superneon_fits = Path(str(superneon_fits))
    peaks_candidates_csv = Path(str(peaks_candidates_csv))
    hand_file = Path(str(hand_file))
    if not superneon_fits.is_absolute():
        superneon_fits = (work_dir / superneon_fits).resolve()
    if not peaks_candidates_csv.is_absolute():
        peaks_candidates_csv = (work_dir / peaks_candidates_csv).resolve()
    if not hand_file.is_absolute():
        hand_file = (work_dir / hand_file).resolve()

    wcfg = cfg.get("wavesol", {}) or {}

    # P0-M: resolve lamp_type and choose the most appropriate line list.
    from scorpio_pipe.lamp_contract import resolve_lamp_type, resolve_linelist_csv_path

    instrument_hint = str(((cfg.get("frames", {}) or {}).get("__setup__", {}) or {}).get("instrument", "") or "")
    lamp_info = resolve_lamp_type(
        cfg,
        arc_path=superneon_fits,
        instrument_hint=instrument_hint,
    )

    # Backward compatibility: "neon_lines_csv" remains supported.
    # New preferred key: wavesol.linelist_csv, or auto by lamp_type.
    if neon_lines_csv is None:
        neon_lines_csv = wcfg.get("linelist_csv") or wcfg.get("neon_lines_csv")
    if neon_lines_csv is None:
        neon_lines_csv = resolve_linelist_csv_path(cfg, lamp_info.lamp_type)

    from scorpio_pipe.resource_utils import resolve_resource

    neon_lines_csv_res = resolve_resource(
        neon_lines_csv,
        work_dir=work_dir,
        config_dir=base,
        project_root=cfg.get("project_root"),
        allow_package=True,
    )
    neon_lines_csv = neon_lines_csv_res.path
    img = fits.getdata(superneon_fits, memmap=False).astype(float)
    prof = _profile_1d(img, y_half=y_half)
    x = np.arange(prof.size, dtype=float)

    pk_x, pk_amp = _read_peaks_candidates(peaks_candidates_csv)

    # Hard guard: line list waveref must match wavesol.waveref to avoid systematic shifts.
    ll_meta = _read_linelist_meta(neon_lines_csv)
    cfg_ref = str(wcfg.get("wave_ref", wcfg.get("waveref", "air")) or "air").strip().lower()
    if cfg_ref not in {"air", "vacuum"}:
        cfg_ref = "air"
    if ll_meta.get("waveref") and ll_meta["waveref"] != cfg_ref:
        raise RuntimeError(
            f"neon_lines.csv WAVEREF mismatch: linelist={ll_meta['waveref']!r}, config={cfg_ref!r}. "
            "Fix one of them (wavesol.waveref or the linelist header) before calibrating."
        )

    ref_lams = _read_neon_lines_csv(neon_lines_csv)

    # --- Atlas (HeNeAr) ---
    atlas_pdf = wcfg.get("atlas_pdf", "HeNeAr_atlas.pdf")
    atlas_path: Optional[Path] = None
    if atlas_pdf:
        from scorpio_pipe.resource_utils import resolve_resource_maybe

        atlas_res = resolve_resource_maybe(
            atlas_pdf,
            work_dir=work_dir,
            config_dir=base,
            project_root=cfg.get("project_root"),
            allow_package=True,
        )
        atlas_path = atlas_res.path if atlas_res else None

    # Try to open the relevant atlas page automatically (0-indexed) + set λ window from instrument DB
    setup = (cfg.get("frames", {}) or {}).get("__setup__", {}) or {}
    disp_raw = str(setup.get("disperser", "") or "").strip()
    disp_norm = "".join(ch for ch in disp_raw.upper() if ch.isalnum())

    atlas_page0: Optional[int] = None
    if disp_norm:
        # SCORPIO-1 (classic pages)
        if "GR300" in disp_norm:
            atlas_page0 = 0
        elif "VPHG550" in disp_norm and "G" in disp_raw.upper():
            atlas_page0 = 1

        # SCORPIO-2 (explicit central λ pages) — check these BEFORE generic "1200"/"1800"
        elif "1200540" in disp_norm or "1200860" in disp_norm:
            atlas_page0 = 5
        elif "1800590" in disp_norm:
            atlas_page0 = 6
        elif "2400415" in disp_norm:
            atlas_page0 = 7

        # Fallbacks
        elif "VPHG1200" in disp_norm:
            atlas_page0 = 2
        elif "VPHG1800" in disp_norm:
            atlas_page0 = 3
        elif "VPHG940" in disp_norm or "VPHG1026" in disp_norm:
            atlas_page0 = 4

    # Wavelength window (Å) for filtering the reference line list
    lam_min_A = None
    lam_max_A = None
    try:
        from scorpio_pipe.instrument_db import find_grism

        inst_name = None
        inst_section = (
            cfg.get("instrument") if isinstance(cfg.get("instrument"), dict) else {}
        )
        if isinstance(inst_section, dict):
            inst_name = str(inst_section.get("name") or "").strip() or None
        if not inst_name:
            inst_name = str(setup.get("instrument") or "").strip() or None

        gr = find_grism(inst_name, disp_raw)
        r = gr.get("range_A") if isinstance(gr, dict) else None
        if isinstance(r, (list, tuple)) and len(r) == 2:
            lam_min_A = float(r[0])
            lam_max_A = float(r[1])
    except Exception:
        lam_min_A = None
        lam_max_A = None
    t = title or f"LineID | {cfg.get('object', 'OBJECT')}"
    min_amp_default = wcfg.get("gui_min_amp", None)
    min_amp_sigma_k = float(wcfg.get("gui_min_amp_sigma_k", 5.0))
    run_lineid_gui(
        LineIdInputs(
            x=x,
            prof=prof,
            pk_x=pk_x,
            pk_amp=pk_amp,
            ref_lams=ref_lams,
            hand_file=hand_file,
            title=t,
            atlas_pdf=atlas_path,
            atlas_page0=atlas_page0,
            disperser=disp_raw,
            lam_min_A=lam_min_A,
            lam_max_A=lam_max_A,
            min_amp_default=(
                float(min_amp_default) if min_amp_default is not None else None
            ),
            min_amp_sigma_k=min_amp_sigma_k,
        )
    )
