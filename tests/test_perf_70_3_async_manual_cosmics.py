from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest


def _process_events_until(app, predicate, *, timeout_s: float = 2.0) -> bool:
    QtCore = pytest.importorskip('PySide6.QtCore')
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def _make_cfg(tmp_path: Path, work_dir: Path) -> dict[str, object]:
    return {
        'config_dir': str(tmp_path),
        'work_dir': str(work_dir),
        'cosmics': {},
    }


def _ensure_clean_files(work_dir: Path, names: list[str]) -> list[Path]:
    clean_dir = work_dir / '05_cosmics' / 'obj' / 'clean'
    clean_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for name in names:
        p = clean_dir / name
        p.write_bytes(b'fake-fits')
        out.append(p)
    return out


def test_manual_cosmics_loads_frames_async_and_keeps_event_loop_alive(monkeypatch, tmp_path: Path) -> None:
    QtCore = pytest.importorskip('PySide6.QtCore')
    QtWidgets = pytest.importorskip('PySide6.QtWidgets')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

    from scorpio_pipe.ui import cosmics_manual as mod

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    work_dir = tmp_path / 'work'
    files = _ensure_clean_files(work_dir, ['a_clean.fits', 'b_clean.fits'])
    cfg = _make_cfg(tmp_path, work_dir)

    mod._LAST_FRAME_CACHE_KEY = None
    mod._LAST_FRAME_CACHE_PAYLOAD = None

    calls: list[str] = []

    def _slow_loader(path: Path, *, work_dir: Path, kind: str):
        calls.append(Path(path).name)
        time.sleep(0.6)
        img = np.full((512, 512), float(len(calls)), dtype=np.float32)
        auto = np.zeros_like(img, dtype=bool)
        manual = np.zeros_like(img, dtype=bool)
        disp = mod._prepare_display_payload(img, auto, manual, max_dim=256)
        return mod._FramePayload(
            path=Path(path),
            key=mod._file_cache_key(Path(path)),
            image=img,
            header={},
            auto_mask=auto,
            manual_mask=manual,
            display=disp,
        )

    monkeypatch.setattr(mod, '_load_frame_payload_sync', _slow_loader)

    t0 = time.monotonic()
    dlg = mod.CosmicsManualDialog(cfg)
    dt = time.monotonic() - t0
    dlg.show()

    assert dt < 0.5, f'CosmicsManualDialog constructor blocked for {dt:.3f}s'
    assert calls == ['a_clean.fits']
    assert dlg._frame_busy is True
    assert 'Loading' in dlg.lbl_help.text()

    fired: list[float] = []
    QtCore.QTimer.singleShot(100, lambda: fired.append(time.monotonic()))
    assert _process_events_until(app, lambda: bool(fired), timeout_s=0.8), 'event loop stalled while first frame was loading'
    assert _process_events_until(app, lambda: dlg._img is not None and dlg._frame_busy is False, timeout_s=2.0), 'first frame never finished loading'

    fired2: list[float] = []
    QtCore.QTimer.singleShot(100, lambda: fired2.append(time.monotonic()))
    t1 = time.monotonic()
    dlg._step_frame(+1)
    dt2 = time.monotonic() - t1
    assert dt2 < 0.5, f'_step_frame(+1) blocked for {dt2:.3f}s'
    assert calls == ['a_clean.fits', 'b_clean.fits']
    assert _process_events_until(app, lambda: bool(fired2), timeout_s=0.8), 'event loop stalled while switching frames'
    assert _process_events_until(app, lambda: dlg.frame_idx == 1 and dlg._img is not None and dlg._frame_busy is False, timeout_s=2.0), 'second frame never finished loading'
    dlg.close()



def test_manual_cosmics_reuses_last_frame_cache_until_mtime_changes(monkeypatch, tmp_path: Path) -> None:
    QtWidgets = pytest.importorskip('PySide6.QtWidgets')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

    from scorpio_pipe.ui import cosmics_manual as mod

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    work_dir = tmp_path / 'work2'
    files = _ensure_clean_files(work_dir, ['same_clean.fits'])
    p = files[0]
    cfg = _make_cfg(tmp_path, work_dir)

    mod._LAST_FRAME_CACHE_KEY = None
    mod._LAST_FRAME_CACHE_PAYLOAD = None

    calls = {'n': 0}

    def _loader(path: Path, *, work_dir: Path, kind: str):
        calls['n'] += 1
        img = np.full((128, 128), float(calls['n']), dtype=np.float32)
        auto = np.zeros_like(img, dtype=bool)
        manual = np.zeros_like(img, dtype=bool)
        disp = mod._prepare_display_payload(img, auto, manual, max_dim=128)
        return mod._FramePayload(
            path=Path(path),
            key=mod._file_cache_key(Path(path)),
            image=img,
            header={},
            auto_mask=auto,
            manual_mask=manual,
            display=disp,
        )

    monkeypatch.setattr(mod, '_load_frame_payload_sync', _loader)

    dlg = mod.CosmicsManualDialog(cfg)
    dlg.show()
    assert _process_events_until(app, lambda: dlg._img is not None and dlg._frame_busy is False, timeout_s=1.0)
    assert calls['n'] == 1

    t0 = time.monotonic()
    dlg._load_frame(0)
    dt = time.monotonic() - t0
    assert dt < 0.2, f'cached _load_frame(0) blocked for {dt:.3f}s'
    app.processEvents()
    assert calls['n'] == 1, 'same path+mtime must use the last-frame cache'

    time.sleep(0.01)
    p.write_bytes(b'fake-fits-v2')
    dlg._load_frame(0)
    assert _process_events_until(app, lambda: calls['n'] == 2 and dlg._frame_busy is False, timeout_s=1.0), 'mtime change must invalidate cache'
    dlg.close()
