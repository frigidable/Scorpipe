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


def test_sky_roi_dialog_quick_peek_runs_async_and_keeps_event_loop_alive(monkeypatch, tmp_path: Path) -> None:
    QtCore = pytest.importorskip('PySide6.QtCore')
    QtWidgets = pytest.importorskip('PySide6.QtWidgets')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

    from scorpio_pipe.ui import sky_roi_dialog as mod

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    fits_path = tmp_path / 'frame.fits'
    fits_path.write_bytes(b'fake')

    monkeypatch.setattr(
        mod,
        '_read_roi_header_info',
        lambda _path: mod.RoiHeaderInfo(ny=4096, nx=4096, hdu_index=0, object_name='NGC2784'),
    )

    calls = {'quick': 0, 'full': 0}

    def _slow_quick(_path, *, max_preview_nx=1024):
        calls['quick'] += 1
        time.sleep(0.6)
        return mod.RoiImagePayload(
            image=np.zeros((4096, 256), dtype=np.float32),
            ny=4096,
            nx=4096,
            hdu_index=0,
            mode='quick',
            x_step=16,
            source_note='Quick peek: every 16-th X column (section)',
            object_name='NGC2784',
        )

    def _slow_full(_path):
        calls['full'] += 1
        time.sleep(0.6)
        return mod.RoiImagePayload(
            image=np.zeros((4096, 4096), dtype=np.float32),
            ny=4096,
            nx=4096,
            hdu_index=0,
            mode='full',
            x_step=1,
            source_note='Full resolution',
            object_name='NGC2784',
        )

    monkeypatch.setattr(mod, '_load_roi_quick_peek', _slow_quick)
    monkeypatch.setattr(mod, '_load_roi_full_resolution', _slow_full)

    t0 = time.monotonic()
    dlg = mod.SkyRoiDialog(fits_path)
    dt = time.monotonic() - t0
    dlg.show()

    assert dt < 0.5, f'SkyRoiDialog constructor blocked for {dt:.3f}s'
    assert calls['quick'] == 1
    assert calls['full'] == 0
    assert dlg.img is None
    assert 'Loading quick peek' in dlg.lbl_status.text()

    fired: list[float] = []
    QtCore.QTimer.singleShot(100, lambda: fired.append(time.monotonic()))
    assert _process_events_until(app, lambda: bool(fired), timeout_s=0.8), 'event loop stalled while quick peek was loading'

    assert _process_events_until(app, lambda: dlg.img is not None, timeout_s=2.0), 'quick peek never finished'
    assert dlg._render_mode == 'quick'
    assert dlg.btn_full_res.isEnabled()
    assert 'Quick peek' in dlg.lbl_status.text()
    dlg.close()



def test_sky_roi_dialog_full_resolution_is_on_demand_and_async(monkeypatch, tmp_path: Path) -> None:
    QtCore = pytest.importorskip('PySide6.QtCore')
    QtWidgets = pytest.importorskip('PySide6.QtWidgets')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

    from scorpio_pipe.ui import sky_roi_dialog as mod

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    fits_path = tmp_path / 'frame_full.fits'
    fits_path.write_bytes(b'fake')

    monkeypatch.setattr(
        mod,
        '_read_roi_header_info',
        lambda _path: mod.RoiHeaderInfo(ny=2048, nx=4096, hdu_index=0, object_name='obj'),
    )

    calls = {'quick': 0, 'full': 0}

    def _fast_quick(_path, *, max_preview_nx=1024):
        calls['quick'] += 1
        return mod.RoiImagePayload(
            image=np.zeros((2048, 512), dtype=np.float32),
            ny=2048,
            nx=4096,
            hdu_index=0,
            mode='quick',
            x_step=8,
            source_note='Quick peek: every 8-th X column (section)',
            object_name='obj',
        )

    def _slow_full(_path):
        calls['full'] += 1
        time.sleep(0.6)
        return mod.RoiImagePayload(
            image=np.zeros((2048, 4096), dtype=np.float32),
            ny=2048,
            nx=4096,
            hdu_index=0,
            mode='full',
            x_step=1,
            source_note='Full resolution',
            object_name='obj',
        )

    monkeypatch.setattr(mod, '_load_roi_quick_peek', _fast_quick)
    monkeypatch.setattr(mod, '_load_roi_full_resolution', _slow_full)

    dlg = mod.SkyRoiDialog(fits_path)
    dlg.show()
    assert _process_events_until(app, lambda: dlg.img is not None, timeout_s=1.0)
    assert calls['quick'] == 1
    assert calls['full'] == 0
    assert dlg.btn_full_res.isEnabled()

    fired: list[float] = []
    QtCore.QTimer.singleShot(100, lambda: fired.append(time.monotonic()))
    t0 = time.monotonic()
    dlg.btn_full_res.click()
    dt = time.monotonic() - t0

    assert dt < 0.5, f'Full-resolution request blocked for {dt:.3f}s'
    assert calls['full'] == 1
    assert _process_events_until(app, lambda: bool(fired), timeout_s=0.8), 'event loop stalled while full resolution was loading'
    assert _process_events_until(app, lambda: dlg._render_mode == 'full', timeout_s=2.0), 'full resolution never finished'
    assert not dlg.btn_full_res.isEnabled()
    assert 'Full resolution' in dlg.lbl_status.text()
    dlg.close()
