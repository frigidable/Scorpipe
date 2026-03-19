from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import pytest

from scorpio_pipe.inspect import EXPECTED_COLUMNS, InspectCancelled, InspectResult, inspect_dataset


def test_inspect_dataset_supports_progress_and_cancel(monkeypatch, tmp_path: Path) -> None:
    files = []
    for i in range(8):
        p = tmp_path / f's{i:04d}.fits'
        p.write_text('stub', encoding='utf-8')
        files.append(p)

    monkeypatch.setattr('scorpio_pipe.inspect.iter_fits_files', lambda _root: iter(files))
    monkeypatch.setattr(
        'scorpio_pipe.inspect._open_fits_header_safe',
        lambda _fp: {'OBJECT': 'obj', 'EXPTIME': 1.0, 'NAXIS1': 4, 'NAXIS2': 2},
    )
    monkeypatch.setattr('scorpio_pipe.inspect.get_frame_meta', lambda *_a, **_kw: None)
    monkeypatch.setattr('scorpio_pipe.inspect.guess_instrument_from_header', lambda _hdr: 'SCORPIO-2')
    monkeypatch.setattr('scorpio_pipe.inspect.find_nightlog', lambda _root: None)

    seen: list[tuple[int, int]] = []
    cancel_after = {'armed': False}

    def _progress(info) -> None:
        seen.append((int(info.n_found), int(info.n_opened)))
        if int(info.n_found) >= 3:
            cancel_after['armed'] = True

    with pytest.raises(InspectCancelled):
        inspect_dataset(
            tmp_path,
            progress=_progress,
            should_cancel=lambda: bool(cancel_after['armed']),
        )

    assert seen, 'progress callback must be invoked'
    assert seen[0] == (0, 0)
    assert any(n_found >= 3 for n_found, _ in seen)
    assert max(n_found for n_found, _ in seen) < 8


def test_inspect_dataset_final_progress_matches_result(monkeypatch, tmp_path: Path) -> None:
    files = []
    for i in range(5):
        p = tmp_path / f's{i:04d}.fits'
        p.write_text('stub', encoding='utf-8')
        files.append(p)

    monkeypatch.setattr('scorpio_pipe.inspect.iter_fits_files', lambda _root: iter(files))
    monkeypatch.setattr(
        'scorpio_pipe.inspect._open_fits_header_safe',
        lambda _fp: {'OBJECT': 'obj', 'EXPTIME': 1.0, 'NAXIS1': 4, 'NAXIS2': 2},
    )
    monkeypatch.setattr('scorpio_pipe.inspect.get_frame_meta', lambda *_a, **_kw: None)
    monkeypatch.setattr('scorpio_pipe.inspect.guess_instrument_from_header', lambda _hdr: 'SCORPIO-2')
    monkeypatch.setattr('scorpio_pipe.inspect.find_nightlog', lambda _root: None)

    seen: list[tuple[int, int]] = []
    res = inspect_dataset(tmp_path, progress=lambda info: seen.append((int(info.n_found), int(info.n_opened))))

    assert isinstance(res, InspectResult)
    assert res.n_found == 5
    assert res.n_opened == 5
    assert seen[0] == (0, 0)
    assert seen[-1] == (5, 5)
    assert list(res.table.columns) == EXPECTED_COLUMNS


def test_launcher_inspect_runs_async_and_keeps_event_loop_alive(monkeypatch, tmp_path: Path) -> None:
    QtCore = pytest.importorskip('PySide6.QtCore')
    QtWidgets = pytest.importorskip('PySide6.QtWidgets')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

    from scorpio_pipe.ui.launcher_window import LauncherWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    data_dir = tmp_path / 'night'
    data_dir.mkdir()

    def _slow_inspect(data_dir: Path, max_files=None, *, progress=None, should_cancel=None):
        if progress is not None:
            class _Info:
                n_found = 0
                n_opened = 0
                current_path = ''

            progress(_Info())
        for _ in range(20):
            time.sleep(0.1)
            if should_cancel is not None and should_cancel():
                raise InspectCancelled('cancelled from test mock')
        return InspectResult(
            table=pd.DataFrame(columns=EXPECTED_COLUMNS),
            n_found=0,
            n_opened=0,
            nightlog_path=None,
            n_nightlog_rows=0,
            open_errors=[],
        )

    monkeypatch.setattr('scorpio_pipe.inspect.inspect_dataset', _slow_inspect)

    win = LauncherWindow()
    win.edit_data_dir.setText(str(data_dir))
    win.show()

    fired: list[float] = []
    QtCore.QTimer.singleShot(100, lambda: fired.append(time.monotonic()))

    t0 = time.monotonic()
    win._do_inspect()
    dt = time.monotonic() - t0

    assert dt < 0.5, f'_do_inspect() blocked the UI thread for {dt:.3f}s'

    deadline = time.monotonic() + 0.8
    while time.monotonic() < deadline and not fired:
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        time.sleep(0.01)

    assert fired, 'event loop did not stay responsive while inspect job was running'
    assert win.btn_inspect.text() in {'Cancel Inspect', 'Cancelling…'}

    win._cancel_inspect()
    deadline = time.monotonic() + 2.5
    while time.monotonic() < deadline:
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        if win._inspect_job is None:
            break
        time.sleep(0.01)

    assert win._inspect_job is None
    assert win.btn_inspect.text() == 'Inspect dataset'
    win.close()
