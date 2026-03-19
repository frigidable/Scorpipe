from __future__ import annotations

import os
import time
from threading import Event

import pytest


def _wait_until(pred, *, timeout_s: float = 2.0, step_s: float = 0.01) -> bool:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(step_s)
    return bool(pred())


def test_ui_job_registry_latest_wins_dedupes_queue(monkeypatch) -> None:
    QtCore = pytest.importorskip('PySide6.QtCore')
    _app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

    from scorpio_pipe.ui.async_worker import (
        _reset_job_registry_for_tests,
        get_job_registry_snapshot,
        run_in_thread,
    )

    monkeypatch.setenv('SCORPIPE_UI_QUEUE_LIMIT', '8')
    _reset_job_registry_for_tests(max_threads=1)

    gate = Event()
    ran: list[int] = []

    blocker = run_in_thread(lambda: gate.wait(2.0), name='blocker')
    assert _wait_until(lambda: get_job_registry_snapshot().running_jobs >= 1, timeout_s=1.0)

    handles = []
    for idx in range(5):
        h = run_in_thread(
            lambda idx=idx: ran.append(idx),
            name='spam',
            dedupe_key='spam-key',
        )
        handles.append(h)
        time.sleep(0.02)

    snap = get_job_registry_snapshot()
    assert snap.running_jobs == 1
    assert snap.queued_jobs <= 1
    assert snap.deduped_total >= 4
    assert snap.cancel_requested_total >= 4

    gate.set()
    assert blocker.wait(1500)
    assert _wait_until(lambda: handles[-1].is_done, timeout_s=2.0)
    assert ran == [4], f'latest-wins dedupe must keep only the newest queued job, got {ran!r}'


def test_ui_job_registry_queue_limit_evicts_oldest_queued(monkeypatch) -> None:
    QtCore = pytest.importorskip('PySide6.QtCore')
    _app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

    from scorpio_pipe.ui.async_worker import (
        _reset_job_registry_for_tests,
        get_job_registry_snapshot,
        run_in_thread,
    )

    monkeypatch.setenv('SCORPIPE_UI_QUEUE_LIMIT', '2')
    _reset_job_registry_for_tests(max_threads=1)

    gate = Event()
    ran: list[str] = []

    blocker = run_in_thread(lambda: gate.wait(2.0), name='blocker')
    assert _wait_until(lambda: get_job_registry_snapshot().running_jobs >= 1, timeout_s=1.0)

    h1 = run_in_thread(lambda: ran.append('a'), name='a', latest_wins=False)
    h2 = run_in_thread(lambda: ran.append('b'), name='b', latest_wins=False)
    h3 = run_in_thread(lambda: ran.append('c'), name='c', latest_wins=False)

    snap = get_job_registry_snapshot()
    assert snap.queued_jobs <= 2
    assert snap.queue_evicted_total >= 1

    gate.set()
    assert blocker.wait(1500)
    assert _wait_until(lambda: h2.is_done and h3.is_done, timeout_s=2.0)
    time.sleep(0.05)

    assert 'a' not in ran, f'oldest queued job should be evicted when queue limit is exceeded, got {ran!r}'
    assert set(ran).issubset({'b', 'c'})
    assert h1.is_done
