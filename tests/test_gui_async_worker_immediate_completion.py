"""Fast worker results must remain observable even if work finishes immediately."""
from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtCore = pytest.importorskip("PySide6.QtCore")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from scorpio_pipe.ui.async_worker import run_in_thread  # noqa: E402


def _wait_for(app, predicate, seconds: float = 3.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def test_immediate_background_results_are_never_lost_before_subscribing() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    results = []
    handles = []
    for i in range(16):
        handles.append(
            run_in_thread(
                lambda value=i: value,
                name="test-immediate-completion",
                on_finished=results.append,
            )
        )

    assert _wait_for(app, lambda: len(results) == 16)
    assert sorted(results) == list(range(16))
    assert all(h.is_done for h in handles)


def test_immediate_background_failure_is_delivered() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    failures = []

    def fail():
        raise ValueError("synthetic-fast-worker-failure")

    handle = run_in_thread(
        fail,
        name="test-immediate-error",
        on_failed=failures.append,
    )
    assert _wait_for(app, lambda: bool(failures))
    assert "synthetic-fast-worker-failure" in failures[0]
    assert handle.is_done


@pytest.mark.parametrize("raise_in_worker", [False, True])
def test_worker_survives_deleted_qt_signal_owner(raise_in_worker: bool) -> None:
    """A cancelled/closing GUI must not crash the background QRunnable."""
    import threading
    import shiboken6

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    started = threading.Event()
    release = threading.Event()

    def job():
        started.set()
        if not release.wait(timeout=3.0):
            raise TimeoutError("test worker was never released")
        if raise_in_worker:
            raise ValueError("expected-after-destroy")
        return "done"

    handle = run_in_thread(job, name="test-deleted-qt-signal-owner")
    try:
        assert started.wait(timeout=2.0)
        shiboken6.delete(handle.worker)
    finally:
        release.set()
    assert handle.wait(timeout_ms=3000)
    assert handle.is_done


@pytest.mark.parametrize("raises", [False, True])
def test_terminal_state_visible_inside_completion_callback(raises: bool) -> None:
    """Callbacks must not race RunHandle._done, even under immediate delivery."""
    import threading

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    started = threading.Event()
    release = threading.Event()
    seen = []

    def job():
        started.set()
        if not release.wait(timeout=3.):
            raise RuntimeError("test worker release timed out")
        if raises:
            raise ValueError("intentional-terminal-state-failure")
        return "completed"

    handle = run_in_thread(
        job,
        name="test-worker-terminal-callback-order",
        on_finished=lambda result: seen.append(("finished", result, handle.is_done)),
        on_failed=lambda traceback: seen.append(("failed", traceback, handle.is_done)),
    )
    try:
        assert started.wait(timeout=2.), "Worker was not scheduled"
    finally:
        release.set()

    assert _wait_for(app, lambda: len(seen) == 1)
    assert handle.is_done
    if raises:
        assert seen[0][0] == "failed"
        assert "intentional-terminal-state-failure" in seen[0][1]
    else:
        assert seen[0][:2] == ("finished", "completed")
    assert seen[0][2] is True, "Terminal callback observed a nonterminal handle"
