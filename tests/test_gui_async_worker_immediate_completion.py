"""Fast worker results must remain observable even if work finishes immediately."""
from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtCore = pytest.importorskip("PySide6.QtCore")

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
    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
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
    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
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
