from __future__ import annotations

import time

import pytest


def _process_events_until(app, predicate, *, timeout_s: float = 2.0) -> bool:
    QtCore = pytest.importorskip('PySide6.QtCore')
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        app.processEvents(QtCore.QEventLoop.AllEvents, 25)
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def test_render_throttle_debounces_bursty_requests() -> None:
    QtCore = pytest.importorskip('PySide6.QtCore')

    from scorpio_pipe.ui.mpl_render import RenderThrottle

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

    fired: list[float] = []
    throttle = RenderThrottle(app, lambda: fired.append(time.monotonic()), max_fps=10.0, debounce_ms=180)

    t0 = time.monotonic()
    for _ in range(4):
        throttle.request()
        app.processEvents(QtCore.QEventLoop.AllEvents, 5)
        time.sleep(0.02)

    assert not fired, 'debounced burst must not render immediately'
    assert _process_events_until(app, lambda: len(fired) == 1, timeout_s=1.0)
    assert len(fired) == 1
    assert fired[0] - t0 >= 0.14


def test_render_throttle_caps_immediate_requests_to_about_10fps() -> None:
    QtCore = pytest.importorskip('PySide6.QtCore')

    from scorpio_pipe.ui.mpl_render import RenderThrottle

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

    fired: list[float] = []
    throttle = RenderThrottle(app, lambda: fired.append(time.monotonic()), max_fps=10.0, debounce_ms=180)

    throttle.request_immediate()
    assert _process_events_until(app, lambda: len(fired) == 1, timeout_s=0.5)

    t1 = time.monotonic()
    throttle.request_immediate()
    app.processEvents(QtCore.QEventLoop.AllEvents, 5)
    time.sleep(0.03)
    app.processEvents(QtCore.QEventLoop.AllEvents, 5)
    assert len(fired) == 1, 'second immediate request must be throttled, not executed instantly'

    assert _process_events_until(app, lambda: len(fired) == 2, timeout_s=0.5)
    assert fired[1] - fired[0] >= 0.08
    assert fired[1] - t1 >= 0.06
