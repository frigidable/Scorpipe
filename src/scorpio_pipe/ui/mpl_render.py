from __future__ import annotations

import time
from typing import Callable

from PySide6 import QtCore


class RenderThrottle(QtCore.QObject):
    """Coalesce expensive UI redraw/recompute requests.

    The scheduler enforces a maximum trigger rate (default 10 fps) and can
    additionally debounce hot UI signals such as slider/ROI updates.
    """

    def __init__(
        self,
        owner: QtCore.QObject | None,
        callback: Callable[[], None],
        *,
        max_fps: float = 10.0,
        debounce_ms: int = 180,
    ) -> None:
        super().__init__(owner)
        self._callback = callback
        self._default_debounce_ms = max(0, int(debounce_ms))
        self._min_interval_ms = 0
        if max_fps > 0:
            self._min_interval_ms = max(0, int(round(1000.0 / float(max_fps))))
        self._last_fire_ts = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._fire)

    def cancel(self) -> None:
        self._timer.stop()

    def request(self, *, debounce_ms: int | None = None) -> None:
        debounce = self._default_debounce_ms if debounce_ms is None else max(0, int(debounce_ms))
        now = time.monotonic()
        remaining_ms = 0
        if self._last_fire_ts > 0.0 and self._min_interval_ms > 0:
            elapsed_ms = int(round((now - self._last_fire_ts) * 1000.0))
            remaining_ms = max(0, self._min_interval_ms - elapsed_ms)
        delay_ms = max(debounce, remaining_ms)
        if delay_ms <= 0:
            self._timer.stop()
            self._fire()
            return
        self._timer.start(delay_ms)

    def request_immediate(self) -> None:
        self.request(debounce_ms=0)

    def flush(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
        self._fire()

    def _fire(self) -> None:
        self._last_fire_ts = time.monotonic()
        self._callback()
