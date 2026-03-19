"""UI freeze watchdog.

Goal
----
When the Qt event loop stalls ("Not responding"), we want actionable evidence:
the Python stack of the main thread at the moment of the stall.

Implementation
--------------
- A small QTimer in the UI thread updates a heartbeat timestamp.
- A dedicated Python thread watches the timestamp.
- If the heartbeat is missing for > threshold seconds, we append a dump file:
  - main-thread stack (via ``sys._current_frames()``)
  - stacks for *all* Python threads (useful when deadlocks happen)
  - snapshot of Scorpipe-managed QThreads (best-effort)
  - log backlog size (queued UI log lines)

The watchdog is intentionally best-effort and never raises.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


def _truthy_env(name: str) -> bool:
    v = os.environ.get(name)
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _user_log_dir() -> Path:
    # Mirror ui.bootstrap logic, but keep this module standalone.
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if base:
        p = Path(base) / "Scorpipe" / "logs"
    else:
        p = Path.home() / ".local" / "share" / "Scorpipe" / "logs"
    try:
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        import tempfile

        p = Path(tempfile.gettempdir()) / "Scorpipe" / "logs"
        p.mkdir(parents=True, exist_ok=True)
        return p


def _default_dump_path() -> Path:
    return _user_log_dir() / "ui_freeze_dump.txt"


def _should_enable(settings: Any | None = None) -> bool:
    """Decide whether to enable the watchdog.

    In developer mode (not frozen) we enable by default.
    In frozen builds we keep it OFF by default to avoid noise.
    Enable via env vars / CLI flags / QSettings.
    """

    if not bool(getattr(sys, "frozen", False)):
        return True

    if _truthy_env("SCORPIPE_UI_WATCHDOG") or _truthy_env("SCORPIPE_FREEZE_WATCHDOG"):
        return True
    if _truthy_env("SCORPIPE_UI_DEBUG") or _truthy_env("SCORPIPE_DEBUG"):
        return True

    argv = " ".join(sys.argv or [])
    if "--watchdog" in argv or "--debug" in argv or "--advanced" in argv:
        return True

    try:
        if settings is not None:
            v1 = settings.value("ui/freeze_watchdog", False, type=bool)
            v2 = settings.value("ui/debug", False, type=bool)
            return bool(v1 or v2)
    except Exception:
        pass
    return False


def _format_stack(frame) -> str:
    try:
        return "".join(traceback.format_stack(frame))
    except Exception:
        return "<failed to format stack>\n"


def _recent_ui_actions() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        from scorpio_pipe.ui import slow_action_profiler  # noqa: PLC0415

        return (
            slow_action_profiler.get_recent_ui_actions(limit=50),
            slow_action_profiler.get_active_ui_actions(limit=20),
        )
    except Exception:
        return ([], [])


def _format_ui_action_line(item: dict[str, Any]) -> str:
    name = str(item.get("name", "?"))
    phase = str(item.get("phase", "?"))
    action_id = item.get("action_id", "?")
    ts = str(item.get("ts_utc", "?"))
    elapsed = item.get("elapsed_ms")
    thr = item.get("threshold_ms")
    thread_name = str(item.get("thread_name", "?"))
    thread_ident = item.get("thread_ident")
    extra = item.get("extra") or {}
    backlog_lines = extra.get("ui_log_backlog_lines")
    backlog_chars = extra.get("ui_log_backlog_chars")
    elapsed_s = "?" if elapsed is None else f"{float(elapsed):.1f}"
    try:
        thr_s = f"{float(thr):.1f}"
    except Exception:
        thr_s = "?"
    return (
        f"id={action_id} ts={ts} phase={phase} name={name!r} elapsed_ms={elapsed_s} "
        f"threshold_ms={thr_s} thread={thread_name!r}/{thread_ident} "
        f"ui_log_backlog_lines={backlog_lines} ui_log_backlog_chars={backlog_chars}"
    )


def _infer_freeze_hints(
    *,
    main_stack: str,
    job_lines: list[str],
    log_backlog: dict[str, int] | None,
    active_actions: list[dict[str, Any]],
    recent_actions: list[dict[str, Any]],
) -> list[str]:
    hints: list[str] = []
    stack_lower = str(main_stack).lower()

    if active_actions:
        names = ", ".join(sorted({str(a.get("name", "?")) for a in active_actions}))
        hints.append(f"UI thread appears to be inside profiled action(s): {names}.")

    if any(tok in stack_lower for tok in ("matplotlib", "draw_idle", "figurecanvas", "imshow", "paintevent")):
        hints.append("Main-thread stack points at matplotlib/paint redraw work on the UI thread.")

    if any(tok in stack_lower for tok in ("astropy", "fits.open", "read_image", "memmap", "hdulist")):
        hints.append("Main-thread stack points at synchronous FITS/file I/O on the UI thread.")

    if any(tok in stack_lower for tok in ("glob", "scandir", "walk", "iterdir", "stat(")):
        hints.append("Main-thread stack points at synchronous filesystem traversal/stat calls.")

    if job_lines:
        running = sum(1 for line in job_lines if " running=True" in line)
        queued = sum(1 for line in job_lines if " queued=True" in line)
        if queued >= 50:
            hints.append(f"Worker backlog is high: {queued} queued UI jobs at dump time.")
        elif queued >= 1 and running == 0:
            hints.append(f"UI jobs are queued ({queued}) but none are running; check pool starvation/deadlock.")

    if log_backlog:
        lines = int(log_backlog.get("lines", 0))
        chars = int(log_backlog.get("chars", 0))
        if lines >= 1000 or chars >= 1_000_000:
            hints.append(
                f"UI log backlog is large ({lines} queued lines, ~{chars} chars); log flood may amplify jank."
            )

    last_action = None
    for item in reversed(recent_actions):
        if str(item.get("phase")) == "begin":
            last_action = str(item.get("name", "?"))
            break
    if last_action and not active_actions:
        hints.append(f"The last profiled UI action before the stall started as: {last_action!r}.")

    if not hints:
        hints.append("No strong heuristic reason inferred; inspect the main-thread stack and recent UI actions below.")
    return hints


class FreezeWatchdog:
    """Watch for UI event-loop stalls and dump stacks to a file."""

    def __init__(
        self,
        *,
        heartbeat_ms: int = 250,
        stall_threshold_s: float = 1.25,
        check_interval_s: float = 0.2,
        dump_path: Path | None = None,
        min_seconds_between_dumps: float = 10.0,
    ) -> None:
        self.heartbeat_ms = int(max(50, heartbeat_ms))
        self.stall_threshold_s = float(max(0.5, stall_threshold_s))
        self.check_interval_s = float(max(0.05, check_interval_s))
        self.dump_path = dump_path or _default_dump_path()
        self.min_seconds_between_dumps = float(max(1.0, min_seconds_between_dumps))

        self._last_beat = time.monotonic()
        self._last_dump = 0.0
        self._stop = threading.Event()

        # UI-thread snapshots (updated from heartbeat, safe for watchdog thread to read).
        self._qt_threadpool_active = 0
        self._qt_threadpool_max = 0

        self._timer = None
        self._thread: threading.Thread | None = None

    def start(self, qt_owner) -> None:
        """Start the watchdog.

        Parameters
        ----------
        qt_owner:
            Any QObject living in the UI thread (e.g. QApplication or main window).
            We parent the heartbeat QTimer to it.
        """

        try:
            from PySide6 import QtCore  # noqa: PLC0415

            self._timer = QtCore.QTimer(qt_owner)
            self._timer.setInterval(self.heartbeat_ms)
            self._timer.timeout.connect(self._beat)
            self._timer.start()
        except Exception:
            return

        self._thread = threading.Thread(
            target=self._monitor,
            name="scorpipe-freeze-watchdog",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._timer is not None:
                self._timer.stop()
        except Exception:
            pass
        try:
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=0.5)
        except Exception:
            pass

    def _beat(self) -> None:
        self._last_beat = time.monotonic()
        # Update Qt thread-pool stats in the UI thread (best-effort).
        try:
            from PySide6 import QtCore  # noqa: PLC0415

            pool = QtCore.QThreadPool.globalInstance()
            self._qt_threadpool_active = int(pool.activeThreadCount())
            self._qt_threadpool_max = int(pool.maxThreadCount())
        except Exception:
            pass

    def _monitor(self) -> None:
        log = logging.getLogger("scorpipe.ui.freeze_watchdog")
        while not self._stop.is_set():
            time.sleep(self.check_interval_s)
            now = time.monotonic()
            dt = now - self._last_beat
            if dt < self.stall_threshold_s:
                continue
            if (now - self._last_dump) < self.min_seconds_between_dumps:
                continue

            self._last_dump = now
            try:
                self._write_dump(stall_seconds=dt)
                log.warning("UI freeze watchdog: stall %.2fs -> wrote %s", dt, self.dump_path)
            except Exception:
                log.exception("UI freeze watchdog: failed to write dump")

    def _write_dump(self, *, stall_seconds: float) -> None:
        self.dump_path.parent.mkdir(parents=True, exist_ok=True)

        main_ident = threading.main_thread().ident
        try:
            frames = sys._current_frames()  # type: ignore[attr-defined]
        except Exception:
            frames = {}

        main_stack = "<main thread frame not found>\n"
        if main_ident is not None and main_ident in frames:
            main_stack = _format_stack(frames[main_ident])

        # Best-effort snapshots (avoid calling Qt from this thread).
        qthreads_info: list[str] = []
        job_lines: list[str] = []
        try:
            from scorpio_pipe.ui import async_worker  # noqa: PLC0415

            qthreads_info = async_worker.snapshot_active_qthreads()
            job_lines = [line for line in qthreads_info if str(line).startswith("job ")]
        except Exception:
            qthreads_info = []
            job_lines = []

        log_backlog: dict[str, int] | None
        try:
            from scorpio_pipe.ui import qt_log  # noqa: PLC0415

            log_backlog = qt_log.get_backlog_snapshot()
        except Exception:
            log_backlog = None

        recent_actions, active_actions = _recent_ui_actions()
        freeze_hints = _infer_freeze_hints(
            main_stack=main_stack,
            job_lines=job_lines,
            log_backlog=log_backlog,
            active_actions=active_actions,
            recent_actions=recent_actions,
        )

        with self.dump_path.open("a", encoding="utf-8") as f:
            f.write("\n" + "=" * 78 + "\n")
            f.write(f"Scorpipe UI FREEZE dump: {datetime.now().isoformat()}\n")
            f.write(f"PID: {os.getpid()}\n")
            f.write(f"argv: {sys.argv!r}\n")
            f.write(f"stall_seconds: {stall_seconds:.3f}\n")
            f.write(
                f"qt_threadpool_active: {int(self._qt_threadpool_active)} / {int(self._qt_threadpool_max)}\n"
            )
            if log_backlog is not None:
                f.write(
                    f"ui_log_backlog: lines={int(log_backlog.get('lines', 0))} chars={int(log_backlog.get('chars', 0))}\n"
                )
            if main_ident is not None:
                f.write(f"main_thread_ident: {main_ident}\n")

            f.write("\n-- Freeze reason hints --\n")
            for hint in freeze_hints:
                f.write(f"- {hint}\n")

            f.write("\n-- Main thread stack --\n")
            f.write(main_stack)

            f.write("\n-- Active QRunnable jobs (Scorpipe-managed) --\n")
            if job_lines:
                for line in job_lines:
                    f.write(line + "\n")
            else:
                f.write("<no active tracked jobs>\n")

            f.write("\n-- UI actions currently in flight --\n")
            if active_actions:
                for item in active_actions:
                    f.write(_format_ui_action_line(item) + "\n")
            else:
                f.write("<no active profiled UI actions>\n")

            f.write("\n-- Last 50 profiled UI actions --\n")
            if recent_actions:
                for item in recent_actions[-50:]:
                    f.write(_format_ui_action_line(item) + "\n")
            else:
                f.write("<no profiled UI actions recorded>\n")

            f.write("\n-- Python threads --\n")
            for th in threading.enumerate():
                f.write(
                    f"- name={th.name!r} ident={th.ident} alive={th.is_alive()} daemon={th.daemon}\n"
                )

            f.write("\n-- All thread stacks (best-effort) --\n")
            for ident, fr in frames.items():
                f.write(f"\n### thread ident: {ident}\n")
                f.write(_format_stack(fr))

            f.write("\n-- Worker pool snapshot (full) --\n")
            if qthreads_info:
                for line in qthreads_info:
                    f.write(line + "\n")
            else:
                f.write("<no tracked QThreads or snapshot unavailable>\n")


def install_if_enabled(qt_owner, *, settings: Any | None = None) -> FreezeWatchdog | None:
    """Create + start watchdog if enabled, else return None."""

    if not _should_enable(settings=settings):
        return None

    wd = FreezeWatchdog()
    wd.start(qt_owner)

    # Stop on exit if possible.
    try:
        from PySide6 import QtCore  # noqa: PLC0415

        app = QtCore.QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(wd.stop)
    except Exception:
        pass

    try:
        logging.getLogger("scorpipe.ui.freeze_watchdog").info(
            "Freeze watchdog enabled (dump=%s, threshold=%.2fs)", wd.dump_path, wd.stall_threshold_s
        )
    except Exception:
        pass
    return wd
