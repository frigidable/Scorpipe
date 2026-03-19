"""UI slow-action profiler.

PERF-00.3
--------
Log any UI-thread action that takes longer than a threshold (default: 200 ms).

Why
---
UI freezes / jank are often caused by a few blocking operations in the Qt event
loop. A lightweight, always-available profiler helps identify *which* action
and *where* it happened without an external profiler.

What it does
------------
Provides :func:`profile_ui_action` as a context manager (and a small decorator
helper).

If elapsed > threshold:
  - log a warning with duration + stack
  - append a JSON line to ``manifest/ui_perf.jsonl`` (best-effort)

The module is dependency-free and never raises from instrumentation.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import traceback
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator


log = logging.getLogger("scorpio_pipe.ui.slow")


_ACTIONS_LOCK = threading.Lock()
_ACTION_COUNTER = 0
_RECENT_ACTIONS: deque["UiActionRecord"] = deque(maxlen=512)
_ACTIVE_ACTIONS: dict[int, "UiActionRecord"] = {}


@dataclass
class UiActionRecord:
    action_id: int
    ts_utc: str
    name: str
    phase: str  # begin|end
    elapsed_ms: float | None
    threshold_ms: float
    pid: int
    thread_name: str
    thread_ident: int | None
    extra: dict[str, Any]
    stack: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _next_action_id() -> int:
    global _ACTION_COUNTER
    with _ACTIONS_LOCK:
        _ACTION_COUNTER += 1
        return int(_ACTION_COUNTER)


def _record_action_begin(*, name: str, threshold_ms: float, extra: dict[str, Any], stack: str) -> int:
    action_id = _next_action_id()
    rec = UiActionRecord(
        action_id=action_id,
        ts_utc=_utc_now_iso(),
        name=str(name),
        phase="begin",
        elapsed_ms=None,
        threshold_ms=float(threshold_ms),
        pid=os.getpid(),
        thread_name=threading.current_thread().name,
        thread_ident=threading.current_thread().ident,
        extra=dict(extra or {}),
        stack=str(stack),
    )
    with _ACTIONS_LOCK:
        _ACTIVE_ACTIONS[action_id] = rec
        _RECENT_ACTIONS.append(rec)
    return action_id


def _record_action_end(*, action_id: int, name: str, elapsed_ms: float, threshold_ms: float, extra: dict[str, Any], stack: str) -> None:
    rec = UiActionRecord(
        action_id=int(action_id),
        ts_utc=_utc_now_iso(),
        name=str(name),
        phase="end",
        elapsed_ms=float(elapsed_ms),
        threshold_ms=float(threshold_ms),
        pid=os.getpid(),
        thread_name=threading.current_thread().name,
        thread_ident=threading.current_thread().ident,
        extra=dict(extra or {}),
        stack=str(stack),
    )
    with _ACTIONS_LOCK:
        _ACTIVE_ACTIONS.pop(int(action_id), None)
        _RECENT_ACTIONS.append(rec)


def get_recent_ui_actions(limit: int = 50) -> list[dict[str, Any]]:
    try:
        lim = max(1, int(limit))
    except Exception:
        lim = 50
    with _ACTIONS_LOCK:
        items = list(_RECENT_ACTIONS)[-lim:]
    return [it.to_dict() for it in items]


def get_active_ui_actions(limit: int = 20) -> list[dict[str, Any]]:
    try:
        lim = max(1, int(limit))
    except Exception:
        lim = 20
    with _ACTIONS_LOCK:
        items = list(_ACTIVE_ACTIONS.values())[:lim]
    return [it.to_dict() for it in items]


def _reset_recent_ui_actions_for_tests() -> None:
    global _ACTION_COUNTER
    with _ACTIONS_LOCK:
        _ACTION_COUNTER = 0
        _RECENT_ACTIONS.clear()
        _ACTIVE_ACTIONS.clear()


def _utc_now_iso() -> str:
    try:
        return datetime.now(timezone.utc).isoformat()
    except Exception:
        # Fallback (avoid timezone-naive timestamps).
        return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()  # noqa: DTZ003


def _safe_stack(skip_tail: int = 0) -> str:
    try:
        st = traceback.format_stack()
        if skip_tail > 0:
            st = st[:-skip_tail]
        return "".join(st)
    except Exception:
        return "<failed to capture stack>\n"


def _infer_work_dir_from_path(p: Path, *, max_up: int = 8) -> Path | None:
    """Infer run root by walking up until a sibling ``manifest/`` exists."""

    try:
        cur = p
        for _ in range(max_up):
            cur = cur.parent
            if (cur / "manifest").exists():
                return cur
    except Exception:
        return None
    return None


def _fallback_log_dir() -> Path:
    """Best-effort location for profiling logs when work_dir is unknown."""

    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if base:
        p = Path(base) / "Scorpipe" / "logs"
    else:
        p = Path.home() / ".local" / "share" / "Scorpipe" / "logs"
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        import tempfile

        p = Path(tempfile.gettempdir()) / "Scorpipe" / "logs"
        p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_out_path(*, work_dir: Path | None, path_hint: Path | None) -> Path:
    wd = None
    if work_dir is not None:
        wd = Path(work_dir)
    elif path_hint is not None:
        wd = _infer_work_dir_from_path(Path(path_hint))

    if wd is not None:
        return wd / "manifest" / "ui_perf.jsonl"
    return _fallback_log_dir() / "ui_perf.jsonl"


@dataclass
class _ProfileEvent:
    name: str
    elapsed_ms: float
    threshold_ms: float
    stack: str
    ts_utc: str
    pid: int
    thread_name: str
    thread_ident: int | None
    extra: dict[str, Any]

    def as_json_line(self) -> str:
        d = {
            "ts_utc": self.ts_utc,
            "name": self.name,
            "elapsed_ms": round(float(self.elapsed_ms), 3),
            "threshold_ms": float(self.threshold_ms),
            "pid": int(self.pid),
            "thread": {"name": self.thread_name, "ident": self.thread_ident},
            "extra": self.extra or {},
            "stack": self.stack,
        }
        return json.dumps(d, ensure_ascii=False) + "\n"


def _emit_event(ev: _ProfileEvent, *, work_dir: Path | None, path_hint: Path | None) -> None:
    # 1) log warning with stack
    try:
        log.warning(
            "UI slow action: %s took %.1f ms (>%s ms)\n%s",
            ev.name,
            ev.elapsed_ms,
            int(ev.threshold_ms),
            ev.stack,
        )
    except Exception:
        pass

    # 2) append to ui_perf.jsonl
    try:
        out_path = _resolve_out_path(work_dir=work_dir, path_hint=path_hint)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(ev.as_json_line())
    except Exception:
        try:
            log.exception("Failed to write ui_perf.jsonl")
        except Exception:
            pass


@contextmanager
def profile_ui_action(
    name: str,
    *,
    threshold_ms: float = 200.0,
    work_dir: str | Path | None = None,
    path_hint: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Profile a UI action.

    If elapsed time exceeds ``threshold_ms``:
    - log + stack
    - append JSON line to ``manifest/ui_perf.jsonl`` (or a fallback log folder)
    """

    t0 = time.perf_counter()
    stack = _safe_stack(skip_tail=2)
    try:
        thr = float(threshold_ms)
    except Exception:
        thr = 200.0

    ex: dict[str, Any] = dict(extra or {})
    # Useful: backlog of queued UI log lines.
    try:
        from scorpio_pipe.ui import qt_log  # noqa: PLC0415

        snap = qt_log.get_backlog_snapshot()
        ex.setdefault("ui_log_backlog_lines", int(snap.get("lines", 0)))
        ex.setdefault("ui_log_backlog_chars", int(snap.get("chars", 0)))
    except Exception:
        pass

    action_id = _record_action_begin(name=str(name), threshold_ms=float(thr), extra=ex, stack=stack)
    try:
        yield
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        _record_action_end(
            action_id=action_id,
            name=str(name),
            elapsed_ms=float(dt_ms),
            threshold_ms=float(thr),
            extra=ex,
            stack=stack,
        )

        if dt_ms <= thr:
            return

        wd = Path(work_dir) if work_dir is not None else None
        ph = Path(path_hint) if path_hint is not None else None

        ev = _ProfileEvent(
            name=str(name),
            elapsed_ms=float(dt_ms),
            threshold_ms=float(thr),
            stack=stack,
            ts_utc=_utc_now_iso(),
            pid=os.getpid(),
            thread_name=threading.current_thread().name,
            thread_ident=threading.current_thread().ident,
            extra=ex,
        )
        _emit_event(ev, work_dir=wd, path_hint=ph)


def profiled_action(
    name: str,
    *,
    threshold_ms: float = 200.0,
    work_dir_getter: Callable[[], str | Path | None] | None = None,
    extra_getter: Callable[[], dict[str, Any] | None] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator helper for UI slots/handlers."""

    def _decor(fn: Callable[..., Any]) -> Callable[..., Any]:
        def _wrap(*args: Any, **kwargs: Any) -> Any:
            wd = work_dir_getter() if work_dir_getter is not None else None
            ex = extra_getter() if extra_getter is not None else None
            with profile_ui_action(
                name,
                threshold_ms=threshold_ms,
                work_dir=wd,
                extra=ex or None,
            ):
                return fn(*args, **kwargs)

        try:
            _wrap.__name__ = getattr(fn, "__name__", "wrapped")
        except Exception:
            pass
        return _wrap

    return _decor
