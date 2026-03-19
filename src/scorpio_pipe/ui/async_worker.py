from __future__ import annotations

import itertools
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

from PySide6 import QtCore

T = TypeVar("T")

# Global UI worker pool:
# - keeps a fixed number of Qt threads alive
# - avoids "QThread per job" explosions
_POOL = QtCore.QThreadPool.globalInstance()
_POOL_CONFIG_LOCK = threading.Lock()
_POOL_CONFIGURED = False

# Global UI job registry:
# - keeps strong refs to unfinished jobs
# - provides diagnostics for the freeze watchdog
# - applies best-effort queue limiting / deduplication
_JOB_COUNTER = itertools.count(1)
_ACTIVE_JOBS: dict[int, "RunHandle[object]"] = {}
_ACTIVE_JOBS_LOCK = threading.Lock()

_STATS_LOCK = threading.Lock()
_STATS_TOTALS: dict[str, int] = {
    "submitted_total": 0,
    "started_total": 0,
    "finished_total": 0,
    "failed_total": 0,
    "cancel_requested_total": 0,
    "stale_requested_total": 0,
    "deduped_total": 0,
    "queue_evicted_total": 0,
}


def _configure_pool_if_needed() -> None:
    global _POOL_CONFIGURED
    if _POOL_CONFIGURED:
        return
    with _POOL_CONFIG_LOCK:
        if _POOL_CONFIGURED:
            return
        cpu = os.cpu_count() or 1
        default_max = min(8, max(1, cpu))
        env = os.environ.get("SCORPIPE_UI_MAX_THREADS")
        if env:
            try:
                default_max = max(1, int(env))
            except Exception:
                pass
        try:
            _POOL.setMaxThreadCount(int(default_max))
        except Exception:
            # Best-effort: never fail GUI startup.
            pass
        _POOL_CONFIGURED = True


def _queue_limit() -> int:
    try:
        return max(1, int(os.environ.get("SCORPIPE_UI_QUEUE_LIMIT", "200")))
    except Exception:
        return 200


def _bump_stat(key: str, delta: int = 1) -> None:
    with _STATS_LOCK:
        _STATS_TOTALS[key] = int(_STATS_TOTALS.get(key, 0)) + int(delta)


@dataclass(frozen=True)
class UiJobRegistrySnapshot:
    pool_max: int
    pool_active: int
    active_jobs: int
    running_jobs: int
    queued_jobs: int
    cancelled_active: int
    stale_active: int
    queue_limit: int
    submitted_total: int
    started_total: int
    finished_total: int
    failed_total: int
    cancel_requested_total: int
    stale_requested_total: int
    deduped_total: int
    queue_evicted_total: int


def get_job_registry_snapshot() -> UiJobRegistrySnapshot:
    try:
        pool_max = int(_POOL.maxThreadCount())
    except Exception:
        pool_max = 0
    try:
        pool_active = int(_POOL.activeThreadCount())
    except Exception:
        pool_active = 0

    with _ACTIVE_JOBS_LOCK:
        jobs = list(_ACTIVE_JOBS.values())
    with _STATS_LOCK:
        totals = dict(_STATS_TOTALS)

    running = sum(1 for h in jobs if h.is_running)
    queued = sum(1 for h in jobs if h.is_queued)
    cancelled_active = sum(1 for h in jobs if (h.is_cancelled and not h.is_done))
    stale_active = sum(1 for h in jobs if (h.is_stale and not h.is_done))

    return UiJobRegistrySnapshot(
        pool_max=pool_max,
        pool_active=pool_active,
        active_jobs=len(jobs),
        running_jobs=running,
        queued_jobs=queued,
        cancelled_active=cancelled_active,
        stale_active=stale_active,
        queue_limit=_queue_limit(),
        submitted_total=int(totals.get("submitted_total", 0)),
        started_total=int(totals.get("started_total", 0)),
        finished_total=int(totals.get("finished_total", 0)),
        failed_total=int(totals.get("failed_total", 0)),
        cancel_requested_total=int(totals.get("cancel_requested_total", 0)),
        stale_requested_total=int(totals.get("stale_requested_total", 0)),
        deduped_total=int(totals.get("deduped_total", 0)),
        queue_evicted_total=int(totals.get("queue_evicted_total", 0)),
    )


def snapshot_active_qthreads() -> list[str]:
    """Return a diagnostic snapshot for the UI freeze watchdog.

    Historical name kept for backward-compat (it used to list per-job QThreads).
    Now it reports:
    - global QThreadPool limits/state
    - UI job registry counters
    - list of active (in-flight) jobs created via :func:`run_in_thread`
    """

    out: list[str] = []
    snap = get_job_registry_snapshot()
    out.append(
        f"QThreadPool max={snap.pool_max} active={snap.pool_active}"
    )
    out.append(
        "job_registry "
        f"active={snap.active_jobs} running={snap.running_jobs} queued={snap.queued_jobs} "
        f"cancelled_active={snap.cancelled_active} stale_active={snap.stale_active} "
        f"queue_limit={snap.queue_limit} deduped_total={snap.deduped_total} "
        f"queue_evicted_total={snap.queue_evicted_total}"
    )
    out.append(
        "job_registry_totals "
        f"submitted={snap.submitted_total} started={snap.started_total} "
        f"finished={snap.finished_total} failed={snap.failed_total} "
        f"cancel_requested={snap.cancel_requested_total} stale_requested={snap.stale_requested_total}"
    )

    try:
        jobs = get_active_job_snapshots(limit=50)
        for h in jobs:
            out.append(
                "job "
                f"id={h.job_id} name={h.name!r} key={h.dedupe_key!r} age_s={h.age_s:.2f} "
                f"running={h.running} queued={h.queued} stale={h.stale} cancelled={h.cancelled}"
            )
        with _ACTIVE_JOBS_LOCK:
            total_jobs = len(_ACTIVE_JOBS)
        if total_jobs > 50:
            out.append(f"... ({total_jobs - 50} more jobs)")
    except Exception:
        pass

    return out




@dataclass(frozen=True)
class UiActiveJobSnapshot:
    job_id: int
    name: str
    dedupe_key: str | None
    age_s: float
    running: bool
    queued: bool
    stale: bool
    cancelled: bool
    done: bool


def get_active_job_snapshots(limit: int = 50) -> list[UiActiveJobSnapshot]:
    try:
        lim = max(1, int(limit))
    except Exception:
        lim = 50
    with _ACTIVE_JOBS_LOCK:
        jobs = list(_ACTIVE_JOBS.values())
    now = time.time()
    out: list[UiActiveJobSnapshot] = []
    for h in jobs[:lim]:
        try:
            age = max(0.0, now - float(h.started_at))
        except Exception:
            age = -1.0
        out.append(
            UiActiveJobSnapshot(
                job_id=int(h.job_id),
                name=str(h.name),
                dedupe_key=(str(h.dedupe_key) if h.dedupe_key is not None else None),
                age_s=float(age),
                running=bool(h.is_running),
                queued=bool(h.is_queued),
                stale=bool(h.is_stale),
                cancelled=bool(h.is_cancelled),
                done=bool(h.is_done),
            )
        )
    return out

class WorkerSignals(QtCore.QObject, Generic[T]):
    """Signals emitted when a background job completes."""

    finished = QtCore.Signal(object)  # result: T
    failed = QtCore.Signal(str)       # traceback str


@dataclass
class RunHandle(Generic[T]):
    """Handle for a submitted background job.

    Cancellation is cooperative/best-effort:
    - we cannot forcibly stop arbitrary Python code, but we can mark results
      as stale and ignore them (prevents UI churn + signal bursts).
    - if the runnable has not started yet, we also try to dequeue it from the
      global ``QThreadPool``.
    """

    job_id: int
    worker: WorkerSignals[T]
    name: str
    dedupe_key: str | None = None
    started_at: float = field(default_factory=time.time)

    _stale: threading.Event = field(default_factory=threading.Event, repr=False)
    _cancelled: threading.Event = field(default_factory=threading.Event, repr=False)
    _done: threading.Event = field(default_factory=threading.Event, repr=False)
    _running: threading.Event = field(default_factory=threading.Event, repr=False)
    _cancel_recorded: threading.Event = field(default_factory=threading.Event, repr=False)
    _stale_recorded: threading.Event = field(default_factory=threading.Event, repr=False)
    _runnable: object | None = field(default=None, repr=False)

    def mark_stale(self) -> None:
        _mark_handle_stale(self, try_dequeue=True)

    def cancel(self) -> None:
        _mark_handle_cancelled(self, try_dequeue=True)

    @property
    def is_stale(self) -> bool:
        return self._stale.is_set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def is_done(self) -> bool:
        return self._done.is_set()

    @property
    def is_running(self) -> bool:
        return self._running.is_set() and not self._done.is_set()

    @property
    def is_queued(self) -> bool:
        return (not self._running.is_set()) and (not self._done.is_set())

    def wait(self, timeout_ms: int | None = None) -> bool:
        """Wait until the job finishes (best-effort).

        Returns True if done, False if timed out.
        """
        if timeout_ms is None:
            return self._done.wait()
        return self._done.wait(timeout=float(timeout_ms) / 1000.0)


class _PoolRunnable(QtCore.QRunnable):
    def __init__(self, fn: Callable[[], T], handle: RunHandle[T]):
        super().__init__()
        self._fn = fn
        self._handle = handle
        try:
            self.setAutoDelete(True)
        except Exception:
            pass

    def run(self) -> None:
        h = self._handle

        # If the job became stale/cancelled while waiting in the pool queue,
        # exit early to avoid wasting resources.
        if h.is_stale or h.is_cancelled:
            _finish_job(h, outcome=("cancelled" if h.is_cancelled else "stale"))
            return

        _mark_job_running(h)
        outcome = "finished"
        try:
            res = self._fn()
            if h.is_cancelled:
                outcome = "cancelled"
            elif h.is_stale:
                outcome = "stale"
            else:
                h.worker.finished.emit(res)
        except Exception:
            tb = traceback.format_exc()
            if h.is_cancelled:
                outcome = "cancelled"
            elif h.is_stale:
                outcome = "stale"
            else:
                outcome = "failed"
                h.worker.failed.emit(tb)
        finally:
            _finish_job(h, outcome=outcome)


def _mark_job_running(h: RunHandle[object]) -> None:
    if h._running.is_set():
        return
    h._running.set()
    _bump_stat("started_total")


def _try_take_queued_job(h: RunHandle[object]) -> bool:
    if h.is_done or h.is_running:
        return False
    runnable = h._runnable
    if runnable is None:
        return False
    try:
        return bool(_POOL.tryTake(runnable))
    except Exception:
        return False


def _mark_handle_stale(
    h: RunHandle[object],
    *,
    try_dequeue: bool,
    deduped: bool = False,
    evicted_for_limit: bool = False,
) -> None:
    if not h._stale.is_set():
        h._stale.set()
    if not h._stale_recorded.is_set():
        h._stale_recorded.set()
        _bump_stat("stale_requested_total")
    if deduped:
        _bump_stat("deduped_total")
    if evicted_for_limit:
        _bump_stat("queue_evicted_total")
    if try_dequeue and _try_take_queued_job(h):
        _finish_job(h, outcome="stale")


def _mark_handle_cancelled(
    h: RunHandle[object],
    *,
    try_dequeue: bool,
    deduped: bool = False,
    evicted_for_limit: bool = False,
) -> None:
    if not h._cancelled.is_set():
        h._cancelled.set()
    if not h._cancel_recorded.is_set():
        h._cancel_recorded.set()
        _bump_stat("cancel_requested_total")
    _mark_handle_stale(
        h,
        try_dequeue=False,
        deduped=deduped,
        evicted_for_limit=evicted_for_limit,
    )
    if try_dequeue and _try_take_queued_job(h):
        _finish_job(h, outcome="cancelled")


def _iter_active_jobs() -> list[RunHandle[object]]:
    with _ACTIVE_JOBS_LOCK:
        return list(_ACTIVE_JOBS.values())


def _prune_duplicate_jobs(dedupe_key: str | None) -> None:
    if not dedupe_key:
        return
    for h in _iter_active_jobs():
        try:
            if h.dedupe_key != dedupe_key or h.is_done:
                continue
        except Exception:
            continue
        _mark_handle_cancelled(h, try_dequeue=True, deduped=True)


def _pick_queue_eviction_candidate() -> RunHandle[object] | None:
    jobs = [h for h in _iter_active_jobs() if h.is_queued]
    if not jobs:
        return None

    # 1) stale/cancelled queued jobs are always safe to drop first.
    for h in jobs:
        if h.is_cancelled or h.is_stale:
            return h

    # 2) queued duplicates (same dedupe key) are next.
    by_key: dict[str, list[RunHandle[object]]] = {}
    for h in jobs:
        key = str(h.dedupe_key or "")
        if not key:
            continue
        by_key.setdefault(key, []).append(h)
    dup_candidates: list[RunHandle[object]] = []
    for lst in by_key.values():
        if len(lst) > 1:
            lst_sorted = sorted(lst, key=lambda hh: float(hh.started_at))
            dup_candidates.extend(lst_sorted[:-1])
    if dup_candidates:
        return sorted(dup_candidates, key=lambda hh: float(hh.started_at))[0]

    # 3) hard cap fallback: evict the oldest queued job globally.
    return sorted(jobs, key=lambda hh: float(hh.started_at))[0]


def _enforce_queue_limit_before_submit() -> None:
    limit = _queue_limit()
    if limit <= 0:
        return
    for _ in range(max(1, limit * 2)):
        snap = get_job_registry_snapshot()
        if snap.queued_jobs < limit:
            return
        cand = _pick_queue_eviction_candidate()
        if cand is None:
            return
        _mark_handle_cancelled(cand, try_dequeue=True, evicted_for_limit=True)
        # If the candidate was already running by the time we tried to dequeue it,
        # just stop here; we did mark it stale/cancelled, so it will not emit late UI updates.
        if cand.is_running and not cand.is_done:
            return


def _finish_job(h: RunHandle[object], *, outcome: str) -> None:
    try:
        h._done.set()
    except Exception:
        pass
    try:
        with _ACTIVE_JOBS_LOCK:
            _ACTIVE_JOBS.pop(int(h.job_id), None)
    except Exception:
        pass

    if outcome == "failed":
        _bump_stat("failed_total")
    else:
        _bump_stat("finished_total")


def _reset_job_registry_for_tests(*, max_threads: int | None = None) -> None:
    """Best-effort test helper: clear counters and optionally reset pool size."""

    with _ACTIVE_JOBS_LOCK:
        jobs = list(_ACTIVE_JOBS.values())
    for h in jobs:
        try:
            h.cancel()
        except Exception:
            pass
    with _ACTIVE_JOBS_LOCK:
        _ACTIVE_JOBS.clear()
    with _STATS_LOCK:
        for k in list(_STATS_TOTALS):
            _STATS_TOTALS[k] = 0
    if max_threads is not None:
        try:
            _POOL.setMaxThreadCount(max(1, int(max_threads)))
        except Exception:
            pass



def run_in_thread(
    fn: Callable[[], T],
    *,
    name: str | None = None,
    dedupe_key: str | None = None,
    latest_wins: bool | None = None,
) -> RunHandle[T]:
    """Submit ``fn`` to the global QThreadPool and return a handle.

    Parameters
    ----------
    dedupe_key:
        Optional logical key for queue coalescing. When provided, the default
        policy is ``latest_wins``: older unfinished jobs with the same key are
        marked stale/cancelled, and if still queued they are removed from the
        pool before the new job is submitted.
    latest_wins:
        Override whether previous jobs with the same ``dedupe_key`` should be
        cancelled before queuing the new one. Defaults to ``True`` when a
        ``dedupe_key`` is provided.

    Notes
    -----
    - Caller should keep the returned handle alive while interested in results.
    - Queue length is best-effort capped via ``SCORPIPE_UI_QUEUE_LIMIT``
      (default: 200). When the queue is at the limit, older queued jobs are
      evicted/cancelled before the new task is enqueued.
    """

    _configure_pool_if_needed()

    if latest_wins is None:
        latest_wins = bool(dedupe_key)
    if latest_wins and dedupe_key:
        _prune_duplicate_jobs(str(dedupe_key))
    _enforce_queue_limit_before_submit()

    job_id = int(next(_JOB_COUNTER))
    sig: WorkerSignals[T] = WorkerSignals()
    h: RunHandle[T] = RunHandle(
        job_id=job_id,
        worker=sig,
        name=str(name or "job"),
        dedupe_key=(str(dedupe_key) if dedupe_key else None),
    )
    _bump_stat("submitted_total")

    try:
        with _ACTIVE_JOBS_LOCK:
            _ACTIVE_JOBS[job_id] = h  # keep strong ref until done
    except Exception:
        pass

    r = _PoolRunnable(fn, h)
    h._runnable = r
    try:
        _POOL.start(r)
    except Exception:
        # Fallback: run immediately (synchronous) but keep API stable.
        try:
            _mark_job_running(h)
            res = fn()
            if not (h.is_stale or h.is_cancelled):
                sig.finished.emit(res)
            _finish_job(h, outcome=("cancelled" if h.is_cancelled else ("stale" if h.is_stale else "finished")))
        except Exception:
            sig.failed.emit(traceback.format_exc())
            _finish_job(h, outcome="failed")

    return h
