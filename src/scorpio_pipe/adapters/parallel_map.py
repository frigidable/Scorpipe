from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Callable, Iterable, Iterator, Literal, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")
ParallelBackend = Literal["auto", "thread", "process", "loky"]
BLAS_THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "GOTO_NUM_THREADS",
)


def child_blas_env(threads: int = 1) -> dict[str, str]:
    """Return environment overrides used for multiprocessing children.

    Keeping BLAS/OpenMP pools at 1 thread per child avoids classic
    oversubscription scenarios like ``16 processes * 16 BLAS threads``.
    """

    value = str(max(1, int(threads)))
    return {key: value for key in BLAS_THREAD_ENV_VARS}


@contextmanager
def process_pool_blas_guard(threads: int = 1) -> Iterator[None]:
    """Temporarily force BLAS/OpenMP thread counts for child-process spawn.

    The variables are set only while the process pool is being created so spawned
    children inherit safe values, then restored in the parent process.
    """

    overrides = child_blas_env(threads)
    old = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, prev in old.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


def limit_child_blas_threads(threads: int = 1) -> None:
    """Apply per-process BLAS/OpenMP limits inside worker processes."""

    overrides = child_blas_env(threads)
    for key, value in overrides.items():
        os.environ[key] = value

    try:
        from threadpoolctl import threadpool_limits  # type: ignore
    except Exception:
        return
    try:
        threadpool_limits(limits=max(1, int(threads)))
    except Exception:
        return


def cpu_count_cap() -> int:
    """Return a conservative worker cap for stage-level parallel work."""

    try:
        n = int(os.cpu_count() or 1)
    except Exception:
        n = 1
    return max(1, min(8, n))



def resolve_n_jobs(n_jobs: int | None, n_items: int) -> int:
    """Resolve a safe worker count for ``parallel_map``.

    If ``n_jobs`` is not provided, the helper uses a conservative cap
    (``min(8, cpu_count)``) and never exceeds the number of items.
    """

    if n_items <= 0:
        return 0
    if n_jobs in {None, 0, "", False}:  # type: ignore[comparison-overlap]
        return max(1, min(cpu_count_cap(), int(n_items)))
    try:
        want = int(n_jobs)
    except Exception:
        want = 1
    return max(1, min(int(n_items), want))



def _iter_serial(items: Sequence[T], fn: Callable[[T], R]) -> Iterator[R]:
    for item in items:
        yield fn(item)



def _thread_map(items: Sequence[T], fn: Callable[[T], R], *, max_workers: int) -> Iterator[R]:
    with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
        yield from ex.map(fn, items)



def _process_map(
    items: Sequence[T],
    fn: Callable[[T], R],
    *,
    max_workers: int,
    chunksize: int,
    backend: str,
) -> Iterator[R]:
    if backend == "loky":
        try:
            from loky import get_reusable_executor  # type: ignore

            with process_pool_blas_guard(1):
                ex = get_reusable_executor(
                    max_workers=int(max_workers),
                    timeout=300,
                    initializer=limit_child_blas_threads,
                    initargs=(1,),
                )
            yield from ex.map(fn, items, chunksize=max(1, int(chunksize)))
            return
        except Exception:
            backend = "process"

    with process_pool_blas_guard(1):
        with ProcessPoolExecutor(
            max_workers=int(max_workers),
            mp_context=mp.get_context("spawn"),
            initializer=limit_child_blas_threads,
            initargs=(1,),
        ) as ex:
            yield from ex.map(fn, items, chunksize=max(1, int(chunksize)))



def parallel_map(
    frames: Iterable[T],
    fn: Callable[[T], R],
    n_jobs: int | None = None,
    backend: ParallelBackend = "auto",
    *,
    chunksize: int = 1,
) -> Iterator[R]:
    """Ordered parallel map with graceful serial fallback.

    Parameters
    ----------
    frames
        Input items.
    fn
        Picklable worker for ``process``/``loky`` backends. Closures/lambdas are
        fine with the ``thread`` backend.
    n_jobs
        Worker count. ``None`` uses a conservative default capped by item count.
    backend
        ``thread`` / ``process`` / ``loky`` / ``auto``.
        ``auto`` currently resolves to ``thread`` because most stage-level hot
        paths are NumPy/Astropy heavy and benefit from shared read-only state.
    chunksize
        Process-pool chunk size (ignored by thread/serial execution).
    """

    items = list(frames)
    if not items:
        return iter(())

    jobs = resolve_n_jobs(n_jobs, len(items))
    if jobs <= 1 or len(items) <= 1:
        return _iter_serial(items, fn)

    chosen = str(backend or "auto").strip().lower()
    if chosen in {"auto", "threads"}:
        chosen = "thread"
    if chosen not in {"thread", "process", "loky"}:
        raise ValueError(f"Unsupported parallel backend: {backend}")

    if chosen == "thread":
        return _thread_map(items, fn, max_workers=jobs)
    return _process_map(items, fn, max_workers=jobs, chunksize=chunksize, backend=chosen)
