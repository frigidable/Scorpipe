from __future__ import annotations

import os

from scorpio_pipe.adapters.parallel_map import (
    child_blas_env,
    parallel_map,
    process_pool_blas_guard,
    resolve_n_jobs,
)


def _square(x: int) -> int:
    return int(x) * int(x)


def test_parallel_map_thread_preserves_order() -> None:
    got = list(parallel_map([0, 1, 2, 3, 4], _square, n_jobs=3, backend="thread"))
    assert got == [0, 1, 4, 9, 16]


def test_parallel_map_serial_fallback() -> None:
    got = list(parallel_map([1, 2, 3], _square, n_jobs=1, backend="thread"))
    assert got == [1, 4, 9]


def test_resolve_n_jobs_caps_to_items() -> None:
    assert resolve_n_jobs(16, 3) == 3
    assert resolve_n_jobs(None, 2) in {1, 2}



def _read_child_blas_env(_: int) -> tuple[str | None, str | None, str | None]:
    return (
        os.environ.get("OPENBLAS_NUM_THREADS"),
        os.environ.get("MKL_NUM_THREADS"),
        os.environ.get("OMP_NUM_THREADS"),
    )


def test_child_blas_env_sets_common_thread_vars() -> None:
    env = child_blas_env(1)
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["OMP_NUM_THREADS"] == "1"


def test_process_pool_blas_guard_restores_parent_env(monkeypatch) -> None:
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "8")
    with process_pool_blas_guard(1):
        assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
        assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ.get("OPENBLAS_NUM_THREADS") is None
    assert os.environ["MKL_NUM_THREADS"] == "8"


def test_parallel_map_process_limits_child_blas_env() -> None:
    got = list(parallel_map([0, 1], _read_child_blas_env, n_jobs=2, backend="process", chunksize=1))
    assert got
    for openblas, mkl, omp in got:
        assert openblas == "1"
        assert mkl == "1"
        assert omp == "1"
