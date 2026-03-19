from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

from scorpio_pipe.adapters.parallel_map import BLAS_THREAD_ENV_VARS


@dataclass(frozen=True)
class StageResources:
    """Best-effort per-stage execution budget.

    Parameters are advisory. Stages should cap parallelism and choose more
    memory-friendly chunking/streaming modes when the budget is tight.
    """

    max_rss_mb: int | None = None
    n_jobs: int | None = None
    blas_threads: int | None = None

    def as_dict(self) -> dict[str, int]:
        out: dict[str, int] = {}
        if self.max_rss_mb is not None:
            out["max_rss_mb"] = int(self.max_rss_mb)
        if self.n_jobs is not None:
            out["n_jobs"] = int(self.n_jobs)
        if self.blas_threads is not None:
            out["blas_threads"] = int(self.blas_threads)
        return out

    def is_empty(self) -> bool:
        return self.max_rss_mb is None and self.n_jobs is None and self.blas_threads is None


_EMPTY_BUDGET = StageResources()


def _as_pos_int(value: Any) -> int | None:
    try:
        if value in (None, "", False):
            return None
        iv = int(value)
    except Exception:
        return None
    return iv if iv > 0 else None


def _first_pos_int(*values: Any) -> int | None:
    for value in values:
        iv = _as_pos_int(value)
        if iv is not None:
            return iv
    return None


def resolve_stage_resources(cfg: Mapping[str, Any] | None, stage_key: str) -> StageResources:
    """Resolve the effective stage-level resource budget from config.

    Precedence (highest first):
      1) <stage>.resources.{...}
      2) <stage>.perf.{n_jobs, blas_threads, max_rss_mb}
      3) top-level resources.{...}
      4) runtime.{...}

    This keeps backward compatibility with existing perf knobs while exposing a
    single normalized object through ``StageContext.resources``.
    """

    if not isinstance(cfg, Mapping):
        return _EMPTY_BUDGET

    runtime = cfg.get("runtime") if isinstance(cfg.get("runtime"), Mapping) else {}
    global_res = cfg.get("resources") if isinstance(cfg.get("resources"), Mapping) else {}
    stage_cfg = cfg.get(stage_key) if isinstance(cfg.get(stage_key), Mapping) else {}
    stage_res = stage_cfg.get("resources") if isinstance(stage_cfg.get("resources"), Mapping) else {}
    stage_perf = stage_cfg.get("perf") if isinstance(stage_cfg.get("perf"), Mapping) else {}

    return StageResources(
        max_rss_mb=_first_pos_int(stage_res.get("max_rss_mb"), stage_perf.get("max_rss_mb"), global_res.get("max_rss_mb"), runtime.get("max_rss_mb")),
        n_jobs=_first_pos_int(stage_res.get("n_jobs"), stage_perf.get("n_jobs"), global_res.get("n_jobs"), runtime.get("n_jobs")),
        blas_threads=_first_pos_int(stage_res.get("blas_threads"), stage_perf.get("blas_threads"), global_res.get("blas_threads"), runtime.get("blas_threads")),
    )


@contextmanager
def stage_blas_guard(threads: int | None) -> Iterator[None]:
    """Best-effort BLAS/OpenMP thread cap for the current stage.

    We update both process environment variables and, when available,
    threadpoolctl limits in the current process.
    """

    limit = _as_pos_int(threads)
    if limit is None:
        yield
        return

    old_env = {key: os.environ.get(key) for key in BLAS_THREAD_ENV_VARS}
    for key in BLAS_THREAD_ENV_VARS:
        os.environ[key] = str(limit)

    with ExitStack() as stack:
        try:
            from threadpoolctl import threadpool_limits  # type: ignore
        except Exception:
            threadpool_limits = None  # type: ignore[assignment]
        if threadpool_limits is not None:
            try:
                stack.enter_context(threadpool_limits(limits=int(limit)))
            except Exception:
                pass
        try:
            yield
        finally:
            for key, prev in old_env.items():
                if prev is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = prev


def _copy_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return dict(cfg)


def _copy_nested_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    cur = parent.get(key)
    out = dict(cur) if isinstance(cur, Mapping) else {}
    parent[key] = out
    return out


def _cap_positive(existing: Any, cap: int | None) -> int | None:
    cap_i = _as_pos_int(cap)
    if cap_i is None:
        return _as_pos_int(existing)
    cur_i = _as_pos_int(existing)
    return cap_i if cur_i is None else max(1, min(cur_i, cap_i))


def _memory_cap_n_jobs(max_rss_mb: int | None) -> int | None:
    rss = _as_pos_int(max_rss_mb)
    if rss is None:
        return None
    if rss <= 1536:
        return 1
    if rss <= 3072:
        return 2
    if rss <= 6144:
        return 4
    return None


def recommend_strip_rows(max_rss_mb: int | None) -> int | None:
    rss = _as_pos_int(max_rss_mb)
    if rss is None:
        return None
    if rss <= 1536:
        return 32
    if rss <= 3072:
        return 64
    if rss <= 6144:
        return 96
    return 128


def recommend_chunk_rows(max_rss_mb: int | None) -> int | None:
    rss = _as_pos_int(max_rss_mb)
    if rss is None:
        return None
    if rss <= 1536:
        return 32
    if rss <= 3072:
        return 64
    if rss <= 6144:
        return 96
    return 128


def apply_stage_resources_to_cfg(
    cfg: Mapping[str, Any],
    *,
    stage_key: str,
    budget: StageResources | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a cfg copy with stage-local perf knobs capped by ``budget``.

    The helper is intentionally conservative: it only tightens existing knobs or
    fills in missing ones with safer values. It never increases parallelism.
    """

    out = _copy_cfg(cfg)
    metrics: dict[str, Any] = {}
    if budget is None:
        budget = _EMPTY_BUDGET

    if budget.max_rss_mb is not None:
        metrics["resource_budget_max_rss_mb"] = int(budget.max_rss_mb)
    if budget.n_jobs is not None:
        metrics["resource_budget_n_jobs"] = int(budget.n_jobs)
    if budget.blas_threads is not None:
        metrics["resource_budget_blas_threads"] = int(budget.blas_threads)

    stage_cfg = _copy_nested_dict(out, stage_key)
    perf_cfg = _copy_nested_dict(stage_cfg, "perf")

    job_cap = _cap_positive(budget.n_jobs, _memory_cap_n_jobs(budget.max_rss_mb))

    if stage_key in {"linearize", "cosmics"}:
        applied = _cap_positive(perf_cfg.get("n_jobs"), job_cap)
        if applied is not None:
            perf_cfg["n_jobs"] = int(applied)
            metrics["resource_budget_applied_n_jobs"] = int(applied)

    if stage_key == "superneon":
        applied = _cap_positive(stage_cfg.get("n_jobs"), job_cap)
        if applied is not None:
            stage_cfg["n_jobs"] = int(applied)
            metrics["resource_budget_applied_n_jobs"] = int(applied)

    if stage_key == "sky":
        applied_workers = _cap_positive(perf_cfg.get("strip_workers"), job_cap)
        if applied_workers is not None:
            perf_cfg["strip_workers"] = int(applied_workers)
            metrics["resource_budget_applied_strip_workers"] = int(applied_workers)
            if int(applied_workers) <= 1:
                perf_cfg["parallel_strips"] = False
                metrics["resource_budget_parallel_strips"] = False
        rec_strip_rows = recommend_strip_rows(budget.max_rss_mb)
        if rec_strip_rows is not None:
            cur_strip_rows = _as_pos_int(perf_cfg.get("strip_rows"))
            applied_strip_rows = rec_strip_rows if cur_strip_rows is None else min(cur_strip_rows, rec_strip_rows)
            perf_cfg["strip_rows"] = int(applied_strip_rows)
            metrics["resource_budget_applied_strip_rows"] = int(applied_strip_rows)

    if stage_key == "stack2d":
        rec_chunk_rows = recommend_chunk_rows(budget.max_rss_mb)
        if rec_chunk_rows is not None:
            cur_chunk_rows = _as_pos_int(stage_cfg.get("chunk_rows"))
            applied_chunk_rows = rec_chunk_rows if cur_chunk_rows is None else min(cur_chunk_rows, rec_chunk_rows)
            stage_cfg["chunk_rows"] = int(applied_chunk_rows)
            metrics["resource_budget_applied_chunk_rows"] = int(applied_chunk_rows)
            metrics["resource_budget_streaming_mode"] = "chunked"

    if stage_key == "wavesolution":
        wave_cfg = _copy_nested_dict(out, "wavesol")
        rss = _as_pos_int(budget.max_rss_mb)
        rec_trace_step = 4 if rss is not None and rss <= 1536 else (2 if rss is not None and rss <= 3072 else None)
        if rec_trace_step is not None:
            cur_trace_step = _as_pos_int(wave_cfg.get("trace_y_step"))
            applied_trace_step = rec_trace_step if cur_trace_step is None else max(cur_trace_step, rec_trace_step)
            wave_cfg["trace_y_step"] = int(applied_trace_step)
            metrics["resource_budget_applied_trace_y_step"] = int(applied_trace_step)

    if stage_key == "geometry_prior":
        geom_cfg = _copy_nested_dict(_copy_nested_dict(out, "wavesol"), "geometry_prior")
        rss = _as_pos_int(budget.max_rss_mb)
        if rss is not None:
            target_block = 48 if rss <= 1536 else (32 if rss <= 3072 else 24)
            cur_block = _as_pos_int(geom_cfg.get("block_height"))
            applied_block = target_block if cur_block is None else max(cur_block, target_block)
            geom_cfg["block_height"] = int(applied_block)
            metrics["resource_budget_geometry_block_height"] = int(applied_block)

    if stage_key == "slit_sanity":
        slit_cfg = _copy_nested_dict(out, "slit_sanity")
        rss = _as_pos_int(budget.max_rss_mb)
        if rss is not None:
            target_frames = 4 if rss <= 1536 else (6 if rss <= 3072 else 8)
            cur_frames = _as_pos_int(slit_cfg.get("max_candidate_frames"))
            applied_frames = target_frames if cur_frames is None else min(cur_frames, target_frames)
            slit_cfg["max_candidate_frames"] = int(applied_frames)
            metrics["resource_budget_slit_max_candidate_frames"] = int(applied_frames)

    if stage_key == "sky_refine":
        refine_cfg = _copy_nested_dict(out, "sky_refine")
        rss = _as_pos_int(budget.max_rss_mb)
        if rss is not None:
            target_detect = 5 if rss <= 1536 else 4
            cur_detect = _as_pos_int(refine_cfg.get("detect_snr"))
            applied_detect = target_detect if cur_detect is None else max(cur_detect, target_detect)
            refine_cfg["detect_snr"] = int(applied_detect)
            metrics["resource_budget_sky_refine_detect_snr"] = int(applied_detect)
            target_top = 4 if rss <= 1536 else (6 if rss <= 3072 else 8)
            cur_top = _as_pos_int(refine_cfg.get("top_n_summary"))
            applied_top = target_top if cur_top is None else min(cur_top, target_top)
            refine_cfg["top_n_summary"] = int(applied_top)
            metrics["resource_budget_sky_refine_top_n_summary"] = int(applied_top)

    return out, metrics
