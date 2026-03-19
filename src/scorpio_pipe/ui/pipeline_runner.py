"""UI compatibility shim for pipeline execution.

Historically the execution engine lived in :mod:`scorpio_pipe.ui.pipeline_runner`.
As of v6.x the core semantics are in :mod:`scorpio_pipe.pipeline.engine`.

P0-ARCH-12
----------
UI code must not import pipeline internals directly. The frozen boundary is
:mod:`scorpio_pipe.pipeline.public_api`.

This module remains only for **backward-compatible imports** in already-shipped
GUI code and external tooling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from scorpio_pipe.pipeline.public_api import (
    CANONICAL_TASKS,
    STAGE_SPECS,
    TASKS,
    CancelToken,
    PlanItem,
    RunContext,
    StageSpec,
    canonical_task_name,
    done_dir_for_task,
    done_json_path_for_task,
    load_context,
    plan_sequence as _plan_sequence,
    run_lineid_prepare,
    run_sequence_locked as _run_sequence_locked,
    run_wavesolution,
)


def plan_sequence(
    cfg_or_path: dict[str, Any] | str | Path | RunContext,
    task_names: Iterable[str],
    *,
    resume: bool = True,
    force: bool = False,
    config_path: Path | None = None,
) -> list[PlanItem]:
    return _plan_sequence(
        cfg_or_path,
        task_names,
        resume=resume,
        force=force,
        config_path=config_path,
    )


def run_sequence(
    cfg_or_path: dict[str, Any] | str | Path | RunContext,
    task_names: Iterable[str],
    *,
    resume: bool = True,
    force: bool = False,
    qc_override: bool = False,
    preflight_override: bool = False,
    preflight_override_reason: str | None = None,
    preflight_override_source: str | None = None,
    cancel_token: CancelToken | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    # Preserve the historical single-writer guarantee for GUI runs.
    return _run_sequence_locked(
        cfg_or_path,
        task_names,
        resume=resume,
        force=force,
        qc_override=qc_override,
        preflight_override=preflight_override,
        preflight_override_reason=preflight_override_reason,
        preflight_override_source=preflight_override_source,
        cancel_token=cancel_token,
        config_path=config_path,
    )


def run_one(
    cfg_or_path: dict[str, Any] | str | Path | RunContext,
    task_name: str,
    *,
    resume: bool = True,
    force: bool = False,
    qc_override: bool = False,
) -> None:
    run_sequence(cfg_or_path, [task_name], resume=resume, force=force, qc_override=qc_override)


__all__ = [
    # UI helpers
    "RunContext",
    "load_context",
    "run_lineid_prepare",
    "run_wavesolution",
    # Core API re-exports
    "CancelToken",
    "PlanItem",
    "StageSpec",
    "CANONICAL_TASKS",
    "STAGE_SPECS",
    "TASKS",
    "canonical_task_name",
    "done_dir_for_task",
    "done_json_path_for_task",
    "plan_sequence",
    "run_sequence",
    "run_one",
]
