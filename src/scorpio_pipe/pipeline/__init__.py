"""Pipeline package.

Public entrypoint
-----------------
Application layers (UI/CLI) must depend on the pipeline only through:

    :mod:`scorpio_pipe.pipeline.public_api`

This ``__init__`` re-exports the frozen API for backward compatibility.
"""

from __future__ import annotations

from .public_api import (
    CANONICAL_TASKS,
    STAGE_SPECS,
    TASKS,
    CancelToken,
    PlanItem,
    StageSpec,
    canonical_task_name,
    done_dir_for_task,
    done_json_path_for_task,
    plan_sequence,
    run_one,
    run_sequence,
    run_sequence_locked,
)

__all__ = [
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
    "run_sequence_locked",
    "run_one",
]
