"""Pipeline execution package.

The core execution semantics live in :mod:`scorpio_pipe.pipeline.engine`.
"""

from .engine import (
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
    "run_one",
]
