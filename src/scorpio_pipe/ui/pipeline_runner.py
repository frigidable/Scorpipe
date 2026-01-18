"""UI compatibility shim for the core pipeline engine.

Historically the execution engine lived in :mod:`scorpio_pipe.ui.pipeline_runner`.
As of v6.0.18 the single source of truth is :mod:`scorpio_pipe.pipeline.engine`.

This module remains to preserve backward-compatible imports for the GUI and
external tooling, and to keep a small amount of UI-friendly helper API
(:class:`RunContext`, :func:`load_context`, :func:`run_lineid_prepare`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from scorpio_pipe.config import load_config_any
from scorpio_pipe.wavesol_paths import wavesol_dir

# Re-export the core engine API
from scorpio_pipe.pipeline.engine import (
    CANONICAL_TASKS,
    STAGE_SPECS,
    TASKS,
    CancelToken,
    PlanItem,
    StageSpec,
    canonical_task_name,
    done_dir_for_task,
    done_json_path_for_task,
    plan_sequence as _plan_sequence,
    run_one as _run_one,
    run_sequence as _run_sequence,
)


@dataclass(frozen=True)
class RunContext:
    """Lightweight wrapper used by the GUI.

    The launcher window keeps a context object around and passes it back to
    :func:`run_sequence` / per-stage helpers.
    """

    cfg_path: Path
    cfg: dict[str, Any]


def load_context(cfg_path: str | Path) -> RunContext:
    """Load config from disk and return a GUI-friendly context."""

    p = Path(cfg_path)
    cfg = load_config_any(p)
    return RunContext(cfg_path=p, cfg=cfg)


def plan_sequence(
    cfg_or_path: dict[str, Any] | str | Path | RunContext,
    task_names: Iterable[str],
    *,
    resume: bool = True,
    force: bool = False,
    config_path: Path | None = None,
) -> list[PlanItem]:
    if isinstance(cfg_or_path, RunContext):
        cfg = cfg_or_path.cfg
        config_path = config_path or cfg_or_path.cfg_path
        return _plan_sequence(cfg, task_names, resume=resume, force=force, config_path=config_path)
    return _plan_sequence(cfg_or_path, task_names, resume=resume, force=force, config_path=config_path)


def run_sequence(
    cfg_or_path: dict[str, Any] | str | Path | RunContext,
    task_names: Iterable[str],
    *,
    resume: bool = True,
    force: bool = False,
    qc_override: bool = False,
    cancel_token: CancelToken | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    if isinstance(cfg_or_path, RunContext):
        cfg = cfg_or_path.cfg
        config_path = config_path or cfg_or_path.cfg_path
        return _run_sequence(
            cfg,
            task_names,
            resume=resume,
            force=force,
            qc_override=qc_override,
            cancel_token=cancel_token,
            config_path=config_path,
        )
    return _run_sequence(
        cfg_or_path,
        task_names,
        resume=resume,
        force=force,
        qc_override=qc_override,
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
    """Compatibility wrapper."""

    run_sequence(cfg_or_path, [task_name], resume=resume, force=force, qc_override=qc_override)


def run_lineid_prepare(ctx: RunContext) -> dict[str, Path]:
    """Run (or skip) ``lineid_prepare`` and return expected output paths."""

    run_sequence(ctx, ["lineid_prepare"], resume=True, force=False, config_path=ctx.cfg_path)

    outdir = wavesol_dir(ctx.cfg)
    return {
        "template": (outdir / "manual_pairs_template.csv"),
        "auto": (outdir / "manual_pairs_auto.csv"),
        "report": (outdir / "lineid_report.txt"),
    }


def run_wavesolution(ctx: RunContext) -> dict[str, Any]:
    """Run (or skip) ``wavesolution`` and return per-task results."""

    return run_sequence(ctx, ["wavesolution"], resume=True, force=False, config_path=ctx.cfg_path)


__all__ = [
    # UI helpers
    "RunContext",
    "load_context",
    "run_lineid_prepare",
    "run_wavesolution",
    # Core engine re-exports
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
