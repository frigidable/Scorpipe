from __future__ import annotations

from pathlib import Path

from scorpio_pipe.pipeline.context import StageContext, StageResult


def run(ctx: StageContext) -> StageResult:
    from scorpio_pipe.stages.science_report import run_science_report

    ctx.ensure_stage_dir()
    return run_science_report(dict(ctx.cfg), run_root=Path(ctx.run_root), stage_dir_path=Path(ctx.stage_dir))
