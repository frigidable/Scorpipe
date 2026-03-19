from __future__ import annotations

from pathlib import Path

from scorpio_pipe.pipeline.context import StageContext, StageResult


def run(ctx: StageContext) -> StageResult:
    from scorpio_pipe.stages.fluxcal_apply import run_fluxcal_apply

    ctx.ensure_stage_dir()
    return run_fluxcal_apply(dict(ctx.cfg), run_root=Path(ctx.run_root), stage_dir_path=Path(ctx.stage_dir))
