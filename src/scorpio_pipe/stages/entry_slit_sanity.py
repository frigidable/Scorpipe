from __future__ import annotations

from pathlib import Path

from scorpio_pipe.pipeline.context import StageContext, StageResult
from scorpio_pipe.pipeline.resources import apply_stage_resources_to_cfg, stage_blas_guard


def run(ctx: StageContext) -> StageResult:
    from scorpio_pipe.stages.slit_sanity import run_slit_sanity

    cfg_eff, budget_metrics = apply_stage_resources_to_cfg(dict(ctx.cfg), stage_key="slit_sanity", budget=ctx.resources)
    if budget_metrics:
        ctx.metrics.update(budget_metrics)
    with stage_blas_guard(ctx.resources.blas_threads):
        res = run_slit_sanity(cfg_eff, run_root=Path(ctx.run_root))
    prods = {
        'slit_sanity_json': str(res.json_path),
        'slit_profile_png': str(res.png_path),
        'slit_consistency_md': str(res.md_path),
    }
    qc = {}
    if res.expected_width_px is not None:
        qc['expected_width_px'] = float(res.expected_width_px)
    if res.median_width_px is not None:
        qc['median_width_px'] = float(res.median_width_px)
    return (StageResult.warn if res.status == 'warn' else StageResult.ok)(products=prods, qc=qc, artifacts=list(prods.values()))
