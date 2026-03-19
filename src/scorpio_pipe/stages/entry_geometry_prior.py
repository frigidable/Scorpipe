from __future__ import annotations

from pathlib import Path

from scorpio_pipe.pipeline.context import StageContext, StageResult
from scorpio_pipe.pipeline.resources import apply_stage_resources_to_cfg, stage_blas_guard


def run(ctx: StageContext) -> StageResult:
    from scorpio_pipe.stages.geometry_prior import build_geometry_prior

    cfg_eff, budget_metrics = apply_stage_resources_to_cfg(dict(ctx.cfg), stage_key="geometry_prior", budget=ctx.resources)
    if budget_metrics:
        ctx.metrics.update(budget_metrics)
    with stage_blas_guard(ctx.resources.blas_threads):
        res = build_geometry_prior(cfg_eff, run_root=Path(ctx.run_root))
    prods = {
        'geometry_model_json': str(res.geometry_model_json),
        'geometry_preview_fits': str(res.geometry_preview_fits),
        'geometry_qc_json': str(res.geometry_qc_json),
        'repers_json': str(res.repers_json),
    }
    qc = {'trust_score': float(res.trust_score)}
    if res.fit_rms_px == res.fit_rms_px:
        qc['fit_rms_px'] = float(res.fit_rms_px)
    return StageResult.ok(products=prods, qc=qc, artifacts=[str(res.geometry_dir)])
