"""Stage entrypoint: sky (Kelson sky subtraction).

P0-ARCH-02.2: lazy entrypoint (string-resolved by the engine).
"""

from __future__ import annotations

from pathlib import Path

from scorpio_pipe.pipeline.context import StageContext, StageResult
from scorpio_pipe.pipeline.resources import apply_stage_resources_to_cfg, stage_blas_guard


def run(ctx: StageContext) -> StageResult:
    from scorpio_pipe.stages.sky_sub import run_sky_sub
    from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG

    ctx.ensure_stage_dir()

    cfg_eff, budget_metrics = apply_stage_resources_to_cfg(dict(ctx.cfg), stage_key="sky", budget=ctx.resources)
    if isinstance(getattr(ctx, "metrics", None), dict):
        ctx.metrics.update(budget_metrics)
    with stage_blas_guard(getattr(ctx.resources, "blas_threads", None) if ctx.resources is not None else None):
        run_sky_sub(cfg_eff, out_dir=Path(ctx.stage_dir))

    prods: dict[str, str] = {}
    for k in (
        "sky_done",
        "qc_sky_json",
        "sky_preview_fits",
        "sky_preview_png",
        "sky_skysub_raw_fits",
    ):
        try:
            p = PRODUCT_CATALOG.best_path(dict(ctx.cfg), k, run_root=Path(ctx.run_root))
        except Exception:
            p = None
        if p is not None and Path(p).exists():
            prods[k] = str(Path(p))

    return StageResult.ok(products=prods, artifacts=[str(Path(ctx.stage_dir))])
