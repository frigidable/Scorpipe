"""Stage entrypoint: linearize.

P0-ARCH-02.2: lazy entrypoint (string-resolved by the engine).
"""

from __future__ import annotations

from pathlib import Path

from scorpio_pipe.pipeline.context import CancelToken, StageContext, StageResult
from scorpio_pipe.pipeline.resources import apply_stage_resources_to_cfg, stage_blas_guard


def run(ctx: StageContext, *, cancel_token: CancelToken | None = None) -> StageResult:
    from scorpio_pipe.stages.linearize import run_linearize
    from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG

    ctx.ensure_stage_dir()

    cfg_eff, budget_metrics = apply_stage_resources_to_cfg(dict(ctx.cfg), stage_key="linearize", budget=ctx.resources)
    if isinstance(getattr(ctx, "metrics", None), dict):
        ctx.metrics.update(budget_metrics)
    with stage_blas_guard(getattr(ctx.resources, "blas_threads", None) if ctx.resources is not None else None):
        res = run_linearize(cfg_eff, out_dir=Path(ctx.stage_dir), cancel_token=cancel_token)

    prods: dict[str, str] = {}
    for k in (
        "linearize_done",
        "lin_preview_fits",
        "lin_preview_png",
        "linearize_qc",
        "linearize_skysub_fits",
        "linearize_skymodel_fits",
    ):
        try:
            p = PRODUCT_CATALOG.best_path(dict(ctx.cfg), k, run_root=Path(ctx.run_root))
        except Exception:
            p = None
        if p is not None and Path(p).exists():
            prods[k] = str(Path(p))

    artifacts = []
    if isinstance(res, dict):
        try:
            pr = res.get('products') if isinstance(res.get('products'), dict) else None
            if pr and isinstance(pr.get('preview_fits'), (str, Path)):
                artifacts.append(str(Path(str(pr['preview_fits']))))
        except Exception:
            pass

    return StageResult.ok(products=prods, artifacts=artifacts)
