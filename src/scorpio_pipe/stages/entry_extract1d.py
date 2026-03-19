"""Stage entrypoint: extract1d.

P0-ARCH-02.2: lazy entrypoint (string-resolved by the engine).
"""

from __future__ import annotations

from pathlib import Path

from scorpio_pipe.pipeline.context import StageContext, StageResult
from scorpio_pipe.pipeline.resources import apply_stage_resources_to_cfg, stage_blas_guard


def run(ctx: StageContext) -> StageResult:
    from scorpio_pipe.stages.extract1d import run_extract1d
    from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG

    ctx.ensure_stage_dir()

    cfg_eff, budget_metrics = apply_stage_resources_to_cfg(dict(ctx.cfg), stage_key="extract1d", budget=ctx.resources)
    if isinstance(getattr(ctx, "metrics", None), dict):
        ctx.metrics.update(budget_metrics)
    with stage_blas_guard(getattr(ctx.resources, "blas_threads", None) if ctx.resources is not None else None):
        res = run_extract1d(cfg_eff, out_dir=Path(ctx.stage_dir))

    prods: dict[str, str] = {}
    for k in ("extract_done", "trace_json", "spec1d_fits", "spec1d_png", "extract1d_done", "extraction_qc_json", "extraction_diagnostics_png"):
        try:
            p = PRODUCT_CATALOG.best_path(dict(ctx.cfg), k, run_root=Path(ctx.run_root))
        except Exception:
            p = None
        if p is not None and Path(p).exists():
            prods[k] = str(Path(p))

    artifacts: list[str] = []
    try:
        if isinstance(res, dict):
            for cand in (res.get('spec1d'), res.get('spec1d_fits')):
                if isinstance(cand, (str, Path)):
                    pp = Path(str(cand))
                    if pp.exists():
                        artifacts.append(str(pp))
    except Exception:
        pass

    return StageResult.ok(products=prods, artifacts=artifacts)
