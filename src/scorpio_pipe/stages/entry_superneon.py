"""Stage entrypoint: superneon.

Historically, SuperNeon artifacts live in the disperser-aware wavesol directory
(08_wavesol[/<slug>]/...).

P0-ARCH-02.1: run(ctx) -> StageResult and engine-controlled run_root.
"""

from __future__ import annotations

from pathlib import Path

from scorpio_pipe.pipeline.context import StageContext, StageResult
from scorpio_pipe.pipeline.resources import apply_stage_resources_to_cfg, stage_blas_guard


def run(ctx: StageContext) -> StageResult:
    from scorpio_pipe.stages.superneon import build_superneon
    from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG
    from scorpio_pipe.wavesol_paths import wavesol_dir

    # SuperNeon lives under the wavesol directory (inside run_root).
    outdir = wavesol_dir(dict(ctx.cfg), run_root=Path(ctx.run_root))
    outdir.mkdir(parents=True, exist_ok=True)

    cfg_eff, budget_metrics = apply_stage_resources_to_cfg(dict(ctx.cfg), stage_key="superneon", budget=ctx.resources)
    if isinstance(getattr(ctx, "metrics", None), dict):
        ctx.metrics.update(budget_metrics)
    with stage_blas_guard(getattr(ctx.resources, "blas_threads", None) if ctx.resources is not None else None):
        res = build_superneon(cfg_eff, run_root=Path(ctx.run_root))

    prods: dict[str, str] = {}
    for k in (
        "superneon_fits",
        "superneon_png",
        "peaks_candidates_csv",
        "superneon_shifts_csv",
        "superneon_shifts_json",
        "superneon_qc_json",
    ):
        try:
            p = PRODUCT_CATALOG.best_path(dict(ctx.cfg), k, run_root=Path(ctx.run_root))
        except Exception:
            p = None
        if p is not None and Path(p).exists():
            prods[k] = str(Path(p))

    msgs = []
    try:
        if getattr(res, "qc", None) is not None:
            pass
    except Exception:
        pass

    return StageResult.ok(products=prods, artifacts=[str(outdir)], messages=msgs)
