"""Stage entrypoint: wavesolution.

Artifacts live in disperser-aware wavesol directory (08_wavesol[/<slug>]/...).

P0-ARCH-02.1: run(ctx) -> StageResult and engine-controlled run_root.
"""

from __future__ import annotations

from pathlib import Path

from scorpio_pipe.pipeline.context import StageContext, StageResult
from scorpio_pipe.pipeline.resources import apply_stage_resources_to_cfg, stage_blas_guard


def run(ctx: StageContext) -> StageResult:
    from scorpio_pipe.stages.wavesolution import build_wavesolution
    from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG
    from scorpio_pipe.wavesol_paths import wavesol_dir

    cfg_eff, budget_metrics = apply_stage_resources_to_cfg(dict(ctx.cfg), stage_key="wavesolution", budget=ctx.resources)
    if budget_metrics:
        ctx.metrics.update(budget_metrics)

    outdir = wavesol_dir(cfg_eff, run_root=Path(ctx.run_root))
    outdir.mkdir(parents=True, exist_ok=True)

    with stage_blas_guard(ctx.resources.blas_threads):
        res = build_wavesolution(cfg_eff, run_root=Path(ctx.run_root))

    prods: dict[str, str] = {}
    for k in (
        "wavesolution_1d_png",
        "wavesolution_1d_json",
        "wavesolution_2d_json",
        "lambda_map",
        "rectification_model_json",
        "wave_done_json",
        "residuals_1d_csv",
        "residuals_2d_csv",
        "control_points_2d_csv",
        "residuals_2d_png",
        "residuals_2d_audit_png",
        "residuals_vs_lambda_png",
        "residuals_vs_y_png",
        "residuals_hist_png",
        "wavesolution_report_txt",
        "line_id_overview_png",
        "geometry_overlay_png",
        "geometry_prior_used_json",
    ):
        try:
            p = PRODUCT_CATALOG.best_path(cfg_eff, k, run_root=Path(ctx.run_root))
        except Exception:
            p = None
        if p is not None and Path(p).exists():
            prods[k] = str(Path(p))

    # Expose key metrics when available.
    qc: dict[str, object] = {}
    try:
        if hasattr(res, "rms1d_A"):
            qc["rms1d_A"] = float(getattr(res, "rms1d_A"))
        if hasattr(res, "rms2d_A"):
            qc["rms2d_A"] = float(getattr(res, "rms2d_A"))
    except Exception:
        pass

    return StageResult.ok(products=prods, qc=qc, artifacts=[str(outdir)], messages=[])
