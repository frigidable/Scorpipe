"""Stage entrypoint: stack2d.

This entrypoint owns the input selection logic that used to live in the engine.

Contract (P1-E)
---------------
Stack2D consumes rectified, sky-subtracted frames, preferring:
  10_linearize/<stem>_skysub.fits

P0-ARCH-02.2: lazy entrypoint (string-resolved by the engine).
"""

from __future__ import annotations

from pathlib import Path

from scorpio_pipe.pipeline.context import StageContext, StageResult
from scorpio_pipe.pipeline.resources import apply_stage_resources_to_cfg, stage_blas_guard


def run(ctx: StageContext) -> StageResult:
    from scorpio_pipe.stages.stack2d import run_stack2d
    from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG
    from scorpio_pipe.product_naming import sky_sub_fits_name
    from scorpio_pipe.workspace_paths import stage_dir, extract_stem_short

    run_root = Path(ctx.run_root)
    lin_stage = stage_dir(run_root, "linearize")

    # Prefer an explicit science exposure list from config.
    frames = ctx.cfg.get("frames") if isinstance(ctx.cfg, dict) else {}
    frames = frames if isinstance(frames, dict) else {}
    obj_list = frames.get("obj") if isinstance(frames.get("obj"), list) else []

    stems = [extract_stem_short(x) for x in obj_list if isinstance(x, str) and x.strip()]

    inputs: list[Path] = []
    if stems:
        missing: list[str] = []
        for stem in stems:
            p = lin_stage / sky_sub_fits_name(stem)
            if p.exists():
                inputs.append(p)
            else:
                missing.append(stem)
        if missing:
            raise FileNotFoundError(
                "Stack2D: missing rectified sky-subtracted inputs for stems: "
                + ", ".join(missing)
                + ". Expected 10_linearize/<stem>_skysub.fits. Run Linearization first."
            )
    else:
        inputs = sorted(p for p in lin_stage.glob("*_skysub.fits") if p.is_file())
        if not inputs:
            raise FileNotFoundError(
                f"No rectified sky-subtracted frames found in {lin_stage}. Expected 10_linearize/*_skysub.fits. Run Linearization first."
            )

    ctx.ensure_stage_dir()
    cfg_eff, budget_metrics = apply_stage_resources_to_cfg(dict(ctx.cfg), stage_key="stack2d", budget=ctx.resources)
    if isinstance(getattr(ctx, "metrics", None), dict):
        ctx.metrics.update(budget_metrics)
    with stage_blas_guard(getattr(ctx.resources, "blas_threads", None) if ctx.resources is not None else None):
        res = run_stack2d(cfg_eff, inputs=inputs, out_dir=Path(ctx.stage_dir))

    prods: dict[str, str] = {}
    for k in (
        "stack2d_done",
        "stack_done",
        "stacked2d_fits",
        "stack2d_fits",
        "eta_lambda_fits",
        "coverage_png",
    ):
        try:
            p = PRODUCT_CATALOG.best_path(dict(ctx.cfg), k, run_root=run_root)
        except Exception:
            p = None
        if p is not None and Path(p).exists():
            prods[k] = str(Path(p))

    artifacts: list[str] = []
    try:
        if isinstance(res, dict):
            for cand in (res.get('stack2d'), res.get('stacked2d'), res.get('stack2d_fits'), res.get('stacked2d_fits')):
                if isinstance(cand, (str, Path)):
                    pp = Path(str(cand))
                    if pp.exists():
                        artifacts.append(str(pp))
    except Exception:
        pass

    return StageResult.ok(products=prods, artifacts=artifacts)
