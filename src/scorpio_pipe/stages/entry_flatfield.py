"""Stage entrypoint: flatfield.

P0-ARCH-02.2: lazy entrypoint (string-resolved by the engine).
"""

from __future__ import annotations

from pathlib import Path

from scorpio_pipe.pipeline.context import StageContext, StageResult


def run(ctx: StageContext) -> StageResult:
    from scorpio_pipe.stages.flatfield import run_flatfield
    from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG

    ctx.ensure_stage_dir()

    res = run_flatfield(cfg=dict(ctx.cfg), out_dir=Path(ctx.stage_dir))

    prods: dict[str, str] = {}
    for k in ("flatfield_done", "flatfield_flat_fits"):
        try:
            p = PRODUCT_CATALOG.best_path(dict(ctx.cfg), k, run_root=Path(ctx.run_root))
        except Exception:
            p = None
        if p is not None and Path(p).exists():
            prods[k] = str(Path(p))

    arts = []
    try:
        if isinstance(res, (str, Path)):
            arts.append(str(Path(res)))
        elif isinstance(res, dict):
            for v in res.values():
                if isinstance(v, (str, Path)):
                    pp = Path(str(v))
                    if pp.exists():
                        arts.append(str(pp))
    except Exception:
        pass

    return StageResult.ok(products=prods, artifacts=arts)
