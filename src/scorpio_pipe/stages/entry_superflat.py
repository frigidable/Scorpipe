"""Stage entrypoint: superflat.

P0-ARCH-02.1: run(ctx) -> StageResult and strict stage I/O.
"""

from __future__ import annotations

from pathlib import Path

from scorpio_pipe.pipeline.context import StageContext, StageResult


def run(ctx: StageContext) -> StageResult:
    from scorpio_pipe.stages.calib import build_superflat
    from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG

    ctx.ensure_stage_dir()

    out = build_superflat(dict(ctx.cfg), run_root=Path(ctx.run_root), stage_dir=Path(ctx.stage_dir), allow_legacy_mirrors=False)

    prods: dict[str, str] = {}
    for k in ("superflat_fits",):
        p = PRODUCT_CATALOG.best_path(dict(ctx.cfg), k, run_root=Path(ctx.run_root))
        if p is not None and Path(p).exists():
            prods[k] = str(Path(p))

    return StageResult.ok(products=prods, artifacts=[str(out)])
