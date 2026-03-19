"""Stage entrypoint: superbias.

P0-ARCH-02.1: run(ctx) -> StageResult and strict stage I/O.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scorpio_pipe.pipeline.context import StageContext, StageResult


def run(ctx: StageContext) -> StageResult:
    from scorpio_pipe.stages.calib import build_superbias
    from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG

    ctx.ensure_stage_dir()

    # Build inside the engine-controlled run_root.
    out = build_superbias(dict(ctx.cfg), run_root=Path(ctx.run_root), stage_dir=Path(ctx.stage_dir), allow_legacy_mirrors=False)

    # Collect declared products (best-effort)
    prod_keys = (
        "superbias_fits",
        "master_bias_fits",
        "master_bias_var_fits",
        "master_bias_dq_fits",
        "master_bias_index",
    )
    prods: dict[str, str] = {}
    for k in prod_keys:
        p = PRODUCT_CATALOG.best_path(dict(ctx.cfg), k, run_root=Path(ctx.run_root))
        if p is not None and Path(p).exists():
            prods[k] = str(Path(p))

    return StageResult.ok(products=prods, artifacts=[str(out)])
