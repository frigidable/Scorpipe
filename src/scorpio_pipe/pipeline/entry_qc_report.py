"""Stage entrypoint: qc_report.

P0-ARCH-02.2: lazy entrypoint (string-resolved by the engine).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scorpio_pipe.pipeline.context import StageContext, StageResult


def run(ctx: StageContext, **_: Any) -> StageResult:
    from scorpio_pipe.work_layout import ensure_work_layout
    from scorpio_pipe.qc_report import build_qc_report

    out_dir = Path(ctx.run_root)
    _ = ensure_work_layout(out_dir)

    res = build_qc_report(dict(ctx.cfg), run_root=out_dir)

    # Historically build_qc_report returned a Path; allow dict too.
    qc_html = None
    if isinstance(res, dict):
        qc_html = res.get("qc_html")
    else:
        qc_html = str(res)

    prods: dict[str, str] = {}
    try:
        from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG

        for k in ("qc_json", "qc_html", "timings", "products_manifest"):
            p = PRODUCT_CATALOG.best_path(dict(ctx.cfg), k, run_root=out_dir)
            if p is not None and Path(p).exists():
                prods[k] = str(Path(p))
    except Exception:
        pass

    artifacts = [str(Path(qc_html))] if qc_html else []
    return StageResult.ok(products=prods, artifacts=artifacts)
