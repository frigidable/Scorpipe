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

    run_root = Path(ctx.run_root).expanduser().resolve()
    _ = ensure_work_layout(run_root)

    # The engine owns the canonical run root; the report builder's out_dir is
    # specifically the QC subdirectory (NOT the run root). Keep both in sync
    # even when the config was originally loaded from a different location.
    cfg = dict(ctx.cfg)
    cfg["work_dir"] = str(run_root)
    qc_dir = run_root / "qc"
    res = build_qc_report(cfg, out_dir=qc_dir)

    # QCReportOutput is a mapping with "json" and "html" keys, while older
    # callers may return the HTML Path directly. Do not confuse "qc_html"
    # (the product catalog key) with the actual result key "html".
    if isinstance(res, dict):
        qc_html = res.get("html") or res.get("qc_html")
        qc_json = res.get("json")
    else:
        qc_html = res
        qc_json = qc_dir / "qc_report.json"

    if not qc_html or not Path(qc_html).is_file():
        raise RuntimeError("QC report builder did not create its HTML output")
    if not qc_json or not Path(qc_json).is_file():
        raise RuntimeError("QC report builder did not create its JSON output")

    prods: dict[str, str] = {}
    from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG

    for k in ("qc_json", "qc_html", "timings", "products_manifest"):
        p = PRODUCT_CATALOG.best_path(cfg, k, run_root=run_root)
        if p is not None and Path(p).exists():
            prods[k] = str(Path(p))

    return StageResult.ok(
        products=prods,
        artifacts=[str(Path(qc_html)), str(Path(qc_json))],
    )
