from __future__ import annotations

from pathlib import Path
from typing import Any

from scorpio_pipe.calib.telluric import build_telluric_model, write_telluric_products
from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG
from scorpio_pipe.io.done_json import write_done_json
from scorpio_pipe.pipeline.context import StageResult


def run_telluric_model(cfg: dict[str, Any], *, run_root: Path, stage_dir_path: Path) -> StageResult:
    stage_dir_path.mkdir(parents=True, exist_ok=True)
    std_path = PRODUCT_CATALOG.best_path(cfg, 'standard_1d_fits', run_root=run_root)
    payload = build_telluric_model(cfg, standard_spec_path=std_path)
    products = write_telluric_products(stage_dir_path, payload)
    qcp = payload.get('qc') or {}
    status = 'warn' if float(qcp.get('median_confidence') or 0.0) < 0.35 else 'ok'
    write_done_json(stage='telluric_model', stage_dir=stage_dir_path, status=status, outputs=products, metrics={'median_confidence': qcp.get('median_confidence')}, qc=qcp)
    return (StageResult.warn if status == 'warn' else StageResult.ok)(products=products, artifacts=list(products.values()), qc=qcp, messages=['telluric model built'])
