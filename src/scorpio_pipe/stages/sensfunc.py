from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scorpio_pipe.calib.sensfunc import build_sensfunc, write_sensfunc_products
from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG
from scorpio_pipe.io.done_json import write_done_json
from scorpio_pipe.pipeline.context import StageResult


def run_sensfunc(cfg: dict[str, Any], *, run_root: Path, stage_dir_path: Path) -> StageResult:
    stage_dir_path.mkdir(parents=True, exist_ok=True)
    std_path = PRODUCT_CATALOG.best_path(cfg, 'standard_1d_fits', run_root=run_root)
    qc_path = PRODUCT_CATALOG.best_path(cfg, 'standard_qc_json', run_root=run_root)
    qc = json.loads(qc_path.read_text(encoding='utf-8')) if qc_path.exists() else {}
    payload = build_sensfunc(cfg, standard_spec_path=std_path, standard_qc=qc)
    products = write_sensfunc_products(stage_dir_path, payload)
    qcp = payload.get('qc') or {}
    status = 'warn' if int(qcp.get('fit_points') or 0) < 100 else 'ok'
    write_done_json(stage='sensfunc', stage_dir=stage_dir_path, status=status, outputs=products, metrics={'fit_points': qcp.get('fit_points'), 'residual_sigma_ln': qcp.get('residual_sigma_ln')}, qc=qcp, extra={'fit_meta': payload.get('fit_meta')})
    return (StageResult.warn if status == 'warn' else StageResult.ok)(products=products, artifacts=list(products.values()), qc=qcp, messages=['sensfunc built'])
