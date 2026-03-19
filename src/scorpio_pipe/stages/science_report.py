from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scorpio_pipe.calib.common import save_json
from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG
from scorpio_pipe.io.done_json import write_done_json
from scorpio_pipe.pipeline.context import StageResult
from scorpio_pipe.science.report import build_science_report_payload, render_science_report_html, render_science_report_markdown


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def run_science_report(cfg: dict[str, Any], *, run_root: Path, stage_dir_path: Path) -> StageResult:
    stage_dir_path.mkdir(parents=True, exist_ok=True)
    flux_qc = _load_json(PRODUCT_CATALOG.best_path(cfg, 'fluxcal_qc_json', run_root=run_root))
    std_qc = _load_json(PRODUCT_CATALOG.best_path(cfg, 'standard_qc_json', run_root=run_root))
    sens_qc = _load_json(PRODUCT_CATALOG.best_path(cfg, 'sensfunc_qc_json', run_root=run_root))
    try:
        tel_qc_path = PRODUCT_CATALOG.best_path(cfg, 'telluric_qc_json', run_root=run_root)
    except Exception:
        tel_qc_path = None
    tel_qc = _load_json(tel_qc_path)
    lines_json = _load_json(PRODUCT_CATALOG.best_path(cfg, 'science_lines_json', run_root=run_root))
    spectrum_preview = None
    line_preview = None
    try:
        spectrum_preview = str(PRODUCT_CATALOG.best_path(cfg, 'fluxcal_preview_png', run_root=run_root))
    except Exception:
        pass
    try:
        line_preview = str(PRODUCT_CATALOG.best_path(cfg, 'science_lines_preview_png', run_root=run_root))
    except Exception:
        pass
    payload = build_science_report_payload(
        fluxcal_qc=flux_qc,
        lines=list(lines_json.get('lines') or []),
        standard_qc=std_qc,
        sensfunc_qc=sens_qc,
        telluric_qc=tel_qc,
        spectrum_preview=spectrum_preview,
        line_preview=line_preview,
    )
    md = render_science_report_markdown(payload)
    html = render_science_report_html(payload)
    md_path = stage_dir_path / 'science_report.md'
    html_path = stage_dir_path / 'science_report.html'
    md_path.write_text(md, encoding='utf-8')
    html_path.write_text(html, encoding='utf-8')
    plan_path = save_json(stage_dir_path / 'science_report_plan.json', {'stage': 'science_report', 'status': 'ok', 'summary': {'calibration_status': payload.get('calibration_status'), 'n_detected_lines': len(payload.get('detected_lines') or [])}})
    products = {
        'science_report_plan_json': str(plan_path),
        'science_report_md': str(md_path),
        'science_report_html': str(html_path),
    }
    qc = {'stage': 'science_report', 'calibration_status': payload.get('calibration_status'), 'n_detected_lines': len(payload.get('detected_lines') or []), 'n_warnings': len(payload.get('warnings') or [])}
    write_done_json(stage='science_report', stage_dir=stage_dir_path, status='ok', outputs=products, metrics=qc, qc=qc)
    return StageResult.ok(products=products, artifacts=list(products.values()), qc=qc, messages=['science_report rendered'])
