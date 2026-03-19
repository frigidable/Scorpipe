from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scorpio_pipe.calib.common import read_spec1d, save_json
from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG
from scorpio_pipe.io.done_json import write_done_json
from scorpio_pipe.pipeline.context import StageResult
from scorpio_pipe.science.lines import DEFAULT_LINES, measure_lines, plot_line_measurements_png, write_lines_csv
from scorpio_pipe.science.reddening import balmer_decrement_summary


def _read_calsys(path: str | Path | None, nlam: int) -> dict[str, np.ndarray]:
    if path is None or not Path(path).exists():
        z = np.zeros(int(nlam), dtype=float)
        return {'rel': z.copy(), 'tell': z.copy(), 'abs': z.copy()}
    from astropy.io import fits
    with fits.open(path, memmap=False) as hdul:
        out = {}
        for ext, key in [('CALSYS_REL', 'rel'), ('CALSYS_TELL', 'tell'), ('CALSYS_ABS', 'abs'), ('CALSYS_TOTAL', 'total')]:
            if ext in hdul:
                out[key] = np.asarray(hdul[ext].data, dtype=float).ravel()
            else:
                out[key] = np.zeros(int(nlam), dtype=float)
        return out


def run_line_measure(cfg: dict[str, Any], *, run_root: Path, stage_dir_path: Path) -> StageResult:
    stage_dir_path.mkdir(parents=True, exist_ok=True)
    spec_path = PRODUCT_CATALOG.best_path(cfg, 'fluxcal_spec1d_fits', run_root=run_root)
    calsys_path = PRODUCT_CATALOG.best_path(cfg, 'calibration_systematics_fits', run_root=run_root)
    fluxcal_qc_path = PRODUCT_CATALOG.best_path(cfg, 'fluxcal_qc_json', run_root=run_root)
    spec = read_spec1d(spec_path, mode='trace')
    calsys = _read_calsys(calsys_path, len(spec['wave']))
    wave = np.asarray(spec['wave'], dtype=float)
    flux = np.asarray(spec['flux'], dtype=float)
    var = np.asarray(spec['var'], dtype=float)
    scfg = cfg.get('science') if isinstance(cfg.get('science'), dict) else {}
    redshift = float((scfg or {}).get('redshift', 0.0) or 0.0)
    rows = measure_lines(
        wave,
        flux,
        var,
        calsys_rel=calsys['rel'],
        calsys_tell=calsys['tell'],
        calsys_abs=calsys['abs'],
        lines=DEFAULT_LINES,
        redshift=redshift,
    )
    lines_by_name = {str(r.get('name')): r for r in rows if isinstance(r, dict)}
    redd = balmer_decrement_summary(lines_by_name)
    csv_path = write_lines_csv(stage_dir_path / 'science_lines.csv', rows)
    json_payload = {
        'stage': 'line_measure',
        'spectrum_path': str(spec_path),
        'calibration_systematics_path': str(calsys_path),
        'lines': rows,
        'reddening': redd,
        'calibration_status': json.loads(fluxcal_qc_path.read_text(encoding='utf-8')).get('status') if fluxcal_qc_path.exists() else None,
        'calibration_uncertainty_summary': {
            'median_rel': float(np.nanmedian(calsys['rel'])) if calsys['rel'].size else 0.0,
            'median_tell': float(np.nanmedian(calsys['tell'])) if calsys['tell'].size else 0.0,
            'median_abs': float(np.nanmedian(calsys['abs'])) if calsys['abs'].size else 0.0,
            'median_total': float(np.nanmedian(calsys.get('total', np.zeros_like(calsys['rel'])))) if calsys['rel'].size else 0.0,
        },
    }
    json_path = save_json(stage_dir_path / 'science_lines.json', json_payload)
    preview_path = plot_line_measurements_png(stage_dir_path / 'science_lines_preview.png', wave, flux, rows)
    detected = [r for r in rows if bool(r.get('detected'))]
    n_det = int(len(detected))
    qc = {
        'stage': 'line_measure',
        'n_lines_total': int(len(rows)),
        'n_lines_detected': n_det,
        'strongest_line': (max(detected, key=lambda r: abs(float(r.get('flux') or 0.0)))['name'] if detected else None),
        'reddening': redd,
    }
    plan_path = save_json(stage_dir_path / 'line_measure_plan.json', {'stage': 'line_measure', 'status': ('ok' if n_det > 0 else 'warn'), 'qc': qc})
    products = {
        'line_measure_plan_json': str(plan_path),
        'science_lines_csv': str(csv_path),
        'science_lines_json': str(json_path),
        'science_lines_preview_png': str(preview_path),
    }
    write_done_json(stage='line_measure', stage_dir=stage_dir_path, status=('ok' if n_det > 0 else 'warn'), outputs=products, metrics={'n_detected': n_det}, qc=qc)
    return (StageResult.ok if n_det > 0 else StageResult.warn)(products=products, artifacts=list(products.values()), qc=qc, messages=[f'line_measure detected {n_det} lines'])
