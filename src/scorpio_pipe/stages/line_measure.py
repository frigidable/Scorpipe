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


def _qualify_lines_for_shared_bias_covariance(
    rows: list[dict[str, Any]], header: Any,
) -> tuple[list[dict[str, Any]], bool]:
    """Withhold formal detections when shared MasterBias VAR is unmodelled.

    The line flux estimate remains diagnostic. Its reported conditional
    S/N and errors are not calibrated for the correlated shared bias.
    """
    shared = (
        str(header.get("SBGSTAT") or "").strip().upper() == "SHARED"
        and str(header.get("SBCOVQ") or "").strip().upper() == "UNMODEL"
    )
    qualified: list[dict[str, Any]] = []
    for entry in rows:
        row = dict(entry)
        row["uncertainty_qualified"] = not shared
        if shared:
            row["conditional_detection_candidate"] = bool(row.get("detected"))
            row["detected"] = False
            row["upper_limit_3sigma"] = None
            row["uncertainty_status"] = "SHARED_MASTERBIAS_COVARIANCE_UNMODELLED"
            # Existing partial-window reasons and mask diagnoses remain.
            if not row.get("reason"):
                row["reason"] = "uncertified_shared_masterbias_uncertainty"
        qualified.append(row)
    return qualified, shared


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
    mask = np.asarray(spec['mask'])
    # The generic table reader retains historical air defaults. For
    # scientific line centres, guessing missing wavelength medium can shift
    # derived velocities: require the actual output FITS to declare it.
    waveref = str(spec['header'].get('WAVEREF') or '').strip().lower()
    if waveref not in {'air', 'vacuum'}:
        raise ValueError(
            'LineMeasure requires explicit WAVEREF=air/vacuum in flux-calibrated spectrum'
        )
    scfg = cfg.get('science') if isinstance(cfg.get('science'), dict) else {}
    redshift = float((scfg or {}).get('redshift', 0.0) or 0.0)
    detection_snr = float((scfg or {}).get('line_detection_snr', 3.0) or 3.0)
    rows = measure_lines(
        wave,
        flux,
        var,
        calsys_rel=calsys['rel'],
        calsys_tell=calsys['tell'],
        calsys_abs=calsys['abs'],
        lines=DEFAULT_LINES,
        redshift=redshift,
        mask=mask,
        coverage_fraction=spec.get('frac_used') if spec.get('dq1d_version') == 1 else None,
        waveref=waveref,
        detection_snr=detection_snr,
    )
    rows, shared_bias_unqualified = _qualify_lines_for_shared_bias_covariance(
        rows, spec["header"]
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
        'dq1d_version': spec.get('dq1d_version'),
        'partial_line_windows': int(sum(
            1 for row in rows if row.get('reason') == 'partial_aperture_or_spectral_coverage'
        )),
        'shared_bias_variance_unqualified': bool(shared_bias_unqualified),
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
        'shared_bias_variance_unqualified': bool(shared_bias_unqualified),
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
