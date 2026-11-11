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
    """Withhold formal detections for explicitly unqualified variance.

    Fixed-weight Stack2D VAR omits data-driven weight-selection uncertainty
    and cross-pixel/cross-exposure covariance. Its S/N can remain useful as a
    *conditional diagnostic*, but cannot certify a line detection or upper
    limit. Keep the old shared-MasterBias return flag for API compatibility.
    """
    status = {
        key: str(header.get(key) or "").strip().upper()
        for key in ("SBGSTAT", "SBCOVQ", "STKVARQ", "STKXCVQ", "STKYCVQ")
    }
    shared = status["SBGSTAT"] == "SHARED" and status["SBCOVQ"] == "UNMODEL"
    causes = [
        label for label, present in (
            ("STACK_FIXED_WEIGHT_CONDITIONAL_VAR", status["STKVARQ"] == "COND_ONLY"),
            ("STACK_SHARED_COVARIANCE_UNMODELLED", status["STKXCVQ"] == "UNMODEL"),
            ("STACK_SPATIAL_COVARIANCE_UNMODELLED", status["STKYCVQ"] == "UNMODEL"),
            ("MASTERBIAS_COVARIANCE_UNMODELLED", status["SBCOVQ"] == "UNMODEL"),
        ) if present
    ]
    model_unqualified = bool(causes)
    qualified: list[dict[str, Any]] = []
    for entry in rows:
        row = dict(entry)
        # No explicit negative flags does not independently prove scientific
        # uncertainty qualification: preserve the existing unknown (None).
        row["uncertainty_qualified"] = False if model_unqualified else None
        row["shared_bias_detection_gate_triggered"] = bool(shared)
        row["uncertainty_gate_triggered"] = model_unqualified
        row["unqualified_uncertainty_causes"] = list(causes)
        if model_unqualified:
            row["conditional_detection_candidate"] = bool(row.get("detected"))
            row["detected"] = False
            row["upper_limit_3sigma"] = None
            row["uncertainty_status"] = (
                "SHARED_MASTERBIAS_COVARIANCE_UNMODELLED"
                if shared else "STACK_OR_CALIBRATION_COVARIANCE_UNMODELLED"
            )
            # Retain more specific reasons such as partial aperture; never
            # convert a conditional 3-sigma error into a formal upper limit.
            if not row.get("reason"):
                row["reason"] = (
                    "uncertified_shared_masterbias_uncertainty"
                    if shared else "uncertified_stack_or_calibration_uncertainty"
                )
        qualified.append(row)
    return qualified, shared


def _science_wavelength_angstrom(
    wave: np.ndarray, header: Any,
) -> tuple[np.ndarray, str, str]:
    """Normalize the wavelength *coordinate* to Å for catalog line science.

    Calibrated F_lambda is already per Angstrom: never scale the flux or
    variance here. Native nm coordinates need conversion so both the
    rest-line match and the physical wavelength integral use Angstrom.
    """
    unit = str(header.get("WAVEUNIT") or header.get("CUNIT1") or "").strip().lower()
    norm = str(header.get("REBNORM") or "").strip().upper()
    if unit in {"a", "aa", "ang", "angstrom", "angstroms", "å", "ångström"}:
        scale, native, qualification = 1.0, "Angstrom", "EXPLICIT"
    elif unit in {"nm", "nanometer", "nanometers"}:
        scale, native, qualification = 10.0, "nm", "EXPLICIT"
    elif not unit and norm != "COUNT_V1":
        # Existing legacy spectra may lack axis metadata; keep the assumption
        # visible and unqualified rather than claiming verified coordinates.
        scale, native, qualification = 1.0, "ASSUMED_ANGSTROM", "LEGACY_ASSUMED"
    elif not unit:
        raise ValueError("COUNT_V1 LineMeasure requires explicit WAVEUNIT")
    else:
        raise ValueError(f"Unsupported LineMeasure wavelength unit: {unit!r}")
    arr = np.asarray(wave, dtype=float)
    if (arr.ndim != 1 or arr.size < 2 or not np.isfinite(arr).all()
            or np.any(arr <= 0) or not np.all(np.diff(arr) > 0)):
        raise ValueError("LineMeasure requires finite positive increasing wavelengths")
    return arr * scale, native, qualification


def run_line_measure(cfg: dict[str, Any], *, run_root: Path, stage_dir_path: Path) -> StageResult:
    stage_dir_path.mkdir(parents=True, exist_ok=True)
    spec_path = PRODUCT_CATALOG.best_path(cfg, 'fluxcal_spec1d_fits', run_root=run_root)
    calsys_path = PRODUCT_CATALOG.best_path(cfg, 'calibration_systematics_fits', run_root=run_root)
    fluxcal_qc_path = PRODUCT_CATALOG.best_path(cfg, 'fluxcal_qc_json', run_root=run_root)
    spec = read_spec1d(spec_path, mode='trace')
    calsys = _read_calsys(calsys_path, len(spec['wave']))
    wave, input_waveunit, coordinate_qualification = _science_wavelength_angstrom(
        spec['wave'], spec['header']
    )
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
    uncertainty_gate_triggered = any(bool(row.get("uncertainty_gate_triggered")) for row in rows)
    n_conditional_candidates = sum(
        bool(row.get("conditional_detection_candidate", row.get("detected")))
        for row in rows
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
        'wavelength_medium': waveref,
        'input_wavelength_unit': input_waveunit,
        'measurement_wavelength_unit': 'Angstrom',
        'coordinate_qualification': coordinate_qualification,
        'partial_line_windows': int(sum(
            1 for row in rows if row.get('reason') == 'partial_aperture_or_spectral_coverage'
        )),
        'shared_bias_variance_unqualified': bool(shared_bias_unqualified),
        'uncertainty_detection_gate_triggered': bool(uncertainty_gate_triggered),
        'n_conditional_detection_candidates': int(n_conditional_candidates),
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
        'n_conditional_detection_candidates': int(n_conditional_candidates),
        'uncertainty_detection_gate_triggered': bool(uncertainty_gate_triggered),
        'wavelength_medium': waveref,
        'input_wavelength_unit': input_waveunit,
        'measurement_wavelength_unit': 'Angstrom',
        'coordinate_qualification': coordinate_qualification,
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
