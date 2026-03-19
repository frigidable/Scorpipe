from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG
from scorpio_pipe.calib.common import save_json
from scorpio_pipe.benchmark.metrics import evaluate_metric
from scorpio_pipe.benchmark.report import write_report_files


@dataclass(frozen=True)
class LongComparisonBundle:
    source: str
    wavelength_solution: dict[str, Any]
    sky_subtraction: dict[str, Any]
    extraction: dict[str, Any]
    response_calibration: dict[str, Any]
    science_outputs: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            'source': self.source,
            'wavelength_solution': self.wavelength_solution,
            'sky_subtraction': self.sky_subtraction,
            'extraction': self.extraction,
            'response_calibration': self.response_calibration,
            'science_outputs': self.science_outputs,
        }


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    title: str
    category: str
    source_type: str
    description: str
    observer_notes: str
    metrics: dict[str, dict[str, Any]]
    manifest_path: Path
    scorpipe_bundle: str | None = None
    long_bundle: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            'case_id': self.case_id,
            'title': self.title,
            'category': self.category,
            'source_type': self.source_type,
            'description': self.description,
            'observer_notes': self.observer_notes,
            'metrics': self.metrics,
            'manifest_path': str(self.manifest_path),
            'scorpipe_bundle': self.scorpipe_bundle,
            'long_bundle': self.long_bundle,
        }


def _load_json_if_exists(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _best_path(cfg: dict[str, Any], key: str, run_root: Path) -> Path | None:
    try:
        return PRODUCT_CATALOG.best_path(cfg, key, run_root=run_root)
    except Exception:
        return None


def _nested_get(obj: Mapping[str, Any] | None, *parts: str) -> Any:
    cur: Any = obj
    for part in parts:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(part)
    return cur


def _first_float(*values: Any) -> float | None:
    for value in values:
        try:
            if value in (None, '', False):
                continue
            fv = float(value)
        except Exception:
            continue
        if np.isfinite(fv):
            return float(fv)
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        try:
            if value in (None, '', False):
                continue
            iv = int(value)
        except Exception:
            continue
        return int(iv)
    return None


def _as_clean_dict(src: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in src.items():
        if value is None:
            continue
        out[str(key)] = value
    return out


def build_scorpipe_observer_bundle(cfg: dict[str, Any], *, run_root: Path | None = None) -> LongComparisonBundle:
    run_root = Path(run_root or cfg.get('work_dir') or '.').expanduser().resolve()

    wave1 = _load_json_if_exists(_best_path(cfg, 'wavesolution_1d_json', run_root))
    wave2 = _load_json_if_exists(_best_path(cfg, 'wavesolution_2d_json', run_root))
    wave_done = _load_json_if_exists(_best_path(cfg, 'wave_done_json', run_root))
    geom_prior = _load_json_if_exists(_best_path(cfg, 'geometry_prior_used_json', run_root))

    sky_done = _load_json_if_exists(_best_path(cfg, 'sky_done', run_root))
    sky_qc = _load_json_if_exists(_best_path(cfg, 'qc_sky_json', run_root))
    sky_refine_qc = _load_json_if_exists(_best_path(cfg, 'qc_sky_refine_json', run_root))
    sky_windows = _load_json_if_exists(_best_path(cfg, 'sky_residual_windows_json', run_root))

    extract_done = _load_json_if_exists(_best_path(cfg, 'extract_done', run_root))
    extraction_qc = _load_json_if_exists(_best_path(cfg, 'extraction_qc_json', run_root))
    sens_qc = _load_json_if_exists(_best_path(cfg, 'sensfunc_qc_json', run_root))
    flux_qc = _load_json_if_exists(_best_path(cfg, 'fluxcal_qc_json', run_root))
    lines_json = _load_json_if_exists(_best_path(cfg, 'science_lines_json', run_root))
    science_report = _load_json_if_exists(_best_path(cfg, 'science_report_json', run_root))
    observer_dash = _load_json_if_exists(_best_path(cfg, 'observer_dashboard_json', run_root))

    wave_metrics = _nested_get(wave_done, 'metrics') or {}
    wavelength_solution = _as_clean_dict(
        {
            'rms': _first_float(wave_metrics.get('rms2d_A'), wave2.get('rms_A'), wave1.get('rms_A')),
            'rms1d_A': _first_float(wave1.get('rms_A')),
            'rms2d_A': _first_float(wave_metrics.get('rms2d_A'), wave2.get('rms_A')),
            'dispersion_A_per_px': _first_float(wave1.get('dispersion_A_per_px')),
            'trust_score': _first_float(wave_metrics.get('wave_trust_score'), wave2.get('wave_trust_score')),
            'line_density': _first_float(wave_metrics.get('wave_line_density')),
            'curvature_smoothness': _first_float(wave_metrics.get('wave_curvature_smoothness')),
            'geometry_prior_median_abs_resid_px': _first_float(geom_prior.get('median_abs_resid_px')),
            'n_pairs': _first_int(wave1.get('n_pairs')),
            'n_used': _first_int(wave2.get('n_used'), wave1.get('n_used')),
        }
    )

    sky_metrics = _nested_get(sky_done, 'metrics') or {}
    sky_refine_metrics = _nested_get(sky_refine_qc, 'aggregate_metrics') or {}
    sky_subtraction = _as_clean_dict(
        {
            'sky_rms': _first_float(sky_metrics.get('robust_sigma_r_after_median'), sky_metrics.get('rms_sky_median')),
            'sky_rms_before': _first_float(sky_metrics.get('robust_sigma_r_before_median')),
            'sky_coverage_frac': _first_float(sky_metrics.get('sky_coverage_frac_median')),
            'object_eating_metric': _first_float(sky_metrics.get('object_eating_metric_median')),
            'flexure_shift_A': _first_float(_nested_get(sky_done, 'extra', 'flexure', 'delta_A_median')),
            'flexure_score': _first_float(_nested_get(sky_done, 'extra', 'flexure', 'flexure_score_median')),
            'refine_sigma_gain': _first_float(sky_refine_metrics.get('sky_sigma_gain_median')),
            'refine_object_delta': _first_float(sky_refine_metrics.get('object_delta_median')),
            'n_refine_windows': _first_int(sky_refine_qc.get('n_windows_total'), len(sky_windows) if isinstance(sky_windows, list) else None),
            'status': str(sky_qc.get('status') or _nested_get(sky_done, 'status') or ''),
        }
    )

    extract_extra = _nested_get(extract_done, 'extra', 'extract') or {}
    extract_metrics = _nested_get(extract_done, 'metrics') or {}
    extract_diag = extraction_qc.get('diagnostics') if isinstance(extraction_qc.get('diagnostics'), Mapping) else {}
    extraction = _as_clean_dict(
        {
            'snr': _first_float(extract_extra.get('snr_trace_median'), extract_extra.get('snr_fixed_median')),
            'snr_trace_median': _first_float(extract_extra.get('snr_trace_median')),
            'snr_fixed_median': _first_float(extract_extra.get('snr_fixed_median')),
            'trace_stability_px': _first_float(extract_diag.get('trace_stability_px'), extract_metrics.get('trace_stability_px')),
            'contamination_risk': _first_float(extract_diag.get('contamination_risk'), extract_metrics.get('contamination_risk')),
            'fixed_vs_trace_disagreement_sigma': _first_float(extract_diag.get('fixed_vs_trace_disagreement_sigma'), extract_metrics.get('fixed_vs_trace_disagreement_sigma')),
            'aperture_sensitivity_frac': _first_float(extract_diag.get('aperture_sensitivity_frac'), extract_metrics.get('aperture_sensitivity_frac')),
            'background_asymmetry_sigma': _first_float(extract_diag.get('background_asymmetry_sigma'), extract_metrics.get('background_asymmetry_sigma')),
            'trace_frac_finite': _first_float(extract_metrics.get('trace_frac_finite')),
        }
    )

    response = _as_clean_dict(
        {
            'smoothness': _first_float(_nested_get(sens_qc, 'fit_meta', 'mad_residual'), _nested_get(sens_qc, 'fit_meta', 'wrms_residual')),
            'median_residual': _first_float(_nested_get(sens_qc, 'fit_meta', 'mad_residual')),
            'standard_status': str(_nested_get(flux_qc, 'standard_qc', 'association') or ''),
            'status': str(flux_qc.get('status') or ''),
        }
    )

    detected_lines = sum(1 for r in (lines_json.get('lines') or []) if isinstance(r, Mapping) and bool(r.get('detected')))
    science_outputs = _as_clean_dict(
        {
            'n_detected_lines': int(detected_lines),
            'calibration_status': str(flux_qc.get('status') or ''),
            'dashboard_trust_score': _first_float(_nested_get(observer_dash, 'trust_score', 'overall')),
            'science_confidence': _first_float(science_report.get('confidence_score')),
        }
    )

    return LongComparisonBundle(
        source='scorpipe',
        wavelength_solution=wavelength_solution,
        sky_subtraction=sky_subtraction,
        extraction=extraction,
        response_calibration=response,
        science_outputs=science_outputs,
    )


def write_observer_bundle(path_or_dir: str | Path, bundle: LongComparisonBundle | Mapping[str, Any]) -> Path:
    payload = bundle.as_dict() if isinstance(bundle, LongComparisonBundle) else dict(bundle)
    dest = Path(path_or_dir)
    if dest.suffix.lower() != '.json':
        dest = dest / 'scorpipe_observer_bundle.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    return save_json(dest, payload)


def _compare_metric(a: Any, b: Any) -> dict[str, Any]:
    try:
        fa = float(a)
        fb = float(b)
        if np.isfinite(fa) and np.isfinite(fb):
            return {'a': fa, 'b': fb, 'delta': fa - fb, 'ratio': (fa / fb if fb not in {0.0, -0.0} else None)}
    except Exception:
        pass
    return {'a': a, 'b': b, 'equal': a == b}


def compare_against_long(scorpipe: LongComparisonBundle | dict[str, Any], long_bundle: LongComparisonBundle | dict[str, Any]) -> dict[str, Any]:
    a = scorpipe.as_dict() if isinstance(scorpipe, LongComparisonBundle) else dict(scorpipe)
    b = long_bundle.as_dict() if isinstance(long_bundle, LongComparisonBundle) else dict(long_bundle)
    dims = ['wavelength_solution', 'sky_subtraction', 'extraction', 'response_calibration', 'science_outputs']
    out: dict[str, Any] = {'sources': {'scorpipe': a.get('source', 'scorpipe'), 'long': b.get('source', 'long')}}
    for dim in dims:
        da = dict(a.get(dim) or {})
        db = dict(b.get(dim) or {})
        comp = {}
        for k in sorted(set(da) | set(db)):
            comp[k] = _compare_metric(da.get(k), db.get(k))
        out[dim] = comp
    return out


def write_comparison_bundle(out_dir: str | Path, payload: dict[str, Any]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return save_json(out_dir / 'long_comparison.json', payload)


def load_long_bundle(path: str | Path) -> LongComparisonBundle:
    p = Path(path)
    data = json.loads(p.read_text(encoding='utf-8'))
    if 'bundle' in data and isinstance(data.get('bundle'), dict):
        data = dict(data['bundle'])
    return LongComparisonBundle(
        source=str(data.get('source') or p.stem),
        wavelength_solution=dict(data.get('wavelength_solution') or {}),
        sky_subtraction=dict(data.get('sky_subtraction') or {}),
        extraction=dict(data.get('extraction') or {}),
        response_calibration=dict(data.get('response_calibration') or {}),
        science_outputs=dict(data.get('science_outputs') or {}),
    )


def load_case_manifest(path: str | Path) -> BenchmarkCase:
    p = Path(path).expanduser().resolve()
    data = json.loads(p.read_text(encoding='utf-8'))
    return BenchmarkCase(
        case_id=str(data.get('case_id') or p.stem),
        title=str(data.get('title') or data.get('case_id') or p.stem),
        category=str(data.get('category') or 'unspecified'),
        source_type=str(data.get('source_type') or 'unknown'),
        description=str(data.get('description') or ''),
        observer_notes=str(data.get('observer_notes') or ''),
        metrics=dict(data.get('metrics') or {}),
        manifest_path=p,
        scorpipe_bundle=(str(data.get('scorpipe_bundle')) if data.get('scorpipe_bundle') else None),
        long_bundle=(str(data.get('long_bundle')) if data.get('long_bundle') else None),
    )


def _resolve_case_path(case: BenchmarkCase, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (case.manifest_path.parent / p).resolve()


def run_case_comparison(
    case: BenchmarkCase,
    *,
    scorpipe_bundle: LongComparisonBundle | Mapping[str, Any],
    long_bundle: LongComparisonBundle | Mapping[str, Any],
) -> dict[str, Any]:
    cur = scorpipe_bundle.as_dict() if isinstance(scorpipe_bundle, LongComparisonBundle) else dict(scorpipe_bundle)
    ref = long_bundle.as_dict() if isinstance(long_bundle, LongComparisonBundle) else dict(long_bundle)
    metrics = {
        metric: evaluate_metric(metric, cur, ref, spec)
        for metric, spec in sorted((case.metrics or {}).items())
    }
    failed = sorted(k for k, row in metrics.items() if str(row.get('status') or '') != 'PASS')
    summary = {
        'status': 'PASS' if not failed else 'FAIL',
        'total_metrics': len(metrics),
        'passed_metrics': len(metrics) - len(failed),
        'failed_metrics': failed,
    }
    return {
        'schema': 'scorpio_pipe.long_parity_report',
        'schema_version': 1,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'case': case.as_dict(),
        'summary': summary,
        'metrics': metrics,
        'bundle': {
            'scorpipe': cur,
            'long': ref,
        },
    }


def write_case_report(out_dir: str | Path, payload: Mapping[str, Any]) -> dict[str, Path]:
    return write_report_files(out_dir, payload)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog='python -m scorpio_pipe.benchmark.long_compare')
    ap.add_argument('--case', required=True, help='Path to a case manifest JSON')
    ap.add_argument('--scorpipe-bundle', default=None, help='Path to Scorpipe observer bundle JSON (overrides manifest)')
    ap.add_argument('--long-bundle', default=None, help='Path to LONG observer bundle JSON (overrides manifest)')
    ap.add_argument('--output', required=True, help='Output directory for JSON + markdown report')
    ap.add_argument('--soft-fail', action='store_true', help='Always exit 0 even when the comparison fails')
    ap.add_argument('--fail-on-regression', action='store_true', help='Exit 2 when any metric fails (default on CI weekly/manual)')
    return ap


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    case = load_case_manifest(ns.case)
    sp_path = Path(ns.scorpipe_bundle).resolve() if ns.scorpipe_bundle else _resolve_case_path(case, case.scorpipe_bundle)
    lg_path = Path(ns.long_bundle).resolve() if ns.long_bundle else _resolve_case_path(case, case.long_bundle)
    if sp_path is None or not sp_path.exists():
        raise SystemExit(f'Missing Scorpipe bundle: {sp_path}')
    if lg_path is None or not lg_path.exists():
        raise SystemExit(f'Missing LONG bundle: {lg_path}')
    payload = run_case_comparison(case, scorpipe_bundle=load_long_bundle(sp_path), long_bundle=load_long_bundle(lg_path))
    write_case_report(ns.output, payload)
    failed = bool(payload.get('summary', {}).get('failed_metrics'))
    if failed and ns.fail_on_regression and not ns.soft_fail:
        return 2
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
