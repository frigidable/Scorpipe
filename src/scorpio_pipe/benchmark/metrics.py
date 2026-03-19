from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def get_nested_metric(payload: Mapping[str, Any], dotted_path: str) -> Any:
    cur: Any = payload
    for part in str(dotted_path or '').split('.'):
        if not part:
            continue
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur.get(part)
    return cur


def compare_scalar(current: Any, reference: Any) -> dict[str, Any]:
    try:
        c = float(current)
        r = float(reference)
        if np.isfinite(c) and np.isfinite(r):
            return {
                'current': c,
                'reference': r,
                'delta': c - r,
                'abs_delta': abs(c - r),
                'ratio': (c / r) if r not in {0.0, -0.0} else None,
                'kind': 'numeric',
            }
    except Exception:
        pass
    return {
        'current': current,
        'reference': reference,
        'equal': current == reference,
        'kind': 'categorical',
    }


def evaluate_metric(
    metric_path: str,
    current_bundle: Mapping[str, Any],
    reference_bundle: Mapping[str, Any],
    spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rule = dict(spec or {})
    prefer = str(rule.get('prefer') or 'equal').strip().lower()
    tol_abs = rule.get('tolerance_abs')
    tol_rel = rule.get('tolerance_rel')
    label = str(rule.get('label') or metric_path)

    current = get_nested_metric(current_bundle, metric_path)
    reference = get_nested_metric(reference_bundle, metric_path)
    cmp = compare_scalar(current, reference)
    out = {
        'metric': metric_path,
        'label': label,
        'prefer': prefer,
        'rule': rule,
        **cmp,
    }

    status = 'PASS'
    reason = 'values match expectation'

    if cmp.get('kind') != 'numeric':
        status = 'PASS' if bool(cmp.get('equal')) else 'FAIL'
        reason = 'categorical match' if status == 'PASS' else 'categorical mismatch'
    else:
        delta = float(cmp.get('delta') or 0.0)
        current_f = float(cmp.get('current'))
        reference_f = float(cmp.get('reference'))
        abs_delta = float(cmp.get('abs_delta') or 0.0)
        rel_delta = abs_delta / abs(reference_f) if reference_f not in {0.0, -0.0} else None
        out['relative_abs_delta'] = rel_delta

        if prefer == 'lower':
            status = 'PASS' if current_f <= reference_f else 'FAIL'
            reason = 'lower is better'
        elif prefer == 'higher':
            status = 'PASS' if current_f >= reference_f else 'FAIL'
            reason = 'higher is better'
        elif prefer == 'equal':
            status = 'PASS' if abs_delta <= 0.0 else 'FAIL'
            reason = 'exact equality required'
        elif prefer == 'abs_le':
            lim = float(rule.get('limit_abs') or 0.0)
            status = 'PASS' if abs(current_f) <= lim else 'FAIL'
            reason = f'absolute value must be <= {lim}'
        else:
            reason = f'unknown rule {prefer}; falling back to equality'
            status = 'PASS' if abs_delta <= 0.0 else 'FAIL'

        if tol_abs is not None:
            try:
                if abs_delta <= float(tol_abs):
                    status = 'PASS'
                    reason = f'within absolute tolerance {float(tol_abs)}'
            except Exception:
                pass
        if tol_rel is not None and rel_delta is not None:
            try:
                if rel_delta <= float(tol_rel):
                    status = 'PASS'
                    reason = f'within relative tolerance {float(tol_rel)}'
            except Exception:
                pass
        if prefer in {'lower', 'higher'} and status == 'PASS' and tol_abs is None and tol_rel is None:
            reason = f'{prefer} is better than reference'

    out['status'] = status
    out['reason'] = reason
    return out
