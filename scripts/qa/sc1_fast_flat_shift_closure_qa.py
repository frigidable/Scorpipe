#!/usr/bin/env python3
"""Read-only triangular-consistency diagnostic for SCORPIO-1 FLAT shift QA.

Input is the 144-row CSV made by sc1_fast_flat_shift_qa.py. Zero residual
is necessary for a single per-FLAT shift, but not sufficient to certify it.
The heuristic threshold is a HOLD flag, not a statistical significance test.
"""
from __future__ import annotations
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

A = 's22800614.fts'
B = 's22800615.fts'
C = 's22800616.fts'
PAIRS = ((A, B), (A, C), (B, C))


def calculate(rows):
    bykey = {}
    for row in rows:
        key = (row['flat_a'], row['flat_b'], int(row['spatial_stripe']), int(row['xwindow']))
        if key in bykey:
            raise ValueError(f'Duplicate: {key}')
        if (key[0], key[1]) not in PAIRS or key[2] not in range(12) or key[3] not in range(4):
            raise ValueError(f'Unexpected row key: {key}')
        eligibility = str(row['eligible']).strip().lower()
        if eligibility not in ('true', 'false'):
            raise ValueError(f'Invalid eligibility: {eligibility}')
        bykey[key] = row
    if len(bykey) != 144:
        raise ValueError(f'Expected 144 distinct trials, found {len(bykey)}')
    output = []
    bywindow = defaultdict(list)
    for band in range(12):
        for window in range(4):
            ar, cr, br = (bykey[p[0], p[1], band, window] for p in PAIRS)
            accepted = all(str(r['eligible']).strip().lower() == 'true' for r in (ar, cr, br))
            item = dict(spatial_stripe=band, xwindow=window, all_three_eligible=accepted,
                        delta_ab_px='', delta_ac_px='', delta_bc_px='', closure_ac_minus_ab_minus_bc_px='')
            if accepted:
                dab = float(ar['best_delta_px'])
                dac = float(cr['best_delta_px'])
                dbc = float(br['best_delta_px'])
                if not np.isfinite([dab, dac, dbc]).all():
                    raise ValueError('Non-finite accepted shift')
                error = dac - dab - dbc
                item.update(delta_ab_px=dab, delta_ac_px=dac, delta_bc_px=dbc,
                            closure_ac_minus_ab_minus_bc_px=error)
                bywindow[window].append(error)
            output.append(item)
    values = np.asarray([r['closure_ac_minus_ab_minus_bc_px']
                         for r in output if r['all_three_eligible']], dtype=float)
    window_stats = {}
    for k in range(4):
        v = np.asarray(bywindow[k], dtype=float)
        window_stats[str(k)] = {'n': int(len(v)),
                                'median_px': float(np.median(v)) if len(v) else None,
                                'median_abs_px': float(np.median(abs(v))) if len(v) else None,
                                'p90_abs_px': float(np.percentile(abs(v), 90)) if len(v) else None}
    enough = len(values) >= 12
    median_abs = float(np.median(abs(values))) if len(values) else None
    gate = ('HOLD_INSUFFICIENT_TRIANGLES' if not enough else
            'HOLD_INCONSISTENT_TRIANGLES' if median_abs > 0.10 else
            'HOLD_UNCERTAINTY_UNCALIBRATED')
    summary = {'status': 'DIAGNOSTIC_ONLY_HOLD', 'gate': gate, 'n_total_triangles': 48,
               'n_eligible_triangles': int(len(values)),
               'median_closure_px': float(np.median(values)) if len(values) else None,
               'median_absolute_closure_px': median_abs,
               'mad_closure_px': float(1.4826*np.median(abs(values-np.median(values)))) if len(values) else None,
               'p90_absolute_closure_px': float(np.percentile(abs(values),90)) if len(values) else None,
               'fraction_abs_closure_gt_0p1': float(np.mean(abs(values)>0.10)) if len(values) else None,
               'per_xwindow': window_stats,
               'threshold_px_heuristic_not_statistical': 0.10,
               'limitations': [
                   'Same-FLAT null controls and reverse-pair antisymmetry have not been tested.',
                   'Multiple windows share spectral and spatial systematic errors; do not treat as independent measurements.',
                   'Pair-specific shift uncertainties are not calibrated; closure mismatch alone does not identify the mechanism.',
                   'No geometric shift or fast-mode overscan-only calibration approved.',
               ]}
    return summary, output


def selftest():
    rows = []
    for band in range(12):
        for window in range(4):
            for (a,b),shift in zip(PAIRS,(0.2,0.15,-0.05)):
                rows.append(dict(flat_a=a,flat_b=b,spatial_stripe=band,xwindow=window,
                                 eligible=True,best_delta_px=shift))
    good, _ = calculate(rows)
    assert good['n_eligible_triangles'] == 48
    assert good['median_absolute_closure_px'] < 1e-10
    assert good['gate'] == 'HOLD_UNCERTAINTY_UNCALIBRATED'
    for row in rows:
        if (row['flat_a'],row['flat_b']) == (A,C):
            row['best_delta_px'] = 0.35
    bad,_ = calculate(rows)
    assert bad['gate'] == 'HOLD_INCONSISTENT_TRIANGLES'
    assert abs(bad['median_absolute_closure_px'] - 0.2) < 1e-12
    print('SELFTEST PASSED')


def run(csv_path):
    with csv_path.open(encoding='utf-8-sig',newline='') as f:
        summary, rows = calculate(list(csv.DictReader(f)))
    out_json = csv_path.with_name('sc1_fast_flat_shift_closure_summary.json')
    out_csv = csv_path.with_name('sc1_fast_flat_shift_closure_48.csv')
    out_json.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with out_csv.open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print('GATE:',summary['gate'])
    print('COMMON INFORMATIVE TRIANGLES:',summary['n_eligible_triangles'])
    print('MEDIAN ABSOLUTE CLOSURE [PIX]:',summary['median_absolute_closure_px'])
    print('JSON:',out_json)
    print('CSV:',out_csv)
    print('READ-ONLY INPUT; NO CALIBRATION APPROVED')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: script.py --selftest | sc1_fast_flat_shift_windows.csv')
    if sys.argv[1] == '--selftest':
        selftest()
    else:
        run(Path(sys.argv[1]))
