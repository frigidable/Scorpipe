from __future__ import annotations

from typing import Any


def calibration_status_card_html(summary: Any) -> str:
    status = str(getattr(summary, 'status', '') or getattr(summary, 'calibration_status', '') or 'UNKNOWN')
    sev = str(getattr(summary, 'severity', '') or getattr(summary, 'max_severity', '') or 'OK')
    return f'<div class="card"><strong>Calibration status:</strong> {status}<br><strong>Severity:</strong> {sev}</div>'


def science_report_card_html(summary: Any) -> str:
    title = str(getattr(summary, 'title', '') or 'Science report')
    n = getattr(summary, 'n_detected_lines', None)
    extra = f'<br><strong>Detected lines:</strong> {n}' if n is not None else ''
    return f'<div class="card"><strong>{title}</strong>{extra}</div>'


def observer_dashboard_html(cards: list[str]) -> str:
    body = '\n'.join(str(c) for c in cards)
    return f'<!doctype html><html><head><meta charset="utf-8"><style>.card{{border:1px solid #ddd;border-radius:10px;padding:12px;margin:8px 0}}</style></head><body>{body}</body></html>'



def standard_prepare_card_html(summary: Any) -> str:
    if summary is None or getattr(summary, 'selected', None) is None:
        return '<div class="card"><strong>Standard prepare:</strong> not available</div>'
    sel = summary.selected
    sev = html_escape(str(getattr(summary, 'max_severity', 'OK') or 'OK'))
    return f'<div class="card"><strong>Standard:</strong> {html_escape(str(getattr(sel, "object_name", "") or getattr(sel, "path", "")))}<br><strong>Role:</strong> {html_escape(str(getattr(sel, "role", "standard")))}<br><strong>Association:</strong> {html_escape(str(getattr(sel, "association_status", "")))}<br><strong>Safe for:</strong> {html_escape(str(getattr(sel, "safe_for", "")))}<br><strong>Severity:</strong> {sev}</div>'


def sensfunc_card_html(summary: Any) -> str:
    if summary is None:
        return '<div class="card"><strong>Sensfunc:</strong> not available</div>'
    extra = ''
    if getattr(summary, 'median_residual', None) is not None:
        extra += f'<br><strong>MAD residual:</strong> {float(getattr(summary, "median_residual")):0.4g}'
    if getattr(summary, 'airmass_provenance', None):
        extra += f'<br><strong>Airmass provenance:</strong> {html_escape(str(getattr(summary, "airmass_provenance")))}'
    if getattr(summary, 'extinction_applied', None) is not None:
        extra += f'<br><strong>Extinction applied:</strong> {bool(getattr(summary, "extinction_applied"))}'
    return f'<div class="card"><strong>Sensfunc waveref:</strong> {html_escape(str(getattr(summary, "waveref", "")))}{extra}</div>'


def telluric_card_html(summary: Any) -> str:
    if summary is None:
        return '<div class="card"><strong>Telluric:</strong> not available</div>'
    return f'<div class="card"><strong>Telluric coverage:</strong> {float(getattr(summary, "coverage_fraction", 0.0) or 0.0):0.3f}<br><strong>Median confidence:</strong> {float(getattr(summary, "median_confidence", 0.0) or 0.0):0.3f}</div>'


def html_escape(s: str) -> str:
    import html
    return html.escape(str(s or ''))
