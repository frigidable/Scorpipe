from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def _status_color(status: str) -> str:
    s = str(status or '').upper()
    if 'TRUSTED' in s:
        return '#1b7f3b'
    if 'APPROX' in s or 'TELLURIC' in s:
        return '#b36b00'
    if 'RELATIVE' in s:
        return '#7a4a00'
    return '#8b1e1e'


def build_science_report_payload(*, fluxcal_qc: dict[str, Any] | None, lines: list[dict[str, Any]] | None, standard_qc: dict[str, Any] | None, sensfunc_qc: dict[str, Any] | None, telluric_qc: dict[str, Any] | None, spectrum_preview: str | None, line_preview: str | None) -> dict[str, Any]:
    fluxcal_qc = dict(fluxcal_qc or {})
    lines = list(lines or [])
    detected = [r for r in lines if bool(r.get('detected'))]
    detected.sort(key=lambda r: abs(float(r.get('flux') or 0.0)), reverse=True)
    warnings: list[str] = []
    for src in (fluxcal_qc, standard_qc or {}, sensfunc_qc or {}, telluric_qc or {}):
        for fl in src.get('flags', []) or []:
            msg = str((fl or {}).get('message') or (fl or {}).get('code') or '').strip()
            if msg and msg not in warnings:
                warnings.append(msg)
    return {
        'calibration_status': str(fluxcal_qc.get('status') or fluxcal_qc.get('calibration_status') or 'UNKNOWN'),
        'calibration_status_2d': str(fluxcal_qc.get('status_2d') or ''),
        'reference_context_id': str(fluxcal_qc.get('reference_context_id') or ''),
        'safe_for': str((standard_qc or {}).get('safe_for') or ''),
        'detected_lines': detected,
        'warnings': warnings,
        'provenance': {
            'standard': {
                'selected_standard': (standard_qc or {}).get('selected_standard'),
                'association': (standard_qc or {}).get('association'),
            },
            'sensfunc': sensfunc_qc or {},
            'telluric': telluric_qc or {},
            'fluxcal': fluxcal_qc,
        },
        'previews': {
            'spectrum_preview': spectrum_preview,
            'line_preview': line_preview,
        },
    }


def render_science_report_markdown(payload: dict[str, Any]) -> str:
    status = str(payload.get('calibration_status') or 'UNKNOWN')
    safe_for = str(payload.get('safe_for') or '')
    lines = list(payload.get('detected_lines') or [])
    warnings = list(payload.get('warnings') or [])
    prev = payload.get('previews') or {}
    prov = payload.get('provenance') or {}
    out = []
    out.append('# Science report')
    out.append('')
    out.append(f'**Calibration status:** `{status}`  ')
    if safe_for:
        out.append(f'**Standard safety class:** `{safe_for}`  ')
    refctid = str(payload.get('reference_context_id') or '')
    if refctid:
        out.append(f'**Reference context:** `{refctid}`')
    out.append('')
    if prev.get('spectrum_preview'):
        out.append(f'![Calibrated spectrum]({Path(str(prev["spectrum_preview"])).name})')
        out.append('')
    if prev.get('line_preview'):
        out.append(f'![Measured lines]({Path(str(prev["line_preview"])).name})')
        out.append('')
    out.append('Calibration and statistical uncertainties are kept separate throughout this report.\n')
    out.append('## Key measured lines')
    out.append('')
    out.append('| Line | Flux | EW | Velocity (km/s) | err_stat | err_cal | err_total |')
    out.append('|---|---:|---:|---:|---:|---:|---:|')
    for row in lines[:12]:
        out.append(
            f"| {row.get('name','')} | {float(row.get('flux') or 0.0):.4g} | {float(row.get('equivalent_width') or 0.0):.4g} | {float(row.get('velocity_kms') or 0.0):.2f} | {float(row.get('err_stat') or 0.0):.4g} | {float(row.get('err_cal') or 0.0):.4g} | {float(row.get('err_total') or 0.0):.4g} |"
        )
    if not lines:
        out.append('| – | – | – | – | – | – | – |')
    out.append('')
    out.append('## Warnings')
    out.append('')
    if warnings:
        out.extend(f'- {w}' for w in warnings)
    else:
        out.append('- No degraded-confidence flags reported.')
    out.append('')
    out.append('## Provenance summary')
    out.append('')
    std = (prov.get('standard') or {})
    if std.get('selected_standard'):
        ss = std.get('selected_standard') or {}
        out.append(f"- Standard: **{ss.get('object_name') or ss.get('path') or 'unknown'}** ({ss.get('role') or 'standard'})")
    assoc = std.get('association') or {}
    if assoc.get('status'):
        out.append(f"- Standard association: `{assoc.get('status')}`")
    sf = prov.get('sensfunc') or {}
    if sf.get('fit_meta'):
        out.append(f"- Sensfunc model: {((sf.get('fit_meta') or {}).get('model') or 'smooth model')}")
    tel = prov.get('telluric') or {}
    if tel.get('coverage_fraction') is not None:
        out.append(f"- Telluric coverage fraction: {float(tel.get('coverage_fraction') or 0.0):.3f}")
    return '\n'.join(out).rstrip() + '\n'


def render_science_report_html(payload: dict[str, Any]) -> str:
    md = render_science_report_markdown(payload)
    status = str(payload.get('calibration_status') or 'UNKNOWN')
    color = _status_color(status)
    lines = list(payload.get('detected_lines') or [])
    warnings = list(payload.get('warnings') or [])
    prev = payload.get('previews') or {}
    rows = ''.join(
        f"<tr><td>{html.escape(str(r.get('name','')))}</td><td>{float(r.get('flux') or 0.0):.4g}</td><td>{float(r.get('equivalent_width') or 0.0):.4g}</td><td>{float(r.get('velocity_kms') or 0.0):.2f}</td><td>{float(r.get('err_stat') or 0.0):.4g}</td><td>{float(r.get('err_cal') or 0.0):.4g}</td></tr>"
        for r in lines[:12]
    ) or '<tr><td colspan="6">No detected lines</td></tr>'
    warn_html = ''.join(f'<li>{html.escape(str(w))}</li>' for w in warnings) or '<li>No degraded-confidence flags reported.</li>'
    imgs = []
    if prev.get('spectrum_preview'):
        imgs.append(f'<div><h3>Calibrated spectrum</h3><img src="{html.escape(Path(str(prev["spectrum_preview"])).name)}" style="max-width:100%"></div>')
    if prev.get('line_preview'):
        imgs.append(f'<div><h3>Measured lines</h3><img src="{html.escape(Path(str(prev["line_preview"])).name)}" style="max-width:100%"></div>')
    return f'''<!doctype html>
<html><head><meta charset="utf-8"><title>Science report</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;line-height:1.45}}
.card{{border:1px solid #ddd;border-radius:12px;padding:16px;margin-bottom:18px}}
.status{{font-weight:700;color:{color}}}
table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #ddd;padding:6px 8px;text-align:left}}
pre{{white-space:pre-wrap;background:#f7f7f7;padding:12px;border-radius:8px}}
</style></head><body>
<h1>Science report</h1>
<div class="card"><div>Calibration status: <span class="status">{html.escape(status)}</span></div></div>
{''.join(imgs)}
<div class="card"><h2>Key measured lines</h2><table><thead><tr><th>Line</th><th>Flux</th><th>EW</th><th>Velocity (km/s)</th><th>err_stat</th><th>err_cal</th></tr></thead><tbody>{rows}</tbody></table></div>
<div class="card"><h2>Warnings</h2><ul>{warn_html}</ul></div>
<div class="card"><h2>Markdown view</h2><pre>{html.escape(md)}</pre></div>
</body></html>'''
