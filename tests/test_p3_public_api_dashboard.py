from __future__ import annotations

import json
from pathlib import Path


def test_build_observer_dashboard_collects_ui_safe_summaries(tmp_path: Path) -> None:
    from scorpio_pipe.pipeline.public_api import build_observer_dashboard
    from scorpio_pipe.workspace_paths import stage_dir

    cfg = {'work_dir': str(tmp_path), 'frames': {}}
    s_flux = stage_dir(tmp_path, 'fluxcal_apply')
    s_flux.mkdir(parents=True, exist_ok=True)
    (s_flux / 'fluxcal_qc.json').write_text(json.dumps({'status': 'RELATIVE_TELLURIC', 'reference_context_id': 'ctx2', 'flags': [{'message': 'telluric degraded'}]}), encoding='utf-8')
    (s_flux / 'fluxcal_preview.png').write_bytes(b'')
    s_rep = stage_dir(tmp_path, 'science_report')
    s_rep.mkdir(parents=True, exist_ok=True)
    (s_rep / 'science_report.md').write_text('# Report\n', encoding='utf-8')
    (s_rep / 'science_report.html').write_text('<html></html>', encoding='utf-8')
    s_lines = stage_dir(tmp_path, 'line_measure')
    s_lines.mkdir(parents=True, exist_ok=True)
    (s_lines / 'science_lines.json').write_text(json.dumps({'lines': [{'name': 'Halpha', 'detected': True}], 'warnings': []}), encoding='utf-8')
    (s_lines / 'science_lines_preview.png').write_bytes(b'')

    dash = build_observer_dashboard(cfg)
    assert dash['fluxcal'] is not None
    assert dash['fluxcal'].status == 'RELATIVE_TELLURIC'
    assert dash['science_report'] is not None
    assert dash['science_report'].n_detected_lines == 1
