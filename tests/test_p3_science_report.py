from __future__ import annotations

from scorpio_pipe.science.report import build_science_report_payload, render_science_report_html, render_science_report_markdown


def test_science_report_renderers_include_status_and_lines() -> None:
    payload = build_science_report_payload(
        fluxcal_qc={'status': 'ABSOLUTE_TRUSTED', 'reference_context_id': 'ctx1'},
        lines=[{'name': 'Halpha', 'detected': True, 'flux': 12.3, 'equivalent_width': 4.2, 'velocity_kms': 15.0, 'err_stat': 0.3, 'err_cal': 0.7}],
        standard_qc={'safe_for': 'absolute_candidate', 'selected_standard': {'object_name': 'AGK+81d266', 'role': 'standard'}, 'association': {'status': 'ABSOLUTE_TRUSTED'}},
        sensfunc_qc={'fit_meta': {'model': 'spline'}},
        telluric_qc={'coverage_fraction': 0.2},
        spectrum_preview='fluxcal_preview.png',
        line_preview='science_lines_preview.png',
    )
    md = render_science_report_markdown(payload)
    html = render_science_report_html(payload)
    assert 'ABSOLUTE_TRUSTED' in md
    assert 'Halpha' in md
    assert 'Science report' in html
    assert 'AGK+81d266' in md
