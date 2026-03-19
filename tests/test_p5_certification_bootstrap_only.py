import json
from pathlib import Path

from scorpio_pipe.benchmark.certification import build_release_certification


def test_bootstrap_only_entries_do_not_count_or_fail(tmp_path: Path):
    reg = tmp_path / 'registry.json'
    case_manifest = tmp_path / 'bootstrap_case.case.json'
    case_manifest.write_text(json.dumps({
        'case_id': 'repeatability_multi_exposure',
        'title': 'Repeatability candidate',
        'category': 'repeatability',
        'source_type': 'bootstrap_from_nightlog',
        'description': 'draft only',
        'observer_notes': '',
        'metrics': {},
        'scorpipe_bundle': None,
        'long_bundle': None,
    }), encoding='utf-8')
    reg.write_text(json.dumps({
        'schema': 'scorpio_pipe.long_parity_registry',
        'schema_version': 1,
        'minimum_real_cases_required': 1,
        'entries': [
            {'case_id': 'mini_night_synthetic', 'status': 'fixture', 'manifest': None, 'category': 'ci-fixture'},
            {'case_id': 'repeatability_multi_exposure', 'status': 'bootstrap-only', 'manifest': str(case_manifest), 'category': 'repeatability'},
        ],
    }), encoding='utf-8')
    payload = build_release_certification(registry_path=reg, reports_root=tmp_path / 'reports')
    s = payload['summary']
    assert s['populated_real_cases'] == 0
    assert s['failed_real_cases'] == 0
    assert s['status'] == 'INCOMPLETE'
