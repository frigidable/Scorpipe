from __future__ import annotations

import json
from pathlib import Path

from scorpio_pipe.benchmark.long_compare import load_case_manifest, load_long_bundle, run_case_comparison, write_case_report


def test_long_parity_registry_has_required_slots() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = repo / 'benchmarks' / 'long_parity' / 'cases' / 'registry.json'
    data = json.loads(p.read_text(encoding='utf-8'))
    entries = list(data.get('entries') or [])
    assert data.get('minimum_real_cases_required') == 8
    assert len(entries) >= 10
    assert any(str(e.get('status')) == 'fixture' for e in entries)


def test_long_parity_fixture_case_writes_reports(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    case = load_case_manifest(repo / 'benchmarks' / 'long_parity' / 'cases' / 'mini_night_synthetic.case.json')
    sp = load_long_bundle(repo / 'benchmarks' / 'long_parity' / 'fixtures' / 'mini_night_scorpipe_bundle.json')
    lg = load_long_bundle(repo / 'benchmarks' / 'long_parity' / 'fixtures' / 'mini_night_long_bundle.json')
    payload = run_case_comparison(case, scorpipe_bundle=sp, long_bundle=lg)
    assert payload['summary']['status'] == 'PASS'
    out = write_case_report(tmp_path, payload)
    assert out['json'].exists()
    assert out['markdown'].exists()
    md = out['markdown'].read_text(encoding='utf-8')
    assert 'Synthetic mini-night harness sanity check' in md
    assert 'Wave RMS [A]' in md
