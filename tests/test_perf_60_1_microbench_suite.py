from __future__ import annotations

import json
from pathlib import Path

from scorpio_pipe.perf.microbench import compare_results_to_baseline, run_microbenchmarks


EXPECTED = {
    "linearize_core",
    "sky_fit_core",
    "cosmics_core",
    "stack_accumulator",
}


def test_perf_60_1_microbench_suite_emits_expected_benchmarks() -> None:
    payload = run_microbenchmarks(names=sorted(EXPECTED))
    assert int(payload.get("schema", 0)) == 1
    benches = dict(payload.get("benchmarks") or {})
    assert set(benches) == EXPECTED
    for name, item in benches.items():
        assert float(item.get("median_ms", 0.0)) > 0.0, name
        assert float(item.get("tolerance_pct", 0.0)) > 0.0, name
        assert int(item.get("repeat", 0)) >= 1, name
        assert int(item.get("loops", 0)) >= 1, name
        assert isinstance(item.get("meta"), dict), name


def test_perf_60_1_compare_detects_regression() -> None:
    current = {
        "schema": 1,
        "benchmarks": {
            "linearize_core": {"median_ms": 110.0, "tolerance_pct": 10.0},
            "sky_fit_core": {"median_ms": 90.0, "tolerance_pct": 20.0},
        },
    }
    baseline = {
        "schema": 1,
        "benchmarks": {
            "linearize_core": {"median_ms": 90.0, "tolerance_pct": 10.0},
            "sky_fit_core": {"median_ms": 100.0, "tolerance_pct": 20.0},
        },
    }
    cmp = compare_results_to_baseline(current, baseline)
    assert cmp["ok"] is False
    assert cmp["regressions"] == ["linearize_core"]
    assert cmp["comparisons"]["sky_fit_core"]["status"] == "ok"


def test_perf_60_1_baseline_json_is_well_formed() -> None:
    p = Path(__file__).resolve().parents[1] / "benchmarks" / "microbench_baseline.json"
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert int(payload.get("schema", 0)) == 1
    benches = dict(payload.get("benchmarks") or {})
    assert EXPECTED.issubset(set(benches))
    for name in EXPECTED:
        item = benches[name]
        assert float(item.get("median_ms", 0.0)) > 0.0
        assert float(item.get("tolerance_pct", 0.0)) > 0.0
