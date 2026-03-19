from __future__ import annotations

import json
from pathlib import Path

from scorpio_pipe.perf import golden_time_budget, microbench


def test_perf_75_1_microbench_soft_fail_writes_compare(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(microbench, "run_microbenchmarks", lambda names=None: {
        "schema": 1,
        "benchmarks": {
            "linearize_core": {"median_ms": 100.0, "tolerance_pct": 1.0},
        },
    })
    out = tmp_path / "cur.json"
    cmp = tmp_path / "cmp.json"
    rc = microbench.main([
        "--output", str(out),
        "--compare-output", str(cmp),
        "--baseline-version", "main",
        "--soft-fail",
    ])
    assert rc == 0
    payload = json.loads(cmp.read_text(encoding="utf-8"))
    assert payload["baseline_path"].endswith("main.json")


def test_perf_75_1_golden_cli_resolves_versioned_baseline(tmp_path: Path) -> None:
    timings = tmp_path / "timings.jsonl"
    timings.write_text('{\"stage\":\"sky\",\"seconds\":1.0}\n', encoding="utf-8")
    out = tmp_path / "cmp.json"
    rc = golden_time_budget.main([
        "--timings", str(timings),
        "--output", str(out),
        "--baseline-version", "main",
        "--soft-fail",
    ])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["baseline_path"].endswith("main.json")
