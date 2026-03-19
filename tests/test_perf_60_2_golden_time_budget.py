from __future__ import annotations

import json
from pathlib import Path

from scorpio_pipe.perf.golden_time_budget import (
    compare_stage_times_to_budget,
    load_stage_seconds,
)


EXPECTED = {
    "superbias",
    "superflat",
    "superneon",
    "wavesolution",
    "sky",
    "linearize",
    "stack2d",
    "extract1d",
    "qc_report",
    "navigator",
}


def test_perf_60_2_load_stage_seconds_from_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "timings.jsonl"
    p.write_text(
        "\n".join(
            [
                '{"stage":"sky","seconds":1.25,"ok":true}',
                '{"stage":"linearize","seconds":0.75,"ok":true}',
                '{"stage":"sky","seconds":0.50,"ok":true}',
                '{"stage":"","seconds":99.0}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    got = load_stage_seconds(timings_path=p)
    assert got == {"sky": 1.75, "linearize": 0.75}


def test_perf_60_2_compare_detects_2x_class_regression() -> None:
    current = {
        "wavesolution": 4.5,
        "sky": 1.6,
        "linearize": 0.8,
    }
    baseline = {
        "schema": 1,
        "default_tolerance_pct": 80.0,
        "default_abs_slack_s": 0.2,
        "default_min_enforced_seconds": 0.2,
        "stages": {
            "wavesolution": {"baseline_s": 2.0},
            "sky": {"baseline_s": 1.0},
            "linearize": {"baseline_s": 0.7},
        },
    }
    cmp = compare_stage_times_to_budget(current, baseline)
    assert cmp["ok"] is False
    assert cmp["regressions"] == ["wavesolution"]
    assert cmp["comparisons"]["sky"]["status"] == "ok"


def test_perf_60_2_baseline_json_is_well_formed() -> None:
    p = Path(__file__).resolve().parents[1] / "tests" / "golden" / "mini_night_time_budget.json"
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert int(payload.get("schema", 0)) == 1
    stages = dict(payload.get("stages") or {})
    assert EXPECTED.issubset(set(stages))
    assert float(payload.get("default_tolerance_pct", 0.0)) > 0.0
    for name in EXPECTED:
        item = dict(stages[name] or {})
        assert float(item.get("baseline_s", 0.0)) > 0.0
