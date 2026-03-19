from __future__ import annotations

import json
from pathlib import Path

from scorpio_pipe.perf.baselines import normalize_baseline_tag, resolve_baseline_path
from scorpio_pipe.version import __version__


def test_perf_75_1_current_version_baselines_exist() -> None:
    p_micro = resolve_baseline_path("microbench", version="current")
    p_golden = resolve_baseline_path("golden_time_budget", version="current")
    assert p_micro.name == f"{__version__}.json"
    assert p_golden.name == f"{__version__}.json"
    assert p_micro.exists()
    assert p_golden.exists()


def test_perf_75_1_unknown_version_falls_back_to_main() -> None:
    p = resolve_baseline_path("microbench", version="999.999.999", fallback_to_main=True)
    assert p.name == "main.json"


def test_perf_75_1_legacy_compat_baselines_still_well_formed() -> None:
    p1 = Path(__file__).resolve().parents[1] / "benchmarks" / "microbench_baseline.json"
    p2 = Path(__file__).resolve().parents[1] / "tests" / "golden" / "mini_night_time_budget.json"
    j1 = json.loads(p1.read_text(encoding="utf-8"))
    j2 = json.loads(p2.read_text(encoding="utf-8"))
    assert j1["baseline_version"] == __version__
    assert j2["baseline_version"] == __version__
    assert j1["family"] == "microbench"
    assert j2["family"] == "golden_time_budget"
