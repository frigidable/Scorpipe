from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scorpio_pipe.perf.baselines import resolve_baseline_path


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _iter_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_stage_seconds(
    *,
    timings_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> dict[str, float]:
    """Load per-stage elapsed seconds from timings.jsonl/json (preferred) or metrics.json.

    Strategy:
      1) If timings_path is given and exists, sum JSONL/JSON records by stage.
      2) Otherwise, fall back to metrics.json stage metrics containing a numeric
         ``seconds`` / ``duration_s`` field.
    """

    out: dict[str, float] = {}

    if timings_path is not None:
        p = Path(timings_path)
        if p.exists():
            if p.suffix.lower() == ".jsonl":
                rows = _iter_jsonl(p)
            else:
                payload = _load_json(p)
                rows = payload if isinstance(payload, list) else []
            for row in rows:
                stage = str(row.get("stage") or "").strip()
                if not stage:
                    continue
                try:
                    sec = float(row.get("seconds", 0.0) or 0.0)
                except Exception:
                    continue
                if sec < 0:
                    continue
                out[stage] = round(float(out.get(stage, 0.0)) + sec, 6)
            if out:
                return out

    if metrics_path is not None:
        p = Path(metrics_path)
        if p.exists():
            payload = _load_json(p)
            stages = dict(payload.get("stages") or {})
            for stage, item in stages.items():
                if not isinstance(item, Mapping):
                    continue
                metrics = dict(item.get("metrics") or {})
                sec_raw = metrics.get("seconds", metrics.get("duration_s"))
                try:
                    sec = float(sec_raw or 0.0)
                except Exception:
                    continue
                if sec < 0:
                    continue
                out[str(stage)] = sec
    return out


def compare_stage_times_to_budget(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    cur = {str(k): float(v) for k, v in dict(current).items()}
    stages = dict(baseline.get("stages") or {})
    default_tol = float(baseline.get("default_tolerance_pct", 80.0) or 0.0)
    default_abs_slack = float(baseline.get("default_abs_slack_s", 0.0) or 0.0)
    default_min = float(baseline.get("default_min_enforced_seconds", 0.0) or 0.0)

    comparisons: dict[str, Any] = {}
    regressions: list[str] = []
    missing_in_current: list[str] = []

    for stage, spec0 in stages.items():
        spec = dict(spec0 or {})
        if stage not in cur:
            missing_in_current.append(stage)
            comparisons[stage] = {
                "status": "missing",
                "current_seconds": None,
            }
            continue

        cur_s = float(cur[stage])
        baseline_s = spec.get("baseline_s")
        max_seconds = spec.get("max_seconds")
        tol = float(spec.get("tolerance_pct", default_tol) or 0.0)
        abs_slack = float(spec.get("abs_slack_s", default_abs_slack) or 0.0)
        min_enforced = float(spec.get("min_enforced_seconds", default_min) or 0.0)

        if max_seconds is not None:
            allowed = float(max_seconds)
            base = float(spec.get("baseline_s", allowed / max(1.0, 1.0 + tol / 100.0)) or 0.0)
        else:
            base = float(baseline_s or 0.0)
            allowed = max(base * (1.0 + tol / 100.0), base + abs_slack)

        delta_pct = ((cur_s - base) / base * 100.0) if base > 0 else 0.0
        enforce = bool(base >= min_enforced or cur_s >= min_enforced)
        is_reg = bool(enforce and cur_s > allowed)
        if is_reg:
            regressions.append(stage)

        comparisons[stage] = {
            "baseline_seconds": round(base, 6),
            "current_seconds": round(cur_s, 6),
            "allowed_seconds": round(float(allowed), 6),
            "delta_pct": round(float(delta_pct), 3),
            "tolerance_pct": round(float(tol), 3),
            "abs_slack_s": round(float(abs_slack), 6),
            "min_enforced_seconds": round(float(min_enforced), 6),
            "status": "regression" if is_reg else "ok",
        }

    return {
        "schema": 1,
        "generated_at": _utc_now(),
        "comparisons": comparisons,
        "regressions": regressions,
        "missing_in_current": sorted(missing_in_current),
        "ok": not regressions and not missing_in_current,
    }


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compare golden-night per-stage timings against a soft budget.")
    ap.add_argument("--timings", default="golden_artifacts/manifest/timings.jsonl", help="Path to timings.jsonl (preferred).")
    ap.add_argument("--metrics", default="golden_artifacts/manifest/metrics.json", help="Optional metrics.json fallback.")
    ap.add_argument("--baseline", default=None, help="Explicit budget baseline JSON path.")
    ap.add_argument("--baseline-version", default=None, help="Resolve a versioned baseline from benchmarks/baselines/golden_time_budget (e.g. main, current, 7.42.11).")
    ap.add_argument("--baseline-root", default=None, help="Optional repository root override for versioned baselines.")
    ap.add_argument("--output", default="golden_artifacts/perf/golden_time_budget_compare.json", help="Where to write comparison JSON.")
    ap.add_argument("--fail-on-regression", action="store_true", help="Exit non-zero on timing regression or missing stage.")
    ap.add_argument("--soft-fail", action="store_true", help="Never exit non-zero on regression; useful for advisory PR checks.")
    return ap


def main(argv: list[str] | None = None) -> int:
    ns = _build_argparser().parse_args(argv)
    current = load_stage_seconds(timings_path=ns.timings, metrics_path=ns.metrics)
    baseline_path = ns.baseline
    if baseline_path is None and ns.baseline_version:
        baseline_path = str(resolve_baseline_path("golden_time_budget", version=ns.baseline_version, root=ns.baseline_root))
    if not baseline_path:
        raise SystemExit("Need either --baseline or --baseline-version")
    baseline = _load_json(baseline_path)
    compare = compare_stage_times_to_budget(current, baseline)
    compare["baseline_path"] = str(baseline_path)
    out = _write_json(ns.output, compare)
    print(f"[golden-time-budget] baseline -> {baseline_path}")
    print(f"[golden-time-budget] wrote comparison -> {out}")
    for stage, item in sorted((compare.get("comparisons") or {}).items()):
        if item.get("status") == "missing":
            print(f"  - {stage}: missing in current timings")
            continue
        print(
            f"  - {stage}: {item['current_seconds']:.3f} s vs baseline {item['baseline_seconds']:.3f} s "
            f"(allow {item['allowed_seconds']:.3f} s, Δ {item['delta_pct']:+.1f}%) => {item['status']}"
        )
    if ns.fail_on_regression and not ns.soft_fail and not bool(compare.get("ok")):
        reg = list(compare.get("regressions") or [])
        missing = list(compare.get("missing_in_current") or [])
        parts = []
        if reg:
            parts.append("regressions=" + ",".join(reg))
        if missing:
            parts.append("missing=" + ",".join(missing))
        print("[golden-time-budget] failing: " + "; ".join(parts))
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
