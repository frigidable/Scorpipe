from __future__ import annotations

import argparse
import json
import math
import os
import socket
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from scorpio_pipe.perf.baselines import resolve_baseline_path
from typing import Any, Callable

import numpy as np


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class BenchSpec:
    name: str
    title: str
    factory: Callable[[], tuple[Callable[[], Any], dict[str, Any]]]
    warmup: int = 1
    repeat: int = 7
    target_ms: float = 150.0
    max_loops: int = 16
    tolerance_pct: float = 40.0


@dataclass(frozen=True)
class BenchResult:
    name: str
    title: str
    median_ms: float
    mean_ms: float
    stdev_ms: float
    loops: int
    repeat: int
    warmup: int
    samples_ms: list[float]
    tolerance_pct: float
    meta: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "median_ms": self.median_ms,
            "mean_ms": self.mean_ms,
            "stdev_ms": self.stdev_ms,
            "loops": self.loops,
            "repeat": self.repeat,
            "warmup": self.warmup,
            "samples_ms": self.samples_ms,
            "tolerance_pct": self.tolerance_pct,
            "meta": self.meta,
        }


def _calibrate_loops(fn: Callable[[], Any], *, target_ms: float, max_loops: int) -> int:
    t0 = time.perf_counter()
    fn()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        return 1
    loops = int(math.ceil(float(target_ms) / float(dt_ms)))
    return max(1, min(int(max_loops), loops))


# -------------------------
# Hot-kernel bench factories
# -------------------------

def _factory_linearize_core() -> tuple[Callable[[], Any], dict[str, Any]]:
    from scorpio_pipe.stages.linearize import (
        _apply_linearize_row_plan,
        _apply_linearize_row_plan_var,
        _get_linearize_grid_plan,
        clear_linearize_grid_cache,
    )

    rng = np.random.default_rng(12041)
    ny, nx = 96, 1536
    nlam = 1700
    wave0 = 4500.0
    dw = 0.82
    x = np.arange(nx, dtype=np.float64)
    lam_base = wave0 + 0.68 * x + 0.00018 * (x - nx / 2.0) ** 2
    y = np.arange(ny, dtype=np.float64)[:, None]
    lam_map = (lam_base[None, :] + (y - ny / 2.0) * 0.017).astype(np.float64)
    sci = rng.normal(1000.0, 15.0, size=(ny, nx)).astype(np.float32)
    var = rng.uniform(2.0, 8.0, size=(ny, nx)).astype(np.float32)
    valid = rng.random((ny, nx)) > 0.02
    edges_out = wave0 + dw * np.arange(nlam + 1, dtype=np.float64)

    clear_linearize_grid_cache()
    plan, _hit = _get_linearize_grid_plan(
        lam_map,
        edges_out=edges_out,
        frame_signature_key="bench:linearize",
        lambda_map_key="bench:linearize:hash",
        wave0=wave0,
        dw=dw,
        nlam=nlam,
    )
    # Warm cache-hit path too.
    _ = _get_linearize_grid_plan(
        lam_map,
        edges_out=edges_out,
        frame_signature_key="bench:linearize",
        lambda_map_key="bench:linearize:hash",
        wave0=wave0,
        dw=dw,
        nlam=nlam,
    )

    def _run() -> tuple[float, float]:
        total_flux = 0.0
        total_var = 0.0
        for yy, row_plan in enumerate(plan.rows):
            if row_plan is None:
                continue
            out_sci, _cov = _apply_linearize_row_plan(sci[yy], row_plan, valid_mask=valid[yy])
            out_var = _apply_linearize_row_plan_var(var[yy], row_plan, valid_mask=valid[yy])
            total_flux += float(np.nansum(out_sci))
            total_var += float(np.nansum(out_var))
        return total_flux, total_var

    meta = {
        "shape": [ny, nx],
        "nlam": nlam,
        "grid_cached": True,
        "valid_fraction": float(valid.mean()),
    }
    return _run, meta


def _factory_sky_fit_core() -> tuple[Callable[[], Any], dict[str, Any]]:
    from scorpio_pipe.stages.sky_sub import _fit_kelson_linear_spline

    rng = np.random.default_rng(8801)
    n = 28000
    lam = rng.uniform(5000.0, 7600.0, size=n).astype(np.float64)
    y_norm = rng.uniform(-1.0, 1.0, size=n).astype(np.float64)
    sky_true = 1200.0 + 110.0 * np.sin(lam / 43.0) + 22.0 * y_norm * np.cos(lam / 61.0)
    var = rng.uniform(4.0, 25.0, size=n).astype(np.float64)
    d = (sky_true + rng.normal(0.0, np.sqrt(var), size=n)).astype(np.float64)

    def _run() -> tuple[int, float]:
        fit = _fit_kelson_linear_spline(
            lam,
            y_norm,
            d,
            var,
            knot_step_A=8.5,
            y_order=1,
            clip_sigma=4.5,
            max_iter=3,
            min_pixels=4000,
            rng=np.random.default_rng(12345),
        )
        coeffs = np.asarray((fit.get("coeff") or {}).get("c0"), dtype=float)
        return int(coeffs.size), float(np.nanmean(coeffs))

    meta = {
        "n_pixels": n,
        "knot_step_A": 8.5,
        "y_order": 1,
        "max_iter": 3,
    }
    return _run, meta


def _factory_cosmics_core() -> tuple[Callable[[], Any], dict[str, Any]]:
    from scorpio_pipe.stages.cosmics import _boxcar_mean2d, _dilate_mask, _laplacian4, _robust_sigma_mad_1d

    rng = np.random.default_rng(991)
    ny, nx = 1024, 1024
    img = rng.normal(1000.0, 6.0, size=(ny, nx)).astype(np.float32)
    # Inject bright spikes/mini-trails.
    yy = rng.integers(2, ny - 2, size=2500)
    xx = rng.integers(2, nx - 2, size=2500)
    img[yy, xx] += rng.uniform(80.0, 450.0, size=yy.size).astype(np.float32)
    img[yy, np.clip(xx + 1, 0, nx - 1)] += rng.uniform(10.0, 70.0, size=yy.size).astype(np.float32)

    def _run() -> tuple[int, float]:
        lap = _laplacian4(img)
        sigma = float(_robust_sigma_mad_1d(lap))
        loc = _boxcar_mean2d(np.abs(lap).astype(np.float32), r=2)
        cand = np.abs(lap) > np.maximum(5.0 * sigma, 4.0 * loc + 2.5 * sigma)
        dil = _dilate_mask(cand, r=1)
        return int(np.count_nonzero(dil)), float(np.mean(dil))

    meta = {
        "shape": [ny, nx],
        "n_spikes": int(yy.size),
        "dilate_r": 1,
        "local_r": 2,
    }
    return _run, meta


def _factory_stack_accumulator() -> tuple[Callable[[], Any], dict[str, Any]]:
    from scorpio_pipe.stages.stack2d import _WeightedMeanAccumulator

    rng = np.random.default_rng(4040)
    nexp = 14
    shape = (192, 2048)
    values = [rng.normal(0.0, 1.0, size=shape).astype(np.float32) for _ in range(nexp)]
    weights = [rng.uniform(0.2, 1.0, size=shape).astype(np.float32) for _ in range(nexp)]
    # A few rejected pixels.
    for w in weights:
        w[rng.random(shape) < 0.04] = 0.0

    def _run() -> tuple[float, float]:
        acc = _WeightedMeanAccumulator(shape)
        for arr, w in zip(values, weights):
            acc.update(arr, w)
        mean, var = acc.finalize()
        return float(np.nanmean(mean)), float(np.nanmean(var))

    meta = {
        "shape": list(shape),
        "n_exp": nexp,
    }
    return _run, meta


BENCHMARKS: tuple[BenchSpec, ...] = (
    BenchSpec(
        name="linearize_core",
        title="Linearize apply core (cached grid)",
        factory=_factory_linearize_core,
        repeat=7,
        warmup=1,
        target_ms=200.0,
        max_loops=8,
        tolerance_pct=35.0,
    ),
    BenchSpec(
        name="sky_fit_core",
        title="Sky fit core (Kelson linear spline)",
        factory=_factory_sky_fit_core,
        repeat=5,
        warmup=1,
        target_ms=250.0,
        max_loops=4,
        tolerance_pct=45.0,
    ),
    BenchSpec(
        name="cosmics_core",
        title="Cosmics detection core",
        factory=_factory_cosmics_core,
        repeat=7,
        warmup=1,
        target_ms=180.0,
        max_loops=8,
        tolerance_pct=35.0,
    ),
    BenchSpec(
        name="stack_accumulator",
        title="Stack2D streaming accumulator",
        factory=_factory_stack_accumulator,
        repeat=7,
        warmup=1,
        target_ms=180.0,
        max_loops=8,
        tolerance_pct=35.0,
    ),
)


def run_one_benchmark(spec: BenchSpec) -> BenchResult:
    fn, meta = spec.factory()
    loops = _calibrate_loops(fn, target_ms=float(spec.target_ms), max_loops=int(spec.max_loops))
    for _ in range(int(max(0, spec.warmup))):
        for _ in range(loops):
            fn()

    samples_ms: list[float] = []
    for _ in range(int(max(1, spec.repeat))):
        t0 = time.perf_counter()
        last: Any = None
        for _ in range(loops):
            last = fn()
        dt_ms = (time.perf_counter() - t0) * 1000.0 / float(loops)
        if last is not None and isinstance(meta, dict):
            meta.setdefault("sanity", repr(last)[:120])
        samples_ms.append(round(float(dt_ms), 6))

    mean_ms = float(statistics.fmean(samples_ms))
    median_ms = float(statistics.median(samples_ms))
    stdev_ms = float(statistics.pstdev(samples_ms)) if len(samples_ms) > 1 else 0.0
    return BenchResult(
        name=spec.name,
        title=spec.title,
        median_ms=round(median_ms, 6),
        mean_ms=round(mean_ms, 6),
        stdev_ms=round(stdev_ms, 6),
        loops=int(loops),
        repeat=int(spec.repeat),
        warmup=int(spec.warmup),
        samples_ms=samples_ms,
        tolerance_pct=float(spec.tolerance_pct),
        meta=dict(meta),
    )


def run_microbenchmarks(*, names: list[str] | None = None) -> dict[str, Any]:
    selected = list(BENCHMARKS)
    if names:
        wanted = {str(n).strip() for n in names if str(n).strip()}
        selected = [spec for spec in BENCHMARKS if spec.name in wanted]
        missing = sorted(wanted - {spec.name for spec in selected})
        if missing:
            raise KeyError(f"Unknown benchmark(s): {', '.join(missing)}")

    results = [run_one_benchmark(spec) for spec in selected]
    payload = {
        "schema": 1,
        "generated_at": _utc_now(),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "pid": os.getpid(),
        "benchmarks": {r.name: r.to_json() for r in results},
    }
    return payload


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_results_to_baseline(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    cur_b = dict(current.get("benchmarks") or {})
    base_b = dict(baseline.get("benchmarks") or {})
    missing_in_current = sorted(set(base_b) - set(cur_b))
    missing_in_baseline = sorted(set(cur_b) - set(base_b))
    comparisons: dict[str, Any] = {}
    regressions: list[str] = []

    for name in sorted(set(cur_b) & set(base_b)):
        cur = dict(cur_b.get(name) or {})
        base = dict(base_b.get(name) or {})
        cur_ms = float(cur.get("median_ms", 0.0) or 0.0)
        base_ms = float(base.get("median_ms", 0.0) or 0.0)
        tolerance_pct = float(cur.get("tolerance_pct", base.get("tolerance_pct", 0.0)) or 0.0)
        allowed_ms = base_ms * (1.0 + tolerance_pct / 100.0)
        delta_pct = ((cur_ms - base_ms) / base_ms * 100.0) if base_ms > 0 else 0.0
        is_regression = bool(base_ms > 0 and cur_ms > allowed_ms)
        if is_regression:
            regressions.append(name)
        comparisons[name] = {
            "baseline_ms": round(base_ms, 6),
            "current_ms": round(cur_ms, 6),
            "delta_pct": round(float(delta_pct), 3),
            "tolerance_pct": round(tolerance_pct, 3),
            "allowed_ms": round(float(allowed_ms), 6),
            "status": "regression" if is_regression else "ok",
        }

    return {
        "schema": 1,
        "generated_at": _utc_now(),
        "missing_in_current": missing_in_current,
        "missing_in_baseline": missing_in_baseline,
        "comparisons": comparisons,
        "regressions": regressions,
        "ok": not regressions and not missing_in_current,
    }


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run Scorpipe microbench suite on hot kernels.")
    ap.add_argument("--output", default="perf_artifacts/microbench_results.json", help="Where to write benchmark results JSON.")
    ap.add_argument("--baseline", default=None, help="Optional explicit baseline JSON path to compare against.")
    ap.add_argument("--baseline-version", default=None, help="Resolve a versioned baseline from benchmarks/baselines/microbench (e.g. main, current, 7.42.11).")
    ap.add_argument("--baseline-root", default=None, help="Optional repository root override for versioned baselines.")
    ap.add_argument("--compare-output", default="perf_artifacts/microbench_compare.json", help="Where to write comparison JSON.")
    ap.add_argument("--fail-on-regression", action="store_true", help="Exit non-zero if current results regress beyond tolerance.")
    ap.add_argument("--soft-fail", action="store_true", help="Never exit non-zero on regression; useful for advisory PR checks.")
    ap.add_argument("--bench", action="append", default=None, help="Run only selected benchmark(s). Can be passed multiple times.")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _build_argparser()
    ns = ap.parse_args(argv)
    current = run_microbenchmarks(names=list(ns.bench) if ns.bench else None)
    out_p = _write_json(ns.output, current)
    print(f"[microbench] wrote results -> {out_p}")

    baseline_path = ns.baseline
    if baseline_path is None and ns.baseline_version:
        baseline_path = str(resolve_baseline_path("microbench", version=ns.baseline_version, root=ns.baseline_root))

    if baseline_path:
        baseline = _load_json(baseline_path)
        compare = compare_results_to_baseline(current, baseline)
        compare["baseline_path"] = str(baseline_path)
        cmp_p = _write_json(ns.compare_output, compare)
        print(f"[microbench] baseline -> {baseline_path}")
        print(f"[microbench] wrote comparison -> {cmp_p}")
        reg = list(compare.get("regressions") or [])
        for name, item in sorted((compare.get("comparisons") or {}).items()):
            print(
                f"  - {name}: {item['current_ms']:.3f} ms vs baseline {item['baseline_ms']:.3f} ms "
                f"(Δ {item['delta_pct']:+.1f}%, tol {item['tolerance_pct']:.1f}%) => {item['status']}"
            )
        if reg and ns.fail_on_regression and not ns.soft_fail:
            print("[microbench] regressions detected:", ", ".join(reg), file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
