from __future__ import annotations

"""Golden-nights E2E regression test (P0-ARCH-06.2).

This test runs the *engine* end-to-end on a tiny, deterministic synthetic "night"
and compares a few key QC numbers within tolerances.

DoD:
  - at least one E2E run exists
  - it executes in CI (pytest)
  - it fails PR on architectural/scientific regressions
"""

# ruff: noqa: E402

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy.io import fits  # noqa: E402

from scorpio_pipe.config import load_config_any
from scorpio_pipe.pipeline.engine import run_sequence
from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG

from tests.golden.mini_night import make_mini_night


GOLDEN = Path(__file__).parent / "golden" / "mini_night_expected.json"


def _get_tol(tol: dict, key: str) -> tuple[float, float]:
    t = tol.get(key) if isinstance(tol.get(key), dict) else {}
    abs_tol = float(t.get("abs", 0.0) or 0.0)
    rel_tol = float(t.get("rel", 0.0) or 0.0)
    return abs_tol, rel_tol


def _assert_close(name: str, got: float, ref: float, *, abs_tol: float, rel_tol: float) -> None:
    if not (np.isfinite(got) and np.isfinite(ref)):
        raise AssertionError(f"{name}: non-finite value(s): got={got!r}, ref={ref!r}")
    lim = abs_tol + rel_tol * abs(ref)
    if abs(got - ref) > lim:
        raise AssertionError(
            f"{name}: got {got:.6g}, expected {ref:.6g} (abs_tol={abs_tol}, rel_tol={rel_tol})"
        )


def _assert_metric(name: str, got: float, expected: object, tol: dict) -> None:
    """Support both golden ranges (min/max) and point+tol comparisons."""

    if isinstance(expected, dict) and ("min" in expected or "max" in expected):
        lo = float(expected.get("min", float("-inf")))
        hi = float(expected.get("max", float("inf")))
        if not (np.isfinite(got) and np.isfinite(lo) and np.isfinite(hi)):
            raise AssertionError(f"{name}: non-finite range/value: got={got!r}, range=({lo!r},{hi!r})")
        if not (lo <= got <= hi):
            raise AssertionError(f"{name}: got {got:.6g}, expected in [{lo:.6g}, {hi:.6g}]")
        return

    # point expectation + tolerances
    ref = float(expected or 0.0)
    abs_tol, rel_tol = _get_tol(tol, name)
    _assert_close(name, got, ref, abs_tol=abs_tol, rel_tol=rel_tol)


@pytest.mark.golden
def test_golden_mini_night_e2e(tmp_path: Path):
    mn = make_mini_night(tmp_path)

    tasks = (
        "manifest",
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
    )

    # Full run (no resume) to ensure the test exercises the current code.
    _ = run_sequence(
        mn.cfg_path,
        tasks,
        resume=False,
        force=True,
        qc_override=False,
        preflight_override=False,
        config_path=mn.cfg_path,
    )

    cfg = load_config_any(mn.cfg_path)

    # --- read golden expectations ---
    assert GOLDEN.exists(), f"Golden file missing: {GOLDEN}"
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    exp = golden.get("metrics") if isinstance(golden.get("metrics"), dict) else {}
    tol = golden.get("tolerances") if isinstance(golden.get("tolerances"), dict) else {}

    # --- Wavesol RMS (Å) ---
    ws1 = PRODUCT_CATALOG.best_path(cfg, "wavesolution_1d_json", run_root=mn.work_dir)
    assert ws1 is not None and Path(ws1).exists()
    ws1_doc = json.loads(Path(ws1).read_text(encoding="utf-8"))
    rms1d = float((ws1_doc.get("fit") or {}).get("rms_A"))

    # --- Sky robust sigma (MAD*1.4826) ---
    sky_done = PRODUCT_CATALOG.best_path(cfg, "sky_done", run_root=mn.work_dir)
    assert sky_done is not None and Path(sky_done).exists()
    sky_doc = json.loads(Path(sky_done).read_text(encoding="utf-8"))
    sky_sigma = float(((sky_doc.get("metrics") or {}).get("robust_sigma_r_after_median")) or 0.0)

    # --- Mask fraction in linearize preview ---
    lin_prev = PRODUCT_CATALOG.best_path(cfg, "lin_preview_fits", run_root=mn.work_dir)
    assert lin_prev is not None and Path(lin_prev).exists()
    with fits.open(lin_prev, memmap=False) as hdul:
        m = np.asarray(hdul["MASK"].data, dtype=np.uint16) if "MASK" in hdul else None
    assert m is not None
    mask_nonzero_frac = float(np.count_nonzero(m != 0) / (m.size if m.size else 1.0))

    # --- compare with tolerances ---
    _assert_metric("wavesol_1d_rms_A", rms1d, exp.get("wavesol_1d_rms_A"), tol)
    _assert_metric(
        "sky_robust_sigma_r_after_median",
        sky_sigma,
        exp.get("sky_robust_sigma_r_after_median"),
        tol,
    )
    _assert_metric(
        "linearize_mask_nonzero_frac",
        mask_nonzero_frac,
        exp.get("linearize_mask_nonzero_frac"),
        tol,
    )

    # CI artifacts (optional): copy into user-specified dir.
    art = os.environ.get("SCORPIPE_CI_ARTIFACTS_DIR")
    if art:
        out_dir = Path(art).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        def _copy(src: str | Path | None, rel: str) -> None:
            if not src:
                return
            p = Path(src)
            if not p.exists():
                return
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)

        # Core UI outputs
        _copy(PRODUCT_CATALOG.best_path(cfg, "qc_json", run_root=mn.work_dir), "qc/qc_report.json")

        nav_dir = PRODUCT_CATALOG.best_path(cfg, "navigator_dir", run_root=mn.work_dir)
        if nav_dir and Path(nav_dir).exists():
            # Keep it small: copy only data.json (enough for debugging a regression).
            data_json = Path(nav_dir) / "data.json"
            _copy(data_json, "ui/navigator/data.json")

        # Key products that drive golden metrics
        _copy(ws1, "products/wavesolution_1d.json")
        _copy(sky_done, "products/sky_done.json")
        _copy(lin_prev, "products/lin_preview.fits")

        # Reproducibility + perf budgets: manifest + lock/snapshot + timings/metrics.
        _copy(PRODUCT_CATALOG.best_path(cfg, "manifest", run_root=mn.work_dir), "manifest/manifest.json")
        _copy(PRODUCT_CATALOG.best_path(cfg, "dependency_lock", run_root=mn.work_dir), "manifest/requirements.lock")
        _copy(PRODUCT_CATALOG.best_path(cfg, "deps_snapshot", run_root=mn.work_dir), "manifest/deps_snapshot.json")

        timings_jsonl = mn.work_dir / "manifest" / "timings.jsonl"
        if timings_jsonl.is_file():
            _copy(timings_jsonl, "manifest/timings.jsonl")

        metrics_json = mn.work_dir / "manifest" / "metrics.json"
        if metrics_json.is_file():
            _copy(metrics_json, "manifest/metrics.json")

        try:
            from scorpio_pipe.benchmark.long_compare import build_scorpipe_observer_bundle, write_observer_bundle

            bundle = build_scorpipe_observer_bundle(cfg, run_root=mn.work_dir)
            write_observer_bundle(out_dir / "long_parity" / "scorpipe_observer_bundle.json", bundle)
        except Exception:
            pass
