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


def _read_wavesol_1d_rms_A(doc: dict) -> float:
    """Read the production wavesolution_1d.json top-level RMS (Angstroms).

    A nested fit.rms_A is not part of the emitted schema. Missing, invalid,
    nonfinite or negative calibration quality must fail Golden explicitly.
    """
    if not isinstance(doc, dict) or "rms_A" not in doc:
        raise AssertionError("wavesolution_1d.json missing top-level rms_A")
    try:
        rms = float(doc["rms_A"])
        n_used = int(doc["n_used"])
        n_pairs = int(doc["n_pairs"])
    except (TypeError, ValueError, KeyError) as exc:
        raise AssertionError(
            "wavesolution_1d.json has invalid rms_A/n_used/n_pairs"
        ) from exc
    if not np.isfinite(rms) or rms < 0:
        raise AssertionError(f"wavesolution_1d.json invalid rms_A={rms!r}")
    if not (5 <= n_used <= n_pairs):
        raise AssertionError(
            f"wavesolution_1d.json insufficient fitted pairs: "
            f"n_used={n_used}, n_pairs={n_pairs}"
        )
    return rms


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

    # The full Golden run must pass the real Stack2D ETA gate. In particular,
    # a failure to create eta_lambda.fits must never be silently classified as
    # "applied", and an uncalibrated noise model must block extraction.
    stack_report_path = mn.work_dir / "11_stack" / "stack_done.json"
    assert stack_report_path.exists(), "Stack2D report missing"
    stack_report = json.loads(stack_report_path.read_text(encoding="utf-8"))
    stack_eta = stack_report.get("eta") or {}
    assert stack_eta.get("applied") is True, stack_eta
    assert not stack_eta.get("error"), stack_eta
    eta_path = mn.work_dir / "11_stack" / "eta_lambda.fits"
    stack_path = mn.work_dir / "11_stack" / "stack2d.fits"
    assert eta_path.exists() and stack_path.exists()
    with fits.open(eta_path, memmap=False) as eta_hdul, fits.open(
        stack_path, memmap=False
    ) as stack_hdul:
        eta_values = np.asarray(eta_hdul[0].data, dtype=float)
        stack_sci = np.asarray(stack_hdul["SCI"].data)
        assert eta_values.ndim == 1 and eta_values.size == stack_sci.shape[1]
        assert np.all(np.isfinite(eta_values) & (eta_values > 0))
        assert bool(stack_hdul[0].header["ETAAPPL"]) is True
        assert stack_hdul[0].header["ETAPATH"] == eta_path.name
        assert "VAR" in stack_hdul and "MASK" in stack_hdul
    sigma_after = float(stack_eta["sigma_after_eta"])
    assert np.isfinite(sigma_after)
    assert abs(sigma_after - 1.0) <= 0.5, (
        f"Normalized sky-noise sigma after eta = {sigma_after:.3f}, "
        "which violates the unchanged scientific Stack2D QC ERROR threshold"
    )

    # Preview is a visualization-only SCI image. Its physical unit must match
    # a real sky-subtracted SCI product, not the lambda_map wavelength unit.
    sky_preview = PRODUCT_CATALOG.best_path(cfg, "sky_preview_fits", run_root=mn.work_dir)
    sky_products = PRODUCT_CATALOG.find_existing_paths(
        cfg, "sky_skysub_raw_fits", run_root=mn.work_dir
    )
    assert sky_preview.exists() and sky_products
    with fits.open(sky_preview, memmap=False) as preview_hdul, fits.open(
        sky_products[0], memmap=False
    ) as science_hdul:
        assert preview_hdul[0].header["SCORPPRV"] is True
        assert "SCI" in preview_hdul and "VAR" not in preview_hdul
        preview_unit = str(preview_hdul["SCI"].header["BUNIT"])
        science_unit = str(science_hdul["SCI"].header["BUNIT"])
        assert preview_unit == science_unit
        assert "VAR" in science_hdul and "MASK" in science_hdul
        # Sky must preserve the explicit source exposure duration; Linearize
        # uses this for electrons/s and electron^2/s^2, never an assumed 1 s.
        assert float(science_hdul[0].header["EXPTIME"]) == 300.0

    # --- read golden expectations ---
    assert GOLDEN.exists(), f"Golden file missing: {GOLDEN}"
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    exp = golden.get("metrics") if isinstance(golden.get("metrics"), dict) else {}
    tol = golden.get("tolerances") if isinstance(golden.get("tolerances"), dict) else {}

    # --- Wavesol RMS (Å) ---
    ws1 = PRODUCT_CATALOG.best_path(cfg, "wavesolution_1d_json", run_root=mn.work_dir)
    assert ws1 is not None and Path(ws1).exists()
    ws1_doc = json.loads(Path(ws1).read_text(encoding="utf-8"))
    rms1d = _read_wavesol_1d_rms_A(ws1_doc)

    # --- Sky robust sigma (MAD*1.4826) ---
    sky_done = PRODUCT_CATALOG.best_path(cfg, "sky_done", run_root=mn.work_dir)
    assert sky_done is not None and Path(sky_done).exists()
    sky_doc = json.loads(Path(sky_done).read_text(encoding="utf-8"))
    # A sky spline fitted to its own training pixels cannot establish
    # absolute flexure. Never transmit an apparently measured zero shift.
    flex = sky_doc.get("flexure") or {}
    # Sky-model FITS cannot inherit detector VAR, because a fitted model
    # has a different covariance. Require honest, incomplete provenance.
    sky_model_path = mn.work_dir / "09_sky" / "obj_01_skymodel_raw.fits"
    with fits.open(sky_model_path, memmap=False) as model_hdul:
        assert "SCI" in model_hdul and "VAR" not in model_hdul
        assert model_hdul[0].header["SKYVQUAL"] == "INCOMP"
        assert model_hdul[0].header["SKYVCOV"] == "UNKNOWN"
    with fits.open(sky_products[0], memmap=False) as sky_science:
        assert "VAR" in sky_science
        assert sky_science[0].header["SKYVSRC"] == "DET_ONLY"
    assert any(f.get("code") == "SKY_MODEL_VAR_UNVERIFIED"
               for f in (sky_doc.get("qc") or {}).get("flags", []))
    assert flex.get("verification") == "unverified"
    assert flex.get("delta_A_median") is None
    assert flex.get("reference_id") is None
    assert any(f.get("code") == "FLEXURE_UNVERIFIED"
               for f in (sky_doc.get("qc") or {}).get("flags", []))
    sky_sigma = float(((sky_doc.get("metrics") or {}).get("robust_sigma_r_after_median")) or 0.0)

    # --- Mask fraction in linearize preview ---
    lin_prev = PRODUCT_CATALOG.best_path(cfg, "lin_preview_fits", run_root=mn.work_dir)
    assert lin_prev is not None and Path(lin_prev).exists()
    with fits.open(lin_prev, memmap=False) as hdul:
        m = np.asarray(hdul["MASK"].data, dtype=np.uint16) if "MASK" in hdul else None
        assert bool(hdul[0].header["NORMEXP"]) is True
        assert float(hdul[0].header["GAIN"]) > 0
        assert float(hdul[0].header["RDNOISE"]) >= 0
        assert str(hdul[0].header["NOISRC"]).strip()
        assert int(hdul[0].header["NOINEXP"]) == len(mn.cfg["frames"]["obj"])
        assert hdul[0].header["NOIDET"] == "linearize_done.json"
        assert str(hdul[0].header["WAVEREF"]).lower() == "air"
        assert str(hdul["SCI"].header["WAVEREF"]).lower() == "air"
        assert str(hdul[0].header["CTYPE1"]).upper() == "AWAV"
    assert m is not None

    # Check a per-exposure SCIENCE output as well as the stacked preview;
    # both must use the wavelength medium declared by lambda_map.
    for idx in (1, 2):
        rect = mn.work_dir / "10_linearize" / f"obj_{idx:02d}_skysub.fits"
        assert rect.exists()
        with fits.open(rect, memmap=False) as rect_hdul:
            assert rect_hdul[0].header["WAVEREF"] == "air"
            assert rect_hdul["SCI"].header["WAVEREF"] == "air"
            assert rect_hdul[0].header["CTYPE1"] == "AWAV"
            assert rect_hdul[0].header["FLXFLAG"] == "UNVERIFIED"
            assert rect_hdul[0].header["DLAMAP"] is False
            assert "DLAM_A" not in rect_hdul[0].header
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
