from __future__ import annotations

"""Synthetic end-to-end smoke test (golden signature).

This is a compact, deterministic run that exercises the main scientific chain
in canonical order:
Sky subtraction → Linearization → 2D stack → 1D extraction.

It also builds:
- QC report (qc/qc_report.json)
- Static navigator (ui/navigator/data.json)

Then compares a stable, high-level signature against a committed golden file.

In CI (BL-P2-CI-010), the QC report and navigator data are copied to a
directory pointed to by the `SCORPIPE_CI_ARTIFACTS_DIR` environment variable.
"""

# ruff: noqa: E402

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy.io import fits  # noqa: E402

from scorpio_pipe.navigator import build_navigator
from scorpio_pipe.qc_report import build_qc_report
from scorpio_pipe.run_passport import ensure_run_passport
from scorpio_pipe.stages.extract1d import run_extract1d
from scorpio_pipe.stages.linearize import run_linearize
from scorpio_pipe.stages.sky_sub import run_sky_sub
from scorpio_pipe.stages.stack2d import run_stack2d
from scorpio_pipe.work_layout import ensure_work_layout


GOLDEN = Path(__file__).parent / "golden" / "smoke_signature.json"


def _write(path: Path, data: np.ndarray, *, header: fits.Header | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=data.astype("float32", copy=False), header=header).writeto(path, overwrite=True)


def _norm_sev(s: str) -> str:
    s = (s or "").strip().upper()
    # Treat INFO as OK for the smoke signature: this is a regression net for big breaks,
    # not for UX-level informational hints.
    if s in {"", "OK", "INFO"}:
        return "OK"
    return s


def _signature(qc_doc: dict, nav_doc: dict) -> dict:
    qc = qc_doc.get("qc") if isinstance(qc_doc, dict) else {}
    qc = qc if isinstance(qc, dict) else {}

    flags = qc.get("flags") if isinstance(qc.get("flags"), list) else []
    codes = sorted(
        {
            str(f.get("code") or "").strip().upper()
            for f in flags
            if isinstance(f, dict)
            and str(f.get("code") or "").strip()
            and str(f.get("severity") or "").strip().upper() in {"WARN", "ERROR"}
        }
    )

    nav_stages = nav_doc.get("stages") if isinstance(nav_doc, dict) else []
    st_status: dict[str, str] = {}
    if isinstance(nav_stages, list):
        for s in nav_stages:
            if not isinstance(s, dict):
                continue
            k = str(s.get("key") or "").strip()
            if k in {"wavesol", "sky", "linearize", "stack2d", "extract1d"}:
                st_status[k] = str(s.get("status") or "").strip()

    qc_summary = nav_doc.get("qc_summary") if isinstance(nav_doc.get("qc_summary"), dict) else {}

    def _i(x, default=0) -> int:
        try:
            return int(x)
        except Exception:
            return int(default)

    return {
        "qc": {
            "max_severity": _norm_sev(str(qc.get("max_severity") or "")),
            "flag_codes": codes,
        },
        "navigator": {
            "qc_summary": {
                "max_severity": _norm_sev(str(qc_summary.get("max_severity") or "")),
                "n_warn": _i(qc_summary.get("n_warn", 0)),
                "n_error": _i(qc_summary.get("n_error", 0)),
            },
            "stages_status": st_status,
        },
    }


@pytest.mark.smoke
def test_smoke_synthetic_chain_golden():
    rng = np.random.default_rng(7)

    with tempfile.TemporaryDirectory(prefix="scorpipe_pytest_smoke_") as td:
        work_dir = Path(td) / "work"
        layout = ensure_work_layout(work_dir)

        # Size chosen to avoid fragile "small sample" heuristics in sky fitting.
        ny, nx = 64, 128
        x = np.arange(nx, dtype=float)[None, :]
        lam = 5000.0 + 2.0 * x
        lambda_map = np.repeat(lam, ny, axis=0)

        # Provide minimal wavelength metadata to avoid QC_LAMBDA_MAP_META warnings.
        hdr = fits.Header()
        hdr["WAVEUNIT"] = "Angstrom"
        hdr["WAVEREF"] = "PIX"
        wavesol_dir = work_dir / "wavesol"
        wavesol_dir.mkdir(parents=True, exist_ok=True)
        _write(wavesol_dir / "lambda_map.fits", lambda_map, header=hdr)

        # Synthetic frame: smooth background + Poisson-like noise + a faint object bump.
        baseline = 100.0
        rn = 3.0
        sigma = float(np.sqrt(baseline + rn * rn))
        img = baseline + rng.normal(0.0, sigma, size=(ny, nx)).astype(float)
        # Add a weak "emission line" bump in the object rows.
        line = 25.0 * np.exp(-0.5 * ((np.arange(nx, dtype=float) - 64.0) / 6.0) ** 2)
        img[28:36, :] += line[None, :]

        raw = layout.raw / "obj_0001.fits"
        _write(raw, img)

        cfg = {
            "work_dir": str(work_dir),
            "frames": {"obj": [str(raw)]},
            "roi": {
                "obj_y1": 28,
                "obj_y2": 35,
                "sky_y1": 5,
                "sky_y2": 20,
                "sky2_y1": 44,
                "sky2_y2": 59,
                "units": "px",
            },
            "sky_sub": {
                "save_per_exp_sky_model": False,
                "kelson_raw": {
                    # Keep the flexure scan cheap and suppress the "low significance" warning
                    # for the expected near-zero shift in synthetic data.
                    "flexure_n_samples": 20000,
                    "delta_score_warn": 0.0,
                    "delta_uncertain_A": 1e9,
                },
            },
            "linearize": {
                "dlambda_A": "auto",
                "save_per_exposure": True,
                "save_preview": True,
            },
            "stack2d": {},
            "extract1d": {"mode": "boxcar", "aperture_half_width": 4},
        }

        # Canonical order.
        run_sky_sub(cfg)
        lin = run_linearize(cfg)

        from scorpio_pipe.workspace_paths import stage_dir

        lin_stage = stage_dir(work_dir, "linearize")
        rectified = sorted(lin_stage.rglob("*_rectified.fits"))
        assert rectified, "Expected per-exposure rectified FITS from linearize stage"

        st = run_stack2d(cfg, inputs=rectified)
        ex = run_extract1d(cfg, stacked_fits=st["stacked2d_fits"])

        assert Path(lin["preview_fits"]).exists()
        assert Path(st["stacked2d_fits"]).exists()
        assert Path(ex["spec1d_fits"]).exists()

        ensure_run_passport(work_dir, overwrite=True)

        qc_out = build_qc_report(cfg)
        qc_json = Path(qc_out["json"])
        assert qc_json.exists()

        nav_html = build_navigator(work_dir, overwrite=True)
        nav_data = nav_html.parent / "data.json"
        assert nav_data.exists()

        qc_doc = json.loads(qc_json.read_text(encoding="utf-8", errors="replace"))
        nav_doc = json.loads(nav_data.read_text(encoding="utf-8", errors="replace"))

        sig = _signature(qc_doc, nav_doc)

        assert GOLDEN.exists(), f"Golden file missing: {GOLDEN}"
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        assert sig == golden

        # CI artifacts (optional): copy into user-specified dir.
        art = os.environ.get("SCORPIPE_CI_ARTIFACTS_DIR")
        if art:
            out_dir = Path(art).expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(qc_json, out_dir / "qc_report.json")
            nav_out = out_dir / "navigator"
            nav_out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(nav_data, nav_out / "data.json")
