import json
from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from scorpio_pipe.frame_signature import FrameSignature
from scorpio_pipe.qc_report import build_qc_report
from scorpio_pipe.work_layout import ensure_work_layout
from scorpio_pipe.workspace_paths import stage_dir
from scorpio_pipe.wavesol_paths import wavesol_dir


def test_qc_report_collects_basic_metrics(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    layout = ensure_work_layout(work_dir)

    # Minimal calibration done markers
    sig = FrameSignature(ny=16, nx=32, bx=1, by=2, window="", readout="RATE=185.0")
    sb_dir = stage_dir(work_dir, "superbias")
    sb_dir.mkdir(parents=True, exist_ok=True)
    (sb_dir / "superbias_done.json").write_text(
        json.dumps({"status": "ok", "frame_signature": sig.to_dict(), "n_inputs": 3}, indent=2),
        encoding="utf-8",
    )
    sf_dir = stage_dir(work_dir, "superflat")
    sf_dir.mkdir(parents=True, exist_ok=True)
    (sf_dir / "superflat_done.json").write_text(
        json.dumps({"status": "ok", "frame_signature": sig.to_dict(), "n_inputs": 5}, indent=2),
        encoding="utf-8",
    )

    cfg = {
        "work_dir": str(work_dir),
        "wavesol": {"disperser": "VPHG1200@540"},
        "frames": {"__setup__": {"binning": "1x2"}},
    }

    wdir = wavesol_dir(cfg)
    wdir.mkdir(parents=True, exist_ok=True)

    # wavesolution JSON products
    (wdir / "wavesolution_1d.json").write_text(
        json.dumps({"deg": 3, "rms_A": 0.18, "n_pairs": 25, "n_used": 22}, indent=2),
        encoding="utf-8",
    )
    (wdir / "wavesolution_2d.json").write_text(
        json.dumps({"kind": "chebyshev", "rms_A": 0.22, "n_points": 120, "n_used": 118}, indent=2),
        encoding="utf-8",
    )
    (wdir / "residuals_2d.csv").write_text(
        "lam,x,y,resid\n5000,10,5,0.10\n5000,12,5,-0.20\n",
        encoding="utf-8",
    )

    # lambda_map FITS (contract header)
    lam = 5000.0 + 2.0 * np.arange(32)[None, :]
    lambda_map = np.repeat(lam, 16, axis=0).astype("f4")
    hdr = fits.Header()
    hdr["CTYPE1"] = "WAVE"
    hdr["WAVEUNIT"] = "Angstrom"
    hdr["WAVEREF"] = "air"
    fits.writeto(wdir / "lambda_map.fits", lambda_map, hdr, overwrite=True)

    # linearize QC
    qc_dir = work_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    (qc_dir / "linearize_qc.json").write_text(
        json.dumps(
            {
                "coverage": {"nonzero_frac": 0.98},
                "stacking": {"method": "sigma_clip", "rejected_fraction": 0.05},
                "exptime_policy": {"normalize_exptime": True},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # manifest (signature consistency)
    (qc_dir / "manifest.json").write_text(
        json.dumps(
            {
                "frames": {
                    "flat": {
                        "n": 5,
                        "frame_signature": sig.to_dict(),
                        "signature_consistent": True,
                        "signature_mismatches": [],
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    out = build_qc_report(cfg)
    assert out["json"].exists()
    payload = json.loads(out["json"].read_text(encoding="utf-8"))

    metrics = payload.get("metrics")
    assert isinstance(metrics, dict)
    assert metrics.get("wavesol_1d", {}).get("rms_A") == 0.18
    assert metrics.get("wavesol_contract", {}).get("unit_ok") is True
    assert metrics.get("linearize", {}).get("coverage", {}).get("nonzero_frac") == 0.98
    assert out["html"].exists()
