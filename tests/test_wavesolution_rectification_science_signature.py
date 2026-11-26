"""Wavelength maps apply to SCI detector geometry even when ARC gain/rate differ."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

fits = pytest.importorskip("astropy.io.fits")

from scorpio_pipe.frame_signature import FrameSignature
from scorpio_pipe.stages.superneon import build_superneon
from scorpio_pipe.stages.wavesolution import build_wavesolution
from tests.golden.mini_night import make_mini_night


def test_rectification_model_records_science_signature_separately_from_arc(
    tmp_path: Path,
) -> None:
    mn = make_mini_night(tmp_path)
    cfg = mn.cfg.copy()
    cfg["frames"] = {**mn.cfg["frames"]}
    cfg["frames"]["__setup__"] = {**mn.cfg["frames"]["__setup__"]}
    cfg["superneon"] = {**mn.cfg["superneon"], "bias_sub": False}

    # ARC readout RATE/GAIN legitimately differ from SCIENCE readout.
    # This is allowed for wavelength geometry but NOT for bias subtraction.
    arc_readout = "nodeE_rate225.0_gain3.5000"
    cfg["frames"]["__setup__"]["readout"] = arc_readout
    for name in cfg["frames"]["neon"]:
        with fits.open(name, mode="update", memmap=False) as hdul:
            hdul[0].header["RATE"] = 225.0
            hdul[0].header["GAIN"] = 3.5
            hdul.flush()

    result = build_superneon(cfg, run_root=mn.work_dir)
    assert result.superneon_fits.exists()
    with fits.open(result.superneon_fits, memmap=False) as hdul:
        assert not bool(hdul[0].header["BIASSUB"])

    wave = build_wavesolution(cfg, run_root=mn.work_dir)
    assert wave.lambda_map_fits.exists()

    # Regression: QC serialization previously referenced an undefined
    # excluded_summary; a broad except used to suppress the NameError,
    # incorrectly returning a successful StageResult without QC products.
    done = json.loads(
        (wave.wavesol_dir / "done.json").read_text(encoding="utf-8")
    )
    wave_done = json.loads(
        (wave.wavesol_dir / "wave_done.json").read_text(encoding="utf-8")
    )
    assert done["stage"] == "wavesolution"
    assert done["compat"]["excluded_summary"] is None
    assert done["ok"] is True
    assert wave_done["status"] in {"ok", "warn"}
    assert (wave.wavesol_dir / "wavesolution_science_qc.json").is_file()
    assert (wave.wavesol_dir / "trace_identity_qc.json").is_file()

    # The production 1D report has top-level RMS/pair-count fields. Guard
    # the schema consumed by Golden and the independent numerical QC report.
    ws1 = json.loads(wave.wavesolution_1d_json.read_text(encoding="utf-8"))
    assert "fit" not in ws1
    assert 0.0 <= float(ws1["rms_A"]) <= 0.05
    assert 5 <= int(ws1["n_used"]) <= int(ws1["n_pairs"])

    model = json.loads(
        (wave.wavesol_dir / "rectification_model.json").read_text(encoding="utf-8")
    )

    science_sig = FrameSignature.from_path(cfg["frames"]["obj"][0])
    arc_sig = FrameSignature.from_path(cfg["frames"]["neon"][0])

    assert model["frame_signature_origin"] == "science"
    assert model["frame_signature"] == science_sig.to_dict()
    assert model["arc_frame_signature"] == arc_sig.to_dict()
    assert science_sig.readout != arc_sig.readout
    assert science_sig.shape == arc_sig.shape
    assert FrameSignature.from_setup(model["frame_signature"]).is_compatible_with(
        science_sig
    )
