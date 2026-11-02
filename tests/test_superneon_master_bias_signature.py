"""SuperNeon must use the supported FrameSignature API and reject CCD mismatch."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

fits = pytest.importorskip("astropy.io.fits")

from scorpio_pipe.frame_signature import FrameSignature
from scorpio_pipe.stages.calib import build_superbias
from scorpio_pipe.stages.superneon import build_superneon
from scorpio_pipe.workspace_paths import stage_dir
from tests.golden.mini_night import make_mini_night


def _mini_night_with_bias(tmp_path: Path):
    mn = make_mini_night(tmp_path)
    bias_path = build_superbias(
        mn.cfg,
        run_root=mn.work_dir,
        stage_dir=stage_dir(mn.work_dir, "superbias"),
        allow_legacy_mirrors=False,
    )
    assert bias_path.exists()
    return mn


def test_superneon_reads_master_bias_header_and_subtracts(tmp_path: Path) -> None:
    mn = _mini_night_with_bias(tmp_path)
    result = build_superneon(mn.cfg, run_root=mn.work_dir)

    assert result.superneon_fits.exists()
    with fits.open(result.superneon_fits, memmap=False) as hdul:
        assert bool(hdul[0].header["BIASSUB"]) is True
        assert int(hdul[0].header["NNEON"]) == len(mn.cfg["frames"]["neon"])

    qc_path = result.superneon_fits.parent / "superneon_qc.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    assert qc["bias_subtracted"] is True
    assert qc["signature"]["readout"] == FrameSignature.from_path(
        mn.cfg["frames"]["neon"][0]
    ).readout


def test_superneon_rejects_different_bias_and_arc_readout(tmp_path: Path) -> None:
    mn = _mini_night_with_bias(tmp_path)
    # Change BOTH arc headers: the arcs remain mutually compatible, but
    # now differ from the fixed synthetic NODE/RATE/GAIN of MasterBias.
    for neon in mn.cfg["frames"]["neon"]:
        with fits.open(neon, mode="update", memmap=False) as hdul:
            hdul[0].header["RATE"] = 225.0
            hdul[0].header["GAIN"] = 3.5
            hdul.flush()

    with pytest.raises(
        ValueError, match="SuperNeon requires superbias to match neon frames"
    ):
        build_superneon(mn.cfg, run_root=mn.work_dir)
