"""Read-only regression contracts for Scorpipe runtime provenance inventory."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from astropy.io import fits

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qa" / "sc1_run_artifacts_inventory_qa.py"


def module():
    spec = importlib.util.spec_from_file_location("sc1_inventory", SCRIPT)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_inventory_selftest():
    module().selftest()


def test_runtime_fits_header_and_stage_marker_without_modification(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    config = root / "config.yaml"
    config.write_text(
        "work_dir: .\nflatfield:\n  enabled: false\n"
        "calib:\n  bias_policy: strict\nsuperneon:\n  bias_sub: false\n",
        encoding="utf-8",
    )
    stage = root / "03_superbias"
    stage.mkdir()
    marker = stage / "superbias_done.json"
    marker.write_text('{"status":"ok","n_outputs":1}', encoding="utf-8")
    product = stage / "master_bias.fits"
    h = fits.PrimaryHDU(np.full((12, 20), 999, dtype=np.float32))
    h.header["RATE"] = 145
    h.header["GAIN"] = 0.9
    h.header["BIASSUB"] = False
    h.writeto(product)
    before = {p: p.read_bytes() for p in (config, marker, product)}
    result = module().inspect(config)
    assert result["stage_count"] == 1
    assert result["stages"][0]["markers"][0]["status"] == "ok"
    assert result["stages"][0]["fits_products"][0]["header"]["RATE"] == "145"
    assert result["stages"][0]["fits_products"][0]["header"]["GAIN"] == "0.9"
    assert result["stages"][0]["fits_products"][0]["shape"] == [12, 20]
    assert result["bias_policy"] == "strict"
    assert result["superneon_bias_sub"] is False
    for p, data in before.items():
        assert p.read_bytes() == data
