from __future__ import annotations

import importlib
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np

from scorpio_pipe.lamp_contract import resolve_linelist_csv
from scorpio_pipe.wavesol_paths import wavesol_dir
from scorpio_pipe.wavesol_presets import load_curated_line_preset


class _FakeHeader(dict):
    def copy(self):
        return _FakeHeader(super().copy())


class _FakePrimaryHDU:
    def __init__(self, data=None, header=None):
        self.data = np.asarray(data) if data is not None else None
        self.header = _FakeHeader(header or {})

    def writeto(self, path, overwrite=True, **kwargs):
        _fake_writeto(path, self.data, header=self.header, overwrite=overwrite, **kwargs)


class _FakeHDUList(list):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _fake_writeto(path, data, header=None, overwrite=True, **kwargs):
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(p)
    with p.open("wb") as f:
        pickle.dump({"data": np.asarray(data), "header": dict(header or {})}, f)


def _fake_getdata(path, memmap=False):
    with Path(path).open("rb") as f:
        return pickle.load(f)["data"]


def _fake_open(path, memmap=False):
    with Path(path).open("rb") as f:
        payload = pickle.load(f)
    return _FakeHDUList([types.SimpleNamespace(data=payload["data"], header=_FakeHeader(payload["header"]))])


def _install_fake_astropy(monkeypatch) -> None:
    fakefits = types.ModuleType("astropy.io.fits")
    fakefits.Header = _FakeHeader
    fakefits.PrimaryHDU = _FakePrimaryHDU
    fakefits.writeto = _fake_writeto
    fakefits.getdata = _fake_getdata
    fakefits.open = _fake_open

    io_mod = types.ModuleType("astropy.io")
    io_mod.fits = fakefits

    astropy_mod = types.ModuleType("astropy")
    astropy_mod.io = io_mod

    monkeypatch.setitem(sys.modules, "astropy", astropy_mod)
    monkeypatch.setitem(sys.modules, "astropy.io", io_mod)
    monkeypatch.setitem(sys.modules, "astropy.io.fits", fakefits)

    for name in [
        "scorpio_pipe.io.atomic_fits",
        "scorpio_pipe.provenance",
        "scorpio_pipe.stages.geometry_prior",
        "scorpio_pipe.stages.slit_sanity",
    ]:
        sys.modules.pop(name, None)


def test_curated_linelist_preset_is_loaded_and_materialized(tmp_path: Path) -> None:
    cfg = {
        "work_dir": str(tmp_path),
        "project_root": str(tmp_path),
        "data_dir": str(tmp_path),
        "wavesol": {"disperser": "VPHG1200@540"},
    }
    preset = load_curated_line_preset(cfg, work_dir=tmp_path, project_root=tmp_path)
    assert preset is not None
    assert preset.slug == "VPHG1200_540"
    assert preset.n_lines > 5
    assert preset.n_safe_auto > 0

    p = Path(resolve_linelist_csv(cfg, lamp_type="HeNeAr"))
    assert p.exists()
    txt = p.read_text(encoding="utf-8")
    assert "# disperser: VPHG1200@540" in txt
    meta = json.loads((p.parent / "line_priors_used.json").read_text(encoding="utf-8"))
    assert meta["slug"] == "VPHG1200_540"
    assert meta["n_lines"] >= preset.n_lines


def test_geometry_prior_stage_builds_model_from_curved_superneon(tmp_path: Path, monkeypatch) -> None:
    _install_fake_astropy(monkeypatch)
    gp = importlib.import_module("scorpio_pipe.stages.geometry_prior")

    cfg = {
        "work_dir": str(tmp_path),
        "project_root": str(tmp_path),
        "data_dir": str(tmp_path),
        "wavesol": {"disperser": "VPHG1200@540", "geometry_prior": {"block_height": 6, "fit_deg": 2}},
    }
    outdir = wavesol_dir(cfg, run_root=tmp_path)
    outdir.mkdir(parents=True, exist_ok=True)
    h, w = 72, 160
    y = np.arange(h, dtype=float)
    x = np.arange(w, dtype=float)
    yy = y[:, None]
    xx = x[None, :]
    shift = 0.0025 * (yy - h / 2.0) ** 2 - 2.0
    img = np.zeros((h, w), dtype=np.float32)
    for x0 in (36.0, 78.0, 116.0):
        img += np.exp(-0.5 * ((xx - (x0 + shift)) / 1.4) ** 2).astype(np.float32)
    _fake_writeto(outdir / "superneon.fits", img, header={})

    res = gp.build_geometry_prior(cfg, run_root=tmp_path)
    payload = json.loads(res.geometry_model_json.read_text(encoding="utf-8"))
    coeff = payload["x_shift_poly_coeffs"]
    assert len(coeff) >= 2
    assert payload["metrics"]["trust_score"] is not None
    assert payload["metrics"]["coverage"] > 0.5
    assert res.geometry_preview_fits.exists()
    preview = _fake_getdata(res.geometry_preview_fits)
    assert preview.shape == img.shape


def test_slit_sanity_reports_expected_width_consistency(tmp_path: Path, monkeypatch) -> None:
    _install_fake_astropy(monkeypatch)
    ss = importlib.import_module("scorpio_pipe.stages.slit_sanity")

    frame = tmp_path / "flat_test.fits"
    h, w = 90, 110
    y = np.arange(h, dtype=float)
    x = np.arange(w, dtype=float)
    yy = y[:, None]
    xx = x[None, :]
    img = np.exp(-0.5 * ((xx - 55.0) / 2.35) ** 2) * np.exp(-0.5 * ((yy - 45.0) / 18.0) ** 2)
    _fake_writeto(frame, img.astype(np.float32), header={"INSTRUME": "SCORPIO-2", "SLITWID": 1.0})

    cfg = {
        "work_dir": str(tmp_path),
        "project_root": str(tmp_path),
        "data_dir": str(tmp_path),
        "frames": {"flat": [str(frame)]},
    }
    res = ss.run_slit_sanity(cfg, run_root=tmp_path)
    payload = json.loads(res.json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["median_width_px"] is not None
    assert abs(float(payload["median_width_px"]) - 5.0) < 2.0
    assert res.expected_width_px is not None
    assert res.png_path.exists()
    assert "Measured median width" in res.md_path.read_text(encoding="utf-8")
