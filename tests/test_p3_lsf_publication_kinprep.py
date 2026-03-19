from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy.io import fits  # noqa: E402

from scorpio_pipe.io.mef import WaveGrid, write_sci_var_mask
from scorpio_pipe.stages.kinematics_prep import run_kinematics_prep
from scorpio_pipe.stages.lsf_model import run_lsf_model
from scorpio_pipe.stages.publication_ready import run_publication_ready
from scorpio_pipe.workspace_paths import stage_dir


def _write_primary(path: Path, data: np.ndarray, **hdr_cards) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = fits.Header()
    for k, v in hdr_cards.items():
        hdr[k] = v
    fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32), header=hdr).writeto(path, overwrite=True)


def test_p3_lsf_publication_ready_and_kinprep_pipeline() -> None:
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory(prefix="scorpipe_pytest_p3_lsf_") as td:
        work_dir = Path(td) / "work"
        wdir = stage_dir(work_dir, "wavesol")
        sdir = stage_dir(work_dir, "stack2d")
        edir = stage_dir(work_dir, "extract1d")
        (work_dir / "lsf_model").mkdir(parents=True, exist_ok=True)
        (work_dir / "publication_ready").mkdir(parents=True, exist_ok=True)
        (work_dir / "kinematics_prep").mkdir(parents=True, exist_ok=True)
        wdir.mkdir(parents=True, exist_ok=True)
        sdir.mkdir(parents=True, exist_ok=True)
        edir.mkdir(parents=True, exist_ok=True)
        (work_dir / "sky_refine").mkdir(parents=True, exist_ok=True)

        ny, nx = 44, 180
        x = np.arange(nx)[None, :]
        y = np.arange(ny)[:, None]
        lam_map = 5000.0 + 0.9 * x + 0.06 * (y - ny / 2.0)
        lam_map = np.asarray(lam_map, dtype=float)
        lam_map[:, :4] = np.nan
        lam_map[:, -4:] = np.nan
        _write_primary(wdir / "lambda_map.fits", lam_map, DISPERSE="VPHG1200@540", SLITWID=1.0, WAVEUNIT="Angstrom", WAVEREF="air")

        line_lams = np.array([5030.0, 5070.0, 5115.0, 5160.0, 5205.0, 5250.0, 5305.0])
        arc = np.zeros((ny, nx), dtype=float)
        for iy in range(ny):
            row_lam = lam_map[iy]
            for lam0 in line_lams:
                ix = int(np.nanargmin(np.abs(row_lam - lam0)))
                sigma_px = 1.4 + 0.15 * (iy / max(1, ny - 1))
                prof = np.exp(-0.5 * ((np.arange(nx) - ix) / sigma_px) ** 2)
                arc[iy] += 120.0 * prof
        arc += rng.normal(0.0, 1.0, size=arc.shape)
        _write_primary(wdir / "superneon.fits", arc, DISPERSE="VPHG1200@540", SLITWID=1.0)
        hand_pairs = "\n".join([f"{20+i*10:.2f} {lam:.3f}" for i, lam in enumerate(line_lams)]) + "\n"
        (wdir / "hand_pairs.txt").write_text(hand_pairs, encoding="utf-8")

        cfg = {
            "work_dir": str(work_dir),
            "frames": {"obj": [str(work_dir / "raw" / "obj1.fits")]},
            "lsf_model": {"fit_half_window_px": 6, "min_line_snr": 3.0, "min_line_measures": 6},
        }
        (work_dir / "raw").mkdir(parents=True, exist_ok=True)
        (work_dir / "raw" / "obj1.fits").write_bytes(b"raw")

        out_lsf = run_lsf_model(cfg, out_dir=work_dir / "lsf_model")
        assert Path(out_lsf["lsf_map"]).exists()
        lsf_map = fits.getdata(out_lsf["lsf_map"], ext=1).astype(float)
        assert np.isfinite(lsf_map[:, 8:-8]).any()
        assert float(np.nanmedian(lsf_map[:, 8:-8])) > 0.2

        grid = WaveGrid(lambda0=5002.0, dlambda=0.9, nlam=160, unit="Angstrom", wave_ref="air")
        sci = 10.0 + 20.0 * np.exp(-0.5 * ((np.arange(ny)[:, None] - ny / 2.0) / 3.0) ** 2)
        sci = np.repeat(sci, grid.nlam, axis=1)
        var = np.full_like(sci, 4.0)
        mask = np.zeros_like(sci, dtype=np.uint16)
        hdr = fits.Header()
        hdr["BUNIT"] = "adu/s"
        write_sci_var_mask(sdir / "stack2d.fits", sci, var=var, mask=mask, header=hdr, grid=grid)

        out_pub = run_publication_ready(cfg, out_dir=work_dir / "publication_ready")
        assert Path(out_pub["outputs"]["spectrum"]).exists()
        assert Path(out_pub["outputs"]["wl"]).exists()
        assert Path(out_pub["outputs"]["dismap"]).exists()
        wl = fits.getdata(out_pub["outputs"]["wl"], ext=1).astype(float)
        dis = fits.getdata(out_pub["outputs"]["dismap"], ext=1).astype(float)
        assert wl.shape == lam_map.shape
        assert dis.shape == lam_map.shape
        assert np.isnan(dis[:, :4]).all()
        assert np.isfinite(dis[:, 20:-20]).all()

        wave = 5002.0 + 0.9 * np.arange(160)
        cols_t = [
            fits.Column(name="LAMBDA", format="D", array=wave),
            fits.Column(name="FLUX_TRACE", format="D", array=np.sin(np.linspace(0, 3, 160)) + 10.0),
            fits.Column(name="VAR_TRACE", format="D", array=np.full(160, 0.5)),
            fits.Column(name="MASK_TRACE", format="J", array=np.zeros(160, dtype=np.int32)),
            fits.Column(name="SKY_TRACE", format="D", array=np.zeros(160)),
            fits.Column(name="NPIX_TRACE", format="I", array=np.full(160, 5, dtype=np.int16)),
        ]
        cols_f = [
            fits.Column(name="LAMBDA", format="D", array=wave),
            fits.Column(name="FLUX_FIXED", format="D", array=np.sin(np.linspace(0, 3, 160)) + 9.8),
            fits.Column(name="VAR_FIXED", format="D", array=np.full(160, 0.8)),
            fits.Column(name="MASK_FIXED", format="J", array=np.zeros(160, dtype=np.int32)),
            fits.Column(name="SKY_FIXED", format="D", array=np.zeros(160)),
            fits.Column(name="NPIX_FIXED", format="I", array=np.full(160, 5, dtype=np.int16)),
        ]
        ph = fits.Header()
        ph["WAVEREF"] = "air"
        ph["WAVEUNIT"] = "Angstrom"
        fits.HDUList([
            fits.PrimaryHDU(header=ph),
            fits.BinTableHDU.from_columns(cols_t, name="SPEC_TRACE"),
            fits.BinTableHDU.from_columns(cols_f, name="SPEC_FIXED"),
        ]).writeto(edir / "spec1d.fits", overwrite=True)
        (edir / "trace.json").write_text(json.dumps({"trace_model": {"method": "centroid_spline"}, "aperture": {"half_width": 2}}, ensure_ascii=False), encoding="utf-8")
        (work_dir / "sky_refine" / "sky_residual_windows.json").write_text(json.dumps([
            {"lambda_min_A": 5060.0, "lambda_max_A": 5070.0, "classification": "sky_residual", "action": "mask"}
        ], ensure_ascii=False), encoding="utf-8")

        out_kin = run_kinematics_prep(cfg, out_dir=work_dir / "kinematics_prep")
        assert Path(out_kin["spec_kin_fits"]).exists()
        hdr_kin = fits.getheader(out_kin["spec_kin_fits"], 0)
        assert hdr_kin["KINPREP"] is True
        assert hdr_kin["KINPROD"] == "TRACE"
        assert "VELSCALE" in hdr_kin
        masks = json.loads((work_dir / "kinematics_prep" / "masks.json").read_text(encoding="utf-8"))
        assert masks["skyline_residual_intervals"]
        assert (work_dir / "kinematics_prep" / "README.md").exists()
