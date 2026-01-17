import json
from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from scorpio_pipe.maskbits import SKYMODEL_FAIL
from scorpio_pipe.stages.extract1d import run_extract1d
from scorpio_pipe.workspace_paths import stage_dir


def _write_min_mef(path: Path, *, ny: int = 32, nx: int = 64) -> None:
    # Synthetic long-slit spectrum: bright stripe at y=ny//2.
    y = np.arange(ny)[:, None]
    profile = np.exp(-0.5 * ((y - (ny / 2.0)) / 2.0) ** 2)
    sci = (1000.0 * profile).astype("f4") * np.ones((ny, nx), dtype="f4")
    var = (np.ones_like(sci) * 10.0).astype("f4")
    mask = np.zeros_like(sci, dtype="u2")
    # Simulate upstream sky pass-through: SKYMODEL_FAIL bit is present.
    mask[:] = np.uint16(SKYMODEL_FAIL)

    ph = fits.Header()
    ph["BUNIT"] = "ADU"

    sh = fits.Header()
    # Minimal linear wavelength axis.
    sh["CRVAL1"] = 5000.0
    sh["CDELT1"] = 1.0
    sh["CRPIX1"] = 1.0
    sh["CTYPE1"] = "WAVE"
    sh["CUNIT1"] = "Angstrom"

    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(header=ph),
            fits.ImageHDU(sci, header=sh, name="SCI"),
            fits.ImageHDU(var, header=sh, name="VAR"),
            fits.ImageHDU(mask, header=sh, name="MASK"),
        ]
    )
    hdul.writeto(path, overwrite=True)


def test_extract1d_raises_qadegrd_from_skymodel_fail(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    out_dir = stage_dir(work_dir, "extract")
    out_dir.mkdir(parents=True, exist_ok=True)

    in_fits = tmp_path / "input2d.fits"
    _write_min_mef(in_fits)

    cfg = {
        "work_dir": str(work_dir),
        "frames": {"__setup__": {"instrument": "scorpio", "mode": "longslit"}},
        # Keep extraction robust on small synthetic frames.
        "extract1d": {"input_mode": "single_frame", "single_frame_path": str(in_fits)},
    }

    out = run_extract1d(cfg, in_fits=in_fits, out_dir=out_dir)
    spec_path = Path(out["spec1d_fits"])
    assert spec_path.exists()

    with fits.open(spec_path, memmap=False) as hdul:
        ph = hdul[0].header
        assert int(ph.get("QADEGRD", 0)) == 1

    done = json.loads((out_dir / "done.json").read_text(encoding="utf-8"))
    flags = (done.get("qc") or {}).get("flags") or []
    codes = {str(f.get("code")) for f in flags if isinstance(f, dict)}
    assert "UPSTREAM_SKY_PASSTHROUGH" in codes
