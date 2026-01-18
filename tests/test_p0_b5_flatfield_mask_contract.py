from __future__ import annotations

import pytest

pytest.importorskip("astropy")
fits = pytest.importorskip("astropy.io.fits")

from pathlib import Path

import numpy as np

from scorpio_pipe.stages.flatfield import apply_flat


def _minimal_hdr(shape: tuple[int, int]) -> fits.Header:
    # Minimal header sufficient for compat checks (fallback path) and noise/unit models.
    hdr = fits.Header()
    hdr["INSTRUME"] = ("TEST", "Synthetic test instrument")
    hdr["MODE"] = ("SPEC", "Synthetic mode")
    hdr["DISPERSE"] = ("GRISM1", "Synthetic disperser")
    hdr["XBIN"] = (1, "X binning")
    hdr["YBIN"] = (1, "Y binning")
    hdr["SLITWID"] = ("1.0", "Slit width")
    hdr["NODE"] = ("A", "Readout node")

    # Noise provenance
    hdr["GAIN"] = (1.0, "e-/ADU")
    hdr["RDNOISE"] = (0.0, "e-")
    hdr["NOISRC"] = ("HEADER", "Noise parameters source")

    # Units
    hdr["BUNIT"] = ("ADU", "Raw detector units")

    # NAXIS1/2 are written by FITS automatically, but keep them explicit for clarity.
    hdr["NAXIS1"] = int(shape[1])
    hdr["NAXIS2"] = int(shape[0])
    return hdr


def test_apply_flat_writes_mask_extension(tmp_path: Path) -> None:
    shape = (10, 12)
    data = (np.ones(shape, dtype=np.float32) * 100.0).astype(np.float32)

    hdr = _minimal_hdr(shape)
    data_path = tmp_path / "frame.fits"
    fits.writeto(data_path, data, hdr, overwrite=True)

    superflat_path = tmp_path / "superflat.fits"
    fits.writeto(superflat_path, np.ones(shape, dtype=np.float32), hdr, overwrite=True)

    out_path = tmp_path / "out_flat.fits"

    cfg = {
        "config_dir": str(tmp_path),
        "work_dir": str(tmp_path / "work"),
        "instrument_hint": "TEST",
        "calib": {},
        "flatfield": {},
    }

    out_f, _sel = apply_flat(
        data_path,
        superflat_path,
        out_path,
        cfg=cfg,
        do_bias_subtract=False,
    )

    assert out_f.exists()

    with fits.open(out_f, memmap=False) as hdul:
        extnames = [str(h.header.get("EXTNAME", "")).strip().upper() for h in hdul]
        assert "SCI" in extnames
        assert "VAR" in extnames
        assert "MASK" in extnames

        m = hdul["MASK"].data
        assert m is not None
        assert m.shape == shape
        assert m.dtype == np.uint16
