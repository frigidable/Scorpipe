"""Production SCORPIO science must not use an unindexed default bias alias."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from scorpio_pipe.stages import calib, cosmics, flatfield


def test_scorpio_cosmics_defers_unindexed_global_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "raw.fits"
    hdr = fits.Header({
        "INSTRUME": "SCORPIO-2", "GAIN": 0.62, "RATE": 185.0,
        "BIASSUB": False,
    })
    fits.writeto(path, np.full((12, 13), 1120.0, dtype="f4"), hdr)
    monkeypatch.setattr(calib, "load_master_bias_index", lambda root: None)
    fallback = np.full((12, 13), 1000.0, dtype="f4")
    chosen = cosmics._resolve_cosmics_superbias_for_kind(
        {}, [path], tmp_path, fallback
    )
    assert chosen is None, "Unknown legacy bias must not be applied to SCORPIO"

    # Preserve the historical numerical helper path for generic non-SCORPIO
    # fixtures without a MasterBias group index.
    with fits.open(path, mode="update") as hdul:
        del hdul[0].header["INSTRUME"]
        hdul.flush()
    assert cosmics._resolve_cosmics_superbias_for_kind(
        {}, [path], tmp_path, fallback
    ) is fallback


def test_scorpio_flatfield_refuses_unindexed_masterbias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "object.fits"
    sf = tmp_path / "superflat.fits"
    master = tmp_path / "legacy_bias.fits"
    hdr = fits.Header({
        "INSTRUME": "SCORPIO-2", "GAIN": 0.62, "RATE": 185.0,
        "RDNOISE": 3.0, "BUNIT": "ADU", "BIASSUB": False,
    })
    fits.writeto(raw, np.full((12, 13), 1120.0, dtype="f4"), hdr)
    fits.writeto(sf, np.ones((12, 13), dtype="f4"))
    fits.writeto(master, np.full((12, 13), 1000.0, dtype="f4"))

    monkeypatch.setattr(flatfield, "ensure_compatible_calib", lambda *a, **kw: None)
    monkeypatch.setattr(
        calib, "resolve_master_bias",
        lambda *a, **kw: calib.BiasSelection(
            sci_path=master, var_path=None, dq_path=None, gid="",
            degraded=True, reason="NOIDX",
        ),
    )
    output = tmp_path / "flat_object.fits"
    with pytest.raises(RuntimeError, match="unindexed SCORPIO MasterBias"):
        flatfield.apply_flat(
            raw, sf, output, cfg={}, do_bias_subtract=True,
        )
    assert not output.exists()
