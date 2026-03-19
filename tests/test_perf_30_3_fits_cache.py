from __future__ import annotations

from pathlib import Path

import pytest


def _require_astropy() -> None:
    pytest.importorskip("astropy")


def _write_valid_fits(path: Path, *, obj: str = "Target") -> None:
    from astropy.io import fits
    import numpy as np

    data = np.zeros((10, 20), dtype=np.int16)
    hdu = fits.PrimaryHDU(data=data)
    hdr = hdu.header
    hdr["INSTRUME"] = "SCORPIO-2"
    hdr["MODE"] = "Spectra"
    hdr["IMAGETYP"] = "OBJECT"
    hdr["OBJECT"] = obj
    hdr["DISPERSE"] = "VPHG1200@540"
    hdr["SLITWID"] = 1.0
    hdr["BINNING"] = "1x1"
    hdr["NODE"] = "A"
    hdr["RATE"] = 1.0
    hdr["GAIN"] = 1.0
    hdr["DATE-OBS"] = "2026-03-06"
    hdr["UT"] = "12:34:56"
    hdu.writeto(path, overwrite=True)


def test_fits_cache_reuses_disk_read_across_inspect_builder_preflight(tmp_path: Path, monkeypatch) -> None:
    _require_astropy()

    from scorpio_pipe.dataset.builder import build_dataset_manifest
    from scorpio_pipe.inspect import inspect_dataset
    from scorpio_pipe.metadata import fits_cache as fc
    from scorpio_pipe.preflight import run_preflight

    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    frame = data_dir / "s0001.fits"
    _write_valid_fits(frame)

    fc.clear_fits_header_meta_cache()
    calls = 0
    orig = fc._load_header_from_disk

    def _wrapped(path: str | Path):
        nonlocal calls
        calls += 1
        return orig(path)

    monkeypatch.setattr(fc, "_load_header_from_disk", _wrapped)

    inspect_dataset(data_dir)
    build_dataset_manifest(data_dir)
    run_preflight({"data_dir": str(data_dir)}, tmp_path / "run1")

    inspect_dataset(data_dir)
    build_dataset_manifest(data_dir)
    run_preflight({"data_dir": str(data_dir)}, tmp_path / "run2")

    assert calls == 1, "same FITS file should be opened once and then served from cache"


def test_fits_cache_invalidates_on_file_change(tmp_path: Path, monkeypatch) -> None:
    _require_astropy()

    from scorpio_pipe.metadata import fits_cache as fc

    frame = tmp_path / "s0002.fits"
    _write_valid_fits(frame, obj="A")

    fc.clear_fits_header_meta_cache()
    calls = 0
    orig = fc._load_header_from_disk

    def _wrapped(path: str | Path):
        nonlocal calls
        calls += 1
        return orig(path)

    monkeypatch.setattr(fc, "_load_header_from_disk", _wrapped)

    h1 = fc.get_header_fields(frame)
    assert str(h1.get("OBJECT")) == "A"
    assert calls == 1

    _write_valid_fits(frame, obj="B")

    h2 = fc.get_header_fields(frame)
    assert str(h2.get("OBJECT")) == "B"
    assert calls == 2, "mtime/size change should invalidate cached entry"


def test_fits_cache_stats_and_launcher_representative_header(tmp_path: Path, monkeypatch) -> None:
    _require_astropy()
    pytest.importorskip("PySide6")

    import pandas as pd

    from scorpio_pipe.metadata import fits_cache as fc
    from scorpio_pipe.ui.launcher_window import _representative_header_from_inspect_df

    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    frame = data_dir / "s0003.fits"
    _write_valid_fits(frame, obj="CachedTarget")

    fc.clear_fits_header_meta_cache_all()
    calls = 0
    orig = fc._load_header_from_disk

    def _wrapped(path: str | Path):
        nonlocal calls
        calls += 1
        return orig(path)

    monkeypatch.setattr(fc, "_load_header_from_disk", _wrapped)

    df = pd.DataFrame([{"path": str(frame), "kind": "obj"}])
    hdr1 = _representative_header_from_inspect_df(df)
    hdr2 = _representative_header_from_inspect_df(df)

    assert hdr1 is not None and hdr2 is not None
    assert str(hdr1.get("OBJECT")) == "CachedTarget"
    assert calls == 1

    stats = fc.get_fits_header_meta_cache_stats()
    assert stats.disk_loads == 1
    assert stats.hits >= 1
    assert stats.entries >= 1
