"""BTA 2024-12-06: strong FITS calibration roles versus SUNSKY nightlog."""
from pathlib import Path

import numpy as np
from astropy.io import fits

from scorpio_pipe.inspect import classify_frame, inspect_dataset
from scorpio_pipe.autocfg import build_autoconfig


def _fits(root: Path, fid: str, name: str, imagetyp: str, exp: float = 10):
    path = root / f"{fid}.fts"
    hdu = fits.PrimaryHDU(np.zeros((10, 12), dtype="float32"))
    hdu.header["OBJECT"] = name
    hdu.header["IMAGETYP"] = imagetyp
    hdu.header["EXPTIME"] = exp
    hdu.header["INSTRUME"] = "SCORPIO-1"
    hdu.header["DISPERSE"] = "VPHG1200B"
    hdu.header["BINNING"] = "1x2"
    hdu.header["DATE"] = "2024-12-06T15:34:12"
    hdu.writeto(path)
    return path


def test_explicit_neon_imagetyp_wins_over_object_name_in_header_only():
    h = fits.Header()
    h["OBJECT"] = "N2146sat"
    h["IMAGETYP"] = "neon"
    assert classify_frame(h) == "neon"
    h["IMAGETYP"] = "flat"
    assert classify_frame(h) == "flat"


def test_nightlog_sunsky_mixed_types_are_unresolved_not_autocalibrated(tmp_path):
    raw = tmp_path / "s241206"
    raw.mkdir()
    # Both FITS and nightlog say SUNSKY, but FITS explicitly declares
    # a calibration: observer must review its actual physical use.
    _fits(raw, "s22800407", "sunsky", "neon", 60.)
    _fits(raw, "s22800408", "sunsky", "flat", 10.)
    _fits(raw, "s22800401", "sunsky", "obj", 180.)
    # Genuine target arc, to verify unambiguous calibration survives.
    _fits(raw, "s22800606", "N2146sat", "neon", 60.)
    _fits(raw, "s22800607", "N2146sat", "obj", 1200.)
    (raw / "s241206.txt").write_text(
        "|s22800407|sunsky|sunsky|14:00:00|60|Spectra|VPHG1200B|1.2|\n"
        "|s22800408|sunsky|sunsky|14:01:00|10|Spectra|VPHG1200B|1.2|\n"
        "|s22800401|sunsky|sunsky|13:59:00|180|Spectra|VPHG1200B|1.2|\n"
        "|s22800606|N2146sat|neon|15:33:00|60|Spectra|VPHG1200B|1.2|\n"
        "|s22800607|N2146sat|obj|15:34:00|1200|Spectra|VPHG1200B|1.2|\n",
        encoding="utf-8"
    )
    result = inspect_dataset(raw)
    assert result.n_opened == 5
    tbl = result.table.set_index("fid")

    assert tbl.loc["s22800407", "kind"] == "unresolved"
    assert tbl.loc["s22800408", "kind"] == "unresolved"
    assert bool(tbl.loc["s22800407", "role_conflict"])
    assert bool(tbl.loc["s22800408", "role_conflict"])
    assert tbl.loc["s22800407", "header_imagetyp"] == "neon"
    assert tbl.loc["s22800408", "header_imagetyp"] == "flat"
    assert tbl.loc["s22800407", "nightlog_kind"] == "sunsky"
    assert "Manual classification required" in tbl.loc[
        "s22800407", "role_conflict_reason"
    ]
    assert tbl.loc["s22800401", "kind"] == "sunsky"
    assert tbl.loc["s22800606", "kind"] == "neon"
    assert not bool(tbl.loc["s22800606", "role_conflict"])
    assert tbl.loc["s22800607", "kind"] == "obj"

    ac = build_autoconfig(
        result.table, raw, "N2146sat", tmp_path / "science_run",
        disperser="VPHG1200B", slit="1.2", binning="1x2",
    )
    assert [Path(p).stem for p in ac.frames["obj"]] == ["s22800607"]
    assert [Path(p).stem for p in ac.frames["neon"]] == ["s22800606"]
    assert ac.frames["flat"] == []
    assert [Path(p).stem for p in ac.frames["sunsky"]] == ["s22800401"]


def test_unambiguous_calibration_kind_no_nightlog(tmp_path):
    raw = tmp_path / "isolated_night"
    raw.mkdir()
    _fits(raw, "s22800606", "N2146sat", "neon")
    _fits(raw, "s22800614", "N2146sat", "flat")
    df = inspect_dataset(raw).table.set_index("fid")
    assert df.loc["s22800606", "kind"] == "neon"
    assert df.loc["s22800614", "kind"] == "flat"
    assert not bool(df.loc["s22800606", "role_conflict"])
    assert not bool(df.loc["s22800614", "role_conflict"])
