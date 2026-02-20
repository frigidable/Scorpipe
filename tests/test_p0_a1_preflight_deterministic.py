from __future__ import annotations

import json
from pathlib import Path

import pytest


def _require_astropy() -> None:
    pytest.importorskip("astropy")


def _write_fits(path: Path, *, good: bool = True) -> None:
    from astropy.io import fits
    import numpy as np

    data = np.zeros((10, 20), dtype=np.int16)
    hdu = fits.PrimaryHDU(data=data)
    hdr = hdu.header

    # Instrument detection
    hdr["INSTRUME"] = "SCORPIO-2"

    # Required keys (strict=True)
    if good:
        hdr["MODE"] = "Spectra"
    # else: missing MODE -> HeaderContractError

    hdr["IMAGETYP"] = "object"
    hdr["DISPERSE"] = "VPHG1200@540"
    hdr["SLITWID"] = 1.0
    hdr["BINNING"] = "1x1"
    hdr["NODE"] = "A"
    hdr["RATE"] = 1.0
    hdr["GAIN"] = 1.0
    hdr["DATE-OBS"] = "2026-02-20"
    hdr["UT"] = "12:34:56"

    # Optional: leave EXPTIME/OBJECT/etc absent to trigger WARN for soft keys.

    hdu.writeto(path, overwrite=True)


def test_preflight_deterministic_json(tmp_path: Path) -> None:
    """P0-A1 DoD: repeat run -> byte-identical JSON for the same inputs."""

    _require_astropy()
    from scorpio_pipe.preflight import run_preflight

    data_dir = tmp_path / "raw"
    run_root = tmp_path / "run"
    data_dir.mkdir(parents=True, exist_ok=True)

    _write_fits(data_dir / "good.fits", good=True)
    _write_fits(data_dir / "bad_missing_mode.fits", good=False)

    cfg = {"data_dir": str(data_dir)}

    run_preflight(cfg, run_root)

    p_report = run_root / "manifest" / "preflight_report.json"
    p_profile = run_root / "manifest" / "night_profile.json"
    p_index = run_root / "manifest" / "quarantine_index.json"

    s_report_1 = p_report.read_text(encoding="utf-8")
    s_profile_1 = p_profile.read_text(encoding="utf-8")
    s_index_1 = p_index.read_text(encoding="utf-8")

    # Re-run to the same run_root.
    run_preflight(cfg, run_root)

    assert s_report_1 == p_report.read_text(encoding="utf-8")
    assert s_profile_1 == p_profile.read_text(encoding="utf-8")
    assert s_index_1 == p_index.read_text(encoding="utf-8")

    # Sanity: quarantine index entries have codes + severity and tickets exist.
    idx = json.loads(s_index_1)
    entries = idx.get("entries")
    assert isinstance(entries, list) and entries, "expected at least one quarantine entry"
    for ent in entries:
        assert ent.get("severity") in ("INFO", "WARN", "ERROR")
        codes = ent.get("codes")
        assert isinstance(codes, list) and len(codes) >= 1
        t = ent.get("ticket")
        assert isinstance(t, str) and t.startswith("quarantine/")
        assert (run_root / t).exists()
