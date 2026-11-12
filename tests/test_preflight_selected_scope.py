"""Regression coverage for SCORPIO-1 imaging and run-scoped Preflight."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scorpio_pipe.metadata import HeaderContractError, parse_frame_meta
from scorpio_pipe.preflight import ensure_preflight, run_preflight


def _sc1_header(*, mode: str, filters: str, disperser: str) -> dict:
    return {
        "INSTRUME": "SCORPIO-1",
        "MODE": mode,
        "IMAGETYP": "obj",
        "DISPERSE": disperser,
        "SLITWID": "",
        "FILTERS": filters,
        "BINNING": "1x2",
        "NAXIS1": 2080,
        "NAXIS2": 1032,
        "NODE": "A",
        "RATE": 145.0,
        "GAIN": 0.9,
        "DATE-OBS": "2024/04/12",
        "UT": "18:02:01.94",
    }


def test_sc1_imaging_does_not_require_spectral_slit() -> None:
    meta = parse_frame_meta(
        _sc1_header(mode="Image", filters="U", disperser=""),
        strict=True,
    )
    assert meta.mode == "Image"
    assert meta.slit_width_key == "nan"
    assert meta.meta_provenance["slit_width_arcsec"] == "not_applicable:imaging"


def test_sc1_spectral_mask_circ_still_requires_slit_width() -> None:
    with pytest.raises(HeaderContractError, match="slit width"):
        parse_frame_meta(
            _sc1_header(
                mode="Spectra", filters="mask_circ", disperser="VPHG1200B"
            ),
            strict=True,
        )


def _write_frame(path: Path, *, bad: bool = False) -> None:
    pytest.importorskip("astropy")
    from astropy.io import fits
    import numpy as np

    hdu = fits.PrimaryHDU(data=np.zeros((10, 20), dtype=np.int16))
    hdr = hdu.header
    hdr["INSTRUME"] = "SCORPIO-2"
    if not bad:
        hdr["MODE"] = "Spectra"
    hdr["IMAGETYP"] = "object"
    hdr["DISPERSE"] = "VPHG1200@540"
    hdr["SLITWID"] = 1.0
    hdr["BINNING"] = "1x1"
    hdr["NODE"] = "A"
    hdr["RATE"] = 1.0
    hdr["GAIN"] = 1.0
    hdr["DATE-OBS"] = "2026-02-20"
    hdr["UT"] = "12:34:56"
    hdu.writeto(path, overwrite=True)


def _report(root: Path) -> dict:
    return json.loads(
        (root / "manifest" / "preflight_report.json").read_text("utf-8")
    )


def test_selected_run_not_blocked_by_unassigned_error(tmp_path: Path) -> None:
    data = tmp_path / "raw"
    data.mkdir()
    good, bad = data / "good.fits", data / "other_bad.fits"
    _write_frame(good)
    _write_frame(bad, bad=True)
    root = tmp_path / "run"

    cfg = {
        "data_dir": str(data),
        "frames": {"obj": [str(good)], "bias": []},
    }
    result = run_preflight(cfg, root)
    report = _report(root)
    assert result["status"] in {"OK", "WARN"}
    assert report["scope"] == "assigned"
    assert report["night_status"] == "ERROR"
    assert report["status"] == report["selection_summary"]["status"]
    assert report["selection_summary"]["n_assigned"] == 1
    assert report["selection_summary"]["n_error"] == 0
    assert any(
        r["code"] == "HEADER_CONTRACT_ERROR"
        for r in report["night_reasons"]
    )
    assert not any(
        r["code"] == "HEADER_CONTRACT_ERROR"
        for r in report["reasons"]
    )

    cached = ensure_preflight(cfg, root)
    assert cached["reused"] is True

    # Reassign the bad exposure: same night inventory, new frame selection.
    cfg["frames"]["obj"] = [str(bad)]
    updated = ensure_preflight(cfg, root)
    assert updated["reused"] is False
    assert updated["status"] == "ERROR"
    assert _report(root)["selection_summary"]["n_error"] == 1


def test_missing_or_excluded_selected_frame_fails_closed(tmp_path: Path) -> None:
    data = tmp_path / "raw"
    data.mkdir()
    good = data / "good.fits"
    _write_frame(good)
    root = tmp_path / "run"
    cfg = {"data_dir": str(data), "frames": {"obj": [str(good)]}}

    assert ensure_preflight(cfg, root)["status"] in {"OK", "WARN"}

    cfg["frames"]["obj"] = [str(data / "missing.fits")]
    res = ensure_preflight(cfg, root)
    assert res["status"] == "ERROR"
    assert any(
        r["code"] == "ASSIGNED_FRAME_NOT_SCANNED"
        for r in _report(root)["reasons"]
    )

    cfg["frames"]["obj"] = [str(good)]
    cfg["exclude_frames"] = [str(good)]
    res = ensure_preflight(cfg, root)
    assert res["status"] == "ERROR"
    assert any(
        r["code"] == "ASSIGNED_FRAME_EXCLUDED"
        for r in _report(root)["reasons"]
    )


def test_night_fingerprint_invalidates_on_new_exposure(tmp_path: Path) -> None:
    data = tmp_path / "raw"
    data.mkdir()
    good = data / "good.fits"
    _write_frame(good)
    root = tmp_path / "run"
    cfg = {"data_dir": str(data), "frames": {"obj": [str(good)]}}

    first = ensure_preflight(cfg, root)
    assert first["reused"] is False
    assert ensure_preflight(cfg, root)["reused"] is True

    _write_frame(data / "other_bad.fits", bad=True)
    changed = ensure_preflight(cfg, root)
    assert changed["reused"] is False
    report = _report(root)
    assert report["night_status"] == "ERROR"
    assert report["selection_summary"]["n_error"] == 0


def test_legacy_night_only_scope_stays_fail_closed(tmp_path: Path) -> None:
    data = tmp_path / "raw"
    data.mkdir()
    _write_frame(data / "good.fits")
    _write_frame(data / "bad.fits", bad=True)
    root = tmp_path / "run"
    report = run_preflight({"data_dir": str(data)}, root)
    assert report["status"] == "ERROR"
    assert _report(root)["scope"] == "night"
