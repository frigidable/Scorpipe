"""Read-only exact-night raw pixel checks: actual bytes, no source mutation."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from tests.test_sc1_first_masterbias_gate import case

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "qa" / "sc1_raw_pixel_readability_qa.py"
)


def load(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("sc1_pixel_readability_qa", SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def inspect_small(m, conf, monkeypatch):
    import sc1_first_masterbias_gate as gate
    original = gate.EXPECTED_SHAPE
    gate.EXPECTED_SHAPE = (6, 8)
    try:
        return m.inspect(conf)
    finally:
        gate.EXPECTED_SHAPE = original


def test_selected_raw_data_hashes_are_deterministic_and_read_only(
    tmp_path, monkeypatch,
):
    m = load(monkeypatch)
    conf, cfg, raw = case(tmp_path)
    selected = sorted(raw.iterdir())
    original = {p: p.read_bytes() for p in selected}
    report, paths = inspect_small(m, conf, monkeypatch)
    assert len(paths) == 32
    assert report["schema"] == "SC1_RAW_PIXEL_READABILITY_V1"
    assert report["status"] == "READ_ONLY_PIXEL_READABILITY_NOT_SCIENCE_QUALIFICATION"
    assert report["input_counts"] == {"bias": 20, "obj": 7, "flat": 3, "neon": 2}
    assert len(report["input_files"]) == 32
    assert report["all_selected_primary_arrays_readable_and_finite"] is True
    assert report["physical_overscan_region_certified"] is False
    assert report["masterbias_and_covariance_validated"] is False
    assert report["raw_pixel_units_and_gain_physically_verified"] is False
    first = report["input_files"][0]
    assert first["source_sha256"] == hashlib.sha256(
        (raw / cfg["frames"]["bias"][0]).read_bytes()
    ).hexdigest()
    assert first["shape_numpy_y_x"] == [6, 8]
    assert first["stored_mean_sample"] == pytest.approx(999.0)
    assert first["fits_checksum_validated"] is False
    again, _ = inspect_small(m, conf, monkeypatch)
    assert again["input_manifest_sha256"] == report["input_manifest_sha256"]
    assert {p: p.read_bytes() for p in selected} == original


def test_nonfinite_raw_pixel_fails_even_when_headers_pass(tmp_path, monkeypatch):
    m = load(monkeypatch)
    conf, cfg, raw = case(tmp_path)
    target = raw / cfg["frames"]["bias"][0]
    with fits.open(target, mode="update") as hdul:
        hdul[0].data[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="Non-finite raw pixels"):
        inspect_small(m, conf, monkeypatch)


def test_raw_pixel_inventory_rejects_truncated_primary_image(tmp_path, monkeypatch):
    m = load(monkeypatch)
    conf, cfg, raw = case(tmp_path)
    target = raw / cfg["frames"]["obj"][0]
    with target.open("r+b") as f:
        f.truncate(2880)  # leave the header but remove its required data
    with pytest.raises(RuntimeError, match="Truncated primary FITS payload"):
        inspect_small(m, conf, monkeypatch)


def test_external_report_only_and_exclusive_create(tmp_path, monkeypatch):
    m = load(monkeypatch)
    conf, cfg, raw = case(tmp_path)
    import sc1_first_masterbias_gate as gate
    orig_shape = gate.EXPECTED_SHAPE
    gate.EXPECTED_SHAPE = (6, 8)
    try:
        for forbidden in (
            raw / "report.json",
            raw / "qa" / "report.json",
            conf.parent / "qa" / "report.json",
        ):
            with pytest.raises(ValueError, match="outside the work and configured RAW"):
                m.run(conf, forbidden)
            assert not forbidden.exists()
        good = tmp_path / "external_qa" / "raw_inventory.json"
        m.run(conf, good)
        report = json.loads(good.read_text(encoding="utf-8"))
        assert len(report["input_files"]) == 32
        assert report["created_utc"]
        assert good.is_file()
        with pytest.raises(FileExistsError):
            m.run(conf, good)
    finally:
        gate.EXPECTED_SHAPE = orig_shape


def test_manifest_changes_if_selected_raw_content_changes(tmp_path, monkeypatch):
    m = load(monkeypatch)
    conf, cfg, raw = case(tmp_path)
    a, _ = inspect_small(m, conf, monkeypatch)
    target = raw / cfg["frames"]["obj"][0]
    with fits.open(target, mode="update") as hdu:
        hdu[0].data[0, 0] += 1
    b, _ = inspect_small(m, conf, monkeypatch)
    assert a["input_manifest_sha256"] != b["input_manifest_sha256"]


def test_integer_fits_blank_sentinel_is_not_treated_as_valid_data(
    tmp_path, monkeypatch,
):
    m = load(monkeypatch)
    conf, cfg, raw = case(tmp_path)
    target = raw / cfg["frames"]["bias"][0]
    with fits.open(target) as source:
        hdr = source[0].header.copy()
    hdu = fits.PrimaryHDU(np.full((6, 8), 999, dtype=np.int16), header=hdr)
    hdu.header["BLANK"] = -32768
    hdu.data[0, 0] = -32768
    hdu.writeto(target, overwrite=True)
    with pytest.raises(RuntimeError, match="Undefined FITS BLANK pixels"):
        inspect_small(m, conf, monkeypatch)


def test_existing_bias_products_do_not_block_readonly_raw_pixel_inventory(
    tmp_path, monkeypatch,
):
    """Existing 03_bias output must remain byte-identical during RAW QA."""
    m = load(monkeypatch)
    conf, cfg, raw = case(tmp_path)
    legacy = conf.parent / "03_bias"
    legacy.mkdir()
    product = legacy / "previous_masterbias.fits"
    product.write_bytes(b"previous science product - do not modify")
    original = {p: p.read_bytes() for p in [conf, product, *raw.iterdir()]}
    report, selected = inspect_small(m, conf, monkeypatch)
    assert len(selected) == 32
    assert report["header_audit_mode"] == "audit_existing_readonly"
    assert report["header_preflight_status"] == (
        "EXISTING_WORKSPACE_HEADER_AUDIT_NOT_SCIENCE_VALIDATION"
    )
    assert str(legacy) in report["existing_stage_products"]
    assert report["stage_execution_authorized"] is False
    assert report["first_build_clear"] is False

    import sc1_first_masterbias_gate as gate
    prior_shape = gate.EXPECTED_SHAPE
    gate.EXPECTED_SHAPE = (6, 8)
    try:
        out = tmp_path / "external" / "pixel_inventory.json"
        m.run(conf, out)
        on_disk = json.loads(out.read_text(encoding="utf-8"))
        assert len(on_disk["input_files"]) == 32
        assert on_disk["header_audit_mode"] == "audit_existing_readonly"
    finally:
        gate.EXPECTED_SHAPE = prior_shape
    assert all(p.read_bytes() == expected for p, expected in original.items())
