"""Synthetic, read-only regression checks for SC1 raw geometry inventory."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load():
    p = (
        Path(__file__).resolve().parents[1]
        / "scripts/qa/sc1_detector_geometry_inventory_qa.py"
    )
    spec = importlib.util.spec_from_file_location("sc1_geometry_inventory_qa", p)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_x_y_bands_cover_entire_raw_geometry_without_overlap():
    m = load()
    assert m.X_BANDS[0][0] == 0
    assert m.X_BANDS[-1][1] == 2080
    assert m.Y_BANDS[0][0] == 0
    assert m.Y_BANDS[-1][1] == 1032
    assert all(a < b for a, b in m.X_BANDS + m.Y_BANDS)
    assert all(b == c for (_, b), (c, _) in zip(m.X_BANDS, m.X_BANDS[1:]))
    assert all(b == c for (_, b), (c, _) in zip(m.Y_BANDS, m.Y_BANDS[1:]))


def test_raw_edge_profile_never_labels_overscan_even_for_synthetic_step():
    m = load()
    a = np.full(m.EXPECTED_SHAPE, 100, dtype=np.int16)
    a[128:896, 256:1792] = 200
    r = m.profile(a)
    assert r["middle_reference_adu"] == 200
    assert r["x_edge_band_medians"][0]["median_adu"] == 100
    assert r["x_edge_band_medians"][-1]["minus_middle_adu"] == -100
    assert r["physical_overscan_region_certified"] is False


def test_raw_profile_fails_closed_on_wrong_shape():
    m = load()
    with pytest.raises(ValueError, match="Unexpected raw detector geometry"):
        m.profile(np.ones((1032, 2068), dtype=np.int16))
    with pytest.raises(ValueError, match="Unexpected raw detector geometry"):
        m.profile(np.ones((1046, 2080), dtype=np.int16))


def test_raw_profile_fails_closed_on_nonfinite_or_nonnumeric():
    m = load()
    a = np.zeros(m.EXPECTED_SHAPE, dtype=np.float32)
    a[0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        m.profile(a)
    with pytest.raises(TypeError, match="Non-numeric"):
        m.profile(np.zeros(m.EXPECTED_SHAPE, dtype="U1"))


def test_raw_header_keywords_not_inferred_when_missing():
    m = load()
    h = {"NAXIS1": 2080, "NAXIS2": 1032, "RATE": 145}
    r = m.header_fields(h)
    assert r["NAXIS1"] == "2080"
    assert r["BIASSEC"] is None
    assert r["DATASEC"] is None
    assert r["TRIMSEC"] is None
    assert r["RATE"] == "145"


def test_geometry_inventory_explicitly_documents_read_only_science_hold():
    m = load()
    source = (
        Path(m.__file__).read_text(encoding="utf-8")
    )
    assert "OVERSCAN_UNVERIFIED_SCIENCE_HOLD" in source
    assert "from sc1_bias_night_timeline_qa" in source
    assert "len(all_paths) != 53" in source
    assert 'report_file.open("x"' in source
