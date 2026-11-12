"""Pointing admission must protect long-slit coadds from unrelated sky areas."""
from __future__ import annotations

import json
from pathlib import Path

from scorpio_pipe.stack_pointing_qc import (
    check_stack_pointing,
    _separation_arcsec,
)


def test_sc1_ngc2146sat_pointing_and_rotangle_mismatch_is_fatal():
    # Exact measured BTA SCORPIO-1 OBJ04 / OBJ12 header fields (2024-12-06).
    h1 = {"RA": "06:28:36.66", "DEC": "+77:55:42.1", "ROTANGLE": 298.4}
    h2 = {"RA": "06:28:12.42", "DEC": "+77:57:00.3", "ROTANGLE": 271.1}
    q = check_stack_pointing([h1, h2], names=["s22800604", "s22800612"])
    assert q["status"] == "REJECT"
    assert 105.0 < q["measurements"][1]["pointing_sep_arcsec"] < 115.0
    assert abs(q["measurements"][1]["rotangle_diff_deg"] - 27.3) < 1e-9
    assert len(q["issues"]) == 2
    assert q["science_spatially_verified"] is False
    assert "not" in q["interpretation"].lower()


def test_identical_header_geometry_passes_without_asserting_spatial_registration():
    h = {"RA": "23:59:59.90", "DEC": "-00:00:00.8", "ROTANGLE": 359.9}
    other = {"RA": "23:59:59.91", "DEC": "-00:00:00.8", "ROTANGLE": -0.1}
    q = check_stack_pointing([h, other])
    assert q["status"] == "PASS"
    assert q["science_spatially_verified"] is False
    assert q["measurements"][1]["pointing_sep_arcsec"] < 0.2
    assert q["measurements"][1]["rotangle_diff_deg"] < 1e-8


def test_missing_or_ambiguous_metadata_is_not_silent_pass():
    a = {"RA": "06:28:12.42", "DEC": "+77:57:00.3", "ROTANGLE": 271.1}
    b = {"RA": 97.0, "DEC": 77.95, "ROTANGLE": 271.1}
    q = check_stack_pointing([a,b])
    assert q["status"] == "UNVERIFIED"
    assert q["measurements"][1]["ra_deg"] is None
    strict = check_stack_pointing([a,b], require_complete_metadata=True)
    assert strict["status"] == "REJECT"
    assert any("Missing or ambiguous" in s for s in strict["issues"])


def test_ra_wrap_and_south_pole_geometry():
    a = {"RA":"23:59:59.90", "DEC":"-60:00:00", "ROTANGLE":2.0}
    b = {"RA":"00:00:00.10", "DEC":"-60:00:00", "ROTANGLE":2.0}
    q = check_stack_pointing([a,b])
    assert q["status"] == "PASS"
    assert q["measurements"][1]["pointing_sep_arcsec"] < 2.0
    assert _separation_arcsec((0,0),(0,0)) == 0


def test_invalid_coordinate_or_threshold_fails_closed():
    import pytest
    a = {"RA":"24:00:00", "DEC":"+77:55:42.1","ROTANGLE":"bad"}
    b = {"RA":"06:28:12.42", "DEC":"+77:55:42.1","ROTANGLE":2.0}
    q = check_stack_pointing([a,b],require_complete_metadata=True)
    assert q["status"] == "REJECT"
    with pytest.raises(ValueError,match="threshold"):
        check_stack_pointing([b,b],max_pointing_sep_arcsec=0)
    with pytest.raises(ValueError,match="names"):
        check_stack_pointing([b,b],names=["only"])


def test_stage_writes_preflight_before_first_stacked_fits(tmp_path):
    # Static contract plus numerical checks above, to catch future refactors
    # that accidentally move the safety gate after product writes.
    from scorpio_pipe.stages import stack2d
    code = Path(stack2d.__file__).read_text(encoding="utf-8")
    stage = code[code.index("def run_stack2d("):]
    assert 'if pointing_qc["status"] == "REJECT":' in stage
    assert stage.index("check_stack_pointing(") < stage.index(
        "_validate_or_raise(\n            what=\"stack2d\""
    )
    assert stage.index("stack_pointing_qc.json") < stage.index(
        'if pointing_qc["status"] == "REJECT":'
    )
