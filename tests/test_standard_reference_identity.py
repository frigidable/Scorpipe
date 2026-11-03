"""Spectrophotometric standards must never be silently misidentified."""

import pytest

from scorpio_pipe.calib.common import find_standard_record
from scorpio_pipe.calib import sensfunc


LIBRARY = {
    "standards": {
        "bd284211": {
            "display_name": "BD+28 4211",
            "aliases": ["BD+28d4211", "BD 28 4211"],
            "table": "standards/bd284211.csv",
        },
        "hz44": {
            "display_name": "HZ 44",
            "aliases": ["HZ44"],
            "table": "standards/hz44.csv",
        },
    }
}


@pytest.mark.parametrize("name,expected", [
    ("BD+28 4211", "bd284211"),
    ("bd284211", "bd284211"),
    ("BD+28d4211", "bd284211"),
    ("HZ44", "hz44"),
    ("HZ 44", "hz44"),
])
def test_known_standard_name_or_curated_alias(name, expected):
    key, rec = find_standard_record(LIBRARY, name)
    assert key == expected
    assert rec == LIBRARY["standards"][expected]


@pytest.mark.parametrize("name", ["Feige 34", "HZ 4", "G191B2B"])
def test_unmatched_name_never_falls_back_to_first_standard(name):
    key, rec = find_standard_record(LIBRARY, name)
    assert key is None
    assert rec is None


def test_missing_name_with_multi_standard_library_is_ambiguous():
    assert find_standard_record(LIBRARY, None) == (None, None)


def test_missing_name_retains_legacy_single_reference_fallback():
    one = {"standards": {"bd284211": LIBRARY["standards"]["bd284211"]}}
    key, rec = find_standard_record(one, None)
    assert key == "bd284211"
    assert rec["display_name"] == "BD+28 4211"


def test_sensitivity_reference_loader_rejects_unknown_selected_standard(
    tmp_path, monkeypatch
):
    library = tmp_path / "library.yaml"
    library.write_text(
        "standards:\n"
        "  bd284211:\n"
        "    display_name: BD+28 4211\n"
        "    table: standards/bd284211.csv\n"
        "    units: f_lambda\n"
        "    wavelength_convention: air\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sensfunc, "reference_path", lambda cfg, **kw: library
    )

    with pytest.raises(ValueError, match="No matching spectrophotometric reference"):
        sensfunc._load_standard_reference({}, selected_name="Feige 34")
