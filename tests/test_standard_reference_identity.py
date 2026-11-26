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


@pytest.mark.parametrize("name", [None, "", "   ", "!!!"])
def test_missing_or_unusable_identity_is_not_guessed_even_with_one_standard(name):
    one = {"standards": {"bd284211": LIBRARY["standards"]["bd284211"]}}
    assert find_standard_record(one, name) == (None, None)


def test_ambiguous_aliases_are_rejected_instead_of_resolved_by_yaml_order():
    library = {
        "standards": {
            "one": {"display_name": "Star One", "aliases": ["HZ 44"]},
            "two": {"display_name": "Star Two", "aliases": ["HZ44"]},
        }
    }
    assert find_standard_record(library, "HZ 44") == (None, None)
    assert find_standard_record(library, "HZ44") == (None, None)


def test_non_mapping_standard_entries_are_never_selected():
    library = {"standards": {
        "broken": "not a record",
        "valid": {"display_name": "A good standard"},
    }}
    assert find_standard_record(library, "broken") == (None, None)
    assert find_standard_record(library, "valid")[0] == "valid"


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

    with pytest.raises(ValueError, match="UNKNOWN_SPECTROPHOTOMETRIC_STANDARD"):
        sensfunc._load_standard_reference({}, selected_name="Feige 34")


def test_sensfunc_alias_resolves_to_validated_reference_without_fallback(tmp_path, monkeypatch):
    from scorpio_pipe.calib import sensfunc
    library = tmp_path / "standards" / "library.yaml"
    library.parent.mkdir()
    library.write_text(
        "standards:\n"
        "  bd284211:\n"
        "    display_name: BD+28 4211\n"
        "    aliases:\n"
        "      - BD+28d4211\n"
        "    scientific_status: validated\n"
        "    source: CALSPEC-source-verified-by-observer\n"
        "    units: f_lambda\n"
        "    wavelength_convention: vacuum\n"
        "    table: standards/bd284211_sed.csv\n",
        encoding="utf-8",
    )
    reference = library.parent / "bd284211_sed.csv"
    reference.write_text(
        "wavelength_A,flux\n"
        "4000,1e-13\n"
        "5000,1e-13\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sensfunc, "reference_path", lambda cfg, **kw: library)
    _lib, ref, units, medium = sensfunc._load_standard_reference(
        {}, selected_name="BD+28d4211"
    )
    assert ref == reference.resolve()
    assert units == "f_lambda"
    assert medium == "vacuum"
