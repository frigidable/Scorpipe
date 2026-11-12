"""Real BTA 2024-12-06: run identity must come from selected science setup."""
from __future__ import annotations

from datetime import date
import pytest

from scorpio_pipe.run_root import (
    night_date_from_header,
    disperser_key_from_header,
    get_or_create_run_root,
)


SCI = {
    "OBJECT": "N2146sat", "DISPERSE": "VPHG1200B",
    "DATE-OBS": "2024/06/12", "DATE": "2024-12-06T15:55:05",
    "BINNING": "1x2", "GAIN": 0.9,
}


def test_scorpio_noniso_date_obs_uses_unambiguous_date():
    assert night_date_from_header(SCI) == date(2024, 12, 6)


def test_scorpio_after_midnight_calibration_belongs_to_previous_night():
    h = dict(SCI, **{
        "DATE-OBS": "2024/07/12",
        "DATE": "2024-12-07T04:14:18",
    })
    assert night_date_from_header(h) == date(2024, 12, 6)


def test_disperse_is_a_first_class_scorpio_disperser_header_key():
    assert disperser_key_from_header(SCI) == "VPHG1200B"


def test_selected_scorpio_run_folder_is_not_unknown_or_current_day(tmp_path):
    result = get_or_create_run_root(
        tmp_path, dataset_path=tmp_path / "s241206", headers=SCI,
        create=False,
    )
    assert result == tmp_path / "06_12_2024" / "N2146sat_VPHG1200B_01"
    assert not result.exists()


def test_science_run_must_not_silently_use_current_date(tmp_path):
    with pytest.raises(ValueError, match="Cannot establish night date"):
        get_or_create_run_root(
            tmp_path, dataset_path=None,
            headers={"OBJECT": "N2146sat", "DISPERSE": "VPHG1200B"},
            create=True,
        )
    assert not list(tmp_path.iterdir())


def test_existing_run_number_increments_only_with_same_signature(tmp_path):
    parent = tmp_path / "06_12_2024"
    (parent / "N2146sat_VPHG1200B_01").mkdir(parents=True)
    (parent / "CSS0854_VPHG1200B_20").mkdir()
    (parent / "N2146sat_VPHG1200R_90").mkdir()
    nxt = get_or_create_run_root(tmp_path, None, SCI, create=False)
    assert nxt.name == "N2146sat_VPHG1200B_02"
    assert not nxt.exists()
