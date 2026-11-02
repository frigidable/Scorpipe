"""The synthetic Golden night must follow the same run-layout contract as real data."""

from __future__ import annotations

from pathlib import Path

from scorpio_pipe.run_passport import ensure_run_passport
from scorpio_pipe.run_validate import validate_run_dir
from scorpio_pipe.work_layout import ensure_work_layout
from tests.golden.mini_night import make_mini_night


def test_mini_night_fixture_matches_strict_run_layout(tmp_path: Path) -> None:
    mn = make_mini_night(tmp_path)
    ensure_work_layout(mn.work_dir)
    passport = ensure_run_passport(mn.work_dir)

    validated = validate_run_dir(mn.work_dir, strict=True)
    assert validated.ok
    assert validated.night_date == "2025-12-16"
    assert validated.object_name == "mini_night"
    assert validated.disperser == "VPHG1200@540"
    assert validated.run_id == "01"
    assert passport["night_date"] == validated.night_date
    assert passport["object"] == validated.object_name
    assert passport["disperser"] == validated.disperser
    assert passport["run_id"] == validated.run_id



def test_mini_night_setup_readout_matches_all_synthetic_fits(tmp_path: Path) -> None:
    from scorpio_pipe.frame_signature import FrameSignature

    mn = make_mini_night(tmp_path)
    setup = mn.cfg["frames"]["__setup__"]
    expected = FrameSignature.from_setup(setup)
    assert expected.readout == "nodeE_rate185.0_gain0.6200"

    for role in ("bias", "flat", "neon", "obj"):
        for name in mn.cfg["frames"][role]:
            observed = FrameSignature.from_path(name)
            assert observed.is_compatible_with(expected), (
                role, name, observed.describe(), expected.describe()
            )

    # A different CCD readout must still fail the strict comparison.
    wrong = FrameSignature.from_setup(
        {**setup, "readout": "nodeE_rate225.0_gain3.5000"}
    )
    assert not FrameSignature.from_path(mn.cfg["frames"]["flat"][0]).is_compatible_with(wrong)
