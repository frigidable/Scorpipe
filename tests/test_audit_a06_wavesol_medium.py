"""Audit A06: wavelength medium is a required categorical science invariant."""
from __future__ import annotations

import pytest

from scorpio_pipe.stages.wavesolution import (
    _validated_wavesol_medium, build_wavesolution,
)


@pytest.mark.parametrize("value", ["air", "AIR", " Air ", "vacuum", "VACUUM"])
def test_valid_explicit_wavelength_media_are_preserved(value):
    assert _validated_wavesol_medium({"wave_ref": value}) == value.strip().lower()


@pytest.mark.parametrize("value", ["", None, "unknown", "barycentric",
                                         "air/vacuum", "vac", "heliocentric"])
def test_unknown_or_empty_medium_rejected(value):
    with pytest.raises(ValueError, match="wave_ref must be explicitly"):
        _validated_wavesol_medium({"wave_ref": value})


def test_legacy_alias_is_supported_but_invalid_alias_is_rejected():
    assert _validated_wavesol_medium({"waveref": "vacuum"}) == "vacuum"
    with pytest.raises(ValueError, match="wave_ref must be explicitly"):
        _validated_wavesol_medium({"waveref": "not_known"})


def test_explicit_invalid_primary_key_cannot_be_overridden_by_valid_alias():
    with pytest.raises(ValueError, match="wave_ref must be explicitly"):
        _validated_wavesol_medium({"wave_ref": "bad", "waveref": "air"})


def test_legacy_absent_setting_keeps_air_default():
    assert _validated_wavesol_medium({}) == "air"


def test_build_stage_rejects_bad_medium_before_reading_or_writing_files(tmp_path):
    for value in ("", "not-a-medium", None):
        work = tmp_path / ("case_" + str(value))
        assert not work.exists()
        with pytest.raises(ValueError, match="wave_ref must be explicitly"):
            build_wavesolution(
                {"work_dir": str(work), "wavesol": {"wave_ref": value}},
                run_root=work,
            )
        assert not work.exists()
