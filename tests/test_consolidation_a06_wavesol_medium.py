"""A06: declared wavelength medium must fail closed at wavesolution entry."""
from __future__ import annotations

import pytest

from scorpio_pipe.stages.wavesolution import (
    _validated_wavesol_medium, build_wavesolution,
)


@pytest.mark.parametrize("value,expected", [
    ("air", "air"), (" AIR ", "air"), ("vacuum", "vacuum"),
    ("VACUUM", "vacuum"),
])
def test_valid_declared_media(value, expected):
    assert _validated_wavesol_medium({"wave_ref": value}) == expected


@pytest.mark.parametrize("value", [
    None, "", "unknown", "barycentric", "air/vacuum", "vac",
    "heliocentric",
])
def test_explicit_unknown_media_cannot_silently_default(value):
    with pytest.raises(ValueError, match="wave_ref must be explicitly"):
        _validated_wavesol_medium({"wave_ref": value})


def test_primary_key_precedes_legacy_alias():
    with pytest.raises(ValueError, match="wave_ref must be explicitly"):
        _validated_wavesol_medium({"wave_ref": "bad", "waveref": "air"})


def test_legacy_alias_and_missing_key_contract():
    assert _validated_wavesol_medium({"waveref": " VaCuUm "}) == "vacuum"
    assert _validated_wavesol_medium({}) == "air"


def test_invalid_medium_blocks_before_output_directory(tmp_path):
    root = tmp_path / "fresh-run"
    with pytest.raises(ValueError, match="wave_ref must be explicitly"):
        build_wavesolution(
            {"work_dir": str(root), "wavesol": {"wave_ref": "unknown"}},
            run_root=root,
        )
    assert not root.exists()
