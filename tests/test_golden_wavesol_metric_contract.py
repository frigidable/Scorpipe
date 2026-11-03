"""Golden numeric assertions must follow the actual Wavesolution JSON schema."""

from __future__ import annotations

import pytest

from tests.test_golden_mini_night_e2e import (
    _assert_metric,
    _read_wavesol_1d_rms_A,
)


def test_golden_reads_top_level_rms_and_retains_scientific_bound() -> None:
    doc = {"rms_A": 0.012, "n_pairs": 7, "n_used": 7, "deg": 4}
    rms = _read_wavesol_1d_rms_A(doc)
    assert rms == pytest.approx(0.012)
    _assert_metric("wavesol_1d_rms_A", rms, {"min": 0.0, "max": 0.05}, {})


@pytest.mark.parametrize(
    "document, reason",
    [
        ({"fit": {"rms_A": 0.01}, "n_pairs": 7, "n_used": 7}, "top-level rms_A"),
        ({"rms_A": None, "n_pairs": 7, "n_used": 7}, "invalid"),
        ({"rms_A": "not a number", "n_pairs": 7, "n_used": 7}, "invalid"),
        ({"rms_A": float("nan"), "n_pairs": 7, "n_used": 7}, "invalid rms_A"),
        ({"rms_A": float("inf"), "n_pairs": 7, "n_used": 7}, "invalid rms_A"),
        ({"rms_A": -0.1, "n_pairs": 7, "n_used": 7}, "invalid rms_A"),
        ({"rms_A": 0.01, "n_pairs": 7, "n_used": 4}, "insufficient"),
        ({"rms_A": 0.01, "n_pairs": 7, "n_used": 8}, "insufficient"),
        ({"rms_A": 0.01, "n_pairs": None, "n_used": 7}, "invalid"),
    ],
)
def test_golden_rejects_missing_or_invalid_wavesolution_rms(
    document: dict, reason: str,
) -> None:
    with pytest.raises(AssertionError, match=reason):
        _read_wavesol_1d_rms_A(document)


def test_golden_does_not_relax_wavesolution_scientific_rms_threshold() -> None:
    doc = {"rms_A": 0.051, "n_pairs": 7, "n_used": 7}
    with pytest.raises(AssertionError, match="expected in"):
        _assert_metric(
            "wavesol_1d_rms_A",
            _read_wavesol_1d_rms_A(doc),
            {"min": 0.0, "max": 0.05},
            {},
        )
