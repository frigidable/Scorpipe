"""Cross-stage Sky -> Linearize flexure provenance and shift admission.

The canonical done.json writer nests stage-specific scientific metadata under
extra. It must not be confused with unverified legacy top-level fields.
"""
from __future__ import annotations

import pytest

from scorpio_pipe.stages.linearize import (
    _has_independent_flexure_reference,
    _read_sky_flexure_summary,
)


def test_linearize_reads_canonical_sky_extra_flexure_not_top_level() -> None:
    current = {
        "schema": "scorpio_pipe.done_json",
        "extra": {
            "flexure": {
                "verification": "unverified",
                "reference_kind": "none",
                "reference_id": None,
                "delta_A_median": None,
                "self_fit_shift_A": 1.3,
            }
        },
    }
    assert _read_sky_flexure_summary(current) == current["extra"]["flexure"]
    assert _has_independent_flexure_reference(
        _read_sky_flexure_summary(current), n_exposures=1
    ) is False


def test_new_extra_metadata_takes_precedence_over_stale_legacy_header() -> None:
    doc = {
        "flexure": {
            "verification": "independent_reference",
            "reference_id": "unsafe-stale-template",
            "delta_A_median": 2.0,
        },
        "extra": {
            "flexure": {
                "verification": "unverified",
                "reference_id": None,
                "delta_A_median": None,
            }
        },
    }
    summary = _read_sky_flexure_summary(doc)
    assert summary is doc["extra"]["flexure"]
    assert not _has_independent_flexure_reference(summary, n_exposures=1)


@pytest.mark.parametrize("document", [
    None, {}, [], {"extra": None},
    {"extra": {"flexure": None}},
    {"extra": {"flexure": "0.0"}},
])
def test_malformed_or_absent_sky_flexure_is_not_verified(document) -> None:
    summary = _read_sky_flexure_summary(document)
    assert summary is None
    assert _has_independent_flexure_reference(summary, n_exposures=1) is False


@pytest.mark.parametrize("n_exposures", [2, 3, 7])
def test_global_median_shift_cannot_qualify_multiexposure_stack(
    n_exposures: int,
) -> None:
    summary = {
        "verification": "independent_reference",
        "reference_id": "verified-sky-template",
        "delta_A_median": 0.3,
    }
    assert not _has_independent_flexure_reference(
        summary, n_exposures=n_exposures
    )


def test_single_exposure_only_accepts_explicit_independent_reference() -> None:
    good = {
        "verification": "independent_reference",
        "reference_id": "lab-sky-template-v1",
        "delta_A_median": 0.25,
    }
    assert _has_independent_flexure_reference(good, n_exposures=1)
    assert not _has_independent_flexure_reference(
        {**good, "verification": "unverified"}, n_exposures=1
    )
    assert not _has_independent_flexure_reference(
        {**good, "reference_id": ""}, n_exposures=1
    )
