"""Golden sky residual noise is calibrated to unity within finite-sample scatter.

The old asymmetric [1,10] band rejected a statistically normal 0.984 while
allowing up to 10x noise-model mismatch. Keep the acceptance band two-sided
and much tighter; this is a Golden reference, not a production QC override.
"""
from __future__ import annotations

import json

import pytest

from tests.test_golden_mini_night_e2e import GOLDEN, _assert_metric


def _sky_metric_reference() -> dict:
    doc = json.loads(GOLDEN.read_text(encoding="utf-8"))
    bound = doc["metrics"]["sky_robust_sigma_r_after_median"]
    assert bound == {"min": 0.85, "max": 1.15}
    return bound


@pytest.mark.parametrize("measured", [0.983963, 1.0, 0.85, 1.15])
def test_golden_accepts_finite_sample_sky_noise_around_unity(
    measured: float,
) -> None:
    _assert_metric(
        "sky_robust_sigma_r_after_median", measured, _sky_metric_reference(), {}
    )


@pytest.mark.parametrize("measured", [0.84, 1.16, 0.5, 1.5, 10.0, float("nan")])
def test_golden_rejects_significantly_miscalibrated_sky_noise(
    measured: float,
) -> None:
    with pytest.raises(AssertionError):
        _assert_metric(
            "sky_robust_sigma_r_after_median", measured, _sky_metric_reference(), {}
        )
