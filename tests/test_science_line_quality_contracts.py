"""Line measurements must obey SCI/VAR/MASK, wavelength medium, and detection S/N.

A positive noise excursion must never become a catalog detection merely
because some continuum-subtracted pixels are positive. Unresolved Halpha/
[NII] blended-window fluxes cannot be used as a quantitative Balmer decrement.
"""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.calib.common import air_to_vacuum
from scorpio_pipe.science.lines import LineSpec, measure_line, measure_lines
from scorpio_pipe.science.reddening import balmer_decrement_summary

OIII = LineSpec(
    name="[OIII]5007",
    rest_wavelength=5006.84,
    window_half_width=12.0,
    continuum_left=(-35.0, -15.0),
    continuum_right=(15.0, 35.0),
)


def _spectrum(*, centre=5006.84, amplitude=0.0):
    wave = np.arange(4960.0, 5055.01, 0.25)
    flux = np.ones_like(wave)
    flux += amplitude * np.exp(-0.5 * ((wave - centre) / 1.0) ** 2)
    var = np.ones_like(wave) * 0.25
    return wave, flux, var


def _measure(w, f, v, *, mask=None, waveref="air", snr=3.0):
    return measure_line(
        w, f, v,
        spec=OIII,
        calsys_rel=None,
        calsys_tell=None,
        calsys_abs=None,
        mask=mask,
        waveref=waveref,
        detection_snr=snr,
    )


def test_real_emission_line_is_detected_from_signed_signal_to_noise() -> None:
    w, f, v = _spectrum(amplitude=15)
    row = _measure(w, f, v)
    assert row["detected"] is True
    assert row["snr_stat"] > 3.0
    assert row["upper_limit_3sigma"] is None
    assert row["centroid"] == pytest.approx(5006.84, abs=0.1)
    assert row["waveref"] == "air"


def test_positive_noise_spike_is_not_a_detection() -> None:
    w, f, v = _spectrum()
    f[np.argmin(abs(w - 5006.84))] += 0.05
    row = _measure(w, f, v)
    assert row["flux"] > 0
    assert row["detected"] is False
    assert row["snr_stat"] < 3
    assert row["centroid"] is None
    assert row["velocity_kms"] is None
    assert row["sigma"] is None
    assert row["sigma_kms"] is None
    assert row["fwhm"] is None
    assert row["upper_limit_3sigma"] > row["flux"]
    import json

    restored = json.loads(json.dumps(row))
    assert restored["detected"] is False
    assert restored["centroid"] is None
    assert restored["velocity_kms"] is None


def test_negative_absorption_does_not_become_emission() -> None:
    w, f, v = _spectrum(amplitude=-10)
    row = _measure(w, f, v)
    assert row["flux"] < 0
    assert row["detected"] is False
    assert row["snr_stat"] < 0
    assert row["centroid"] is None
    assert row["velocity_kms"] is None
    assert row["sigma_kms"] is None


def test_finite_cosmic_spike_is_rejected_by_integer_mask() -> None:
    w, f, v = _spectrum()
    target = np.argmin(abs(w - 5006.84))
    f[target] += 5000.0
    mask = np.zeros(w.shape, dtype=np.uint16)
    mask[target] = 1
    bad = _measure(w, f, v)
    good = _measure(w, f, v, mask=mask)
    assert bad["detected"] is True  # Legacy failure mode is reproducible.
    assert good["detected"] is False
    assert good["n_masked_in_window"] == 1
    assert good["flux"] == pytest.approx(0, abs=1e-12)


def test_nonpositive_variance_is_never_treated_as_zero_noise() -> None:
    w, f, v = _spectrum()
    f[np.argmin(abs(w - 5006.84))] += 5000.0
    v[:] = 0.0
    row = _measure(w, f, v)
    assert row["detected"] is False
    assert row["reason"] == "insufficient_valid_unmasked_pixels"


@pytest.mark.parametrize("medium", ["air", "vacuum"])
def test_observed_line_center_and_velocity_use_declared_wavelength_medium(
    medium: str,
) -> None:
    rest = 5006.84 if medium == "air" else float(
        air_to_vacuum(np.array([5006.84]))[0]
    )
    w, f, v = _spectrum(centre=rest, amplitude=20.0)
    row = _measure(w, f, v, waveref=medium)
    assert row["detected"] is True
    assert row["centroid"] == pytest.approx(rest, abs=0.15)
    assert row["velocity_kms"] == pytest.approx(0, abs=10.0)


@pytest.mark.parametrize("medium", ["unknown", "", "Vac"])
def test_science_lines_reject_unknown_wavelength_reference(medium: str) -> None:
    w, f, v = _spectrum(amplitude=10)
    with pytest.raises(ValueError, match="WAVEREF"):
        _measure(w, f, v, waveref=medium)


def test_science_lines_require_nontrivial_detection_threshold() -> None:
    w, f, v = _spectrum(amplitude=10)
    with pytest.raises(ValueError, match=">= 3 sigma"):
        _measure(w, f, v, snr=2.0)


def test_balmer_decrement_requires_two_detected_unblended_lines() -> None:
    ha = {"flux": 286.0, "detected": True, "blend_warning": False}
    hb = {"flux": 100.0, "detected": True, "blend_warning": False}
    assert balmer_decrement_summary({"Halpha": ha, "Hbeta": hb}) is not None
    assert balmer_decrement_summary({
        "Halpha": {**ha, "blend_warning": True},
        "Hbeta": hb,
    }) is None
    assert balmer_decrement_summary({
        "Halpha": ha,
        "Hbeta": {**hb, "detected": False},
    }) is None


def test_halfa_window_is_explicitly_tagged_as_unresolved_blend() -> None:
    from scorpio_pipe.science.lines import DEFAULT_LINES

    w = np.arange(6500.0, 6650.0, 0.25)
    f = 1.0 + 20.0 * np.exp(-0.5 * ((w - 6562.8) / 1.0) ** 2)
    v = np.ones_like(w) * 0.25
    rows = measure_lines(
        w, f, v,
        calsys_rel=None, calsys_tell=None, calsys_abs=None,
        lines=[s for s in DEFAULT_LINES if s.name == "Halpha"],
    )
    assert rows[0]["detected"] is True
    assert rows[0]["blend_warning"] is True
    assert rows[0]["flux_kind"] == "window_integral_not_deblended"
