"""Audit A02/A03 regression: line detections require signed significance and DQ."""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.contracts.maskbits import COSMIC, EDGE, SKYMODEL_FAIL
from scorpio_pipe.science.lines import LineSpec, measure_line, measure_lines
from scorpio_pipe.science.reddening import balmer_decrement_summary


SPEC = LineSpec("test_line", 5000.0, 8.0, (-30., -15.), (15., 30.))


def _spectrum():
    wave = np.arange(4950., 5051., dtype=float)
    continuum = np.ones_like(wave)
    var = np.ones_like(wave)
    mask = np.zeros_like(wave, dtype=np.uint16)
    return wave, continuum, var, mask


def _measure(wave, flux, var, mask=None):
    return measure_line(
        wave, flux, var, spec=SPEC, redshift=0.,
        calsys_rel=None, calsys_tell=None, calsys_abs=None,
        mask=mask,
    )


def test_positive_noise_residual_is_not_an_emission_detection():
    w, flux, var, mask = _spectrum()
    flux[(w >= 4998.) & (w <= 5002.)] += .10
    result = _measure(w, flux, var, mask)
    assert result["flux"] > 0
    assert result["snr_stat"] < 3
    assert result["detected"] is False
    assert result["reason"] == "below_detection_threshold"
    assert result["centroid"] is None
    assert result["velocity_kms"] is None


def test_negative_absorption_feature_does_not_count_as_emission():
    w, flux, var, mask = _spectrum()
    flux[(w >= 4998.) & (w <= 5002.)] -= 10.
    result = _measure(w, flux, var, mask)
    assert result["flux"] < 0
    assert result["snr_stat"] < 0
    assert result["detected"] is False


def test_unmasked_strong_line_remains_a_detection():
    w, flux, var, mask = _spectrum()
    flux += 10. * np.exp(-.5 * ((w - 5000.) / 1.5)**2)
    result = _measure(w, flux, var, mask)
    assert result["detected"] is True
    assert result["snr_stat"] > 3
    assert abs(result["centroid"] - 5000.) < .1


def test_fatal_cosmic_outlier_is_excluded_and_does_not_expand_bin_width():
    w, flux, var, mask = _spectrum()
    flux[(w >= 4998.) & (w <= 5002.)] += 10.
    mask[w == 5000.] = COSMIC
    flux[w == 5000.] = 10000.  # A huge fake emission line.
    result = _measure(w, flux, var, mask)
    assert result["detected"] is True
    assert result["n_excluded_window"] == 1
    np.testing.assert_allclose(result["flux"], 40., rtol=0, atol=1e-10)
    # The skipped bin is absent, not spuriously replaced with an oversized delta-lambda.


def test_edge_quality_bit_is_informational_not_fatal():
    w, flux, var, mask = _spectrum()
    flux[(w >= 4998.) & (w <= 5002.)] += 10.
    mask[w == 5000.] = EDGE
    result = _measure(w, flux, var, mask)
    assert result["n_excluded_window"] == 0
    np.testing.assert_allclose(result["flux"], 50., atol=1e-10)


def test_sky_model_failed_pixel_is_excluded():
    w, flux, var, mask = _spectrum()
    flux[(w >= 4998.) & (w <= 5002.)] += 10.
    flux[w == 5000.] = 10000.
    mask[w == 5000.] = SKYMODEL_FAIL
    assert _measure(w, flux, var, mask)["flux"] == pytest.approx(40.)


def test_one_sided_masked_continuum_refuses_scientific_detection():
    w, flux, var, mask = _spectrum()
    flux[(w >= 4998.) & (w <= 5002.)] += 10.
    mask[(w >= 4970.) & (w <= 4985.)] = COSMIC
    result = _measure(w, flux, var, mask)
    assert result["detected"] is False
    assert result["reason"] == "insufficient_unmasked_continuum"


def test_invalid_variance_pixel_does_not_create_undefined_weight():
    w, flux, var, mask = _spectrum()
    flux[(w >= 4998.) & (w <= 5002.)] += 10.
    flux[w == 5000.] = 10000.
    var[w == 5000.] = -1.
    result = _measure(w, flux, var, mask)
    assert result["n_excluded_window"] == 1
    assert result["flux"] == pytest.approx(40.)


def test_legacy_maskless_measure_lines_is_still_supported():
    w, flux, var, _ = _spectrum()
    flux[(w >= 4998.) & (w <= 5002.)] += 10.
    result = measure_lines(
        w, flux, var, calsys_rel=None, calsys_tell=None,
        calsys_abs=None, lines=[SPEC]
    )
    assert len(result) == 1 and result[0]["detected"] is True


def test_input_mask_shape_and_detection_threshold_are_validated():
    w, flux, var, _ = _spectrum()
    with pytest.raises(ValueError, match="Line mask"):
        _measure(w, flux, var, np.zeros(5, dtype=np.uint16))
    with pytest.raises(ValueError, match="detection_snr_min"):
        measure_line(
            w, flux, var, spec=SPEC, calsys_rel=None,
            calsys_tell=None, calsys_abs=None, detection_snr_min=0.0,
        )


def test_balmer_reddening_never_uses_explicitly_undetected_hbeta():
    result = balmer_decrement_summary({
        "Halpha": {"flux": 10., "detected": True},
        "Hbeta": {"flux": 1., "detected": False},
    })
    assert result is None
