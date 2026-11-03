"""Unmodelled shared MasterBias must not yield certified line detections."""
from astropy.io import fits
from scorpio_pipe.stages.line_measure import _qualify_lines_for_shared_bias_covariance


def _line():
    return {
        "name": "Hbeta", "flux": 12.5, "err_stat": 1.3, "snr_stat": 9.6,
        "detected": True, "upper_limit_3sigma": None, "reason": None,
    }


def test_shared_bias_uncertainty_suppresses_formal_line_detection():
    h = fits.Header()
    h["SBGSTAT"] = "SHARED"
    h["SBCOVQ"] = "UNMODEL"
    rows, blocked = _qualify_lines_for_shared_bias_covariance([_line()], h)
    assert blocked
    assert rows[0]["detected"] is False
    assert rows[0]["conditional_detection_candidate"] is True
    assert rows[0]["upper_limit_3sigma"] is None
    assert rows[0]["uncertainty_qualified"] is False
    assert rows[0]["reason"] == "uncertified_shared_masterbias_uncertainty"
    assert rows[0]["flux"] == 12.5 and rows[0]["snr_stat"] == 9.6


def test_distinct_bias_groups_do_not_trigger_specific_shared_bias_gate():
    h = fits.Header()
    h["SBGSTAT"] = "DISTINCT"
    h["SBCOVQ"] = "UNMODEL"
    rows, blocked = _qualify_lines_for_shared_bias_covariance([_line()], h)
    assert not blocked
    assert rows[0]["detected"] is True
    assert rows[0]["uncertainty_qualified"] is False
    assert rows[0]["shared_bias_detection_gate_triggered"] is False


def test_missing_group_provenance_preserves_legacy_but_does_not_certify_it():
    h = fits.Header()
    rows, blocked = _qualify_lines_for_shared_bias_covariance([_line()], h)
    assert not blocked
    assert rows[0]["detected"] is True
    assert rows[0]["uncertainty_qualified"] is None


def test_shared_bias_does_not_override_prior_partial_spectral_failure():
    h = fits.Header()
    h["SBGSTAT"] = "SHARED"
    h["SBCOVQ"] = "UNMODEL"
    row = _line()
    row["detected"] = False
    row["reason"] = "partial_aperture_or_spectral_coverage"
    rows, blocked = _qualify_lines_for_shared_bias_covariance([row], h)
    assert blocked
    assert rows[0]["detected"] is False
    assert rows[0]["reason"] == "partial_aperture_or_spectral_coverage"
    assert rows[0]["uncertainty_status"] == "SHARED_MASTERBIAS_COVARIANCE_UNMODELLED"
