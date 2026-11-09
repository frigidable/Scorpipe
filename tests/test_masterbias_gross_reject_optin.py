"""Scoped gross-transient MasterBias variance fix; default remains unchanged.

This is engineering validation of an opt-in finite-N Gaussian approximation.
It is not a claim of publication-grade ADC-quantization coverage or overscan.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from scorpio_pipe.bias_uncertainty import (
    master_bias_estimator_variance,
    normal_median_variance_factor,
)


def test_clean_opt_in_median_matches_default_science_and_variance():
    rng = np.random.default_rng(198224)
    raw = 999.0 + rng.normal(0.0, 3.0, (20, 5, 9))
    sci_old, var_old, m_old = master_bias_estimator_variance(
        raw, combine="median"
    )
    sci_new, var_new, m_new = master_bias_estimator_variance(
        raw, combine="median", gross_reject_adu=50.0
    )
    assert np.array_equal(sci_old, sci_new)
    assert np.allclose(var_old, var_new, rtol=1e-6, atol=1e-7)
    assert m_old["variance_model"] == "GAUSSMED"
    assert m_new["variance_model"] == "GROSSMED"
    assert m_new["gross_rejected_samples"] == 0
    assert m_new["gross_affected_pixels"] == 0
    assert m_new["n_effective_min"] == 20
    assert not m_new["invalid_support"].any()


def test_one_very_bright_outlier_no_longer_inflates_variance():
    rng = np.random.default_rng(200615)
    raw = 999.0 + rng.normal(0.0, 3.0, (20, 2, 3))
    raw[0, 0, 0] += 17_166.0
    _, old_var, _ = master_bias_estimator_variance(
        raw, combine="median"
    )
    sci, var, meta = master_bias_estimator_variance(
        raw, combine="median", gross_reject_adu=50.0
    )
    expected = np.median(raw[1:, 0, 0])
    expected_var = (
        np.var(raw[1:, 0, 0], ddof=1)
        * normal_median_variance_factor(19)
    )
    assert sci[0, 0] == pytest.approx(expected, abs=1e-4)
    assert var[0, 0] == pytest.approx(expected_var, rel=2e-6)
    assert old_var[0, 0] > 1000.0
    assert var[0, 0] < 10.0
    assert meta["gross_rejected_samples"] == 1
    assert meta["gross_affected_pixels"] == 1
    assert meta["n_effective_min"] == 19
    assert not meta["invalid_support"].any()


def test_one_bad_first_row_rejects_just_anomalous_samples():
    raw = np.full((20, 3, 4), 999., dtype=np.float64)
    raw[0, 0, :] += 559
    sci, var, meta = master_bias_estimator_variance(
        raw, combine="median", gross_reject_adu=50.0
    )
    assert np.array_equal(sci, np.full((3, 4), 999., dtype=np.float32))
    assert np.all(var == 0.0)
    assert meta["gross_rejected_samples"] == 4
    assert meta["gross_affected_pixels"] == 4
    assert meta["n_effective_min"] == 19


def test_bad_effective_support_is_explicitly_flagged():
    raw = np.array([[[0.0]], [[1000.0]], [[2000.0]]])
    sci, var, meta = master_bias_estimator_variance(
        raw, combine="median", gross_reject_adu=50.0
    )
    assert sci[0, 0] == pytest.approx(0.0)
    assert var[0, 0] == pytest.approx(0.0)
    assert meta["invalid_support"][0, 0]
    assert meta["n_effective_min"] == 1
    assert meta["gross_rejected_samples"] == 2


@pytest.mark.parametrize("bad", [-1.0, np.inf, np.nan])
def test_gross_reject_invalid_threshold_fails_closed(bad):
    with pytest.raises(ValueError, match="gross_reject_adu"):
        master_bias_estimator_variance(
            np.zeros((3, 2, 2)), combine="median", gross_reject_adu=bad
        )


@pytest.mark.parametrize("combine,sigma_clip", [
    ("mean", 0.0), ("median", 3.0),
])
def test_gross_median_cannot_be_combined_with_other_estimator(combine, sigma_clip):
    with pytest.raises(ValueError, match="median|clipping"):
        master_bias_estimator_variance(
            np.zeros((3, 2, 2)), combine=combine,
            sigma_clip=sigma_clip, gross_reject_adu=50.0,
        )


def test_legacy_median_sigma_clip_fails_instead_of_being_silently_ignored():
    with pytest.raises(ValueError, match="refusing to silently ignore"):
        master_bias_estimator_variance(
            np.zeros((20, 2, 3)), combine="median", sigma_clip=3.0,
        )


def test_median_quantized_integer_data_returns_finite_nonnegative_var():
    rng = np.random.default_rng(18487)
    raw = np.rint(999.0 + rng.normal(0, 3.0, (20, 4, 5)))
    raw[7, 0, 0] += 559
    sci, var, meta = master_bias_estimator_variance(
        raw, combine="median", gross_reject_adu=50.0
    )
    assert np.isfinite(sci).all()
    assert np.isfinite(var).all()
    assert np.all(var >= 0.0)
    assert meta["gross_rejected_samples"] == 1


def test_stage_provenance_records_explicit_gross_mode(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    from scorpio_pipe.stages.calib import build_superbias
    from tests.golden.mini_night import make_mini_night

    mn = make_mini_night(tmp_path)
    mn.cfg.setdefault("calib", {})["bias_gross_reject_adu"] = 50.0
    stage = mn.work_dir / "03_bias"
    output = build_superbias(
        mn.cfg, run_root=mn.work_dir, stage_dir=stage,
        allow_legacy_mirrors=False,
    )
    assert output.is_file()
    info = json.loads((stage / "master_bias_index.json").read_text(encoding="utf8"))
    assert info["gross_reject_adu"] == pytest.approx(50.0)
    assert info["groups"][0]["variance"]["model"] == "GROSSMED"
    assert info["groups"][0]["variance"]["gross_reject_adu"] == pytest.approx(50.0)
    assert "PROVISIONAL" in info["groups"][0]["variance"]["assumptions"]
    vp = mn.work_dir / info["groups"][0]["files"]["var"]
    with fits.open(vp) as hdu:
        assert hdu[0].header["SB_VMOD"] == "GROSSMED"
        assert hdu[0].header["SB_GROS"] == pytest.approx(50.0)
        assert int(hdu[0].header["SB_GCNT"]) >= 0
        assert np.isfinite(hdu[0].data).all()


def test_more_than_ten_percent_gross_outliers_invalidates_pixel():
    # With N=20, 1-2 gross transients are supported; 3 are no longer
    # safely characterized as a rare contamination episode.
    raw = np.full((20, 1, 2), 999.0)
    raw[0:3, 0, 0] = 16000.0
    sci, var, meta = master_bias_estimator_variance(
        raw, combine="median", gross_reject_adu=50.0
    )
    assert meta["gross_min_required_support"] == 18
    assert meta["invalid_support"][0, 0]
    assert not meta["invalid_support"][0, 1]
    assert sci[0, 0] == 0.0
    assert var[0, 0] == 0.0
    assert meta["gross_rejected_samples"] == 3


@pytest.fixture
def gross_indexed_masterbias(tmp_path):
    """The real producer's v3 index and FITS products, not a mocked schema."""
    fits = pytest.importorskip("astropy.io.fits")
    from scorpio_pipe.stages.calib import build_superbias
    from tests.golden.mini_night import make_mini_night

    mn = make_mini_night(tmp_path)
    mn.cfg.setdefault("calib", {})["bias_gross_reject_adu"] = 50.0
    stage = mn.work_dir / "03_bias"
    build_superbias(
        mn.cfg, run_root=mn.work_dir, stage_dir=stage,
        allow_legacy_mirrors=False,
    )
    index = json.loads((stage / "master_bias_index.json").read_text(encoding="utf8"))
    group = index["groups"][0]
    var_file = mn.work_dir / group["files"]["var"]
    return index, group, var_file, mn.work_dir


def test_grossmed_consumer_accepts_producer_estimator_provenance(
    gross_indexed_masterbias,
):
    from scorpio_pipe.stages.calib import _require_estimator_masterbias_index

    index, group, var_file, work_dir = gross_indexed_masterbias
    fits = pytest.importorskip("astropy.io.fits")
    with fits.open(var_file) as hdu:
        assert hdu[0].header["SB_VMOD"] == "GROSSMED"
        assert "SB_VFAC" not in hdu[0].header  # Per-pixel finite-N factors
    _require_estimator_masterbias_index(index, group, work_dir)


@pytest.mark.parametrize("mutate", [
    "missing_gross_cutoff",
    "incorrect_index_minimum",
    "fake_fixed_factor",
    "missing_provisional_assumption",
    "insufficient_support",
])
def test_grossmed_consumer_rejects_inconsistent_or_unsafe_provenance(
    gross_indexed_masterbias, mutate,
):
    from scorpio_pipe.stages.calib import _require_estimator_masterbias_index

    index, group, _, work_dir = gross_indexed_masterbias
    if mutate == "missing_gross_cutoff":
        del group["variance"]["gross_reject_adu"]
    elif mutate == "incorrect_index_minimum":
        group["variance"]["min_effective_n"] -= 1
    elif mutate == "fake_fixed_factor":
        group["variance"]["factor"] = 0.1
    elif mutate == "missing_provisional_assumption":
        group["variance"]["assumptions"] = "independent_gaussian_read_noise"
    elif mutate == "insufficient_support":
        group["variance"]["bad_support_pixels"] = 1

    with pytest.raises(RuntimeError, match="Gross-rejected"):
        _require_estimator_masterbias_index(index, group, work_dir)


@pytest.mark.parametrize("card,new_value", [
    ("SB_GROS", 999.0),
    ("SB_NMIN", 1),
    ("SB_GCNT", 999),
    ("SB_METH", "mean"),
])
def test_grossmed_consumer_rejects_fits_header_mismatch(
    gross_indexed_masterbias, card, new_value,
):
    from scorpio_pipe.stages.calib import _require_estimator_masterbias_index

    index, group, var_file, work_dir = gross_indexed_masterbias
    fits = pytest.importorskip("astropy.io.fits")
    with fits.open(var_file, mode="update") as hdu:
        hdu[0].header[card] = new_value
    with pytest.raises(RuntimeError, match="Gross-rejected|sample-count"):
        _require_estimator_masterbias_index(index, group, work_dir)
