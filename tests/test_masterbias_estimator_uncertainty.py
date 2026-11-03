"""MasterBias VAR must represent the estimator, not individual-frame noise."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from scorpio_pipe.bias_uncertainty import (
    master_bias_estimator_variance,
    normal_median_variance_factor,
)
from scorpio_pipe.stages.calib import build_superbias
from tests.golden.mini_night import make_mini_night


@pytest.mark.parametrize(
    "n, expected",
    [
        (2, 0.5),
        (3, 0.4486711045782159),
        (4, 0.2981996184753104),
        (5, 0.2868336616058763),
        (6, 0.214742666581499),
        (9, 0.16610128135888091),
        (10, 0.13832643583621765),
    ],
)
def test_gaussian_median_finite_sample_variance_factors(
    n: int, expected: float,
) -> None:
    assert normal_median_variance_factor(n) == pytest.approx(
        expected, rel=2e-4,
    )


def test_large_n_median_is_documented_asymptotic() -> None:
    from math import pi

    assert normal_median_variance_factor(65) == pytest.approx(pi / 130)


@pytest.mark.parametrize("n", [3, 4, 9])
@pytest.mark.parametrize("method", ["mean", "median"])
def test_estimator_var_matches_empirical_monte_carlo(
    n: int, method: str,
) -> None:
    # Each independent spatial pixel is a Monte Carlo replication of
    # a *constant* electronic bias pedestal plus Gaussian CCD read noise.
    rng = np.random.default_rng(18046 + n)
    sigma = 5.0
    raw = 1000.0 + rng.normal(0, sigma, size=(n, 1, 60000))
    sci, var, meta = master_bias_estimator_variance(
        raw, combine=method,
    )
    empirical = float(np.var(sci, ddof=1))
    reported = float(np.mean(var))
    theoretical = (
        sigma**2 / n if method == "mean"
        else sigma**2 * normal_median_variance_factor(n)
    )
    assert reported == pytest.approx(theoretical, rel=0.04)
    assert reported == pytest.approx(empirical, rel=0.06)
    assert meta["independent_gaussian_assumption"] is True
    assert meta["n_effective_min"] == n


def test_mean_var_is_unbiased_sample_variance_over_n() -> None:
    stack = np.array([[[0.0, 2.0]], [[2.0, 4.0]], [[4.0, 6.0]]])
    sci, var, meta = master_bias_estimator_variance(
        stack, combine="mean"
    )
    assert np.allclose(sci, [[2.0, 4.0]])
    assert np.allclose(var, [[4.0 / 3.0, 4.0 / 3.0]])
    assert meta["variance_model"] == "MEANSEM"
    assert meta["variance_factor"] == pytest.approx(1.0 / 3.0)


def test_sigma_clipped_mean_records_approximation_and_support() -> None:
    stack = np.array([
        [[0.0, 0.0]], [[0.0, 100.0]],
        [[0.0, 200.0]], [[0.0, 300.0]],
        [[100.0, 400.0]],
    ])
    sci, var, meta = master_bias_estimator_variance(
        stack, combine="mean", sigma_clip=3.0,
    )
    assert meta["variance_model"] == "CLIPAPPX"
    assert not meta["invalid_support"].any()
    assert meta["n_effective_min"] >= 2
    assert np.all(np.isfinite(var) & (var >= 0))


def test_sigma_clipping_single_inlier_cannot_masquerade_as_valid() -> None:
    stack = np.array([[[0.0]], [[100.0]], [[200.0]]])
    _, _, meta = master_bias_estimator_variance(
        stack, combine="mean", sigma_clip=0.01,
    )
    assert meta["invalid_support"][0, 0]
    assert meta["n_effective_min"] == 1


@pytest.mark.parametrize("method", ["mean", "median"])
def test_master_bias_requires_more_than_one_source_frame(method: str) -> None:
    with pytest.raises(ValueError, match=">=2"):
        master_bias_estimator_variance(
            np.zeros((1, 4, 5)), combine=method,
        )


def test_nonfinite_and_unknown_method_fail_closed() -> None:
    bad = np.zeros((3, 2, 2))
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        master_bias_estimator_variance(bad, combine="median")
    with pytest.raises(ValueError, match="Unknown"):
        master_bias_estimator_variance(np.zeros((3, 2, 2)), combine="weighted")


def test_masterbias_pipeline_writes_estimator_var_and_provenance(
    tmp_path: Path,
) -> None:
    mn = make_mini_night(tmp_path)
    output = build_superbias(
        mn.cfg, run_root=mn.work_dir,
        stage_dir=mn.work_dir / "03_bias",
        allow_legacy_mirrors=False,
    )
    assert output.is_file()
    index = json.loads(
        (mn.work_dir / "03_bias" / "master_bias_index.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(index["groups"]) == 1
    group = index["groups"][0]
    definition = group["variance"]
    assert definition["definition"] == "uncertainty_of_combined_master_bias"
    assert definition["model"] == "GAUSSMED"
    assert definition["factor"] == pytest.approx(
        normal_median_variance_factor(3),
    )
    assert definition["min_effective_n"] == 3
    var_path = mn.work_dir / group["files"]["var"]
    with fits.open(var_path) as hdul:
        hdr = hdul[0].header
        assert hdr["BUNIT"] == "ADU^2"
        assert hdr["SB_VMOD"] == "GAUSSMED"
        assert hdr["SB_VFAC"] == pytest.approx(
            normal_median_variance_factor(3),
        )
        assert hdr["NBIAS"] == 3
        var = np.asarray(hdul[0].data, dtype=float)
    # The Golden bias has only Gaussian read noise. The expected uncertainty
    # of the *median master* is RN_e^2/gain^2 times finite-N factor.
    model = (3.0 / 0.62)**2 * normal_median_variance_factor(3)
    assert np.mean(var) == pytest.approx(model, rel=0.07)
