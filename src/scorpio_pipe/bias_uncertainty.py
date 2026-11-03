"""Uncertainty of a *combined* MasterBias image, in ADU squared.

A temporal variance across raw bias frames, sigma_readout**2, is NOT the
uncertainty of the combined master offset. The latter depends on the estimator
(mean versus median) and number of independent bias exposures.

Assumptions:
* Independent, identically distributed approximately Gaussian bias read noise
  at a fixed pixel, with no drift or temporal correlation.
* np.var(stack, ddof=1) estimates the per-frame variance.
* An un-clipped mean has estimator variance s**2 / N.
* An un-clipped Gaussian sample median has estimator variance
  s**2 * c_N where c_N is the exact finite-N normal order-statistic
  variance for N <= 64; for larger N use pi/(2N) asymptotically.
* A sigma-clipped mean is nonlinear. We retain the *unclipped* sample
  variance to avoid hiding contaminated pixels, divide by the number of
  accepted samples, and label this an APPROXIMATION. Fewer than two retained
  values are explicitly invalid and must receive the BADPIX DQ bit.

Neither static bias pattern drift nor common-mode temporal correlations are
captured. Independent calibration QA is required for publication-grade errors.
References: Gaussian order-statistic distributions and their Beta transforms,
MIT 18.650 lecture notes "Order statistics, quantiles, and sample quantiles";
Davison, Statistical Models, section on Normal sample median.
"""
from __future__ import annotations

from functools import lru_cache
from math import pi

import numpy as np


@lru_cache(maxsize=128)
def normal_median_variance_factor(n: int) -> float:
    """Var(median(Z_1,...,Z_n)) for i.i.d. standard-normal samples.

    For even n the median is the average of the two central order statistics,
    following NumPy's definition. This exact finite-N calculation uses Beta
    marginal and adjacent-order-statistic joint PDFs. It is performed once
    per distinct N and cached; N>64 uses its well-known asymptotic form.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("Median variance needs at least two bias frames")
    n = int(n)
    if n == 2:
        return 0.5  # A two-sample median is precisely the arithmetic mean.
    if n > 64:
        return float(pi / (2.0 * n))

    from scipy.integrate import quad
    from scipy.special import gammaln, ndtri
    from scipy.stats import beta

    if n % 2:
        k = (n + 1) // 2
        second, _ = quad(
            lambda u: float(ndtri(u) ** 2 * beta.pdf(u, k, k)),
            0.0, 1.0, epsabs=1e-10, epsrel=1e-9,
        )
        return float(second)

    # n=2*k: median=(Z_(k:n)+Z_(k+1:n))/2. By symmetry,
    # Var(median) = (E[Z_(k:n)^2] + E[Z_(k:n) Z_(k+1:n)])/2.
    k = n // 2
    marginal, _ = quad(
        lambda u: float(ndtri(u) ** 2 * beta.pdf(u, k, k + 1)),
        0.0, 1.0, epsabs=1e-10, epsrel=1e-9,
    )
    from numpy.polynomial.legendre import leggauss

    nodes, weights = leggauss(192)
    v = (nodes + 1.0) / 2.0
    w = weights / 2.0
    u = v[:, None] * v[None, :]
    # Joint density of adjacent uniform order statistics U_(k),U_(k+1)
    # over 0<u<v<1; substitute u=v*t to integrate over a unit square.
    log_density = (
        gammaln(n + 1) - 2.0 * gammaln(k)
        + (k - 1) * np.log(u)
        + (k - 1) * np.log1p(-v[:, None])
        + np.log(v[:, None])
    )
    cross = float(np.sum(
        ndtri(u) * ndtri(v[:, None]) * np.exp(log_density)
        * (w[:, None] * w[None, :])
    ))
    factor = (float(marginal) + cross) / 2.0
    if not (np.isfinite(factor) and factor > 0):
        raise RuntimeError("Failed normal-median order-statistic quadrature")
    return factor


def master_bias_estimator_variance(
    stack_adu: np.ndarray,
    *,
    combine: str,
    sigma_clip: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build MasterBias SCI/VAR from a same-readout CCD stack.

    Returns estimator SCI, its ADU^2 variance, and provenance including
    per-pixel accepted-count limits and a boolean bad-support mask.
    """
    data = np.asarray(stack_adu, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError("MasterBias stack must be a (N,y,x) array")
    n = data.shape[0]
    if n < 2:
        raise ValueError(
            "MasterBias requires >=2 frames for an empirical uncertainty; "
            "one input is not a statistically calibrated master"
        )
    if not np.all(np.isfinite(data)):
        raise ValueError("MasterBias source images contain nonfinite pixels")
    method = str(combine).strip().lower()
    if method not in {"mean", "median"}:
        raise ValueError(f"Unknown MasterBias combination method: {combine!r}")
    if not np.isfinite(sigma_clip) or sigma_clip < 0:
        raise ValueError("MasterBias sigma_clip must be finite and nonnegative")

    # ddof=1 is unbiased for the individual-exposure noise variance under
    # independent, equal-variance normal read-noise samples.
    individual_var = np.var(data, axis=0, ddof=1)
    invalid = np.zeros(data.shape[1:], dtype=bool)

    if method == "median":
        sci = np.median(data, axis=0)
        factor = normal_median_variance_factor(n)
        var = individual_var * factor
        model = "GAUSSMED" if n <= 64 else "MEDASYMP"
        n_eff_min = n
    elif sigma_clip > 0:
        median = np.median(data, axis=0)
        mad = np.median(np.abs(data - median), axis=0)
        sigma = 1.4826 * mad
        sigma = np.where(sigma > 0, sigma, 1.0)
        keep = np.abs(data - median) <= sigma_clip * sigma
        counts = keep.sum(axis=0).astype(np.int32)
        invalid = counts < 2
        sci = np.sum(np.where(keep, data, 0.0), axis=0) / np.maximum(counts, 1)
        var = individual_var / np.maximum(counts, 1)
        factor = None
        model = "CLIPAPPX"
        n_eff_min = int(counts.min())
    else:
        sci = np.mean(data, axis=0)
        var = individual_var / n
        factor = 1.0 / n
        model = "MEANSEM"
        n_eff_min = n

    if not np.all(np.isfinite(var)) or np.any(var < 0):
        raise RuntimeError("MasterBias estimator variance is not physical")
    return (
        np.asarray(sci, dtype=np.float32),
        np.asarray(var, dtype=np.float32),
        {
            "variance_model": model,
            "variance_factor": float(factor) if factor is not None else None,
            "n_effective_min": int(n_eff_min),
            "invalid_support": invalid,
            "independent_gaussian_assumption": True,
        },
    )
