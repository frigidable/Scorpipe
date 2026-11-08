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
* Optional gross-rejected median (explicit cutoff in ADU, default disabled)
  recalculates SCI and approximate VAR on the same accepted samples; assumes
  the cutoff is far above read-noise and does not calibrate ADC quantization.
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
    gross_reject_adu: float = 0.0,
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
    if not np.isfinite(gross_reject_adu) or gross_reject_adu < 0:
        raise ValueError("MasterBias gross_reject_adu must be finite and nonnegative")
    if method == "median" and sigma_clip > 0:
        raise ValueError(
            "MasterBias bias_sigma_clip>0 is not implemented for "
            "bias_combine=median: refusing to silently ignore clipping"
        )
    if gross_reject_adu > 0 and (method != "median" or sigma_clip > 0):
        raise ValueError(
            "MasterBias gross_reject_adu>0 is supported only with "
            "unclipped median; do not mix rejection rules"
        )

    # ddof=1 is unbiased for the individual-exposure noise variance under
    # independent, equal-variance normal read-noise samples.
    individual_var = np.var(data, axis=0, ddof=1)
    invalid = np.zeros(data.shape[1:], dtype=bool)
    rejected_samples = 0
    affected_pixels = 0

    if method == "median" and gross_reject_adu > 0:
        # This mode is deliberately opt-in. A generous absolute cutoff removes
        # gross transients, *not* ordinary Gaussian read-noise fluctuations.
        # Recalculate both SCI and estimator VAR on the SAME accepted samples.
        centre = np.median(data, axis=0)
        accepted = np.abs(data - centre) <= float(gross_reject_adu)
        counts = accepted.sum(axis=0).astype(np.int32)
        invalid = counts < 2
        rejected_samples = int(np.count_nonzero(~accepted))
        affected_pixels = int(np.count_nonzero(counts < n))

        # NaN/inf sentinel is never allowed into a retained median.
        # Sorting places inf after valid values, so middle retained order
        # statistics are accessed directly without masked-array warnings.
        ordered = np.sort(np.where(accepted, data, np.inf), axis=0)
        idx_lo = np.maximum((counts - 1) // 2, 0)
        idx_hi = np.maximum(counts // 2, 0)
        lower = np.take_along_axis(ordered, idx_lo[None], axis=0)[0]
        upper = np.take_along_axis(ordered, idx_hi[None], axis=0)[0]
        sci = 0.5 * (lower + upper)
        sci[invalid] = 0.0

        # Compute the retained sample variance about its own mean. Center
        # at the per-pixel original median for numerical stability, then
        # apply finite-N Gaussian order-statistic factors at each n_eff.
        centred = np.where(accepted, data - centre, 0.0)
        shifted_mean = np.sum(centred, axis=0) / np.maximum(counts, 1)
        ss = np.sum(
            np.where(accepted, (centred - shifted_mean) ** 2, 0.0),
            axis=0,
        )
        sample_var = ss / np.maximum(counts - 1, 1)
        factors = np.zeros(counts.shape, dtype=np.float64)
        for n_eff in np.unique(counts):
            if n_eff >= 2:
                factors[counts == n_eff] = normal_median_variance_factor(
                    int(n_eff)
                )
        var = sample_var * factors
        var[invalid] = 0.0
        factor = None
        model = "GROSSMED"
        n_eff_min = int(counts.min())
    elif method == "median":
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
            "gross_reject_adu": float(gross_reject_adu),
            "gross_rejected_samples": int(rejected_samples),
            "gross_affected_pixels": int(affected_pixels),
            "gross_var_assumption": (
                "provisional Gaussian finite-N approximation after removing "
                "gross contaminants; quantized-ADC coverage not calibrated"
                if gross_reject_adu > 0 else None
            ),
        },
    )
