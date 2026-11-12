# Stack2D variance: fixed-weight conditional propagation (v1)

## Applicability

For independent input images with *deterministic* weights w_i and marginal
per-pixel input variances V_i, the weighted-mean conditional variance is

    VAR_fixed = Σ(w_i² V_i) / [Σw_i]²

The familiar `1/Σw` is exact only when the weights equal 1/V_i (up to a
common factor, properly accounted for). Huber factors, variance floors, and
non-optimal weights violate that assumption. The current correction
propagates the **actual input VAR** rather than the variance floor that was
used only to choose the averaging weight.

The FITS output declares:

- `STKVRALG=W2VAR`: fixed-weight propagation;
- `STKVARQ=COND_ONLY`: no guarantee for the full sampling variance of the
  robust estimator when weights are estimated from the same noisy data;
- `STKXCVQ=UNMODEL`: cross-exposure shared calibration errors (notably the
  shared MasterBias) have not been propagated as covariance;
- `STKYCVQ=UNMODEL` when Y alignment is enabled: spatial covariance from
  interpolation is not published; `NONE` otherwise does not certify the
  absence of earlier wavelength/spatial resampling covariance.

No upgrade to the global product/unit metadata and no guarantee of correct
`Extract1D` science uncertainty follow solely from this change.

## Remaining P0 requirements

1. Separate independent detector VAR from shared MasterBias estimator VAR
   through all detector, Flat, Cosmics, Linearize and Stack2D stages.
   For a normalized weighted mean with common bias error V_B, the correct
   term is V_B, not V_B*Σalpha_i². Never double count it.
2. Carry the covariance of fractional Y alignment through spatial extraction
   (banded or matrix-free operator), not just each stacked pixel's VAR.
3. Characterize true Huber/sigma-clipped estimator sampling uncertainty
   with controlled injection/recovery, along with the selection policy.
4. Propagate local sky-model estimator error, wavelength resampling
   correlation, and flat/spectrophotometric systematic uncertainty separately.
5. Only publish a scientifically qualified line-flux error after validated
   instrument-realistic ensemble tests and real-science LONG comparisons.

## References

- PypeIt coaddition and separate variance propagation:
  https://pypeit.readthedocs.io/en/soar_triplespec/api/pypeit.core.coadd.html
- Bolton & Schlegel, 2010, covariance-aware spectroscopy:
  https://arxiv.org/abs/0911.2689

This patch is adapted from the independent draft PR #115 and does not
merge its older main-based branch.
