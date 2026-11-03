# P1.4d8 — synthetic grouped MasterBias + frozen Kelson sky Monte Carlo

**Status:** pure synthetic statistical benchmark. No real data are used
or required. Parent PR #154 passed independent Windows QA. This
benchmark is NOT production error propagation or a scientific S/N claim.

## Question

Does the analytic fixed-weight, fixed-selection SkySub covariance formula
with a shared MasterBias noise field reproduce the actual sampling
covariance and frequentist Gaussian-interval coverage after fitting
the sky independently in multiple exposures?

For exposures `e` and MasterBias groups `g`, the Monte Carlo data
generation explicitly draws

```text
D[e, trial, pixel] = sky_true[e, pixel]
  + independent_detector_noise[e, trial, pixel]
  + bias_sensitivity[e, pixel] * shared_bias_noise[g, trial, pixel].
```

MasterBias errors are **identical between exposures sharing the same
group in each trial**. They are independently re-drawn across Monte
Carlo trials, independently across different groups, and independently
across different CCD pixels. This is a *defined synthetic model*, not
an assertion about every real calibration estimator.

Each trial contains a non-zero synthetic sky and is reduced using
**dense QR weighted least squares**, separately for each exposure,
on a selected sky ROI with a full 2D wavelength map. There is no
application of the sparse `FrozenSkyWLS.apply` or adjoint to generate
Monte Carlo sample measurements. Model coefficients are obtained
from the fitted observations and subtracted before computing two
different synthetic object-aperture observables.

An independent analytic prediction calls
`covariance_of_frozen_sky_grouped` and preserves the three
decomposed components: independent detector covariance,
shared MasterBias covariance, and total covariance.

The comparison reports:
- analytic and empirical 2x2 covariance matrices, including a
  non-trivial cross-covariance;
- fractional variance differences;
- cross-covariance error divided by geometric mean of predicted
  standard deviations;
- standardized estimator means and the empirical fraction of samples
  inside nominal 1 sigma and central 95% Gaussian intervals;
- the deliberately WRONG coverage when only the independent detector
  variance is used and shared calibration noise is suppressed.

The oracle tests two true MasterBias sharing patterns
`(A,A)` and `(A,B)`, with separate `y_order=0/1` splines.
The analytic cross-exposure shared covariance MUST change while the
independent variance component does not. Confidence bands on
Monte Carlo discrepancies explicitly scale with realization count.

## Scientific boundaries

The result is a check **conditioned** on known fixed variance maps,
fixed wavelength solution, fixed spline knots and fixed sky pixel
membership, and the assumed per-pixel Gaussian noise field. In
particular this does not estimate the full covariance of actual
iterated clipping or clipping/mask selection, the variance in
MasterFlat response, heterogeneous readout correlations, resampled
pixel covariance, sky-spectrum misspecification, or velocity/line
fit confidence intervals. The instrument/noise model is
synthetic by design; no SCORPIO observational datasets are used.
Output always has `science_uncertainty_qualified=false`.
The test uses at most 20,000 trials, 240 CCD pixels and two exposures
to keep the QR oracle bounded.

This new diagnostic does not alter production SkySub or any FITS,
`VAR`, `SBCOVQ=UNMODEL` or `VCMPQ=SKY_LOST` product.

## Follow-up without observational data

After independent Windows QA: extend to repeated full `kelson_raw`
LSQR refits with fresh sigma-clipping decisions on each Monte Carlo
realization, and measure where full-refit sampling distributions
depart from the frozen WLS benchmark. Avoid treating any number
of synthetic configurations as proof of observational calibration.
