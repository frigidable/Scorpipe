# P1.4e3 — synthetic same-data profile-endogeneity Monte Carlo

**Parent:** independently Windows-validated PR #163.
**Status:** opt-in scientific diagnostic; neither a science FITS writer
nor a general observation-specific uncertainty calibration.

## Motivation

The frozen SkySub→Extract1D operator (P1.4e1) and the full synthetic
Kelson re-clipping chain through a fixed Extract1D map (P1.4e2) both
condition on deterministic extraction coefficients. In actual Horne-like
optimal extraction, Scorpipe's `_optimal_extract` **learns the spatial
profile from the same noise-affected SCI being extracted** and may clip
negative values in that profile. The resulting estimator is nonlinear
in the pixel data. Its `1 / sum(P**2 / VAR)` formal variance is
conditional on the estimated profile, not automatically its
unconditional sampling variance.

## Independent reference versus production extractor

`calib.optimal_profile_mc.synthetic_optimal_profile_coverage` creates
a known normalized Gaussian spatial profile on a seven-row aperture
at one central wavelength bin of a 9×25 synthetic image. The true
object flux, read variance, trace, sky subtraction, aperture and
Boolean validity are fixed. Gaussian pixel noise is newly sampled
for each of 200–1500 independent Monte Carlo science realizations.

On identical noise draws, the experiment independently computes a
**known-profile** WLS flux estimator with analytical variance
`1 / sum(P_true**2 / VAR)`, and calls production `_optimal_extract`
with `backend='numpy'`, a profile freshly inferred from each noisy
frame, and the actual positive-profile clipping and optional
wavelength-domain profile smoothing.

It measures the known-profile empirical variance/bias/68.3%/95%
frequentist coverage against an exact Gaussian WLS reference,
and the adaptive extractor's bias, empirical variance, *reported*
conditional variance, 68.3%/95% interval coverage centered on the
**true injected flux**, and cross-wavelength covariance of adjacent
bins. Adjacent spectral bins are reported, but **only the central
bin** is used for coverage, to avoid treating spectrally smoothed
profile estimates as independent across wavelength.

The adversarial zero-flux case is scientifically informative:
if positive-clipped profile estimates preferentially follow the
**same positive noise fluctuations** used for extraction, the
weighted flux is expected to have a positive mean and nominal
Gaussian intervals may undercover even with perfectly specified
CCD read variance. This is a mathematical estimator-property
check, NOT a statement that any actual SCORPIO spectrum is biased.

Statuses:
- `REFERENCE_MC_MISMATCH`: independent known-profile Gaussian oracle
  fails numerical/statistical consistency checks.
- `ADAPTIVE_PROFILE_DISCREPANCY`: the data-derived profile shows
  significant mean bias or 95% undercoverage against its reported
  conditional variance within the test's finite Monte Carlo threshold.
- `NO_DETECTED_ADAPTIVE_DISCREPANCY`: no discrepancy detected with
  this synthetic noise level, profile and number of draws; **not**
  a science qualification.

An adversarial detection is a **successful test** of an incomplete
variance model. Test assertions do not demand or falsely claim
nominal adaptive coverage in a known pathological regime.

## Restrictions

Only the extraction profile adapts per noise realization. The sky,
spatial trace, aperture, CCD read noise, wavelength solution and
masks remain fixed. This test does not run SkySub, Stack2D, or
any actual final FITS publication path, and deliberately does not
pretend to combine all nonlinearities already measured in
P1.4d9/e2. Interpolation covariance, shared MasterBias, MasterFlat
noise, Poisson signal dependence and uncertain source aperture
selection remain out of scope here.

`science_uncertainty_qualified=false` and
`production_variance_release_allowed=false` always.
Production SCI/VAR/MASK or `SBCOVQ=UNMODEL`/
`VCMPQ=SKY_LOST` are unchanged.

Background method: Horne (1986), *An optimal extraction algorithm
for CCD spectroscopy*, PASP 98, 609. The same-data profile issue
is tested here in Scorpipe's actual implementation rather than
assumed from that source.
