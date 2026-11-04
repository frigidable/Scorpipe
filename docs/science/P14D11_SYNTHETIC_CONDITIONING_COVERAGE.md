# P1.4d11 — synthetic conditioning versus covariance coverage

**Status:** independent diagnostic only. Based on Windows-verified PR #157.
No observational data; no production variance or sky stage change.

## The scientific problem

Normal-equation LU is potentially inaccurate when the whitened design
`A = sqrt(W) Xfit` is ill-conditioned: the condition number of
`A.T A` is approximately `kappa(A)**2`. The P1.4d10 SVD oracle
checks whether the *operator action* is accurate, but a separate
question is whether the resulting **scalar scientific observables**
happen to show a measurable variance or coverage error for a given
measurement geometry.

There is **no universal condition number at which confidence
intervals suddenly become wrong**. Some observables can be relatively
insensitive to unstable coefficients; others may be extremely
sensitive. Passing a finite Monte Carlo coverage check does not
override an independent numerical-stability failure.

## Experiment: independent SVD sampling versus normal-equation covariance

`calib.sky_conditioning_coverage_sweep.
synthetic_wls_conditioning_coverage_sweep` constructs a bounded,
16-pixel **algebraic proxy** for a degenerating sky fit, with
three fitting columns. The second and third columns become almost
collinear as the sweep perturbation decreases. These are controlled
algebraic stress data, NOT a literal SCORPIO/SCORPIO-2 B-spline
simulation or a claim about its physical condition spectrum.

Two exposures receive distinct heteroskedastic CCD noise and share
a pixel-wise MasterBias error field only when they have the same
explicit `group_ids`. Each has a known synthetic sky and nonzero
object flux in detector pixels excluded from the sky training
selection.

The oracle independently computes:

1. The **SVD reference** weighted-least-squares residual projector
   and its exact (up to floating-point arithmetic) 2x2 covariance
   for two science-pixel measurements, including shared MasterBias.
2. The alternative **production mathematical covariance prediction**
   using `FrozenSkyWLS` normal-equation LU and
   `covariance_of_frozen_sky_grouped` on those same measurements.
3. Thousands of independent noisy science realizations, with every
   realization re-fitted using dense **SVD** and no reuse of the
   LU normal-equation operator. Empirical covariance and 95% normal
   interval coverage are compared independently to the SVD reference
   and LU prediction.

Every sweep point also runs P1.4d10 SVD conditioning checks. A
`NUMERIC_SAFE` result requires that both operators pass that
numerical gate and the **reference SVD Monte Carlo** is consistent.
A `NUMERIC_BLOCKED` result is reported if either operator fails
the numerical gate even when the particular aperture's empirical
coverage still looks nominal. `REFERENCE_MC_MISMATCH` means the
independent statistical oracle itself disagrees with its SVD model.
A failed normal-equation factorization is explicitly reported.

Outputs contain condition numbers, independent SVD diagnostic
details, SVD-predicted and LU-predicted covariance matrices,
empirical covariance, separate 95% coverages, LU/SVD variance
ratios and reference MC statistical checks. The experiment has
strict simulation-size and parameter bounds. Reproducible seed.

## Restrictions

The normal-equation condition limit (default `1e5`) is a
conservative engineering control, NOT an instrument-specific
calibration threshold. The toy algebraic model fixes training
samples, weights, wavelength-independent design geometry and
Gaussian noise model. It does not reproduce Kelson sigma clipping,
astrophysical line profiles, real flat-field uncertainty,
resampling, spatial CCD noise correlation or final 1-D extraction.

This diagnostic **must not** remove production uncertainty flags
`SBCOVQ=UNMODEL` or `VCMPQ=SKY_LOST`. All outputs state
`science_uncertainty_qualified=false`. This test is designed to
prevent unjustified covariance claims, not to calibrate a general
reduction pipeline.

Reference: Higham, *Accuracy and Stability of Numerical Algorithms*
(2nd ed.), SIAM (2002), least-squares conditioning.
