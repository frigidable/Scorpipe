# P1.4e7 — 1D spectral-bin covariance and integrated-line uncertainty

**Status:** independent *frozen conditional* diagnostic only.
**Parent:** PR #167 passed independent pinned Windows QA.
No real observations. No production SCI/VAR/MASK or pipeline flags changed.

## Scientific question

After Kelson SkySub and flux-conserving COUNT_V1, adjacent
spectral bins need not have independent errors. A common native CCD
pixel may contribute to more than one output bin, and the fitted
sky model couples multiple pixels; MasterBias can also correlate
different exposures. A diagonal FITS `VAR` extension cannot encode
these **inter-wavelength covariance terms**.

For an integrated line `L = sum_k a_k F_k`, where `a_k` includes
any wavelength interval, throughput, or continuum-subtraction
weights, the conditional full uncertainty is

```math
Var(L) = a^T C a = sum_k a_k^2 C_kk + 2 sum_{k<l} a_k a_l C_kl.
```

Using `sum_k a_k^2 C_kk` alone discards the second term.
**The difference can have either sign**, depending on the
covariance matrix and signed integration weights. No universal
correction factor exists.

## New opt-in matrix-free wrapper

`calib.spectral_covariance_kernel.
frozen_sky_1d_spectral_covariance_kernel` accepts 2–12 strictly
increasing native output-bin indices and their finite, potentially
signed integration weights. It evaluates the existing verified
`covariance_of_frozen_sky_chain_grouped_pair` on basis vectors
corresponding to each requested channel pair.

The result contains separate independent-detector, shared-MasterBias,
and total symmetric spectral covariance matrices, with numerical
positive-semidefiniteness checks. It also reports the correct
integrated-line variance, the diagonal-only shortcut, their ratio,
the corresponding 1-sigma ratio, and the aggregate signed
off-diagonal contribution. Invalid groups, missing variance,
unsupported bins and conflicting geometries fail closed.

No full CCD×CCD covariance is ever materialized. This bounded
small-matrix diagnostic is not a general-purpose publication FITS
covariance writer.

## Independent validation

`tests/test_p14e7_spectral_covariance_kernel.py` uses the existing
P1.4e1 algebraic 4×4 two-exposure CCD proxy with known sky-training
rows, independent native variances, differently weighted stacked
rows, subpixel shifts, detector-validity masks, transfer and explicit
common/split MasterBias groups. The expected 3×3 spectral covariance
is constructed by **an independent dense forward COUNT_V1 +
spatial registration map and dense weighted-least-squares projector**,
not by the newly added kernel or the sparse frozen SkySub adjoint.

A separate 20,000-realization Monte Carlo re-fits a known injected
sky with **dense QR in every realization** and measures all three
spectral output channels jointly. The empirical covariance and
integrated three-channel variance must agree with the independent
matrix and the wrapper.

In this particular small algebraic example, off-diagonal terms
increase the variance of the equally weighted integrated three-bin
measurement by about **18–19%** relative to a diagonal-only sum,
depending on MasterBias grouping. This is a *counterexample* to
assuming bin independence, not a SCORPIO-instrument calibration.

Signed channel weights are also tested for continuum subtraction.
An alternative grouping checks that independent noise remains fixed
while shared calibration covariance changes appropriately.

## Scope and qualification

The frozen projector, final selected sky pixels, wavelength solution,
COUNT_V1 overlap geometry, Stack2D weights, extraction weights and
fluxcal transfer are conditional inputs. The test does not re-run
Kelson clipping, source trace fitting, Huber weight updates or
adaptive profile estimation per draw in this experiment (those
nonlinearities have dedicated synthetic experiments P1.4e6). It
does not include wavelength-solution calibration uncertainty,
MasterFlat estimation, telluric model uncertainty or real data.

Every report has `science_uncertainty_qualified=false` and
`production_variance_release_allowed=false`. A plausible-looking
spectral covariance must not override `SBCOVQ=UNMODEL` or
`VCMPQ=SKY_LOST`, and does not authorize declaring a measured
line-flux error calibrated for actual SCORPIO spectra.
