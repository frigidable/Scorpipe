# P1.4d2 - shared MasterBias covariance through frozen reduction operators

**Status:** exact conditional numerical primitive for a restricted model; not
called in the real Stack2D/Extract1D/LineMeasure reduction. This builds
on existing P1.4d1 rather than reimplementing wavelength, Y interpolation
or extraction operators.

## Model

Each exposure e has a fixed complete linear science measurement derivative
g_e[j,i] with respect to the pre-rectified CCD pixel x_e[j,i]. It includes
COUNT_V1 wavelength overlap, per-exposure output scaling, spatial alignment,
final normalized stacking weights, extraction coefficients and fixed
flux calibration. P1.4d1 already contracts independent detector variances:

    C_ind = sum_e,j,i V_ind,e[j,i] g_L,e[j,i] g_R,e[j,i].

A common MasterBias estimator for group k is a *single shared random CCD
image* B_k[j,i]. Its covariance is assumed diagonal in the native bias CCD
grid for this contract, with per-pixel variance V_B,k[j,i]. Each science
exposure supplies the independently verified sensitivity of its pre-rectified
science pixels to that common error, h_e[j,i]. The covariance contribution is

    C_bias = sum_k,j,i V_B,k[j,i] *
             [sum_{e in k} h_e[j,i] g_L,e[j,i]] *
             [sum_{e in k} h_e[j,i] g_R,e[j,i]].

This captures the terms missed by treating a MasterBias group as independent
across its science exposures, including covariance caused by wavelength
resampling and fractional Y shifts. The projection reuses the existing
COUNT_V1 adjoint, keeping memory proportional to CCD pixels rather than
constructing a dense output wavelength covariance matrix. The two functionals
may represent distinct TRACE and FIXED extraction maps.

## Explicit required inputs

- Per-exposure detector uncertainty with the shared MasterBias term
  **already separated out**. Passing historical total SCI `VAR`, which
  includes BVAR, would double count the bias component.
- One unambiguous MasterBias calibration group ID per exposure, and exact
  native CCD grid shape/unit for all exposures sharing a group.
- Per-exposure sensitivity maps from actual FlatField/SkySub calibration
  operators. A constant of one cannot generally replace these because flat
  division, gain, masking and sky-model fits change the response.
- One group MasterBias estimator variance map, with its estimator-noise
  semantics independently verified. Groups must be statistically independent.
- P1.4d1's strict and *actual frozen* row plans, valid masks, spatial shifts,
  final normalized weights, extraction operators and fluxcal transfer.

## Not included

- Non-diagonal MasterBias covariance, shared input biases between nominally
  separate MasterBias groups, data-derived sky-model covariance or cancellation,
  flatfield shared covariance, trace/PSF estimation error, adaptive Huber/clip
  selection uncertainty, wavelength solution and flux-response uncertainty.
- Current production FITS products do NOT persist the complete frozen operator
  inputs or independent-vs-shared variance decomposition; callers cannot use
  production `VAR` alone to invoke this oracle honestly.
- This patch does not alter science FITS values or turn off existing
  `SBCOVQ=UNMODEL` qualification gates.

## Independent validation

- Exact dense forward-matrix vs matrix-free adjoint for mixed-group and
  single-group TRACE/FIXED cross-covariance with non-integer Y shifts,
  nontrivial COUNT_V1 plans, masks and per-row scales.
- Joint Gaussian Monte Carlo draws of one shared bias image plus independent
  science noise, observed with different exposure response maps.
- Exact one-group N=7 and distinct-group comparisons.
- Zero calibration variance, invalid map/lineage/sensitivity contracts,
  and finite-vs-masked CCD behavior.

The next meaningful milestone is *capturing operator provenance* and
safely transporting independent detector variance and group sensitivities
through actual FlatField, SkySub and Linearize. This demands integration tests
before reporting calibrated astrophysical line/kinematic uncertainties.
