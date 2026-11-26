# P1.4e2 — synthetic Kelson re-clipping Monte Carlo across 2-D→1-D chain

**Status:** independent synthetic-only diagnostic, parent independently
Windows-validated PR #161. No real exposures and no science FITS edits.

## Purpose

P1.4e1 computes conditional analytic SkySub covariance across frozen
Linearize, Stack2D, Extract1D and FluxCal transformations, including
explicit shared MasterBias. P1.4d9 separately samples real production
Kelson clipping/LSQR but measures only native detector image
functionals. The additional question is whether the *known-truth*
statistical coverage of extracted 1-D quantities remains nominal
after each Monte Carlo exposure re-fits its sky with clipping.

`calib.sky_chain_mc_reclipping.synthetic_reclipped_sky_to_1d_coverage`
constructs two small synthetic CCD exposures (10×20 pixels) with
known nonzero B-spline sky, known spatially extended object flux
in non-sky rows, deliberately signed MasterBias sensitivity, per-pixel
independent Gaussian noise, and either shared or distinct per-draw
MasterBias error fields. The wavelengths differ by exposure.

For a deterministic noiseless baseline, the *real production*
`_fit_kelson_bspline` establishes the final retained sky samples.
That fixed mask defines the analytic `FrozenSkyWLS` projector.
The **P1.4e1** `covariance_of_frozen_sky_chain_grouped_pair`
then predicts two correlated 1-D spectral measurements after the
whole frozen linear chain and separate independent/shared bias terms.

Every new Monte Carlo draw:
1. generates new independent native CCD noise for each exposure and
   one shared MasterBias noise field per named group;
2. re-runs production Kelson B-spline + LSQR + sigma clipping
   independently for each exposure's sky-window pixels;
3. subtracts the fitted sky on **every original CCD pixel**;
4. calls production `_apply_linearize_row_plan` for every CCD row;
5. performs explicit frozen subpixel y interpolation, normalized
   Stack2D weighting, TRACE/FIXED-like deterministic extraction maps
   and fluxcal transfer, then compares against the independently
   transformed known object truth.

Reported are analytic/empirical 2×2 covariance, independent/shared
MasterBias terms, centered standardized errors, 68.3%/95% normal
interval coverage, relative variance and cross-covariance errors,
final sky-selection change frequency, LSQR failure frequency and
finite-sample tolerances.

The `SELECTION_VARIABLE` status takes precedence whenever any
realization changes the fitted sky sample selection. Statistical
coverage is still reported even in that regime; it is *not*
automatically claimed to be incorrect just because a sample was
reselected. `NUMERIC_UNSTABLE` takes precedence when any LSQR
termination is nonnominal. `MC_CONSISTENT` requires no re-selection
and agreement of the measured and conditional predicted covariance
and coverage within explicit sampling tolerances.

## Explicit scientific limitations

This is **not** a full execution of production Stack2D or
Extract1D. Their final weights, y alignment, extraction profiles
and fluxcal transfer are fixed, not re-derived on each draw. The
simulation reuses production Kelson and COUNT_V1 row-plan execution,
but intentionally holds the subsequent maps deterministic. It does
not model sky-window discovery, flat-field estimator uncertainty,
cosmic rejection, data-driven Stack2D weighting, aperture profiling,
calibration-standard errors, or uncertainty in wavelength solutions.

The total covariance assumes Gaussian independent CCD pixels given
the shared MasterBias field. The test may validate a particular
controlled synthetic truth, not the distribution of arbitrary
astronomical measurements. It never changes production
`SCI/VAR/MASK`, never removes `SBCOVQ=UNMODEL` or
`VCMPQ=SKY_LOST`, and always reports
`science_uncertainty_qualified=false` and
`production_variance_release_allowed=false`.
