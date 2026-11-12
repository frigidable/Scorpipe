# P1.4e6 — native synthetic Kelson→COUNT_V1→Stack2D→Extract1D Monte Carlo

**Status:** opt-in full mathematical-chain simulation; not an observation-
calibrated uncertainty model. Parent PR #166 independently passed user-run
Windows QA. No changes to production pipeline, FITS science products,
MasterBias flags or `SBCOVQ=UNMODEL`/`VCMPQ=SKY_LOST`.

## Motivation

PRs #163 and #166 test, respectively, (i) full production re-clipped
Kelson sky fitting followed by **fixed** Stack2D/extraction coefficients,
and (ii) actual **adaptive** Huber stacking and same-data optimal
profile extraction from **already rectified** images. Their separate
success does not demonstrate the combined sampling statistics when the
upstream sky model and both downstream robust estimators are data-derived
from *the same noise realization*.

## Native CCD synthetic experiment

`calib.full_nonlinear_chain_mc.synthetic_full_nonlinear_chain_coverage`
constructs three 10×20 native CCDs with an exact known B-spline sky
(`y_order=0` or 1), known spatial source outside the 120 sky-training
samples, fixed wavelength map including slit tilt, unit Gaussian
independent CCD variance, and a unit-variance per-detector-pixel
MasterBias field. MasterBias is explicitly **shared** across exposures
only when group identifiers match; it is resampled independently for
each of 200–800 Monte Carlo trials. The known deterministic sky may
contain an artificial outlier near the iterative clipping threshold.

Each **newly sampled** full-chain trial executes:

1. **Production Kelson** `_fit_kelson_bspline` including finite
   SciPy LSQR iteration, sigma-clipping and the actual fit diagnostics
   for each exposure; subtracts the model over the native 2D CCD.
2. **Production COUNT_V1** `_apply_linearize_row_plan` with row-wise
   wavelength-overlap plans built by the real constructor.
3. **Production Stack2D streaming** `_stack_chunk_streaming` twice:
   equal-noise `invvar` and reweighted `invvar_huber`, using in-memory
   synthetic FITS HDULists (no disk files).
4. **Independent known-profile WLS extraction** and **production
   optimal extraction** `_optimal_extract`, both on the same stacked
   SCI; the latter re-estimates its positive-clipped spatial profile.

All four estimates are evaluated against a known, separately transformed
object flux at one fixed central wavelength. Trials are not multiplied
by a count of neighboring wavelength bins when measuring coverage.

## Independent covariance positive control

Before the Monte Carlo trial loop, the deterministic noiseless
baseline sky fit establishes the final sky sample mask and the
production `FrozenSkyWLS` residual projector. The P1.4e1
`covariance_of_frozen_sky_chain_grouped_pair` computes the **exact
conditional reference variance** of `invvar_known` through the
native pixels, fixed wavelength interpolation and equal-weight Stack2D,
including the group-specific MasterBias cross-exposure terms.

This reference is independent of production Stack2D and Extract1D
adaptive calculations. In a **stable sky selection** regime its
predicted Gaussian 95% interval coverage, Monte Carlo sample variance
and centered mean must agree within finite-sample diagnostic
tolerances. If the fitted sky pixels change across realizations,
the function reports `SKY_SELECTION_VARIABLE`; a frozen estimator
can no longer be assumed globally valid. A nonnominal LSQR termination
overrides this with `NUMERIC_UNSTABLE`.

Only when the numerical and selection assumptions are stable and
the reference agrees does the report use
`JOINT_ADAPTIVE_MC_REPORTED`. This means the known-profile control
is consistent — **not** that any adaptive estimator's 95% coverage
is correct. Each adaptive case independently reports its own bias,
empirical variance/mean reported conditional variance and coverage.

The same trial's paired differences isolate the effect of re-estimating
the extraction profile under invvar and Huber stacking, and the
interaction `(huber_adaptive - huber_known)
- (invvar_adaptive - invvar_known)` is reported without assuming
its value or sign. No general correction factor is inferred.

## Scientific scope exclusions

The experiment is mathematically joined, not the full public
end-to-end production CLI. It holds wavelength map, source-exclusion
sky windows, object location, observation masks, incoming pixel
variances, flatfield, spatial alignment, variance floors and source
aperture fixed. Input spectroscopic frames are synthetic and
in-memory; in particular all three exposures share the same
known wavelength geometry. The pipeline's production sky-fitting,
linearization, stack and extraction helper functions are actually
invoked, but their file-based stages, dynamic trace discovery,
sky-geometry selection, calibration-standard uncertainty, background
estimation and final FITS writer are **not** invoked.

The reported conditional `VAR` at Stack2D/Extract1D ignores the
full SkySub-generated wavelength/pixel covariance and shared bias
cross-exposure terms by design; this is why comparison against
the separate conditional full-chain reference is necessary.
Neither a nominal synthetic result nor successful Python tests
can certify published SCORPIO S/N values.

Every output has `science_uncertainty_qualified=false` and
`production_variance_release_allowed=false` regardless of status.
