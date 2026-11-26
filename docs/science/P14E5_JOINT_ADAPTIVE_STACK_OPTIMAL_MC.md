# P1.4e5 — paired adaptive Stack2D + optimal Extract1D Monte Carlo

**Status:** opt-in synthetic-only science diagnostic.
**Parent:** PR #165, independently verified by user-run Windows QA.
No observational inputs; no changes to production products or QA flags.

## Why the combined experiment matters

P1.4e3 demonstrated estimator self-noise effects from re-estimating
the *optimal extraction spatial profile* on the same noisy SCI data.
P1.4e4 separately checked data-dependent Stack2D Huber weights and
common MasterBias calibration errors. Because one nonlinear estimator
feeds another, their total bias and conditional-variance discrepancy
cannot generally be inferred by summing two separate experiments.

The new `calib.joint_adaptive_stack_optimal_mc.
synthetic_joint_stack_optimal_coverage` creates three synthetic,
pre-rectified 9×21 frames with a known seven-row Gaussian spatial
object profile. An object with known true flux is identical in every
frame. Each frame has independent Gaussian detector noise of
variance 1, plus a separate Gaussian MasterBias error map for each
**explicit** group ID. One MasterBias noise realization is shared
across multiple exposures **only** within the same calibration
group; all realizations are independent across Monte Carlo trials.
Every frame has its correct *diagonal* VAR=2.

Each newly sampled 3-exposure realization passes through two
**actual production** Stack2D `_stack_chunk_streaming` modes:
unweighted-by-residual inverse variance and iterative robust Huber,
with the latter recalculating its sample-dependent weights every
draw. Each stacked frame then produces TWO final 1-D flux estimates:
the independent, known-true-profile WLS estimator and production
`extract1d._optimal_extract` with positive-clipped spatial profile
re-learned from the same stacked SCI and optionally smoothed across
wavelengths.

All four estimators are evaluated on the **same source and noise
realization**, enabling statistically efficient paired contrasts:

```text
                           Known true profile     Adaptive profile
Fixed inverse-var Stack2D    invvar_known           invvar_adaptive
Adaptive Huber Stack2D       huber_known            huber_adaptive
```

The paired interaction is defined as
`(huber_adaptive - huber_known) -
 (invvar_adaptive - invvar_known)`.
The experiment reports its signed mean and finite-sample standard
error, but **does not assume** it must be nonzero or have a
particular sign.

## Independent exact reference and scientific interpretation

For the *fixed invvar Stack2D and true-profile WLS estimator*, with
three equal-variance exposures and known Gaussian spatial shape,
the exact grouped variance is

```text
   [1/3 + sum_group (n_group/3)**2] / sum_y(P_true[y]**2).
```

The published independent-input conditional variance predicts
`[2/3] / sum_y(P_true[y]**2)`. They agree when all group IDs
differ; the exact variance is **twice** that prediction when
all three science exposures reuse the *same* MasterBias field.

Independent Monte Carlo variance, mean and 95% coverage of the
`invvar_known` control are checked against the exact grouped
Gaussian reference before any adaptive result is interpreted.
Every final reported conditional variance is evaluated against
the injected known truth, and only ONE fixed wavelength bin is
used per draw; spatial or wavelength pixels are not incorrectly
counted as separate Monte Carlo trials.

A synthetic no-source experiment (`true_flux=0`) checks whether
the production spatial profile's nonnegative truncation causes
a statistically significant *positive* extraction bias, including
after adaptive Huber stacking. For shared MasterBias groups the
known-profile control deliberately demonstrates how interval
undercoverage can arise **even without profile self-noise**.

Statuses: `PAIRED_MC_REPORTED` means the independent Gaussian
control agrees with exact grouped variance. It **does not** mean
that Huber or adaptive extraction coverage is correct.
`REFERENCE_MC_MISMATCH` blocks interpretation if that positive
control is inconsistent with its known distribution.

## Strict limitations

The input frames are **already rectified**; they do not invoke
SkySub/Kelson/COUNT_V1 on every draw. Spatial shifts, wavelength
maps, Stack2D input masking, trace, aperture width, variance maps,
detector response and local sky correction are all fixed.
The adaptive Stack2D and Extract1D internal calculations are real,
but this is NOT a full end-to-end native CCD reduction.

The calibration-group model is deliberately simple: common bias
noise at matching pixels only, with a single linear +1 sensitivity
and independent Gaussian detector noise. It omits flat-field
covariance, signal-dependent Poisson noise, sky-subtracted
pixel correlations, trace estimation, missing coverage and
calibration-standard uncertainty.

The production algorithm is not modified. Extraction receives the
actual Stack2D mask with REJECTED interpreted as *informational
contributor provenance* and NO_COVERAGE as fatal, as in a verified
stack. That is a synthetic controlled call, not an assertion about
the provenance of arbitrary FITS files.

Outputs always include
`science_uncertainty_qualified=false` and
`production_variance_release_allowed=false`. The experiment
cannot override production `SBCOVQ=UNMODEL` or
`VCMPQ=SKY_LOST` statuses. No files are read or written.
