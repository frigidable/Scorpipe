# P1.4e4 — adaptive production Stack2D Huber and shared-MasterBias MC

**Status:** opt-in diagnostic on a synthetic 1×2 rectified-image proxy.
Parent PR #164 independently passed Windows QA. No real SCORPIO inputs,
production stage edits, or science FITS modifications.

## Problem

Production `_stack_chunk_streaming` forms a robust weighted average
whose Huber multipliers depend on the observed pixel residuals, and
propagates an **independent-input, fixed-final-weight conditional**
variance, `sum_e w_e**2 V_e / (sum_e w_e)**2`.

Two important effects are outside that expression:
(1) data-derived weights can change the unconditional estimator variance,
and (2) a single MasterBias field can appear **in multiple science
exposures**: those calibration errors must not be treated as
independent merely because they were measured on separate frames.

## Controlled simulation and exact independent reference

`calib.adaptive_stack_mc.synthetic_adaptive_stack2d_coverage` generates
three science exposures, each 1×2 pixels, for 200–2000 independent
Monte Carlo trials. Each exposure has independent unit Gaussian noise
and a unit-variance MasterBias error field. The same bias noise is
reused **only across exposures with the same explicit calibration group**,
but drawn independently in every Monte Carlo trial. Every exposure
advertises `VAR=2` (independent+MasterBias variance).

At every realization, the function updates three **in-memory FITS
HDULists**, calls the **real** `stack2d._stack_chunk_streaming` and
records its combined SCI and published conditional VAR. The worker
uses its actual initial estimate, Huber reweighting, stopping and
`_WeightedMeanAccumulator`, not a separately coded robust estimator.
No FITS are written to disk. Spatial alignment, masks and scales are
intentionally fixed to isolate stacking weights and grouped errors.

A positive control runs production `method='invvar'` with equal
weights. Its *independently derivable* true sampling variance is

```text
Var(mean) = V_ind/3 + V_bias * sum_group (n_group/3)**2.
```

For `V_ind=V_bias=1`, this is `2/3` for three independent MasterBias
groups and `4/3` for one shared MasterBias group, whereas the
production fixed-weight independent-input `VAR` yields `2/3`
for both. The Monte Carlo verifies both cases.

The Huber experiment, with **actually data-dependent weights**, reports
empirical variance, mean nominal `VAR`, the variance ratio, interval
coverage for one and two fixed pixel channels, mean standardized error
and fraction of channels with robust downweighting. The exact simple
grouped-variance formula above is **NOT** claimed to be an analytic
law for random adaptive weights. Coverage is evaluated on independent
realizations (no bin-level pseudoreplication).

`MC_CONSISTENT` means no discrepancy was detected against the
published conditional `VAR` in this scenario. The adversarial
`CONDITIONAL_VAR_UNDERCOVERAGE` means both variance inflation and
two-sided 95% interval undercoverage are detected relative to explicit
Monte Carlo uncertainty thresholds. An intermediate
`MC_DISCREPANCY_OR_INCONCLUSIVE` prevents forcing either assertion.

The shared-bias case is **not evidence that Huber alone caused
undercoverage**; the independent-vs-common group comparison isolates
the shared calibration error, while the independent-group Huber
comparison tests the remaining effect of adaptive weights.

## Scope

The test is a pure synthetic, bounded single-row two-wavelength
experiment. It does not model upstream SkySub covariance, COUNT_V1
spectral interpolation, uncertain y registration, cosmic masking,
Poisson noise, MasterFlat errors, unknown bias-group provenance, or
adaptive Extract1D profiling. It cannot infer any universal Stack2D
coverage correction factor for actual SCORPIO frames.

Output always sets
`science_uncertainty_qualified=false` and
`production_variance_release_allowed=false`; it must not override
`SBCOVQ=UNMODEL`, `VCMPQ=SKY_LOST` or production `VAR`.
