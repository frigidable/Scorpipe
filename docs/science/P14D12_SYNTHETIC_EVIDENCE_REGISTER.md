# P1.4d12 — fail-closed synthetic SkySub scientific-evidence register

**Status:** portfolio-level QA evidence only. Parent: PR #158, independently
Windows QA verified. No real-science input or production output is changed.

## Why an evidence register is necessary

P1.4d3–d11 validated several different, valuable aspects of a *specific*
model of SkySub noise: saved-fit integrity, LSQR derivative replay,
independent SVD of a fixed weighted fit, fixed-mask and re-clipped
Monte Carlo covariance, and numerical conditioning stress tests.

These analyses deliberately use **different synthetic scenes and
fitting geometries**. Passing all these tests is *not* equivalent to
demonstrating that a particular SCORPIO exposure has passed a single,
co-registered, fully propagated error-budget validation.

`calib.sky_evidence_gate.assess_synthetic_sky_evidence(reports)`
collects nine independent synthetic diagnostics:
`fidelity`, `lsqr`, `svd`, `frozen_mc`,
`reclipped_mc`, `stable_clipping`, `changed_clipping`,
`changed_reclipping`, and `condition_sweep`.

Expected adversarial scenarios are **not treated as science failures**:
correctly detecting re-selected sky pixels and correctly blocking a
near-degenerate LU solve are essential positive evidence for a good
guardrail. Conversely, no clipping variability in the *stable*
experiment must be reported separately from deliberate adversarial
clipping instability.

The register refuses unknown categories, checks expected algorithmic
scope, re-evaluates several quantitative constraints instead of trusting
a literal `status=PASS` string, distinguishes absent from malformed/
contradictory evidence, and returns:

- `SYNTHETIC_CONTRACTS_SUPPORTED`: all nine synthetic test families
  supplied coherent, positive **synthetic** evidence;
- `EVIDENCE_INCOMPLETE`: some required scenarios have not been supplied;
- `EVIDENCE_INVALID`: a supplied report is malformed, contradictory
  or numerically exceeds the specified engineering diagnostic limits.

The output is always `CROSS_SCENARIO_SYNTHETIC_REGRESSION_ONLY` and always
sets `observation_specific_qualification=NOT_ASSESSED`,
`science_uncertainty_qualified=false`,
`production_variance_release_allowed=false`, and
`override_science_status_supported=false`.

## Explicit limitations

The register is **not a cryptographically authenticated evidence store**:
it cannot establish that a supplied Python dictionary was produced by
a specific commit, controlled hardware, laboratory, or observational
dataset. Pinned Windows QA SHA and test logs remain separate provenance
requirements. It MUST NOT be used to override `SBCOVQ=UNMODEL`,
`VCMPQ=SKY_LOST` or to promote a simulated covariance into a
publication-quality per-pixel `VAR`.

It is not an end-to-end test of all stages and does not imply all
instrument effects are simulated. Spatial read noise, hot pixels,
cosmic-ray masking, wavelength-model covariance, flat estimator noise,
blending of sky/object windows, resampling, Stack2D and Extract1D
remain separately scoped, potentially material sources of uncertainty.
The synthetic evidence register therefore *cannot* certify observational
S/N, regardless of the number of green synthetic regression tests.
