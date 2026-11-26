# P1.4g — captured Kelson SkySub + indexed MasterBias: conditional bridge

**Scope:** read-only, opt-in, explicitly bounded **sky-output** diagnostic.
**SCIENCE HOLD.** Does not alter SkySub, FlatField, COUNT_V1,
Stack2D, Extract1D, flux calibration, published VAR, or pPXF LSF.

## Sources

The vNext pipeline already writes:
- An ACTUAL Kelson LSQR solver snapshot with native wavelength map,
  post-clipping final fit pixels, selected variances and spline basis
  (P1.4d3; `_kelson_frozen_fit_v1.npz`).
- SHA256-bound selected SkySub input, Kelson archive and sky FITS, with
  independent FITS fidelity replay (PR #201).
- Indexed FlatField variance component SCI/VAR/VBIAS/VREST/BSENS/BVARADU
  maps on native detector geometry (P1.4f source check, PR #202).

`load_captured_sky_bias_conditional_basis` now uses the existing
cryptographic SkySub and source-component admission audits, then:
1. Reads exactly those actual selected source MEF maps (in e- and ADU
   variance units) with identical source-file and capture lineage.
2. Verifies **every final fitted sky pixel's variance** is numerically
   consistent with the original source `VAR`. An unrecorded gain,
   variance-floor, sky-input transformation or noise estimate is
   rejected rather than silently assumed.
3. Recreates the already available `FrozenSkyWLS` operator from
   the actual archived fit geometry and final mask, without inventing
   new clipping masks, sky weights or calibration groups.
4. Returns authentic *signed* BSENS coefficients and identical
   per-group BVARADU maps, and retains `VREST` under the explicit
   name `vrest_if_independent`. Resource caps default to 131,072
   detector pixels per exposure and eight exposures, to keep the
   normal-equation sparse factorization diagnostic bounded.

`evaluate_captured_sky_bias_conditional_pair` accepts explicitly
chosen functionals on the **pre-rectification, sky-subtracted CCD**.
It calls the previously tested `covariance_of_frozen_sky_grouped`.
For a functional gradient `g_e = R_e^T u_e`, the diagnostic uses

```text
V_conditional = Σ_e g_eᵀ diag(VREST_e) g'_e
              + Σ_g (Σ_(e∈g) BSENS_e * g_e)ᵀ
                    diag(BVARADU_g)
                    (Σ_(e∈g) BSENS_e * g'_e)
```

**The independent part is an explicit mathematical assumption, not an
established property of VREST.** In particular, VREST may contain
additional shared MasterFlat uncertainty. This can change both signed
cross-covariance and total variance; these formulas are not guaranteed
lower or upper bounds.

## Limitations and gates

- This is an **exact frozen-WLS normal-equation oracle** given fixed
  spline geometry and weights. The production fit is iterative SciPy
  LSQR and selects support/weights based on noisy data: its finite
  stopping and nonlinear selection derivatives are NOT modelled.
  Merely replaying output sky FITS does not establish operator
  derivative correctness at finite iteration.
- The source map may already have undergone cosmic cleaning, gain
  normalization, FlatField calibration or other correlated
  transformations. The selected source is not automatically the
  original raw CCD, and component-source closure does not validate
  upstream cross-exposure correlations.
- The evaluator accepts **only native SkySub CCD output functionals**.
  There is no bridge yet to actual COUNT_V1 row maps, geometric Y
  shifts, normalized Stack2D weights, TRACE/FIXED extraction maps or
  the real wavelength-by-wavelength flux-calibration transfer. It
  MUST NOT be interpreted as final 1D uncertainty.
- Gaussian Monte Carlo tests of the frozen linear map introduce an
  explicitly *independent toy VREST* and one shared bias random field
  and check the conditional formula. They do not independently
  reproduce the entire adaptive SkySub run for every realization.
- Default resource bounds are intentionally too small for full CCD
  science exposures; larger WLS factorization requires an independent
  memory/time and numerical conditioning study.

The only output status is
`HOLD_CONDITIONAL_SOURCE_NOISE_NOT_FULL_COVARIANCE`.
`science_uncertainty_qualified=false` and
`production_variance_release_allowed=false` unconditionally.

**Next:** independently verify the exact production COUNT_V1 validity
masks and per-row operators and bind output-shift/scale maps using
P1.4d2 artifacts to *these same input exposure hashes*. Then compare
cross-stage conditional sampling coverage with independent
noise-injection ensembles (especially shared flat-field noise and
data-adaptive weighting), and determine effective extracted LSF.
