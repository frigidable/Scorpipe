# P1.4f — read-only admission of upstream FlatField variance components

**Status:** opt-in audit, not a science covariance product. SCIENCE HOLD.

The actual vNext FlatField stage can save a component-resolved MEF per
exposure when the indexed MasterBias has a physical `SB_GID`:
`VBIAS` and `VREST` (e-²), signed `BSENS` (e-/ADU),
and `BVARADU` (ADU²). All are in native source CCD geometry.
The numerical contract for every detector pixel is

```text
VAR = VBIAS + VREST
VBIAS = BSENS² × BVARADU
```

These relations are preserved **before** sky subtraction. The
production SkySub deliberately drops component planes and marks its
output `VCMPQ=SKY_LOST`: its FITS VAR is detector-only, not a
validated propagated residual-sky covariance.

## New bounded check

`calib.sky_masterbias_source_audit.audit_sky_masterbias_group_sources`
accepts an explicit list of one or more actual
`_kelson_frozen_fit_lineage_v1.json` manifests saved by the production
SkySub. It performs, in order:

- Replay the frozen Kelson fit against both Sky FITS files while verifying
  SHA-256 of the actual selected source, solver archive and outputs.
- Reopen that **exact selected input**, not a guessed RAW or restored
  post-sky source. If it has no versioned component planes, return a
  science HOLD; never reconstruct from final `VAR` alone.
- For a declared `VCMPVER=1` require complete source SCI/VAR/4
  component HDUs, `VCMPQ=LOCAL`, `FLATCOR/BIASSUB/BVARINC=True`,
  explicit `SCORPUM=ELECTRON`, `BUNIT=e-`, matching nonempty
  `SB_GID=VCMPGID` and exact per-component BUNIT. Reject any
  nonfinite or negative variance, inconsistent CCD shape, non-normalized
  decomposition or inconsistent signed MasterBias response.
- For exposures sharing one `SB_GID`, demand exactly matching *full
  2D* `BVARADU` maps and CCD geometry. The same group label alone is
  not evidence of the same MasterBias estimator. Reject use of the
  same source file twice as independent exposures.
- Enforce explicit input count and per-exposure pixel limits.

Even with verified local components, the only possible overall success
status is **`HOLD_PRE_SKY_VARIANCE_COMPONENTS_PRESENT_BUT_NOT_PROPAGATED`**.
Absent/incomplete components lead to an explicit HOLD or a fail-closed
exception. The report always has
`science_uncertainty_qualified=false` and
`production_variance_release_allowed=false`.

## What remains unproven

`VREST` is the variance remaining after removing the MasterBias term
**in each individual image**, not automatically an independent noise
field across exposures. It can contain shared MasterFlat noise or
other calibration systematics. The signed `BSENS` and the group-specific
`BVARADU` are physically necessary to model shared-bias covariance,
but none of these labels alone establishes that the original detector
variance is complete.

The new check does **not** differentiate noisy sky pixel selection,
prove exact LSQR normal-equation convergence, reconstruct the full
unit/cosmic transformation, bind every COUNT_V1 row operator and final
Stack2D/extraction calibration to one run, or propagate covariance
through the chain. Nor does it produce certified per-wavelength
uncertainties, PSF/LSF measurements or publication-ready 1D error
arrays.

Tests execute an instrument-free synthetic observing night through the
actual SkySub entrypoint with explicit flat-corrected component planes,
and then test inconsistent group maps, missing planes, unsigned or
wrong-unit data and repeated exposure identity. **No original observed
SCORPIO FITS are touched.**
