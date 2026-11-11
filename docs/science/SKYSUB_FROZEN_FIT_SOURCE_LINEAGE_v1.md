# SkySub Kelson frozen-fit archive — exact source and output lineage (v1)

**Status: diagnostic-only; SCIENCE HOLD.** This contract is opt-in through
`sky.covariance_capture.enabled: true` (or existing `sky_sub` compatibility
configuration). It modifies neither production FITS SCI/VAR/MASK nor the
Kelson solver, sky selection, optical wavelength solution or observing
geometry. The existing archive is not made scientifically qualified.

## Existing science path, confirmed directly in vNext

The production `_fit_kelson_bspline` performs data-derived masking,
iterative sigma clipping and finite-iteration SciPy LSQR on a weighted
B-spline basis with optional linear y-dependence. When capture is enabled,
`capture_kelson_fit` already writes the **actual final selection of native
CCD sky pixels**, fitted detector variance, evaluated wavelength map,
spline knots, fitted coefficients and LSQR diagnostics to
`<stem>_kelson_frozen_fit_v1.npz`. The FITS products record
`SKYFRZQ=FIT_ONLY`. The independent
`audit_frozen_kelson_products` checks native geometry, the reconstructed
model and LSQR residual norm against the resulting FITS.

This existed *before* the present change. It is not correct to describe
the actual post-clipping frozen fit as missing from Scorpipe.

## New v1 binding of a single observed reduction

Immediately after SkySub writes both
`<stem>_skymodel_raw.fits` and `<stem>_skysub_raw.fits`, an enabled
capture writes `<stem>_kelson_frozen_fit_lineage_v1.json`, atomically,
with full-file **SHA-256 and byte size** for four distinct files:

1. The actual **selected SkySub input file path** (which may be
   cosmics-cleaned or otherwise adapted, not necessarily original RAW CCD);
2. The actual `_kelson_frozen_fit_v1.npz` snapshot;
3. The production sky-model FITS;
4. The production sky-subtracted SCI/VAR/MASK FITS.

The path recorded for the selected input is the one used by SkySub, not
an inferred filename from the object manifest. The three derived files
must share the sky-stage directory. The archive and outputs must agree
on the same exposure stem and the FITS headers must identify the named
snapshot via `SKYFRZP`, version and method cards.

A separate read-only
`calib.sky_fit_lineage.audit_frozen_kelson_fit_lineage` checks all file
hashes and sizes, path and naming identity, and FITS-to-frozen-archive
header cards. Its default invokes the existing independent
`audit_frozen_kelson_products` fidelity replay; a changed model cannot
be rehabilitated simply by altering a JSON SHA256 value.

The manifest is optional and is not a claimed requirement of otherwise
valid sky-subtracted data. If its write fails, the pipeline records
`SKY_FROZEN_LINEAGE_FAILED` and omits the new manifest, rather than
fabricating source integrity or replacing valid science with a
soft-failure passthrough. Captures remain disabled by default.

## Crucial covariance limits

**A file hash binds bytes, not detector-noise provenance.** The selected
input frame could be post-cosmics or transformed by a unit adapter.
The new JSON file does not encode the entire data transformation, source
read noise and Poisson components, the signed response to a shared
MasterBias, or an independently validated link to the COUNT_V1
linearization and registered Stack2D/Extract1D/FluxCal operators.

In particular, finite-iteration LSQR is not automatically identical to
an exact solution of the normal equations used in the conditional
`FrozenSkyWLS` adjoint; final fit-mask and robust weight choices vary
with the noisy science frame. The archive supports frozen conditional
inference only. No exposure-time or instrument-specific error model
is inferred from it.

A successful read-only audit reports
`HOLD_UNBOUND_DETECTOR_AND_SKY_COVARIANCE` with
`science_uncertainty_qualified=false` and
`production_variance_release_allowed=false`. The existing
`SKYVQUAL=INCOMP` and wider SCIENCE HOLD remain unchanged. A subsequent
science task must bind the actual detector noise components and
Kelson→COUNT_V1→Stack2D→Extract1D chain **from one and the same validated
run**, then test true injected-noise coverage and pPXF-effective LSF
independently.

## Synthetic validation

`tests/test_kelson_fit_lineage_sha256.py` executes the existing actual
SkySub synthetic-observing-night fixture, verifies FITS and frozen-fit
identities, and deliberately tampers with selected source, snapshot,
sky model and residual FITS. It also checks that re-hashing altered
sky-model data cannot bypass independent model replay, wrong output
directories are rejected, and optional provenance write failures
never alter the science array. **No observer FITS were used or modified.**
