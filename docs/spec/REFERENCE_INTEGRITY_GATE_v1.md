# Reference Integrity & Fail-Closed Calibration v1

**Scope:** Sci-safe calibration gates in PR built on `0e8bca4`. This is a
first-stage safety contract, not certification of flux or telluric accuracy.

## Provenance-validated spectrophotometric standards

A standard-star spectrum can be used for `sensfunc` only if all conditions
hold:

1. The observed standard has a nonempty object identity, matching the selected
   library key, `display_name` or explicitly curated `aliases` exactly after
   normalization. The missing-name or first-entry fallback is forbidden.
2. The matched record declares `scientific_status: validated` (or `verified`)
   and a nonempty `source` with identifiable, independently checked origin.
3. The current implementation accepts only **absolute** physical
   `f_lambda` in erg s^-1 cm^-2 Å^-1, with explicitly declared `air`
   or `vacuum` wavelengths. Data in Jy, f_nu or AB magnitudes require
   a deliberate conversion before import.
4. The exact named table exists. No fallback to another standards.* table
   is allowed.

These metadata requirements are *necessary, not sufficient*: a library author
must independently validate the file against its named source, verify its
wavelength coordinate convention and flux scale, and preserve the reference
snapshot and SHA-256 in the pipeline reference context.

The bundled BD+28°4211 four-point table is marked
`scientific_status: unvalidated_placeholder` and is **blocked** by default.
It remains in the package temporarily for provenance/diagnostic comparison,
not for scientific calibration. To support BD+28°4211, supply an independent,
fully sampled, checked CALSPEC/ESO reference via `fluxcal.standard_library`
and the matching reference table; do not simply relabel the four-point data
as validated.

## Standard-star flat-fielding

If `fluxcal.standard_apply_flat=true`, a missing or failed flat-field
correction terminates the standard preparation. An explicit
`standard_apply_flat=false`, or reusing an already rectified file whose flat
provenance is not verified, is allowed only for diagnostic/relative response:
`standard_prepare.safe_for=response_only` with
`STANDARD_FLATFIELD_UNVERIFIED`.

This gate does NOT yet establish whether a standard global SuperFlat and
science per-set MasterFlat have the same spectral transfer function. The
cross-stage flat policy remains a separate scientific P0 issue (#94).

## Telluric model

The existing sparse bundled O2 template cannot be interpreted as physical
transmission between O2 B and A. The diagnostic model therefore does not
blend that template, preserves transmission 1 and confidence 0 outside the
configured telluric windows, and stamps the output FITS
`TELQUAL=DIAGNOSTIC`. The stage returns WARN.

`fluxcal_apply` **refuses to apply** a diagnostic-tagged model to a
science spectrum. To flux-calibrate without a qualified telluric model,
explicitly omit the telluric input and treat affected bands as uncorrected.
A future scientifically qualified mode must incorporate time/airmass,
atmospheric state, line saturation and instrument resolution with independent
held-out spectral tests. The current template is not silently promoted to it.

## Release restrictions and remaining work

- No claim that `scientific_status=validated` alone proves correct CALSPEC
  content or aperture/slit-loss calibration.
- Bring generic standard lookup utility in PR #106 into parity with this
  strict Sensfunc identity gate before merging.
- Eliminate untagged legacy telluric model application in a follow-up
  migration, with upgraded product schema and explicit compatibility handling.
- Require a trusted standard/science Flat transfer contract and matching
  per-science atmospheric extinction before automatic ABSOLUTE_TRUSTED use.
- Tests must pass at an exact pinned SHA and then be challenged by a real
  instrument-specific reference observation.
