# Issue #45 — Independent TRACE and FIXED flux calibration

## Confirmed defect on main before this change

The old calibration reader selected the TRACE spectrum only and the writer
copied its calibrated flux and variance into **both** SPEC_TRACE and SPEC_FIXED
and both FLUX2/VAR2 preview rows. This overwrote the science identity of the
fixed-aperture extraction and potentially misled later interpretation.

## Corrected contract

- Read TRACE and FIXED as separate extraction products. In COUNT_V1 mode,
  retain the existing standard-to-science bin width, BUNIT and wavelength
  reference guards. TRACE and FIXED must share the same science wavelength
  axis to better than 1e-9 Å (in the existing file coordinate units);
  no implicit interpolation between two different grids.
- Treat response and optional telluric transmission as the **same
  deterministic multiplicative transfer** for each extraction mode:
  F_cal,m = F_raw,m / response / transmission and
  VAR_cal,m = VAR_raw,m / response² / transmission².
  The two modes have different input signals and variances, so neither
  may be copied from the other.
- Preserve each mode's MASK and NPIX values. Calibrate the stored SKY
  residual value by the same physical transfer but retain its **per spatial
  pixel** meaning: it is not the integrated aperture sky flux.
  Stamp SKYUNIT for the two tables; update FLUX2/VAR2/SKY2 preview data
  independently and their BUNIT headers. MASK2 and NPIX2 must match
  their originating table rows.
- A legacy TRACE-only spectrum can still be used with the mathematical
  `apply_fluxcal_to_spec1d` API (for compatibility and unit guard tests).
  It cannot be written as a qualified two-mode calibrated FITS. The writer
  explicitly raises rather than fabricating FIXED data.
- Response-fit and telluric calibration uncertainties can be spectrally
  correlated and shared by TRACE/FIXED. This patch only propagates the
  **conditional detector VAR** for a fixed transfer function; CALSYS and
  the pipeline's broader covariance modeling are separate concerns in #42.
- Existing calibration status classifications are unchanged. The FITS
  header EXTID=TRACE+FIXED records the new independent-mode contract.

## Regression

Tests use deliberately different TRACE/FIXED fluxes, variances, sky
residuals, masks and aperture sample counts, wavelength-varying response,
and optional wavelength-varying telluric transmission. They check both
tables, every preview row, and the calibrated 2D companion. Negative tests
cover the absence of FIXED, a mismatched science wavelength axis, a bad
FIXED array length, a corrupted MASK2 preview and incompatible COUNT_V1
standard unit. These are programmatic/synthetic tests, not a validation
using real SCORPIO-2 observations.

## Migration and downstream scientific use

Previously written fluxcal_spec1d.fits files may have duplicated TRACE into
FIXED and must be treated as unqualified for fixed-aperture science. Rerun
calibration from the original, distinct 12_extract/spec1d.fits after this
fix. If the original extraction was lost, the missing FIXED spectrum cannot
be recovered from an old duplicated calibrated product.

This change is separate from covariance transport P1.4d: no changes to
`measure_line`, pPXF, extraction, or robust stacking uncertainties.
