# Audit A02/A03 — signed emission detection and spectral DQ

**Affected code:** `science/lines.py`, `stages/line_measure.py`,
`science/reddening.py`.

## Defects

The legacy rule was `detected = (abs(integrated_flux)/sigma >= 2) OR
(any positive line residual)`. Consequently even a tiny positive
noise residual caused a declared detection. Large negative absorption
could also be treated as a detection because the S/N was absolute.
The stage discarded the input `SPEC_TRACE.MASK_TRACE` entirely, so an
outlier in a saturated/cosmic/failed-sky pixel contributed to the
measured line, centroid, S/N and Balmer decrement.

## Revised detection contract

For the **diagnostic** line-integral mode, `detected` requires positive
continuum-subtracted integrated flux with *signed* statistical S/N
at least `science.line_detection_snr_min` (default 3.0).
Non-detections retain a signed flux measurement and `reason` but
publish no centroid, dispersion or velocity.

Samples marked `NO_COVERAGE`, `BADPIX`, `COSMIC`, `SATURATED`,
`USER`, `REJECTED`, `OUTSIDE_SLIT`, `INVALID_WAVELENGTH`, or
`SKYMODEL_FAIL` are excluded. Informational `EDGE` alone is allowed.
Only finite positive variance is usable. Both sideband continuum
windows require at least two usable samples (numerical guard, not a
validated science criterion). Spectral widths are calculated **before**
excluding pixels to prevent a masked gap from increasing the
integration weights of its neighbors.

The Balmer decrement will not use explicitly non-detected line fluxes
even when the numerical flux happens to be positive.

## Scientific limitations — separate qualification required

- Default 3-sigma is a nominal diagonal-noise criterion, **not** a
  controlled global false discovery rate. In particular, scanning
  multiple lines requires multiple-testing correction if used for
  statistical catalog selection.
- This is *not* a profile fit, continuum-uncertainty propagation, line
  deblending or complete covariance-aware upper limit method.
- The default spectral-line windows may overlap nearby lines (A10).
  Continuum placement, stellar Balmer absorption and local blends may
  bias recovered line flux even with corrected DQ.
- Air/vacuum compatibility, velocity reference frame and redshift
  conventions still require the A05 audit.
- Existing saved line products are not rewritten; reprocess affected
  spectra before using their detection flags or reddening estimates.

## Gates

Independent synthetic tests cover low positive noise, strong negative
absorption, a real positive emission line, huge masked cosmic/sky
outliers, EDGE-only cases, one-sided continuum and invalid variance;
a stage-level replay verifies actual DQ handoff. Real SCORPIO
observations, injected lines and Monte Carlo coverage remain required
before promoting diagnostic measurements to science-qualified line
catalogs.
