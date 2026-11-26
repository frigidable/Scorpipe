# DQ1D v1 — aperture-input provenance vs output science usability

**First implementation** in the branch based on PR #120. The canonical
2D maskbits schema (`SCORPMKV=2`) is unchanged. `DQ1DVER` is a separate
per-spectrum semantic version. Never infer it from the global 2D schema.

## FITS contract

When `DQ1DVER=1` and `DQ1DSEM=FATAL` are present:

- `MASK_TRACE`, `MASK_FIXED`: fatal bits for the **produced 1D channel**.
  The current extraction filters and output conditions can reliably express
  `NO_COVERAGE` and `SKYMODEL_FAIL`. Unknown bits are retained as fatal,
  never silently cleared. All other recognized excluded-input bits belong to
  provenance and are not a reason by themselves to discard an accepted bin.
- `DQ_ANY_TRACE`, `DQ_ANY_FIXED`: bitwise OR of original 2D pixel DQ
  **inside the geometric aperture** at each wavelength, including pixels
  excluded from flux/variance. The source image is not modified.
- `FRAC_TRACE`, `FRAC_FIXED`: number of valid spatial contributors
  divided by the **nominal** `2*APHW+1` spatial aperture rows. This is a
  geometric good-pixel fraction (0..1), not a measured flux fraction.
- The preview `DQANY2` and `FRAC2` images contain TRACE and FIXED rows;
  `MASK2` retains the output science usability mask.
- 1D FITS readers and product validators reject malformed/unknown versioned
  schemas. Untagged historical FITS keep their original conservative
  mixed-mask interpretation and are **not** silently upgraded.

## Cross-stage science policy

1. Extract1D excludes input `NO_COVERAGE`, `BADPIX`, `COSMIC`,
   `SATURATED`, `USER`, `INVALID_WAVELENGTH` and `OUTSIDE_SLIT`
   from the weighted flux. `REJECTED` is excluded for single frames, but
   informational in provenance-qualified Stack2D, as in PR #120.
2. Partial input `NO_COVERAGE` must not create output `NO_COVERAGE`
   when enough valid spatial contributors remain. An output-fatal
   `NO_COVERAGE` remains when extraction coverage/profile constraints fail.
3. The extraction may produce a finite, statistically usable spectrum with
   `FRAC<1`; **this does not imply that full-aperture object flux was
   recovered**, particularly for boxcar. A flux fraction cannot generally
   be inferred from a pixel fraction.
4. Sensfunc excludes incompletely sampled standard-star apertures from
   fitting response. Fluxcal copies `DQ1DVER` and per-extraction DQ/FRAC
   columns unchanged. LineMeasure receives the spatial fraction only for tagged v1 spectra.
   If any pixel in the *line integration window* has `FRAC<1` or fails
   output spectral quality, the integrated line flux is marked partial and
   automatic `detected=True` is forbidden, even if the S/N is formally >3.
   Such a partial integral cannot be advertised as a full-flux upper limit.
5. Sky failure (including `QADEGRD`) remains fatal to science lines.

## Remaining P0 qualification gaps

- The profile-dependent effective fraction of retained **source flux** is
  not generally identifiable from `NPIX`; model or hold-out estimates
  are required to recover partial line flux scientifically.
- This version is deliberately conservative for emission-line measurement.
  A full scientific line model must handle core masking, LSF/shape and
  continuum uncertainty, rather than publish partial window integrals.
- Existing independent legacy extraction-kernel tests still verify their
  mixed-provenance return values. They are **not** FITS output semantics.
- Common calibration covariance (#117), subpixel spatial covariance (#112),
  local sky estimate uncertainty (#113) and optimal extraction self-profile
  selection bias (#98) remain independent blockers for calibrated errors.
