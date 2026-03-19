# Calibration Stack v1

This note documents the stable calibration stack semantics used from the v7.70 hardening pass onward.

## Stage split

1. `standard_prepare` — prepare a flux-honest standard spectrum
2. `sensfunc` — model the smooth instrumental response
3. `telluric_model` — model atmospheric absorption explicitly
4. `fluxcal_apply` — apply those layers to science outputs

## Semantic freeze

The calibration stack now uses one frozen vocabulary across code, validators, QC and reports:

- `RELATIVE`
- `RELATIVE_TELLURIC`
- `ABSOLUTE_APPROX`
- `ABSOLUTE_TRUSTED`

`RAW` is reserved for uncalibrated semantics and is not legal for validated calibrated products.

## Product-kind policy

### 1D calibrated products (`fluxcal_spec1d.fits`)

Allowed claims:
- `RELATIVE`
- `RELATIVE_TELLURIC`
- `ABSOLUTE_APPROX`
- `ABSOLUTE_TRUSTED`

### 2D calibrated products (`fluxcal_2d.fits`)

Allowed claims:
- `RELATIVE`
- `RELATIVE_TELLURIC`
- `ABSOLUTE_APPROX`

`ABSOLUTE_TRUSTED` is intentionally disallowed for the 2D calibrated product contract.
Even when 1D extraction can justifiably reach trusted absolute semantics, the 2D product remains a more conservative representation.

## Error-model rule

Statistical variance remains in the ordinary science `VAR` plane. Calibration uncertainty is written separately into `calibration_systematics.fits` and is not hidden inside per-pixel noise.

`calibration_systematics.fits` is part of the first-class contract, not an optional diagnostic.
It must carry at least:
- `CALSYS_REL`
- `CALSYS_TELL`
- `CALSYS_ABS`
- `CALSYS_TOTAL`

with `CALSYS_TOTAL = quadrature(CALSYS_REL, CALSYS_TELL, CALSYS_ABS)`.
