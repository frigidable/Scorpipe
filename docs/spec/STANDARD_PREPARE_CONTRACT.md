# Standard Prepare Contract

**Scope:** `standard_prepare` outputs used by future `sensfunc` / `telluric_model` stages.

## Intent

`standard_prepare` exists to reduce a standard star as a calibration object, not as
an ordinary science target. The primary product is optimized for **flux honesty**.

## Primary semantics

The stage MUST produce:

- `standard_1d.fits`
- `standard_trace.json`
- `standard_qc.json`

The preferred default extraction policy is:

- traced aperture;
- wide box extraction as the primary output;
- local sky correction allowed;
- optimal extraction may be generated only as a diagnostic.

## QC expectations

`standard_qc.json` should expose enough information for UI/automation to answer:

- which standard was selected,
- whether association to the science setup is trusted / approximate / degraded,
- whether the extraction aperture is likely too narrow,
- whether detector-edge truncation or contamination is plausible,
- whether the resulting product is safe only for relative response or for
  stronger absolute-calibration claims.
