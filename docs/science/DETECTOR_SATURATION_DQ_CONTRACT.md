# Detector saturation qualification (Issue #38)

State: synthetic qualification PR. This is a safety/correctness improvement,
not proof that the complete SCORPIO-1/2 raw CCD saturation and nonlinear response
are fully characterized.

## Why linearize must not re-detect saturation

The original detector sample is ADC output (ADU) **before** bias subtraction,
flat-field correction, cosmic-ray replacement and sky subtraction. After any of
these operations, SCI no longer equals that ADC value. `linearize` was comparing
`_guess_saturation_level_adu` against electron-converted, often sky-subtracted
SCI. Even gain conversion alone cannot recover saturation after sky subtraction.

## Revised contract

- `carry_detector_saturation` can classify pixels *only* if BUNIT explicitly
  declares ADU and processing provenance indicates no bias/flat/cosmic/sky
  transformation. The detector threshold and margin are compared in ADU.
- An explicit `detector.saturation_adu` takes priority; legacy
  `linearize.saturation_adu` is supported on **raw inputs**.
  `SATADU` from header is also permitted; generic `SATURATE/SATLEVEL`
  is permitted only with `SATUNIT=ADU`.
  No unqualified 16-bit or DATAMAX heuristic establishes saturation.
- `SATQ=RAW_EVAL`: classification performed on raw ADC ADU, even if no
  saturated pixels found. `SATQ=PROP`: upstream SATURATED evidence, but
  detection provenance is not independently verified here.
  `SATQ=UNKNOWN`: no trustworthy assessment.
- `flatfield.apply_flat` now classifies raw inputs **before** calibration.
  `sky_sub._prepare_sky_input_planes` classifies only raw ADU direct inputs,
  otherwise just propagates mask bits.
- `linearize._process_linearize_frame` no longer classifies from an
  already corrected sky spectrum. It propagates SATURATED and excludes those
  source pixels from valid SCI/VAR overlap contributions.
- A prior `SATQ=RAW_EVAL` is preserved through subsequent transformations.
  Existing SATURATED bits are preserved even if the processed SCI residual
  is small or negative.

## Explicit limitations / future blocking work

- The existing cosmics implementations may consume raw ADC data and write
  interpolated or bias-subtracted clean frames without retaining a fully
  qualified saturation DQ. Those modes need their own RAW-source matching
  and output bitmask propagation audits; no saturation inference is allowed
  from cleaned samples.
- Absence of the SATURATED bit is **not** proof of unsaturated data unless
  the output provenance demonstrates a full raw assessment.
- `SATURATE` can mean ADC or full-well electrons depending on instrument;
  requiring an explicit ADU unit prevents hidden unit conflation.
  SCORPIO-specific instrument saturation and nonlinearity characterization
  must be verified independently against detector documentation.
- No bad pixel, blooming, persistence, crosstalk, or full-well electron
  nonlinear-response model is implemented by this PR.
- Historical reduced FITS do not gain valid upstream saturation masks merely
  by being reread. They must be reprocessed from the original detector frames
  once the full calibration path is qualified.
- Before scientific merge of P1.3c (#35), evaluate whether the affected
  observations followed a qualified raw saturation path or remain SATQ=UNKNOWN.

Test oracles use threshold margins, gains 0.9 and 2 e-/ADU, raw and processed
input paths, sky SCI counterexamples, FITS MEF propagation and explicit
absence of raw threshold evidence.
