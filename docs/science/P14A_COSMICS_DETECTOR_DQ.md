# P1.4a — detector DQ handoff across cosmics (draft)

Science scope: raw saturation evaluated before any bias subtraction, flatfield
or pixel inpainting. For recognizable raw `BUNIT=ADU` and a calibrated ADU
threshold (explicit detector config, SATADU, or SATUNIT=ADU), `SATQ=RAW_EVAL`
means the original ADC values were checked; it is **not** a claim that no pixel
ever exceeded the CCD nonlinearity threshold. Other input types retain
`SATQ=UNKNOWN` or `PROP` and require separate qualification.

All cosmics methods (two-frame difference, stack MAD, Laplacian, LA Cosmic)
continue to write a primary-image `*_clean.fits` for the historical GUI.
They must also write a mandatory typed `uint16` `*_clean_mask.fits` containing
original detector bits OR the independent `COSMIC` pixel-replacement bit.
Both files advertise `SATMSK=DQ_V1` and matching `SATQ` provenance; for
`RAW_EVAL`, a missing or malformed companion is an error, **not** a zero mask.
GUI/manual cosmic masks remain separate binary masks; manual edits replace
only `COSMIC` and preserve other detector DQ bits.

The flatfield and direct sky input routes inspect the qualified sidecar and
preserve upstream DQ. `SATQ=RAW_EVAL` survives a processed primary-only
science frame only with the verified DQ_V1 sidecar; a copied FITS header by
itself cannot establish detector quality.

Validation scope: synthetic raw ADU CCD data with known saturation flags,
single/two/three frame cosmic cleaning, manual GUI postprocessing, bias
subtraction, flatfield input mask, direct sky and negative sidecar corruption
tests. Real SCORPIO-1/2 runs, instrument-specific saturation calibration,
whole-pipeline variance modeling and covariance after inpainting remain
outside this change. Legacy output without `DQ_V1` is not retroactively
qualified. Regenerate old products for any published DQ claim.
