# A01: Sensfunc grid and variance validity (vNext consolidation)

**Source reviews:** historical audit [PR #54](https://github.com/frigidable/Scorpipe/pull/54) and independent full-grid [PR #104](https://github.com/frigidable/Scorpipe/pull/104). **Integration:** [PR #183](https://github.com/frigidable/Scorpipe/pull/183) into Draft vNext [PR #180](https://github.com/frigidable/Scorpipe/pull/180). Status: **SCIENCE HOLD**.

The current vNext code already fits only accepted standard-star pixels while evaluating the response on the complete wavelength grid; it also verifies finite positive output and writes numerical FITS mask columns. Those earlier main-based patch variants must not be blindly merged into this divergent history.

A unique missing condition remained: `isfinite(VAR)` admits nonpositive variances. These samples were assigned an artificially large inverse error through numerical clipping and could distort the sensitivity fit. The integration fix requires `VAR > 0` to train, rejects fewer than eight physically valid training samples and records training-point count and evaluated wavelength domain in QC.

The synthetic regression checks zero, negative and nonfinite input variance, complete response grid retention and fail-closed behavior when all variances are invalid. It does **not** certify the eight-sample threshold as sufficient for actual spectrophotometric accuracy, nor resolve reference CALSPEC provenance, extinction, slit losses, tellurics, flat-field compatibility or covariance of the fitted response.

Before closing historical PRs as superseded, confirm the exact integration commit and CI; retain their source branches and audit history. No observed spectra were recalibrated by this change.
