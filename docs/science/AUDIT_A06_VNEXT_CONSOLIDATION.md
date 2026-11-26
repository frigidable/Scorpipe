# A06 — strict wavelength-medium validation in WaveSolution

**Source:** main-based [PR #57](https://github.com/frigidable/Scorpipe/pull/57). **Scoped integration:** vNext [PR #180](https://github.com/frigidable/Scorpipe/pull/180). **SCIENCE HOLD**.

Wavelength map medium is a physical spectral coordinate, not merely a label. Previously `build_wavesolution` converted any unrecognized explicit `wavesol.wave_ref` to `air` before writing `lambda_map.fits`. The FITS file could therefore claim an incorrect physical convention while numerically passing other QC.

A06 now validates `wave_ref` (primary key) or `waveref` (legacy alias) before the stage even creates directories. Only explicit `air` or `vacuum` (case-insensitive) is accepted. An absent setting uses the documented historical air default. Explicit null, empty, unknown and conflicting-key combinations fail closed. The resulting value propagates unchanged into the stage's FITS metadata and downstream data contract.

Regression covers valid/invalid values, alias precedence, backward compatibility and zero output-directory side effects on rejection. An accepted keyword alone does **not** establish that input ARC hand pairs and reference line lists are physically in that medium: independent input-line-medium provenance remains a science gate. No data have been recalibrated, no global default has been changed, and the source PR history remains preserved.
