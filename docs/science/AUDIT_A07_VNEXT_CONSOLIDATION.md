# A07 — Sensfunc reference wavelengths in native science coordinate units

**Review source:** original [audit A07 PR #59](https://github.com/frigidable/Scorpipe/pull/59), based on the old main-based A01 implementation. **New integration:** Scorpipe vNext [PR #180](https://github.com/frigidable/Scorpipe/pull/180). **Status: SCIENCE HOLD.**

A standard-star `SPEC_TRACE.LAMBDA` may be saved in **nm**, while the scientific reference-star table (`wavelength_A`), extinction-curve table (`wavelength_A`), Balmer intervals and telluric intervals use **Å**. Prior vNext code compared these coordinate arrays directly and wrote `WAVEUNIT=Angstrom` even for nm input. That can silently misalign or invalidate response, extinction and masks.

This scoped change identifies the standard's declared wavelength unit and converts **reference coordinates only** from Å to the standard's native unit before interpolation/masking. It does not multiply reference numerical `F_lambda` or change the intended physical output unit, erg/s/cm²/Å. Native `WAVEUNIT` is retained in the sensitivity FITS, while standard science `INWUNIT` and `INSTEP` metadata remain linked to the native count/bin grid. Diagnostic Balmer spans are plotted in the same unit as the data.

Explicit unknown units fail closed. Modern `COUNT_V1` spectra must declare a supported unit; legacy missing-unit spectra retain an assumed Å path for compatibility but must not be treated as externally science-validated.

Regression uses injected smooth throughput, a nonconstant extinction curve, airmass 1.25, masked Balmer/telluric bins, and writes/reloads the FITS sensitivity table on a 0.2 nm grid. It checks correct response and native coordinate metadata. Other independent problems remain, particularly reference-star CALSPEC validity, atmospheric transmission for the science exposures, standards/objects flat lineage, spectral response covariance and slit losses. Green CI does not release absolute spectrophotometry.

Do not merge original #59 wholesale: its patch is stacked on historical A01 and would replace later strict vNext semantics. Preserve the original PR and its commit history; close it as **superseded** only after the focused integration has passed exact-HEAD CI.
