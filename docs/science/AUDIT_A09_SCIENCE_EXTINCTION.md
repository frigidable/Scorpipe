# A09 — independent science-target atmospheric extinction (2026-11-11)

**Source review:** historic [PR #61](https://github.com/frigidable/Scorpipe/pull/61), adapted for vNext. **Integration target:** [Draft PR #180](https://github.com/frigidable/Scorpipe/pull/180). **SCIENCE HOLD**.

## Physical transfer

A standard-star response built from above-atmosphere standard counts has the form

    R(lambda) = N_std(lambda) * 10**(0.4 * k(lambda) * X_std) / F_std(lambda)

where k is the atmospheric extinction coefficient in mag per airmass. The science spectrum must be transformed by its own (possibly different) airmass:

    F_sci = N_sci / R(lambda) * 10**(0.4 * k(lambda) * X_sci)
    VAR_sci = VAR_counts / R(lambda)**2 * 10**(0.8 * k(lambda) * X_sci)

The transformation must also account for any separately applied telluric correction and must carry the same multiplicative factor into both TRACE and FIXED extraction, their variance, local sky residuals, and the 2D SCI/VAR. These are the standard observational flux-calibration conventions; see the PypeIt fluxing reference, https://pypeit.readthedocs.io/en/stable/fluxing.html .

## Narrow, fail-closed implementation

- The low-level numerical interface accepts an explicit `science_extinction` model with **finite X in [1,5]** and monotonically increasing **air-Angstrom** reference wavelengths plus k(lambda) in mag/airmass. Convert the reference to the science FITS WAVEREF (air/vacuum) and WAVERUNIT (Angstrom/nm) before interpolation. There is no extrapolation, negative extinction coefficient or missing unit default.
- Science extinction is applied only if `sensfunc_qc.extinction_applied=true` explicitly establishes a separately de-extincted standard. A missing standard correction cannot silently be compensated by applying only the science correction.
- The stage's automatic admission requires a single configured OBJ RAW FITS, an Extract1D `INPUT2D` pointing to the current Stack2D FITS, a successful Stack2D report with **exactly one accepted input**, and a matching processed exposure filename stem. A parser-supplied science airmass must have `header` or `derived` provenance. It must match a reference-context extinction curve containing `wavelength_A` and `k_lambda`. Multiple science frames, missing report, stale/mismatched input lineage or unverified airmass cause **no numerical science extinction correction**; no arithmetic-mean airmass is ever invented.
- The single-stack stem match is a **limited admission check, not cryptographic RAW→processed provenance**. Its qualification is `SINGLE_STACK_STEM_MATCH_ONLY`; without a complete exposure-by-exposure operator history it does not certify final scientific flux calibration. The output remains `ABSOLUTE_APPROX` even after applying a verified numeric factor. Slit loss, throughput temporal changes, calibration-curve uncertainty, shared noise and photometric cross-checks are not resolved.
- TRACE and FIXED are independently transformed. The 2D factor is only applied when its wavelength grid is identical to the 1D factor grid. FITS primary headers state `ATMCORR`, `ATMXSCI`, `ATMPROV`, and JSON QC includes applied status, provenance and warning. `ATMCORR=true` certifies only that a numerical multiplicative factor was applied; never that the uncertainty is scientifically complete.

## Limits requiring further changes

1. Robust Stack2D of multiple exposures needs per-exposure transmission factors or exposure-specific frozen coefficient/weight maps. Spatial/spectral variations in weights invalidate one scalar mean airmass.
2. The existing standard-Sensfunc extinction-table interpolation still needs **independent** verification of air/vacuum transformation, wavelength coverage, reference metadata and time dependence. An empirical k(lambda) model itself carries uncertainty. This A09 numerical science transfer does not validate upstream standard correction.
3. The output `ABSOLUTE_APPROX` designation is qualitative; it does **not** guarantee a bounded 12% absolute calibration error or promote line-flux significance. The current instrument-specific slit-loss/covariance/LSF validation and original-FITS E2E remain science blockers.

Tests cover exactly known science/standard airmasses, nonconstant extinction, 1D and 2D signal/variance consistency, retained independent TRACE/FIXED data, nm vacuum conversion, incomplete curves, provenance failure modes and status qualification.

No original observations or `main` code modified.
