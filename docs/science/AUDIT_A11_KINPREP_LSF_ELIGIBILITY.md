# A11 — detector arc-line widths are NOT the final extracted LSF

**Historical basis:** [PR #63](https://github.com/frigidable/Scorpipe/pull/63), reviewed and reimplemented against the current A11b native-unit fixes. **Target:** [vNext PR #180](https://github.com/frigidable/Scorpipe/pull/180), Draft / SCIENCE HOLD.

## P0 finding

The former `kinematics_prep` extracted the `summary_1d.sigma_A` spatial median of **detector-coordinate** neon-arc line fits from `lsf_model/lsf_model.json`. It interpolated this to the rectified/stacked/extracted 1D wavelength array (with edge extrapolation), then exported finite `SIGMA_INST_A`/`SIGMA_INST_KMS` values and `SIGINST`/`FWHMINST` header keywords as if these were the *effective extracted science LSF*. This is invalid without a verified forward operator.

Changes in the science LSF include detector pixel integration, slit illumination, wavelength rectification, residual flexure, subpixel shifts, robust stacking, weight selection and source-profile extraction. Existing A11b corrects native nm→Å in the detector-arc map and stamps `LSFGEOM=DETECTOR`, `LSFQUAL=F`; that does **not** magically supply an independently validated final extracted LSF.

MaNGA DR17 explicitly distinguishes `LSFPRE` versus `LSFPOST` for pixel-integration and effective fitted resolution, and transfers resolution through its own resampling operator (https://www.sdss4.org/dr17/manga/manga-data/working-with-manga-data/). A detector-arc median cannot be substituted for this process.

## Strict downstream export contract

- The output `SPEC_KIN.SIGMA_INST_A` and `SIGMA_INST_KMS` are **all NaN** until a separately scientifically validated post-reduction effective LSF exists. Both inherited `SIGINST` and `FWHMINST` FITS header cards are deleted, not merely omitted.
- When the source JSON genuinely declares `instrument.wave_unit=Angstrom` (including after A11b converts a native nm input) and valid medium `WAVEREF`, an arc detector proxy can be shown separately as `SIGMA_ARC_PROXY_A`/`SIGMA_ARC_PROXY_KMS`. It is a **diagnostic only**. Invalid or older `wave_unit=nm` source files are not interpreted as Angstrom. Medium conflicts are blocked.
- Proxy wavelengths must be finite strictly increasing, with strictly positive finite sigma and a matching number of samples. Interpolation is restricted to the arc-covered wavelength span; **no proxy edge extrapolation**.
- FITS `LSFQUAL=F`, `LSFGEOM=UNQUAL_1D`, and `LSFSTAT` specify the actual situation. `ARCSIG`, where present, is clearly an arc-only proxy. `lsf_provenance.json` states `effective_1d_lsf_qualified=false`, available detector diagnostic, and the missing rectification/stack/extraction operator assumptions. `done.json` also states `lsf_qualified=false`.
- The 1D science flux, variance and mask are copied without numerical correction or implicit Gaussian LSF approximation. Advisories about `VELSCALE` do not qualify any pPXF sigma.

## Future physical qualification

1. Fit a detector-arc LSF with empirical hold-out lines, slit width, spectral position and air/vacuum/native-unit QC.
2. Propagate an input monochromatic kernel through **the same COUNT_V1 wavelength overlap/resampling operator**, flexure correction, alignment, stacked weights, DQ and TRACE/FIXED extraction.
3. Compare against observed sky/arc line shapes where illumination and extraction geometry permit. Slit-filling arcs and point sources cannot automatically share the same slit illumination function.
4. Validate residual kernel shape and wavelength dependence; a Gaussian `sigma_A` is only an approximation, especially after robust stacking. Quantify uncertainty in the LSF and its effect on recovered dispersions in pPXF versus E-MILES template resolution.
5. Only then populate science `SIGMA_INST` with an explicit qualified spectral kernel or Gaussian approximation and tested uncertainty.

Automated tests cover Angstrom/nm science axes, native nm converted by A11b, FITS LSFQUAL and NaN science LSF, medium/unit mismatch, missing products and invalid arc proxy, no edge extrapolation and retained numerical science data. No real SCORPIO exposure or pPXF fitting was validated. **SCIENCE HOLD remains.**
