# P3 products: LSF, publication-ready export, kinematics prep

## `lsf_model/`

First-class instrumental-resolution products:

- `lsf_map.fits` — per-pixel instrumental sigma map in detector geometry, stored in Angstrom (`LSFUNIT=sigma_A`).
- `lsf_model.json` — measured/fallback model summary, line fits, 1D median resolution summary.
- `lsf_qc.json` — QC summary (coverage, median sigma/FWHM, source).

Semantics:

- `lsf_map.fits` is authoritative for `sigma_instr(lambda, y)`.
- If arc-line fitting is insufficient, the stage falls back to manual grism priors scaled by slit width.

## `publication_ready/`

Science-facing export products with explicit semantics:

- `Spectrum.fts` — linearized stacked science frame on the exact linear wavelength grid.
- `wl.fts` — authoritative wavelength map in detector geometry. Pixel values are authoritative; helper WCS is approximate.
- `dismap.fts` — authoritative instrumental sigma map in detector geometry. No values are extrapolated outside finite wavelength support.
- `publication_provenance.json` — sidecar explaining inputs, outputs and semantics.

## `kinematics_prep/`

External-kinematics handoff:

- `spec_kin.fits` — preferred 1D spectrum plus `SIGMA_INST_A`/`SIGMA_INST_KMS` columns.
- `masks.json` — wavelength intervals to inspect or mask (including skyline-residual intervals where available).
- `README.md` — human-readable guidance for cross-correlation / pPXF workflows.

## Extraction diagnostics

`extract1d` now also writes:

- `extraction_qc.json`
- `extraction_diagnostics.png`

with explicit metrics for:

- trace stability
- contamination risk
- fixed vs trace disagreement
- aperture sensitivity
- local background asymmetry
