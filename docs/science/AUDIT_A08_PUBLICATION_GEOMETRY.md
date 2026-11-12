# A08 — pixel geometry qualification for the publication export

**Source:** historic [PR #60](https://github.com/frigidable/Scorpipe/pull/60). **Target:** [vNext PR #180](https://github.com/frigidable/Scorpipe/pull/180) (Draft; SCIENCE HOLD).

The exporter writes `Spectrum.fts` from a **rectified, wavelength-linearized Stack2D**; `wl.fts` and `dismap.fts` retain the **detector sampling** of the upstream wavelength/LSF models. Their FITS arrays cannot be combined pixelwise merely because their `NAXIS1/NAXIS2` happen to agree.

## Output contract

1. Every primary and SCI image header contains `PIXGEOM=RECTIFIED` for `Spectrum.fts` or `PIXGEOM=DETECTOR` for `wl.fts` and `dismap.fts`. `MAPALGN=F` explicitly forbids treating either detector map as aligned with the rectified Spectrum.
2. `publication_provenance.json` publishes `geometry_contract_version=1`, image geometry classes, a `direct_pixelwise_comparison_allowed=false` gate, and the fact that equal array shapes do not establish registration.
3. Even `wl.fts` and `dismap.fts` having identical detector-type dimensions **does not independently verify** that their stages used identical pixel coordinates/origin and detector frame. The manifest therefore marks `wl_dismap_coregistered=null` and `detector_map_alignment_qualification=NOT_INDEPENDENTLY_VERIFIED`. An independent physical geometry/lineage check is needed to upgrade that statement.
4. Explicit incompatible `WAVEREF` (air/vacuum), `WAVEUNIT` and `CUNIT1` declarations are rejected. Missing information is stated as `coordinate_declarations=INCOMPLETE` and **never** inferred from array shape.
5. The exporter does not change the source SCI, wavelength map or instrumental sigma values except the existing finite-wavelength support masking of `dismap`. The detector sigma is **not** a scientifically qualified effective 1D LSF after rectification and stacking.
6. Preserve the newer vNext export safety contract exactly: `EXPORTOK=T` only means files were written; `PUBRDY=F`, `SCIQPASS=F` and `VALLEVEL=EXPORT_ONLY` remain. Never copy historic #60's outdated `PUBRDY=T` into vNext.

## Standards and validation

The FITS WCS conventions describe the mapping from **each dataset's** pixel coordinates to world coordinates; they do not assert a shared resampling operator or physical pixel co-registration. See Greisen & Calabretta (2002), FITS WCS Papers I–III, https://science.data.nasa.gov/fits/documentation/wcs .

Independent synthetic FITS tests cover both unequal and matching shapes, PRIMARY and SCI headers, numerical data preservation including NaN detector support, contradictory wavelength media/units and incomplete declarations. No original SCORPIO exposure or measured LSF transfer has been validated.

**Still required:** instrument-specific detector-to-rectified geometry operator, uncertainty/covariance propagation, physical LSF convolution model, known reference frame and original-FITS science comparison. This change does not lift SCIENCE HOLD.
