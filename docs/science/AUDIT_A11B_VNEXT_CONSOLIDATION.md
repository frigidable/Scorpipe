# A11b: detector ARC-LSF native units (vNext consolidation)

**Source review:** [PR #64](https://github.com/frigidable/Scorpipe/pull/64), based on historical main. **New implementation:** [PR #184](https://github.com/frigidable/Scorpipe/pull/184), targeting Draft vNext [PR #180](https://github.com/frigidable/Scorpipe/pull/180). Status: **SCIENCE HOLD**.

Wavelength-solution maps may be stored in `nm` or `Angstrom`, while arc-line catalogues, resolution priors, output `sigma_A`, empirical line brackets and published wavelengths are all in Angstrom. Fitting a stored nm map without conversion makes physical widths and line matching inconsistent by a factor of ten.

The new boundary converts a finite positive native map to Angstrom before ARC profile fitting, fallback, line-support testing, approximate spectral WCS, QC and export. Unsupported/missing units fail closed. Output `WAVEUNIT`, `CUNIT1` and `LAMUNIT` refer to Angstrom; `NATWUNIT` and JSON scaling metadata preserve native units. `LSFGEOM=DETECTOR`, `LSFSCI=False` and `LSFQUAL=False` explicitly reject science-extracted LSF qualification.

The test compares physically identical synthetic arc profiles saved as nm and Angstrom maps, including fitted widths and resulting metadata. Passing it is an algorithmic unit-contract check, **not** evidence that the detector arc LSF represents the resolution of a rectified or extracted galaxy spectrum. Slit illumination, object centering, spatial structure, fit support, wave-calibration quality and covariance remain independent science requirements.

Do not cherry-pick the older main-based PR wholesale: vNext has substantially stricter arc-line profile gates and wavelength-support accounting, which must be preserved.
