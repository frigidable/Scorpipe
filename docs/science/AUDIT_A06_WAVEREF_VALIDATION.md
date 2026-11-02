# Audit A06 — reject unknown wavelength-medium labels

A wavelength solution and its downstream products must preserve the
physical medium of the wavelengths. FITS spectral-WCS conventions
distinguish wavelength coordinates, and air/vacuum is a physically
measurable transformation, not a display preference:
[Greisen et al. 2006, A&A 446, 747](https://doi.org/10.1051/0004-6361:20053818).

Previously `wavesolution.py` accepted an explicit unrecognized
`wavesol.wave_ref` and wrote `WAVEREF=air` in its place. That could
silently mislabel a wavelength map and lead to velocity biases.

The new validation runs at `build_wavesolution` entry, before creating
stage artifacts, and only accepts explicit `air` or `vacuum` for
configured values, ignoring case and surrounding whitespace. The
legacy `waveref` alias works. Absent settings retain the historical
default `air`; a supplied empty, null or unknown setting fails closed.

**Remaining science limitation:** even a syntactically valid
`wave_ref=vacuum` is not proof that the arc input line list or manually
identified lines were expressed in vacuum. The physical conversion or
provenance matching of *input line medium* to *output medium* needs
another scoped validation. This PR solves only false defaulting from
an unknown label, not all spectral-coordinate mismatches.

Tests: accepted values, rejected values, alias precedence, backward
compatibility, stage-level failure before any output directory.
