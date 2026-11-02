# Audit A11b — native nm wavelength maps and arc-LSF Angstrom widths

The `wavesolution` stage can write a detector-geometry `lambda_map`
in **nm**, whereas hand-pair arc line wavelengths and instrument
resolution priors are in **Angstrom**. The former `lsf_model` used
`sigma_px * d(lambda_map)/dx`, then unconditionally labelled the
product `sigma_A`. For a nm wavelength map this introduced a
**factor-of-ten** physical unit error; line location matching to
Angstrom hand pairs could also fail and trigger a misleading manual
resolution prior.

A11b explicitly converts detector lambda_map coordinate values to Å
before measuring arc centers, local dispersions, line-profile widths
and fallback coverage. `lsf_model.json.summary_1d.lambda_A` and
`sigma_A` are guaranteed to be numeric Å. The LSF FITS BUNIT,
LSFUNIT, approximate WCS WAVEUNIT/CUNIT1 and LAMUNIT all agree, while
`NATWUNIT` and JSON `native_wave_unit` preserve the original wavelength
mapping. Unknown units fail closed.

The result remains **detector-geometry** and is explicitly marked
`LSFGEOM=DETECTOR`, `LSFQUAL=F`; neither this fix nor a successful
arc fit provides post-rectification/science-extracted LSF.

Regression compares the same synthetic arc lines and a known pixel
width in native Å versus nm FITS, including full measured-stage
execution, finite output line sigmas, model summary and FITS headers.
This is a unit-invariance test, **not** real-instrument science
validation or slit-illumination qualification.

References: MaNGA DR17 LSF pre/post pixelization conventions:
https://www.sdss4.org/dr17/manga/manga-data/working-with-manga-data/
and MUSE pipeline instrumental LSF modeling:
https://doi.org/10.1051/0004-6361/202037855 .
