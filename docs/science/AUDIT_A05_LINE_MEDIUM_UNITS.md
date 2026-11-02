# Audit A05 — spectral line air/vacuum and wavelength unit alignment

The default atomic rest wavelengths in `science/lines.py` are catalog
**standard-air Angstrom** values. For a science spectrum with
`WAVEREF=vacuum`, its expected observed line center must be derived as

```
lambda_expected_in_spectrum = air_to_vac(lambda_rest_air) * (1 + z)
```

The velocity and sigma-to-km/s conversions must use the **same vacuum
rest wavelength**. Using air rest lambda with a measured vacuum
centroid leaves a spurious apparent radial velocity (~80 km/s optical).

The fitting/integration wavelength grid is now consistently in
Angstrom. The stage converts `WAVEUNIT=nm` coordinates to Angstrom,
without scaling already calibrated `F_lambda` (which retains
`erg s^-1 cm^-2 A^-1`); it accepts only explicit air/vacuum media
for COUNT_V1 science products. Legacy unqualified products without
explicit coordinate provenance are still accepted under an
`LEGACY_ASSUMED_AIR_ANGSTROM` label, **not** science-qualified.

This stage does not itself apply barycentric correction, re-redshift
the observed data, or automatically match source and template
velocity reference frames. Its `science.redshift` must already be
compatible with the measured science coordinate frame, not merely a
literature redshift of unspecified convention.

Source: FITS spectral coordinate conventions (Greisen et al. 2006,
https://doi.org/10.1051/0004-6361:20053818) and the Scorpipe
`airvac` conversion kernel, previously qualified separately.

Synthetic tests include zero-velocity lines in either medium,
redshifted vacuum lines, an actual stage invocation with an nm vacuum
grid, unknown medium, missing COUNT_V1 metadata, and unsupported
wavelength units.

This is dependent on A02/A03 PR #55. It must not be merged before
that dependency, an updated CI run, Windows regression, and scientific
review of all coordinate-system metadata.

CI verification note: this stacked branch is tested against the current
`main` integration snapshot and must be retargeted to its parent feature
branch before review/merge. The code and associated source provenance
remain unchanged by this administrative check.
