# Audit A07 — Angstrom reference tables on nm spectral grids

This PR is stacked on A01 PR #54, which fixes a separate
`fit_mask` vs response-grid length bug.

Scorpipe's standards CSV `wavelength_A`, extinction-curve CSV
`wavelength_A`, Balmer masking intervals, and telluric masking
intervals all express **Angstrom** coordinates. But extracted standard
spectra may express wavelength in **nm**, as supported by the upstream
wavesolution and COUNT_V1 contracts.

Before this patch, `build_sensfunc` would compare nm standard-star
wavelengths to Angstrom reference coordinates, giving an invalid
reference flux or spline. It also masked inappropriate wavelength
ranges, and `write_sensfunc_products` hardcoded
`WAVEUNIT=Angstrom` even when the table itself had nm values.

The pipeline now validates the standard's spectral-axis unit early,
converts all Å reference *coordinate values* to the source unit,
and records the correct coordinate unit in the response FITS table.
It deliberately **does not** convert the reference F_lambda numerical
values into flux per nm: the calibrated physical flux quantity
remains per Å; the response is fitted for the standard's actual
count/rate unit divided by `F_lambda,Å`, preserving the output flux
density `erg s^-1 cm^-2 Å^-1`. COUNT_V1 bin widths and unit
signatures remain in their native source-axis unit and must match
the science observation.

This behavior is tested with a synthetic standard on 0.2 nm bins,
smooth known throughput, an injected nonconstant extinction curve,
Balmer feature exclusion and a telluric-mask interval specified in Å,
plus full FITS write/read and `_load_sensfunc` round-trip. Unknown
spectral units are rejected.

**Scope limitation:** physically matching AIR/VAC reference line
coordinates, atmospheric extinction at the science-object airmass
(A09), standard-star slit losses and actual instrument throughput
remain separate qualification tasks. Legacy products with missing
coordinate units should not be promoted to qualified science.

Source: FITS spectral-coordinate representation,
Greisen et al. (2006) doi:10.1051/0004-6361:20053818.

CI verification note: this stacked branch is tested against the current
`main` integration snapshot and must be retargeted to its parent feature
branch before review/merge. The code and associated source provenance
remain unchanged by this administrative check.
