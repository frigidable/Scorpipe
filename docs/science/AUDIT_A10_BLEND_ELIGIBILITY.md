# Audit A10 — integrated emission-line windows are not component fits

The current Scorpipe measurement is a continuum-subtracted, fixed-window
integral, *not* a simultaneous model of Halpha+[NII] or [SII].
The nominal Halpha ±18 Å integration overlaps the [NII]6548
window and the [NII]6583 neighborhood. The ±12 Å [SII]6716 and
[SII]6731 windows overlap one another.

As a fail-closed diagnostic safeguard, `measure_lines` now compares
all its integration intervals to the default physical-line catalogue
even when a caller requests only one line. Overlapping intervals
produce `blend_flag`, `blend_partners`, and
`flux_interpretation=BLENDED_DIAGNOSTIC_ONLY`. The integrated numerical
flux, S/N and detection flag are retained as a *blended-window signal*;
none is automatically reinterpreted as an individual astrophysical
component. `blend_safe_for_physical_ratios` represents only absence of
a **known window overlap**, not complete photometric or kinematic
qualification. The stage QA counts detected blended and blend-safe
windows separately.

The Balmer-decrement helper refuses Halpha if the measured window
has a known blend, even if Halpha and Hbeta each have significant
integrated signal. Externally supplied, independently deblended fluxes
remain supported and are not automatically rejected by this field.

**Required next science method:** joint continuum plus constrained
Gaussian/LSF-aware fit to Halpha,[NII]6548,[NII]6583, with common
kinematics and physically justified [NII] doublet ratio, then joint
[SII]6716/6731; propagate fit covariance, stellar Balmer absorption,
and spectral covariance from P1.4. Validation needs injection/recovery
over variable LSF, S/N and line ratios, and comparison to repeat spectra.

Reference examples include the simultaneous gas-template practice in
q3dfit and recent spectroscopic catalogues:
https://www.aanda.org/articles/aa/pdf/2025/04/aa53668-25.pdf
https://doi.org/10.1093/mnras/stag939

This PR is stacked after #55 and #58. A normal `main` CI run tests the
integrated combined branch only; no individual component should be
merged independently. Windows and real SCORPIO tests remain outstanding.
