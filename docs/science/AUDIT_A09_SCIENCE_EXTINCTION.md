# Audit A09 — science-target atmospheric extinction

The standard-star sensitivity function fits
`R(lambda) = N_std(lambda) * 10^(0.4 k(lambda) X_std) / F_std(lambda)`,
where `k(lambda)` is the extinction coefficient in mag per airmass
and `X_std` is the standard-star airmass. A science spectrum must
then be transformed as

```
F_science_above_atmosphere = N_science / R(lambda) * 10^(0.4*k(lambda)*X_science)
VAR_output = VAR_input / R(lambda)^2 * 10^(0.8*k(lambda)*X_science)
```

This contract is described in the PypeIt fluxing documentation:
https://pypeit.readthedocs.io/en/stable/fluxing.html .
Before this patch, Scorpipe applied the standard correction but
never the science correction, producing the wrong spectral shape.
The error can be substantial when standard and science airmasses differ.

## Implementation limits

This PR supports a **single raw science exposure only** and only if
its parser reports `airmass` with `header` or `derived` provenance.
It reuses the same atmosphere curve selected by the reference context
for the standard. The wavelength curve is interpreted in standard
air Angstrom and transformed to the science spectral medium and unit.
Extrapolation beyond curve support, invalid values or missing
spectral-coordinate declarations cause an error when a correction
was requested.

Corrected scale factors are propagated coherently to TRACE, FIXED,
1D and 2D SCI/VAR, including local sky residuals (no double counting).
`ATMCORR` and `ATMXSCI` describe the actual correction in FITS,
and `fluxcal_qc.json` records the applied state and provenance.

**Multi-exposure Stack2D outputs with distinct airmasses MUST NOT be
corrected with a guessed arithmetic mean airmass**. A robust stack has
spatially and spectrally varying effective weights, so its atmospheric
transmission must be applied per input exposure or computed using the
frozen Stack2D coefficient maps and proper source provenance.
When no justified science correction is available, the actual
flux values are left unchanged and any would-be
`ABSOLUTE_TRUSTED` calibration is downgraded to
`ABSOLUTE_APPROX`. This is a degraded calibration designation, not
a numerical correction or a guarantee of fractional accuracy.

An explicitly supplied science extinction model with a manually
chosen airmass may be used by the low-level numerical interface,
but it is **never** sufficient for `ABSOLUTE_TRUSTED` status.
This PR does not fix slit losses, temporal throughput variability,
extinction-curve model uncertainty, telluric absorption below the
transmission clip, or airmass exposure integration for long integrations.

## Tests

Synthetic noiseless standards/science of different airmass, known
nonconstant extinction, correct 1D/2D flux and variance transfer,
TRACE/FIXED independence, FITS/QC stamps, invalid/unsupported
airmass, incomplete extinction coverage, and a multi-frame refusal.

The nominal `ABSOLUTE_TRUSTED` status still requires additional
real spectrophotometric qualification (standard slit losses and
comparison against calibrated photometry). GitHub CI and Windows
checks are necessary but not sufficient.
