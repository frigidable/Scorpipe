# A09b — standard-star extinction curve: air/vacuum and coverage

**Companion to [A09 #194](https://github.com/frigidable/Scorpipe/pull/194). Target:** [vNext Draft #180](https://github.com/frigidable/Scorpipe/pull/180). **SCIENCE HOLD**.

## Scientific defect

The previous Sensfunc code sampled the BTA atmosphere curve as `np.interp(wave, extinction_wavelength_A * unit_scale, k_lambda, left=median(k), right=median(k))`. This has two methodological problems:

1. The BTA reference file `resources/extinction/bta.csv` explicitly declares `wavelength_convention: air`, while a qualified standard spectrum may declare `WAVEREF=vacuum`. Comparing the two unconverted grids shifts the wavelength dependence of the atmospheric attenuation.
2. Returning the *median* k(lambda) outside the calibrated reference table extrapolates a physically unsupported extinction value while reporting `extinction_applied=true`.

The correction evaluates the standard's native unit-grid wavelengths in **air Angstrom** and performs interpolation in the reference's *original air coordinates*, not on knots transformed nonlinearly to vacuum. When the response is de-extincted, the reference must have strictly increasing, finite positive wavelengths, finite nonnegative k(lambda), coverage of **every standard wavelength sample**, and actual source-file comment declarations `units: mag_per_airmass` and `wavelength_convention: air`. Reference-context fallback defaults cannot substitute for declarations in the input file.

If no validated standard-star airmass is available, the normal relative-calibration behavior remains; the numerical standard extinction factor is not applied. This patch does not try to guess an uncalibrated missing reference.

The current A09 science correction uses the same native-air interpolation policy, with its own exposure airmass. Both separate corrections are required to interpret the response physically, as described in PypeIt's fluxing documentation (https://pypeit.readthedocs.io/en/stable/fluxing.html).

## Validation and limitations

Synthetic tests include known k(lambda) under vacuum-nm and air-Angstrom coordinates, nonmonotone/negative/nonfinite attenuation tables, incomplete reference support, and invalid spectral-medium declarations. These are mathematical transfer-contract checks only. The finite wavelength-independent precision of the atmospheric curve, time variability, spectral slit losses, standard-star absolute flux identity, telluric features, true covariance and original SCORPIO data remain unqualified.

No raw FITS edited; `ABSOLUTE_APPROX` and central SCIENCE HOLD are not upgraded by this fix.
