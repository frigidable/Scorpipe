# Audit A01 — sensitivity training/evaluation grid contract

**Scope:** `calib/sensfunc.py` only; diagnostic P0 correction.
**Reference code baseline:** `main@dee41ed3`.

## Mathematical definition

Observed standard counts/rates `N_i` and reference spectral fluxes
`F_lambda,i` yield a response model, in the same input unit convention
as the standard, `R(lambda_i) = N_i / F_lambda,i` (subject to the
current extinction convention, which is under separate review A09).
A good-pixel mask selects the reliable samples for *fitting* a smooth
log-response. This mask **must not reduce the output wavelength grid**.

The previous implementation did:

```python
model_log = fit(wave[fit_mask], log_ratio[fit_mask])
response_abs = exp(model_log)
clean = fit_mask & finite(response_abs)
```

When `fit_mask` excludes a Balmer-line, telluric or bad pixel, the
model had `sum(fit_mask)` samples but `clean` had `len(wave)` samples.
This usually produced a shape error and prevented writing a scientifically
usable response. When a degenerate constant fallback occurred, a misleading
partial response could be emitted.

The fixed contract separates training `(wave[fit_mask], log_ratio[fit_mask])`
from full-grid evaluation `wave`, retains all original wavelength bins,
refuses fewer than eight valid training samples, and forbids nonfinite
or nonpositive response functions.

## Limitations and required follow-up

- The number **8** is only a minimal numerical viability guard. It is not
  a science-qualified criterion of wavelength coverage or spline accuracy.
  P1 method selection and holdout standard-star validation remain required.
- The masked intervals can interpolate across actual throughput structure.
  They should not automatically be considered equally trustworthy.
- Wavelength-unit conversion for reference tables (`A07`), absolute
  extinction handling (`A09`), slit-loss corrections, and physical
  calibration uncertainty are separate issues, **not** fixed here.
- Historical sensfunc outputs from the previous implementation are not
  silently rewritten. Qualify/rebuild any affected products with explicit
  reference and response provenance.

## Regression acceptance

- Synthetic standard with excluded Balmer/bad-pixel bins yields full-length
  response, finite strictly positive coefficients, matching FITS table length.
- Smooth injected response is recovered within an explicit 0.2% numerical
  tolerance under the tested noiseless model; this is not a real-instrument
  spectrophotometric accuracy claim.
- No usable standard samples fails closed rather than supplying a constant
  response; masked nonpositive variance cannot enter the fit.
- Preserve the originally selected fit mask and its count in QC.

## Literature anchors

[PypeIt fluxing overview](https://pypeit.readthedocs.io/en/stable/fluxing.html)
distinguishes the fit on reliable standard-star pixels from the sensitivity
model evaluated along a wavelength grid.
[PypeIt core flux calibration API](https://pypeit.readthedocs.io/en/stable/api/pypeit.core.flux_calib.html)
describes wavelength-aligned zero-point modeling, masks for stellar absorption
lines and good-pixel selection.
These are methodological comparison sources, not a claim that PypeIt uses
this implementation or validates Scorpipe science products.
