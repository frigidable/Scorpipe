# Sky residual refine

`sky_refine` is a second-pass cleanup stage that runs **after** raw-domain `sky` subtraction and **before** `linearize`.

It does **not** replace Kelson RAW modelling. Instead, it measures residual skyline structure only on the sky stripes and applies a small correction model on top of the existing `*_skysub_raw.fits` products.

## Inputs

- `09_sky/*_skysub_raw.fits`
- `09_sky/*_skymodel_raw.fits`
- `08_wavesol/lambda_map.fits`
- ROI / sky geometry from config

## Outputs

- `sky_refine/*_skycorr_raw.fits`
- `sky_refine/*_skysub_refined_raw.fits`
- `sky_refine/*_sky_windows.csv`
- `sky_refine/*_sky_windows.json`
- `sky_refine/*_sky_diagnostics.png`
- `sky_refine/sky_residual_windows.csv`
- `sky_refine/sky_residual_windows.json`
- `sky_refine/sky_observer_pack.md`
- `sky_refine/qc_sky_refine.json`
- `sky_refine/done.json`

## Behaviour

- re-computes sky/object geometry on the sky-subtracted frame;
- expands the object-protection mask conservatively using continuum and line cues;
- auto-detects problematic skyline windows from aggregated upper/lower sky stripes;
- classifies residuals into approximate families:
  - amplitude mismatch
  - wavelength mismatch
  - LSF mismatch
  - sky-window contamination
  - cosmic leakage
- applies corrections only for windows that look physically safe to correct;
- writes an observer-facing summary of the worst skyline windows.

## Safety rules

- the stage never learns on the protected object aperture;
- contamination-like windows are reported but skipped, not "fixed";
- `linearize` prefers `sky_refine/*_skysub_refined_raw.fits` when present, otherwise it falls back to `09_sky/*_skysub_raw.fits`.
