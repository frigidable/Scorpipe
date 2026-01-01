# Developer reminders (keep it quick)

When you touch Wavelength Solution / Linearization, re-check only these:

1) **File contract**: `08_wavesol/{lambda_map.fits, rectification_model.json, done.json}` exists.
2) **Units**: `wave_unit` and `wave_ref` are consistent between the JSON + FITS header.
3) **Provenance**: `done.json` hashes recorded; no silent fallbacks without an explicit warning.
4) **VAR/MASK math**: VAR uses Σ a^2 Var; MASK uses OR over fatal bits + NO_COVERAGE on gaps.
5) **Flexure shift**: if `delta_lambda` is present, record `{value, score, flag, applied, policy}` and never apply when flag=UNCERTAIN unless user explicitly forces.
6) **Post-sky cleanup**: AUTO must not worsen object metric; report both output metrics and (if rejected) candidate metrics.
