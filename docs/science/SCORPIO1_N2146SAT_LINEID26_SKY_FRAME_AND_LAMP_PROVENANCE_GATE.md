# SCORPIO-1 / N2146sat LineID26 — science-frame sky and calibration source gate

**2026-11-11 — SCIENCE HOLD. Diagnostic-only independent research.**

## Instrument-level provenance

SAO's [2013 SCORPIO-1 manual](https://www.sao.ru/hq/lsfvo/devices/scorpio/manuals/scorpio_manual_2013.pdf) and [prime-focus adapter documentation](https://www.sao.ru/hq/lsfvo/adap/index-en.html) specify the nominal **NEON = He–Ne–Ar lamp**, confirming the correct *element family* searched in LineID23–25, not actual 2024 tube inventory or relative strengths. The manual advises a new NEON frame after changes in **zenith distance** around **10–15°**; **ROTANGLE (ARC06 295.2°, ARC13 266.1°) is not zenith distance**. Source-verified 2024 ARC headers have `DATE=2024-12-06T...` and `DATE-OBS=2024/06/12`; without instrument-specific format documentation these are not auto-reinterpreted as equivalent observation dates. `DATE` may be different semantics.

## Atmospheric alternatives are weak/conditional

Actual historic N2146sat 5341.60 Å anchor is at x≈2048, near the detector's red boundary. Verify full 2D map bounds per row; earlier full NIST atlas 3800–5400 Å was an *index interval*, not guaranteed recorded physical coverage. The strong atmospheric [O I] 5577 Å is expected outside the source pixels. An in-range possibility is weak **[N I] 5197.9+5200.3 Å**, normally unresolved and not a fixed-centroid line. Hg I 4046.56/4358.33 Å are artificial-light candidates, not assured at SAO and not part of a He–Ne–Ar arc. The presence of sky emission, an uncontaminated sky slit region, and source-specific calibration have never yet been verified for this night.

## Standalone LineID26 read-only engineering package (no production integration)

`Scorpipe_LineID26_ScienceSky_ReadOnly_20261111.zip` — SHA256 `64a0b992cd718d47ff435a57c3357caeabbb4e2d10e6a00efb9539d8a3ec8be2`, 6 ZIP files, independently re-extracted/manifest-hashed and **22/22 synthetic tests PASS**. Build environment lacks Astropy and original OBJ/ARC FITS, so genuine Windows FITS I/O and OBJ sky detection are **not yet exercised**. Existing user's local `.venv` is used for actual Windows run.

Two phases: **Check** (six original source SHA256 and bundle hash, Astropy import and pytest isolated basetemp) → **Run** (inspect all original night `.fts` frame headers; select original N2146sat OBJ frames with SCORPIO-1 VPHG1200B, record exact filenames and raw DATE/DATE-OBS, grism/filter, gain, rotator angle, any actual ZD or airmass; check original science geometry and full frozen original map; export **unaltered native detector values** in 37-pixel windows around three *candidate* sky features on ten manually-unvalidated near-slit-edge rows per OBJ).

Exports: night header inventory, sky candidate native counts, candidate row metrics, per-row old-map wavelength coverage, SHA-guarded JSON. Local linear baseline + MAD **only for morphology triage**. Never assert independent emission-line identification, measured object flexure, positive sky detection, calibrated S/N, validated science LSF or new absolute wavelength zero. No source FITS/production Wave2D/LSF writes; outputs only to a new directory. Explicitly leave all `sky_line_centroids_approved`, `wave2d_approved`, `lsf_approved` false.

## Gate still required

Human inspect outer slit sky regions / distinguish galaxy contamination; independently validate the *existence and chemical identity* of any apparent sky lines; handle [N I] two-component intensity-dependent effective center; secure original observing log to confirm actual lamp, exposure chronology and telescope zenith distance; independently identify ≥10 TRAIN and ≥4 genuinely new HOLDOUT ARC reference features across the full native wavelength range. Do not run LineID25 `Fit` and do not create `_97`; PR remains Draft. [NIST ASD](https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html); [Airglow observations](https://aas.aanda.org/articles/aas/pdf/1998/01/ds1449.pdf); [5199 doublet physics](https://doi.org/10.1088/0004-637x/757/1/79).