# SCORPIO-1 N2146sat — LineID25a native peak discovery and manually held-out calibration gate

Date: 2026-11-10. **STATUS: RESEARCH DIAGNOSTIC ONLY — SCIENCE HOLD.**

## Scientific rationale

LineID24 tested native detector pixels but did not validate absolute wavelength accuracy: 0/66 proposed genuinely new Ne I holdouts obtained credible centroids. Five conditional references were part of the historically trained Wave2D and showed wavelength-dependent offsets. The three-component Ne I model for the 5341 Å region won a descriptive held-column RMSE comparison in 22/22 patches, but its model-dependent component ratio cannot be adopted as a laboratory wavelength or scientific LSF.

## LineID25a standalone Windows read-only design

A **separate, not integrated into production**, SHA-guarded Windows ZIP `Scorpipe_LineID25_FullRangeNativeDiscovery_ReadOnly_20261110.zip` was built and independently ZIP-extracted+verified; archive SHA-256 **`e634b4a0d35d9cf01dc6a85cdb5db544c33d6d888eb5d1b9c3dabddb0dc0f0fc`**, 29/29 **local synthetic** tests PASS. Real NIST full-band Atlas and real Windows ARC discovery under LineID25 **not yet run**. Original ARC, _95/_96 map, LSF and hand pairs remain immutable.

Phases:
1. **Check**: validate 6 immutable original source SHA-256 files and every bundled script/test against a manifest; run isolated pytest `--basetemp`.
2. **Atlas**: fetch 16 full NIST ASD **5.12** direct HTML pages in contiguous 10 nm bins over **380–540 nm**, for He I/II, Ne I/II, Ar I/II. Store original HTML, original URLs and SHA-256, compare NIST declared line counts to parsed stored-transition rows, and separate **observed** air wavelengths from Ritz-only data. No inferred lamp strength prior.
3. **Discover**: find positive emission peaks separately in original unstacked ARC06/ARC13 **native CCD row y=479** using raw x (no historic map in peak detection), fit raw x Gaussian+linear continuum, pair two ARC native x centroids within 3 pixels. Export paired features, raw 31-pixel profile stamps from both ARC frames, and a **blank/manual-review-required** species and lab-wavelength CSV. Only *after* detection use old `lambda_map.fits` to list alternative NIST candidates ±5 Å. No auto IDs or automatic use for calibration.
4. **Fit** (*separate human gate, not authorized from discovery alone*): require ≥10 manually reviewed TRAIN observed NIST lines and ≥4 *truly new* HOLDOUT lines, with a documented blend check and wavelength coverage/spacing constraints. Explicitly exclude all holdout feature IDs from the polynomial training. Measure native peak x(y) at 11 y positions in each ARC, using old map for **search localization only**, then fit a new robust Chebyshev `lambda(x,y)` (degree 3 in x, 2 in y) **only from TRAIN CCD positions and selected laboratory λ**, separately for ARC06 and ARC13. Evaluate HOLDOUT line-level residuals and descriptive Gaussian width/skew/wing proxies without claiming a validated science LSF.

NIST source scope is all *stored* transitions for the six chosen species in 380–540 nm, **not** every theoretically possible Ritz transition or all lamp impurity species. Relative intensities in NIST are source-dependent and cannot be used as a cross-species line ratio. Manual review still uses a historic map-derived prior and hence is not completely blind, although the optional polynomial does not use the historical map's values as calibration equations.

**All phase outputs default to SCIENCE HOLD.** A paired native peak is not an atomic ID. Two ARC frames, 11 slit samples and repeated detector pixels share systematics; a line-centroid RMS is not an independent absolute wavelength standard. No 2024 lamp-filling proof, science OBJECT sky-line zero check, telescope flexure transfer or empirical non-Gaussian LSF validation has been performed. Do not modify production Wave2D, LSF, hand pairs, source FITS or create _97.

Primary sources: [NIST ASD spectral-line conventions](https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html), [NIST ASD 5.12](https://physics.nist.gov/PhysRefData/ASD/lines_form.html), [FLAME 2D λ(x,y) wavelength solution](https://academic.oup.com/mnras/article/478/2/2097/4995920), [ESO grism-specific arc frame caution](https://www.eso.org/sci/facilities/lasilla/instruments/efosc/inst/CalSpecHeAr.html).

**PR #178 remains Draft and science gated.**