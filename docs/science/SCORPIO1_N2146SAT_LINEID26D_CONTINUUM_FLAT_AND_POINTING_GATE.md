# LineID26d — full-slit multi-band science/FLAT independent audit gate (2026-11-11)

**SCIENCE HOLD — diagnostic only; this is a source-backed report, not an authorized sky mask or a calibration step.**

## Evidence independently replayed

From real user-uploaded LineID26d 7 OBJ ×8 non-overlapping ~30 Å native windows and 3 FLAT ×8 windows, independently verified all **1,987,230 raw OBJ CCD samples, 851,670 raw FLAT samples, 57,792/24,768 per-row medians+maxima+pixel-starts+lengths+MAD, 3,640 16-row block medians and all 65 review labels**. All 5 CSV + NPZ + 2 short PNG exact SHA checks pass; `lineid26d_per_exposure_excess_maps.png` is re-encoded/downscaled on upload to **1717×2048** from original 1820×2170, and does **not** preserve output PNG SHA. Numerical outputs remain verified. Original raw FITS were not independently opened in this analysis runtime; source SHA checks for them are documented by the successful user's Windows report.

Previous `lineid26d_report.json` says 56 `PROVISIONAL_LOW_EXCESS_REVIEW`, 8 `AMBIGUOUS_REVIEW`, 1 `OBJECT_LIKE_REVIEW` for 65 16-row y blocks. **0/65 SKY_ONLY_APPROVED.** The word LOW is not a sky identity; it measures the residual against a fitted empirical sky/FLAT shape and uncalibrated MAD proxy.

## Critical: OBJ04 is a different pointing

Authenticated LineID26 night header inventory: `s22800604.fts` (600s) telescope RA=06:28:36.66, DEC=+77:55:42.1 vs `s22800607.fts` (1200s) RA=06:28:11.57, DEC=+77:56:57.5. **Spherical header-pointing separation 108.95 arcsec** (~1.82 arcmin). OBJ08..12 differ from OBJ07 by 0.80,1.89,1.81,3.26,3.86 arcsec. Initial 5s imaging OBJ03 shares the OBJ04 coordinates while 30s acquisition OBJ05 and ARC06 match OBJ07, consistent with a deliberate pointing change. User's prior 7-frame 5199 analysis must NOT interpret OBJ04 absence of the y900 feature as time variation in the same field, and OBJ04 must not be silently coadded or used to train the common-frame sky mask. Observation log/acquisition astrometry required to label the separate field; it might be a deliberate offset, but not yet established.

## Two distinct spatial structures

* Broad excess y≈400–511 has median 0.959 in units of **edge MAD proxy (NOT S/N)** across all 7×8 combinations. It is distributed across wavelength bands in OBJ07..12 and suggests possible target continuum, but **0/56 band-frame region summaries exceeds the overly strict 3-MAD classification gate**. Therefore low-excess label falsely suggests absence of objects; there is source-like morphology below that threshold.
* Narrow positive broadband structure y≈896–911 has median 4.418 edge-MAD proxies; **43/48** band-frame medians above 3 for OBJ07..12, **0/8** for distinct-pointing OBJ04. In the same detector region the flat lamp normalized profile actually declines by 0.018–0.035 relative to y864–879 rather than showing a positive bump. Atmospheric [N I] 5199 **independent prior LineID26c raw sample** likewise has a background continuum increase +4.9 to +13.0 ADU in OBJ07..12 at y896–911 relative to y800–860, vs −2.7 ADU in OBJ04. This is a potential contaminant to sky-line centroids, not yet proven stellar vs detector/scattered-light in origin. **Exclude or quarantine for sky-mask review.**
* FLAT spatial shape median y0 and y1031 ~0.077 of normalized interior vs y480 ~1.095; last y1024–1031 median residual −5.85 MAD. CCD edge rows cannot be certified pure sky on that basis; validate actual active detector/slit coverage.
* Recomputed for OBJ07..12 only with same heuristic thresholds: **53 low / 11 ambiguous / 1 source-like** blocks. Still ZERO scientifically approved sky-only positions.

## Model assumptions that still fail the science gate

OBJ GAIN 0.9 e−/ADU vs 3 FLAT GAIN 3.5 e−/ADU; no matching bias/overscan correction, so raw OBJ/FLAT division invalid. The FLAT shape is a lamp-specific illumination pattern, not a calibrated object slit function. Its 3-frame normalized median spread for y64..969 ≈0.00237 and max 0.00913: repeatable under lamp conditions, not an estimate of sky-flat accuracy. Provisional baseline fits use outer regions y≈62–237 and 794–969; the latter explicitly contains the y900 anomaly, so continuum object flux/ghosts may be fitted away. None of the 8 bands was independently shown free of object or sky lines. Historical Wave2D is used as pixel selection prior, and there is no independently validated science LSF, sky identity, noise variance or relative OBJECT flexure.

## Primary 2013 SAO manual corrections

Instrument manual [SAO SCORPIO](https://www.sao.ru/hq/lon/SCORPIO/manuals/scorpio_manual_2013.pdf) §4.6 separately defines **PA(table)** and the celestial slit angle **PA(slit)**. Appendix H FITS card inventory calls `ROTANGLE` a *field rotation angle*, `PARANGLE` a *parallactic angle*, `Z` a *zenith distance*, and `GAIN` electrons per ADU. The header template explicitly records **DATE-OBS=YYYY/DD/MM**; `2024/06/12` is therefore consistent with **6 Dec 2024** (the DATE card), not 12 June. Varying ROTANGLE **by itself** proves neither a changing celestial slit PA nor distinct sky y positions; sky registration and the log are needed. This corrects earlier overly restrictive/ambiguous statements.

## Gate / disposition

Treat OBJ04 separately; re-establish acquisition pointing groups and sky position angle. Audit same-readout BIAS/overscan and calibrated flat before scientific 2D tracing. Independently trace N2146sat/other objects using unbiased multiple spectral bands and inspect y≈320–620, y≈880–927 and detector edges. Cross-validate any candidate sky mask using 5199 raw line-continuum maps *outside* the selection bands, with rigorous spatial/exposure holdouts. **No automatic sky mask, no science wavelength correction, no LSF production, no LineID25 Fit, no `_97`. PR remains Draft, SCIENCE HOLD.**