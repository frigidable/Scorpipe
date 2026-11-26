# SCORPIO-1 N2146sat v19: original ARC profile-model systematics

**Status: READ-ONLY ENGINEERING AUDIT; SCIENCE HOLD.** This audit does
not modify original NEON06/NEON13 FITS, the frozen _95 Wave2D map or
the _96 LSF products. The v18 candidate is NOT adopted by production.

## Direct primary inputs and references

Original lamp s22800606.fts SHA-256:
54a6d638d2284b9bfd1d947df02399eaf0264f998667a830e144b1c437a61dac

Original lamp s22800613.fts SHA-256:
9e2c98f2d2ba1125e18f84722854099f73e435a1fd7810f0b4df84ea07ab0c05

Frozen _95/_96 lambda_map.fits SHA-256:
37808e73c00279ced67ea520e2245767427bd71dc7edec9af22177fbc8df07ab

The v18 Windows-generated source-controlled JSON has status
DIAGNOSTIC_CANDIDATE, 10 accepted features and source SHA-256 proof.
Independent original FWHM extractions were reused only for the
secondary surface comparison; primary v19 profiles were **re-fitted
from both actual raw ARC image arrays** with an original map read-only.

SCORPIO-1/VPHG1200B nominal FWHM=5.5 A, dispersion=0.87 A/pixel:
https://www.sao.ru/hq/lsfvo/devices/scorpio/grisms/grisms.html

Spectral LSF can be non-Gaussian, vary across wavelength and slit,
and differ between lamp and science target illumination:
https://doi.org/10.1051/0004-6361/200811467

## 1. Direct two-Gaussian profile sensitivity

Two positive Gaussians with common width and common pixel shift,
independently fitted line amplitudes and either linear or quadratic
continuum. Gaussian-equivalent FWHM comes from locally measured
d(lambda)/dx of the frozen map. Fit windows were +/- 11, 13,
16, 20, 24 pixels around the *individual members* of each pair.
At 33 spatial y locations from 74 through 938 in steps of 27,
a 21-row raw detector median was used per original ARC.
This is not y-rectified, so tilt smearing is analyzed separately.
Robust soft-L1 fitting and explicit optimizer/wing/weak-line rejection.

Four pairs x 33 y x 2 originals x 5 windows x 2 continuum orders
= **2640 trials; 2425 numerically acceptable**. These are not
2425 independent measurements and 2425 successes are **not**
a scientific validation. Across the whole line + both originals,
the median fitted widths under **linear continuum** for each
selected window range as follows:

| Lab Ne line pair (air A) | Median Gaussian FWHM range A | Interpretation |
|---|---|---|
| 4181.88 + 4190.71 | 5.02-6.13 | VERY window-dependent |
| 4333.56 + 4345.17 | 6.05-6.33 | Moderate window dependence |
| 5037.75 + 5047.74 | 5.49-5.92 | Model/neighbor dependent |
| 5330.78 + 5341.60 | **5.96-6.02** | Linear-background stable |

Adding a QUADRATIC continuum at +/-11 px changes the
across-bin red FWHM median from ~5.976 to ~6.494 A;
at +/-24 px the corresponding values are ~6.021 and
~5.995 A. Small-window continuum curvature can trade
off strongly with blended Gaussian wings. A lower residual
in a higher-order continuum fit does NOT make its LSF
more physically reliable.

At center row **y=506**, fixed +/-16 pixel, linear continuum:
NEON06 FWHM **6.214 A**, NEON13 **6.184 A**; v18
five-parameter global smooth candidate gives **5.793 A** at
the red pair laboratory center 5336.19 A. The source
widths exceed this model by **0.421/0.391 A**, even though
the two original exposures agree. This discrepancy is NOT
explained by the smooth model's own training residual QC.

## 2. Spatial-median / trace-curvature sensitivity

The frozen wavelength map predicts a shift in a *21-row*
slit strip near y=74..911 of up to ~2.2 px for 3889 A,
~1.9 px near 4186 A, but ~0.8-1.0 px at 5336 A.
This is an empirical wavelength trace difference, not
necessarily equivalent to full LSF broadening.

Using half-strip sizes 0,1,3,5,10,15 rows with a fixed
+/-16 px joint Gaussian/linear model:
- Blue 4186 A median FWHM: single row 5.209 A,
  21-row median 5.248 A; median *paired absolute
  bin difference* ~0.053 A, p90 ~0.158 A.
- Red 5336 A median FWHM: single row 5.940 A,
  21-row median 5.959 A; median paired absolute
  difference ~0.022 A, p90 ~0.066 A.
These spatial-strip effects are far smaller than the
~0.4 A red-pair-v18 model mismatch at y506. They
could matter along the slit and must be propagated
or eliminated by curvature-aware extraction.

## 3. Four competing FWHM(wavelength,y) surfaces

Only the ten features selected by v18 were used for
this exploratory separate comparison. Each original-ARC
repeat was paired at matching y before fitting. Each
laboratory feature gets equal total weight; robust IRLS
downweights outliers. Four models:
  A: linear wavelength + linear/quadratic y + lambda*y
     (five-parameter *v18-style*, refit with slightly
     different robust weighting, not identical coefficients);
  B: A plus quadratic wavelength term (six parameters);
  C: B plus cubic wavelength term (seven parameters);
  D: quadratic fit in y to each individual feature and
     piecewise linear interpolation in laboratory wavelength.

Leave-one-FEATURE-out scoring uses all spatial rows and
both lamp repeats of the held-out feature. The two
extreme laboratory anchors are intrinsically *outside*
the wavelength span after exclusion and require invalid
edge extrapolation; these are marked explicitly.
On the **eight interior features**, equal-feature RMS
of median-absolute held-out errors, in A/B/C/D order:
**0.285 / 0.274 / 0.312 / 0.343 A**.
There is no compelling evidence to prefer an extra
spectral polynomial order: all models have line-by-line
systematics and the difference in scores is small.
When the red outer pair is held out, cubic model C's
median absolute prediction error reaches ~**1.170 A**,
but this is a warning about extrapolation rather than
a valid science cross-validation datum.

**Selection leakage limitation:** the ten features were
selected using v18 LOFO on the very same N2146sat night.
All subsequent comparisons are conditional and exploratory,
NOT a blind test of model family, a statistically calibrated
error budget or a justification to substitute the model.

## Implemented in PR #178

Pure scorpio_pipe.lsf_profile_systematics diagnostic module:
fit_neon_pair_hypothesis() and evaluate_neon_pair_systematics()
refuse incomplete profiles and explicitly return
science_lsf_validated=false / auto_apply_allowed=false.
Production lsf_model, Sky, Linearize, Stack2D and kinematics
are unchanged.

Source-SHA-locked scripts/qa/lsf19_profile_probe.py
re-fits both original raw ARC FITS arrays and writes only
a diagnostic CSV and JSON into an exclusive new directory.
The simple FITS reader is intentionally *only* for these
hardcoded immutable primary HDU images; normal Scorpipe
must continue to use Astropy FITS I/O.

The prior v18 Windows audit (124 targeted tests PASS)
remains separately pinned. New v19 unit tests require
fresh Windows regression testing before accepting changes.

## Decision / next science gate

**Do NOT launch _97 as production science LSF.**
Investigate residual line-shape asymmetry, weak unlisted
features and alternate centroid/relative amplitudes in
the red pair, physically motivated slit illumination and
2D wavelength-map uncertainty. Only independent-night
model selection can establish whether a smooth polynomial,
spline or local empirical model is defensible.
A diagnostic-only isolated future pilot may be prepared
after the current regressions, but cannot approve
science data or zero-point flexure corrections.
