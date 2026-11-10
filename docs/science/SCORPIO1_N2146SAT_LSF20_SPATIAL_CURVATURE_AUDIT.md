# SCORPIO-1 N2146sat v20: pair-dependent spatial LSF curvature

**Status: DIAGNOSTIC_ONLY / SCIENCE HOLD.** Original _95 Wave2D and
_96 isolated LSF maps remain immutable; NO SCIENCE FITS are written.
Prior Windows v19 completed 135/135 PASS + engineering selftest,
and re-fitted two source-sha-locked original NEON06/NEON13 lamps:
2425/2640 numerically valid hypothesis fits.

## Scientific diagnosis

The v18 smooth five-term FWHM(lambda,y) model uses
u=(lambda_air-4600 Å)/1000 Å and v=(y-515.5)/515.5, with a
**shared** quadratic spatial term coefficient +1.3753 Å
across ALL wavelengths. The real original-ARC profiles instead
show strongly wavelength/line-dependent quadratic spatial shape.

Using solely original-ARC joint-Gaussian/linear-background
profile fits (spectral half window 16 px; 21 spatial rows
per sample, step 27; y from 74 to 938), independent quadratic
fit to EACH original lamp and each laboratory pair gives:

| Lab pair air A | NEON06 v² coefficient A | NEON13 v² coefficient A |
|---|---:|---:|
| 4181.88 + 4190.71 | +2.0281 | +2.1388 |
| 4333.56 + 4345.17 | +2.2643 | +2.3245 |
| 5037.75 + 5047.74 | +0.1278 | +0.2692 |
| **5330.78 + 5341.60** | **-0.4383** | **-0.3554** |

Red curvature sign is OPPOSITE to v18's shared +1.3753.
Source raw pair at y=506 gives FWHM 6.2138/6.1843 Å,
where the existing v18 candidate predicts only 5.7928 Å.
The old v18 smooth model has central-region positive residuals
(+~0.37 Å) but opposite negative residuals near y extremes
(-~0.55 Å). This morphology cannot be solved by simply shifting
the model up or down globally. Source lamp repeatability
does not imply scientific LSF-model accuracy.

## Diagnostic alternative model comparison

On ten earlier v18-selected features (not a blind independent
sample), five robust low-order surfaces were compared in
leave-one-WHOLE-FEATURE-out cross-validation. The two extreme
lines were excluded from score because their removal demands
unconstrained extrapolation. Eight interior features yielded
equal-feature RMS of medians of |prediction error|:

| Basis extension relative to v18 five-term model | RMS A |
|---|---:|
| Baseline (5 terms) | 0.285 |
| + u² | 0.274 |
| + u v² | 0.266 |
| + u v² + u²v² | **0.257** |
| + u² + u v² + u²v² | 0.270 |

The u v² interaction allows the curvature along the slit
to vary with wavelength. Full-fit red-pair prediction at
y=506 improves from ~5.735 A (a separately refitted v18
five-term baseline) to ~6.013 A (with u v² + u²v²).
The original v18 coefficient vector predicts 5.793 A;
do not mistake the separately refit five-term baseline
for the unmodified original v18 stored model.

This reduction in score and improvement at red y=506
are **not a model selection approval**: feature selection,
thresholds and family comparison all used the same
N2146sat data. Curvature may be line-specific contamination,
not a smooth physical wavelength trend. Model complexity
must be externally validated against independent setups
and independent original lamp frames/nights.

## Code boundary (PR #178)

New pure src/scorpio_pipe/lsf_spatial_curvature_audit.py checks
per-lamp quadratic FWHM(y) and compares each laboratory pair
to the original v18 shared-y² model; reports DIAGNOSTIC_ONLY only when all four original-ARC pairs
have sufficient matched spatial coverage; otherwise global REJECT,
with per-pair failure reasons. No science-ready result is possible. Seven spatial-shape synthetic regression tests check sign reversal, two distinct lamps,
duplicate samples, insufficient common spatial rows,
provenance status, non-finite widths and non-adoption by
production LSF. No production model is changed.

New read-only scripts/qa/lsf20_spatial_curvature_probe.py:
- SHA256-verifies BOTH original source ARC FITS and original _95/_96
  wavelength map using their canonical exact digests.
- Requires existing independent JSON audit v18 and JSON+CSV
  profile results from the successfully run v19 original-ARC audit;
  checks original source provenance and trial counts, verifies
  the complete SHA of the v18 report embedded in v19 JSON,
  and cross-checks every per-line/window/continuum summary
  median against the original profile CSV, including the red
  pair y=506 from both lamps. Four additional QA tests enforce
  this provenance contract and the canonical 64-digit map SHA.
- Fits spatial quadratic to existing original-ARC v19 profiles,
  writes an *exclusive new JSON* outside any source directory;
  does not refit raw pixels in this follow-on run.
- Reports repeated lamp profile shapes separately and never
  emits approved science LSF, sky zero-points or FITS products.

See SCORPIO source grism specifications:
https://www.sao.ru/hq/lsfvo/devices/scorpio/grisms/grisms.html
Nominal VPHG1200B FWHM ~5.5 Å is not a guarantee of
width for these particular observations or a point source.

## v20b schema-reconciliation and Windows packaging correction

Original v20 launcher was placed in the main repository working directory,
where another copy already existed: Expand-Archive without -Force therefore
refused to extract it. Subsequent pytest startup was interrupted
(KeyboardInterrupt before test collection, no scientific failure established).

More seriously, the original v20 real-data QA reader expected synthetic
v19 fields source_files_sha256, group_summary, n_accepted and
observed_fwhm_A. Actual source-controlled Windows v19 diagnostic JSON
uses original_source_sha256 (with wavelength_map key),
feature_summaries / pair_A / n_good,
median_fwhm_by_hypothesis_A keys degree1_window11, etc.,
and original_lamp_fwhm_A. This mismatch would have rejected the
valid original-ARC input at RealData. Updated
scripts/qa/lsf20_spatial_curvature_probe.py to validate these **exact**
recorded v19 field names and all 40 reported per-pair spectral-window
medians from the 2640-row CSV. New regression fixtures reproduce actual
v19 JSON shape and test rejected fake field aliases, wrong laboratory
pair identities and altered width medians. The actual supplied Windows
v19 CSV/JSON was checked independently: accepted per-pair counts
659/632/476/658, all available 40 medians match, and red pair y=506
matches NEON06 6.2138312453 A / NEON13 6.1843358131 A.

v20b has its own fresh pinned r20b worktree and a separate extracted
PowerShell filename. It does not resume or modify the interrupted
r20 worktree, _95/_96 input archives or previous v18/v19 diagnostic
outputs. v20b Windows tests and RealData remain pending.

## Next decision gate

**Do not create a science _97 LSF or combine N2146sat spectra yet.**
The immediate decision is not which polynomial yields smallest
training residual, but whether the observed pair-dependent
spatial curvature represents the instrumental LSF, real
unlisted ARC blends, spectrum tilt, or spatially-dependent
profile/background/systematics. Require independent
line-by-line profile evidence and external-night checks.
The true science-source LSF and wavelength zero point
remain SCIENCE HOLD.
