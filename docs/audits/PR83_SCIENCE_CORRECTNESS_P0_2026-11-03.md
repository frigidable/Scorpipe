# Scorpipe Science Correctness P0 — code review and release gate
Date: 2026-11-03. Branch: `fix/typed-config-manifest-runtime-meta` (PR #83).

This note reconciles the user's independent Scorpipe audit of `main@dee41ed3`
with the newer unmerged PR #83. **A finding in the old main code is not
evidence that the same fault remains in all later revisions.** The listed
changes have been pushed to PR #83 but have **not yet passed Windows QA**.
Keep this PR Draft and do not promote the source products to publication-grade
until both scientific and integration gates are satisfied.

## Last verified baseline, before these fixes

Windows Python 3.12.10, pytest 9.2.0 at
`0b2dd18d812c624ab3a6a1a597cdc90df8bd05a4`:

- MasterBias index v3 and estimator-variance contracts: **54 passed**.
- Eleven-stage Golden mini-night E2E: **1 passed**.
- Fast regression: **656 passed, 2 skipped, 2 deselected**.
- Synthetic Smoke: **1 passed**.

Total **712 passed**, zero failures; untracked `workspace/` preserved.
These tests certify the exercised synthetic/scaffold cases, not real
SCORPIO-1/2 spectra. The revised science correctness changes below require
a fresh pinned run; the above green status does **not** transfer.

## P0 corrections in this change set (awaiting Windows QA)

1. **Optical barycentric Doppler sign.**
   `src/scorpio_pipe/barycentric.py` had
   `lambda_bary=lambda_obs/(1+v_corr/c)` while Astropy's
   `SkyCoord.radial_velocity_correction` convention adds a *positive*
   optical correction to observed radial velocity. The wavelength must be
   multiplied by `1+v_corr/c`. The helper now multiplies and fails closed
   for invalid factors/grids. Tests include positive/negative 30 km/s and
   exact sign checks. Official reference:
   https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html

2. **Failed Sky pass-through must not enter a science stack.**
   `src/scorpio_pipe/stages/sky_sub.py` intentionally has a
   `failure_policy='soft'` diagnostic path that copies the uncorrected
   science frame and stamps `SKYOK=0` and/or `SKYMODEL_FAIL`. Stack2D
   previously merely propagated QADEGRD, still accepting its flux. Now
   `run_stack2d` rejects any such exposure before accumulation,
   **regardless of** user `reject_if_mask_bits`. If none remains, raise a
   hard error. The diagnostic Sky output remains available for inspection;
   it no longer pollutes the combined scientific spectrum. Tests exercise
   SKYOK failure alone, MASK-bit failure alone, both, and all-bad inputs.

3. **Mixed science setups in Wavesolution are fatal.**
   `src/scorpio_pipe/stages/wavesolution.py` raised `RuntimeError`
   for multiple distinct science setups, but its subsequent
   `except Exception` downgraded the mismatch to a compatibility warning.
   It now raises `ScienceSetupMismatchError`, caught by the explicit
   hard-failure branch along with `CalibrationMismatchError`.
   Integration regression alters a Golden science FITS grism or slit width,
   ensures valid ARC products exist, and requires a fatal mismatch.

4. **Emission-line significance and FITS mask.**
   `science/lines.py` declared `detected=True` if `sn>=2 OR
   weight_sum>0`, allowing random positive noise pixels. It also accepted
   zero/negative input VAR via clipping, never received the extracted MASK,
   and assumed air rest wavelengths even for vacuum SCI. It now:
   - requires positive net flux and signed `S/N >= 3` (preliminary
     single-line threshold, not a multi-line catalog false-discovery guarantee);
   - filters integer MASK and invalid/nonpositive VAR from both sideband
     continuum and integration;
   - reads explicit `WAVEREF` in `stages/line_measure.py` and converts
     air rest wavelengths to vacuum when necessary;
   - computes observed velocity width with the *observed* centroid and labels
     it `observed_including_instrument_LSF`, not intrinsic gas dispersion;
   - records a 3-sigma emission upper limit for measured nondetections;
   - marks Hα/[N II] windows as *not deblended*. `science/reddening.py`
     now refuses automated Balmer decrements based on nondetections or
     unresolved blended Hα.
   These changes do **not** replace proper simultaneous continuum+emission
   profile fitting, LSF deconvolution, or multiple-hypothesis selection.

5. **Export is not science certification.**
   `stages/publication_ready.py` stamped `PUBRDY=True` merely on
   creation of three FITS exports. Now headers use `EXPORTOK=True`,
   `PUBRDY=False`, `SCIQPASS=False`,
   `VALLEVEL=EXPORT_ONLY`, plus explicit JSON provenance.
   Product file names and science arrays remain unchanged. Existing
   LSF+publication integration test now asserts these conditions.

## Confirmed outstanding issues — *not yet solved*

- **Stack2D Huber / sigma-clipped uncertainty:** the
  `_WeightedMeanAccumulator.finalize` rule `VAR=1/sum(w)` assumes
  inverse-variance weights and independent samples. Robust data-dependent
  weights break that exact identity. Even a fixed-weight sandwich
  `sum(w_i² sigma_i²)/(sum(w_i))²` is only a conditional variance;
  robust-estimator frequentist coverage requires Monte Carlo validation.
- **Shared MasterBias covariance** between science exposures and spatial/
  spectral noise covariance introduced by rectification. An accurate
  MasterBias per-pixel VAR alone cannot guarantee calibrated stacked errors.
- **Flux calibration extinction convention:** Sensfunc corrects its
  standard for atmospheric extinction; apply-fluxcal must use the object's
  airmass and a consistent extinction reference, or refuse absolute
  outside-atmosphere flux claims. Compare resulting physical fluxes with
  standards and independent imaging; slit losses remain a separate systematic.
- **Barycentric RA-unit ambiguity:** the existing `_parse_radec` heuristically
  treats numerical RA≤24 as hours in some headers. This is ambiguous without
  explicit units. Instrument-aware FITS coordinate parsing and regression
  examples must precede general barycentric automation.
- **Unreadable wavesolution science headers:** while mixed *parsed* setups
  are now fatal, an inner exception handler still skips some unparseable
  science frames. Scientific certification should require all relevant
  input headers to be parseable or explicitly excluded.
- **Line blending and reddening:** mark unresolved windows, do not publish
  line-by-line Hα/[N II] deblended fluxes or Balmer E(B−V) until a jointly
  fitted, uncertainty-propagating model exists. The 3σ threshold alone
  is not sufficient for a large automatic line catalogue.
- **LSF validation, overscan/RAW input state, spatial sky residuals,
  sky-refine behavior, flux standard slit losses and absolute photometry**
  still need targeted injection and external reference testing.

## Immediate QA / merge gate

Run new targeted tests:
`tests/test_barycentric_science_convention.py`,
`tests/test_stack2d_sky_failure_science_gate.py`,
`tests/test_wavesolution_mixed_science_setup_hard_error.py`,
`tests/test_science_line_quality_contracts.py`,
`tests/test_p3_science_lines.py`,
`tests/test_p3_lsf_publication_kinprep.py`,
plus prior Sky/Wavesolution regression and MasterBias v3 contracts.
Then rerun Golden, fast and Smoke on a **single pinned HEAD** without QC
overrides. Do not alter real N2146sat FITS, existing reductions, or
untracked `workspace/`. Do not merge while covariance/statistical
uncertainty model and real instrumental validation remain unresolved.

## Follow-up: fail-closed QC evaluation (pending Windows QA)

At `8037e2e38ce4ffbcf4fee3670168f37b09b9fc85`, independent Windows tests confirmed **47/47 Science Correctness P0**, **Golden 1/1**, **686 fast passed, 2 skipped, 2 deselected**, and **Smoke 1/1**, for **735 passing test executions** without failure. All 11 Golden stages ran without scientific QC overrides. Local untracked `workspace/` was retained.

A subsequent review confirmed another old-main P0 persisted on that revision: `pipeline/engine.py` had `except Exception: pass` around the mandatory `check_qc_gate` invocation. I/O exceptions, JSON parse errors, or programming defects in the checker could therefore permit downstream science stages without an operative QC gate. `qc/gate.py:_read_done_flags` also silently converted malformed upstream JSON into zero QC flags.

The next PR changes (which **have not yet passed Windows QA**) now:

- Fail closed and mark the stage `blocked` with error code `QC_GATE_EVALUATION_FAILED` if the QC check itself raises unexpectedly; preserve the original exception as `__cause__` and never execute the science task.
- Reject existing upstream `done.json` files that cannot be parsed as JSON objects, contain an invalid QC object, invalid flag list, non-object flags, missing or unrecognized severity. Reports *without* a QC subsection remain allowed for stages that do not emit science flags.
- Add targeted malformed-report tests and a real `run_sequence` integration regression that simulates an I/O failure during QC evaluation and verifies the stage was not executed.

**Remaining limitation:** the engine currently plans `resume/skip` decisions ahead of executing upstream tasks. This can create stale reuse decisions if products are regenerated during the same run. Resolving this requires dependency-aware live revalidation, rather than disguising stale products as successfully checked stages. Also, missing upstream reports where reports are mandatory need a separate per-stage requirement specification; absence is not automatically evidence of an error for every optional stage.

Run `tests/test_qc_gate_fail_closed_engine.py`, `tests/test_qc_gate_exception_diagnostics.py`, complete Golden, fast regression, and Smoke on the **same new HEAD** before updating the verified baseline. Real SCORPIO FITS and user workspaces remain untouched.


## Follow-up: external wavelength-reference and Sky/CCD contracts (pending Windows QA)

**Baseline:** the last locally verified Windows commit was `51363331648eba63db95a0ddf5cc4c39a440f369`: QC fail-closed contracts 24 passed, Golden E2E 1 passed, fast regression 695 passed (2 skipped, 2 deselected), synthetic smoke 1 passed — **721 test executions passed**, zero failures. The changes below are **new and not yet Windows-tested**.

### #87: Sky self-fit cannot establish absolute flexure

The prior audit was **partly incorrect** when it said Linearize did not apply Sky's `SKYDLAM`: the current `linearize.py` contained an active block that read the median `sky_done.json.flexure.delta_A_median`, shifted the common wavelength grid and `lambda_map`, and wrote `DLAM_A` / `DLAMAP`. That transfer existed, but it had no independent laboratory-reference verification.

The `_estimate_flexure_delta` in `sky_sub.py` compares a BSpline model to the *same sky data used to fit the BSpline*. Without an external spectral reference, the absolute shift is not identifiable. It is now retained solely as `self_fit_shift_A` diagnostic; Sky reports `FLEXURE_UNVERIFIED`, `delta_A=None`, `SKYFLX=UNVERIF`, `SKYDAPP=False`, and uses `delta_A=0` when evaluating its fitted sky model. Linearize rejects unverified / legacy no-reference shifts by default; the explicit `delta_lambda_policy=always` requires a verified independent reference rather than bypassing this contract. The Golden asserts that no self-fit correction is applied. This is an **interim fail-safe**, not a replacement for a physically anchored flexure algorithm.

Reference: PypeIt flexure documentation https://pypeit.readthedocs.io/en/state/calibrations/flexure.html — PypeIt cross-correlates observed sky lines against an archived independent sky spectrum and validates the resulting shift. A future Scorpipe external-reference algorithm must specify laboratory line list/template, wavelength medium (air/vac), spectral resolution, significance, uncertainty, per-exposure correction, and independent injection tests.

### #88: fitted-sky variance is not raw detector variance

The previous `skymodel_raw.fits` writer fell back to the detector `VAR` if an empirical residual variance was missing, and sometimes used in-sample `max(sigma_residual^2 - median(VAR_input), 0)` as if it were `Var(M_sky)`. This is not a valid fitted-model uncertainty estimator: `Var(D-M)=Var(D)+Var(M)-2Cov(D,M)`.

For now, Sky exports `skymodel_raw.fits` with SCI/MASK and **no VAR extension** rather than invented model uncertainty. Sky-subtracted SCI keeps its **detector-only** VAR and explicitly stamps `SKYVSRC=DET_ONLY`, `SKYVCOV=UNKNOWN`, `SKYVQUAL=INCOMP`, `SKYVADD=False`, and QC `SKY_MODEL_VAR_UNVERIFIED` WARN. In-sample residual excess is diagnostics `SKYVEMP` only; it is **not added** to science VAR. The Golden asserts the new FITS metadata. This is honest uncertainty provenance, **not a complete sky-model covariance solution**. Publication-grade sky covariance still needs held-out sky pixels / injections / parameter covariance, and input detector/resampling covariance.

### #89: Astro-SCRAPPY protected-line SCI and detector noise

`astroscrappy.detect_cosmics` returns the *already inpainted* science image. Simply unsetting its returned cosmic mask for `line_protect` pixels previously concealed interpolation. The new `_reconcile_lacosmic_line_protection` restores the observed SCI on overlaps and flags ambiguous line/CR pixels as `BADPIX` in the mandatory DQ sidecar. Unprotected cosmic pixels stay interpolated and marked `COSMIC`. The actual frame's gain and read noise (resolved from its FITS/config) are passed to Astro-SCRAPPY, rather than relying on defaults of GAIN=1 e-/ADU, RN=6.5 e-. A further single-frame double-bias guard was fixed. Synthetic tests cover known narrow-line SCI, real injected impulse and complete DQ provenance. Ambiguous overlaps intentionally remain masked in downstream photometry until corroborated by another exposure.

Reference: Astro-SCRAPPY API https://astroscrappy.readthedocs.io/en/latest/api/astroscrappy.detect_cosmics.html.

### #90: count versus count-rate dimensional consistency

Without `normalize_exptime`, Stack2D now rejects `ADU` mixed with `ADU/s` or `e-` mixed with `e-/s` even though the *base* unit matches. It also rejects `NORMEXP=True` coupled to an integrated-count BUNIT. With explicit EXPTIME normalization, SCI scales as 1/t and VAR as 1/t². Tests include both file orderings, valid conversion and bad metadata.

### Additional time-ordering issue: Sky must not copy future Linearize results

Compatibility aliases `09_sky/<stem>_skysub.fits` and `*_sky_sub.fits` are now **always** copies of the new RAW-geometry `*_skysub_raw.fits`. The code no longer preferentially copies any stale `10_linearize/<stem>_skysub.fits`. An existing synthetic Sky regression injects an incorrect previous Linearize FITS product and verifies that current Sky aliases contain only current RAW data.

### Validation and release status

PR #83 remains **Draft**. The above new changes must be run through their targeted tests and the full Windows Golden, regression and Smoke on one exact SHA. The prior 721-pass baseline is not evidence of success of these changes. **Issues #87 and #88 remain methodologically open** after the fail-safe patches; #89 and #90 should remain open until QA and physical-line injections confirm the new behavior. Spatial slit coordinate map #91, shared MasterBias covariance, robust Stack2D variance, atmospheric extinction and real SCORPIO benchmarks remain independent work packages. Raw N2146sat FITS/config and untracked QA `workspace/` must not be altered.


## Windows QA follow-up: Sky flexure JSON schema mismatch

The user's pinned Windows run at `dae32b089429f3ebf8f40665fd2ad7dcf79d9730`
passed **39/39 targeted P0 tests**, including LA-Cosmic protected-line DQ, input ADU
semantics, Stack2D count/rate units, previous Sky geometry, and QC blocking.
The eleven-stage Golden **ran through the pipeline but failed its new
provenance assertion**: `sky_done.json` had no top-level `flexure`.
The fast and Smoke suites were therefore not rerun on that SHA.

Root cause: the canonical `write_done_json(..., extra={"flexure": ...})`
writer serializes the scientific report under **`extra.flexure`**, not
`flexure`. Both the Golden and the Linearize consumer read the incorrect
top-level field. This was a cross-stage JSON-schema mismatch: fixing only the
Golden test would have left Linearize unable to read the proper provenance.

- Golden now explicitly asserts `extra.flexure.verification=unverified`, a
  null measured shift and the same absent applied shift on per-exposure FITS.
- Linearize now reads `extra.flexure` first, with a legacy top-level fallback
  that can never bypass the independent-reference requirement.
- A median shift for **multiple** science exposures is categorically
  rejected by the current scalar-application path, even if someone marks
  the summary as independently referenced: true per-exposure flexure mapping
  is required for science-series application.
- New `tests/test_linearize_sky_flexure_schema.py` covers the canonical
  namespace, stale conflicting top-level values, malformed metadata,
  independent-reference requirements, and 2+/multi-exposure global-median
  rejection. No scientific QC thresholds have been relaxed.

**Current fixes pending Windows QA**: rerun the new reader-contract unit tests,
the full synthetic Golden, then full fast suite and Smoke on a single pinned
SHA. PR #83 stays Draft, and #87/#93 remain open pending an independently
calibrated per-exposure algorithm.


## Latest Windows QA baseline and spectral-frame consistency (#92)

**Verified Windows QA at `bfea816d70c4b7b1ecc4e5a6e60891a6256f59d6`:**
- Navigator and Smoke scientific WARN contracts: **10 passed**, 2 expected FITS comment warnings.
- Full 11-stage Golden E2E: **1 passed**, 3 FITS comment warnings.
- Full fast regression: **725 passed, 2 skipped, 2 deselected**, 19 warnings (mostly long FITS comments; one intentionally damaged FITS header test).
- **736 successful test executions, zero failures**, with untracked `workspace/` untouched.

This fixes the former `qc_report.json.qc.flags` versus Navigator root-flags mismatch and preserves the two honest Sky WARNs. **The baseline applies only to that commit**, not the following changes.

### #92: air/vacuum, observer frame and correction provenance in Stack2D

The earlier Stack2D verified image shapes, numerical `WaveGrid` equality
within a wavelength tolerance, count/rate units and `REBNORM`, but
not the **spectral coordinate semantics**: numerically identical
`CRVAL1/CDELT1` can mean physically distinct air versus vacuum wavelengths
or topocentric versus barycentric coordinates.

The new `src/scorpio_pipe/spectral_axis_contract.py` defines
`SpectralAxisContract` and validates:

- FITS `CTYPE1=AWAV` indicates air wavelengths, `CTYPE1=WAVE` indicates
  vacuum wavelengths; these must agree with `WAVEREF=air|vacuum`.
  Nonlinear or nonwavelength `CTYPE1` values (`WAVE-LOG`, `AWAV-TAB`,
  `FREQ`, `LINEAR`) require validated WCS conversions and are rejected.
- `SPECSYS` must name a recognized FITS spectral reference frame and all
  inputs must share the same frame, medium, and spectral axis type.
- Mixed `DLAMAP=True/False/absent` provenance is rejected, as is
  `DLAMAP=True` with `FLXFLAG=UNVERIFIED`.
- Production inputs with `INSTRUME=SCORPIO*` require explicit, consistent
  `WAVEREF`, `CTYPE1`, `SPECSYS` and `DLAMAP`; missing fields fail
  closed. `DLAMAP=True` additionally requires `FLXFLAG=OK` and a finite
  `DLAM_A`; conflicting or unverified correction state is forbidden.
  These are **provenance consistency checks**, not proof of external line
  calibration.
- Anonymous historical minimal test FITS with **all** spectral metadata
  absent may still run but are explicitly marked `SPCQUAL=UNQUAL`
  (not guessed air, vacuum, or TOPOCENT). Partially specified metadata
  fail even for anonymous files.
- The original numerical-grid test, `BUNIT` and `REBNORM` gates remain
  mandatory. The output FITS and Stack2D report carry spectral coordinate
  qualification and per-input provenance.

New tests `tests/test_stack2d_spectral_axis_contract.py` cover identical
numeric wavelengths but incompatible air/vacuum, topo/bary/helio, invalid
CTYPE1/WAVEREF pairing, missing fields, mixed flexure state, valid
topocentric air/vacuum barycentric pairs, and explicitly unqualified
historical fixtures.

Authority: Greisen et al., A&A 446 (2006) 747 (FITS spectral WCS);
FITS Standard section 8.4.1; Astropy SpectralCoord documentation.
See https://fits.gsfc.nasa.gov/fits_wcs.html and
https://docs.astropy.org/en/stable/coordinates/spectralcoord.html.

**Not completed by this fix:** actual per-exposure barycentric/heliocentric
transformation, injection-validated line velocities and line-shape
preservation across resampling, and reliable provenance guarding against
double-applied corrections even when `SPECSYS` is deceptively unchanged.
Do not close #92 or promote this to science-qualified until a pinned Windows
test run and physical wavelength injection checks are complete.

The new HEAD remains Draft and unverified on Windows. No actual FITS
N2146sat/config or untracked user workspaces have been altered.
