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
