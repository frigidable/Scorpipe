# Scorpipe PR #83 — integration and CCD science-readiness audit (2026-11-03)

**Scope:** code review against the PR base `fix/qt-async-test-cleanup-and-deleted-signals`; real-data bias subtraction, noise/variance provenance, LSF warnings, and integration gates. This is a review of implementation and synthetic tests, **not** a reduction of actual SCORPIO-1/2 frames.

## Verified baseline

Windows Python 3.12.10 / pytest 9.2.0 at commit `3f7756f2c32bb49e443c9d0107de046e9df959fa`:

| Suite | Result |
| --- | --- |
| Golden numeric contracts | 21 passed |
| Full Golden mini-night | 1 passed |
| Fast non-smoke/non-golden tests | 591 passed, 2 skipped, 2 deselected |
| Synthetic smoke | 1 passed |

The Golden run covers 11 pipeline stages, strict scientific product validation and QC-gate **without overrides**. These are test results on a deterministic, *zero-electronic-pedestal* synthetic CCD dataset. They do not by themselves establish scientific correctness for actual uncalibrated SCORPIO images.

## Corrected by this audit — requires fresh Windows QA

### P0 — Flatfield trusted Cosmics configuration instead of the actual image (fixed)

**Location:** `src/scorpio_pipe/stages/flatfield.py:run_flatfield`, before the `apply_flat` call.

Prior behavior: when a `*_clean.fits` existed and `cfg.cosmics.bias_subtract=True`, Flatfield forced `do_bias_subtract=False`. Cosmics can generate a cleaned file *without* subtracting bias (if its optional MasterBias is unavailable or incompatible); in those cases its FITS header does **not** claim `BIASSUB=True`. The prior logic misclassified such frames as calibrated. Their ~1000 ADU electronics offset could then inflate estimates of Poisson noise and contaminate SCI.

**Correction:** always pass the configured Flatfield bias-subtraction intent; `apply_flat` already checks the actual `BIASSUB` FITS flag and subtracts only when absent. The original MasterBias selection uses detector geometry/readout and propagates available bias VAR and DQ. Tests simulate cleaned files with `BIASSUB=False` and `BIASSUB=True` under the same Cosmics configuration using a **1000 ADU pedestal**, a 120 ADU photon signal, GAIN=0.62 e−/ADU and RDNOISE=3 e−. They require no double subtraction, and for fresh bias subtraction check the model `VAR_e2 = N_e + RN_e² + VAR_masterbias,ADU² × gain²`.

### P2 — LSF all-NaN wavelength-column warning (fixed)

**Location:** `src/scorpio_pipe/stages/lsf_model.py`. The code called `np.nanmedian(lam_map, axis=0)` and `np.nanmedian(lsf_map, axis=0)` even where every spatial row was NaN at a wavelength pixel. That is an expected uncovered-pixel condition, not a reason to fill values. New helper computes finite medians on columns with actual data and retains all-invalid columns as NaN, without generating `RuntimeWarning: All-NaN slice encountered`. Tests cover all-invalid planes, valid+invalid mixtures, and linear-grid reconstruction from partly masked maps.

## Open science-readiness risks (not certified; do not hide)

### P0 — Cosmics MasterBias selection is not readout-aware

`src/scorpio_pipe/stages/cosmics.py:_load_superbias` loads one global `superbias.fits` array if present. In contrast, `stages.calib.resolve_master_bias` supports an instrument, ROI, binning and CCD NODE/RATE/GAIN keyed MasterBias group index and explicitly handles degraded group fallback.

All `cosmics` cleaning methods receive the **single global bias array**; they do not independently validate that its readout mode matches each science input. Mixed-readout observations can thus be subtracted with an inappropriate bias; the `BIASSUB=True` card is insufficient evidence of matching calibration. This requires per-exposure group selection plus stamped `BIASGID`/readout selection provenance, or strict rejection of mixed-group input until supported.

### P0 — Missing MasterBias VAR after bias subtraction inside Cosmics

Cosmics can subtract MasterBias while writing a primary-only `*_clean.fits`, without a propagated `VAR` plane. Flatfield sees `BIASSUB=True` and correctly does not subtract a second time, but when input `VAR` is absent it estimates only source shot noise + CCD read noise. The available MasterBias variance is **not automatically added** at this boundary, and cannot be identified confidently without recording the original bias selection and uncertainty provenance. This can underestimate `VAR`. Must carry and validate a bias variance map through the Cosmics→Flatfield boundary and guard against double counting.

### P0 — Direct raw Sky input and Poisson noise

`src/scorpio_pipe/stages/sky_sub.py:_prepare_sky_input_planes` estimates variance from the input SCI whenever `VAR` is missing. `estimate_variance_adu2` assumes that counts have a zero electronic pedestal (photon counts + read noise), not raw detector values containing, e.g., a 1000 ADU bias offset. When Sky selects raw frames and those frames have not undergone a compatible bias subtraction, the variance model is physically invalid. The Golden fixture has deliberately zero-offset images and therefore **does not test** this route. Require verified upstream bias status, safe per-frame debias with uncertainty propagation, or a fail-closed error on uncertain raw inputs before science production.

### P1 — Bias policy defaults to degraded when readout group does not match

`resolve_master_bias` defaults to `calib.bias_policy=degraded`. When geometry matches but readout group differs, it chooses a degraded fallback with marked provenance. This is an explicit existing policy, **not** an exact-calibration guarantee. Use `strict` for real-science validation; test incompatible NODE/RATE/GAIN and explicit degraded behavior separately.

### P2 — FITS metadata comment truncation

The Windows pytest run showed repeated Astropy `VerifyWarning: Card is too long, comment will be truncated`. FITS keyword values are not shown to be corrupted, but long comments can be lost. Identify originating header cards with a provenance/round-trip test and shorten comments without losing scientific metadata. Do not broadly silence FITS warnings.

### P2 — Other suite warnings and test selection

The previous fast suite reported 2 skipped tests, 2 deselected tests (the latter are intentionally excluded by its `-m` filter) and a GUI test with deliberately malformed header size. Classify skips individually. Run GUI/integration under a separately controlled platform environment as needed.

## Integration review and release gate

- PR #83 at the baseline had **52 commits and 31 changed files**, with its base set to `fix/qt-async-test-cleanup-and-deleted-signals`, **not main**. Validate that intended branch ancestry remains correct before merging; never infer main compatibility from `mergeable=true` alone.
- PR is **Draft**; do not merge solely because the synthetic Golden passed.
- Complete a new pinned Windows QA after the two audit fixes, including `tests/test_flatfield_actual_bias_provenance.py`, `tests/test_lsf_invalid_wavelength_columns.py`, `tests/test_p3_lsf_publication_kinprep.py`, full Golden and fast+smoke regressions.
- For real science, execute a controlled small SCORPIO-2 dataset with nonzero detector bias, readout-specific flats and bias, and independent detector-noise references. Compare the observed sky normalized residuals, spatial variance, wavelength solution and extracted 1D flux/uncertainties against a trusted reduction. Keep full SCI/VAR/MASK/ETA/FITS provenance.
- Raw N2146sat frames and existing validated reductions are **not modified** by this audit.

**Release interpretation:** PR #83 has a validated synthetic E2E baseline. It is **not yet a production science certification** for real SCORPIO-1/2 exposures until the three P0 remaining risks above have an implemented, tested, and reviewed solution.
