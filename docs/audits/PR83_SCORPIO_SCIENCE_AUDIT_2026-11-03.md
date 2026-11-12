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


## Follow-up implementation — indexed Cosmics→Flatfield/Sky bias variance (pending Windows QA)

The original risk descriptions above record the **pre-remediation state**. The follow-up changes below have been implemented after the verified baseline, but have **not yet passed Windows testing**.

1. **Exact indexed MasterBias selection for Cosmics.** For each reduction group the stage reads input FITS headers, distinguishes already-debiased from raw input, requires exact indexed MasterBias group resolution with `policy_override="strict"`, and rejects mixed CCD readout groups or mixed `BIASSUB` states before science output. The old global default alias is no longer used when an index exists. For instrument-labelled SCORPIO data without an index, bias subtraction is deferred rather than claiming unknown legacy group compatibility. Non-SCORPIO legacy helper runs retain their historical fallback.
2. **MasterBias variance handoff.** For successful indexed subtraction, `*_clean_biasvar.fits` stores the selected MasterBias uncertainty in ADU². `CLEAN` remains a primary-image FITS to preserve GUI compatibility and references its sidecar by `BVARREF`, `SB_GID` and `BVARINC=False`. The shared sidecar reader strictly validates the exact adjacent filename, units, dimensions, finite nonnegative values, schema marker, group identity and single-addition state. Flatfield and direct Sky now combine this variance with independently modeled photon and read-noise contributions, stamp `BVARINC=True` and discard the no-longer-applicable relative sidecar path in their own outputs.
3. **Raw SCORPIO Sky gate.** `_prepare_sky_input_planes` rejects an instrument-labelled SCORPIO input without `BIASSUB=True`, regardless of whether an input VAR plane exists. This blocks unverified raw electronic pedestal and its physically incorrect Poisson noise. The Golden synthetic science images explicitly declare their zero electronic pedestal, as required by this new contract. Generic unlabeled legacy test images retain the old low-level path.
4. **Unindexed Flatfield guard.** Real SCORPIO science no longer accepts a `resolve_master_bias` result with `degraded=True, reason=NOIDX`; building exact detector/readout groups is required.

### Outstanding scientific questions after this follow-up

- **Definition of MasterBias variance:** `stages/calib.py:_combine_group` currently writes pixelwise `(1.4826 × MAD)²` over the source bias-frame stack. That estimates the scatter of individual frames, not generally the statistical variance of the *combined MasterBias estimator*. For (N) independent bias inputs, the uncertainty of a mean master decreases approximately as (1/N), while a median estimator has a different finite-(N) factor. The newly implemented sidecar faithfully transmits the **existing upstream** variance product, not a newly validated estimator-uncertainty model. Publication-grade uncertainty claims require a separate Monte Carlo/calibration audit and a correction of this definition as appropriate for the chosen combine method.
- **Interpolation/inpainting covariance:** replacement of cosmic-affected pixels using another exposure or a local median changes per-pixel variance and induces correlations. The new sidecar addresses the MasterBias term only; covariance induced by cosmic rejection, subsequent rectification and combination must be separately characterized.
- **Calibration provenance for older processed files:** legacy `BIASSUB=True` FITS without `SB_GID` or linked uncertainty may predate these contracts. Such files cannot automatically be promoted to fully provenance-certified science products. Require a targeted legacy-migration/QA policy.
- **Identity-free FITS inputs:** the strict raw-bias gate currently activates for `INSTRUME` beginning with `SCORPIO`. Anonymous generic CCD arrays can still exercise legacy helpers; do not claim the guard certifies arbitrary unlabelled real observations.
- **FITS comment-length warnings:** header comment truncation remains unresolved; do not globally suppress.

### Regression tests added

`tests/test_cosmics_masterbias_readout_selection.py`,
`tests/test_cosmics_masterbias_variance_handoff.py`,
`tests/test_scorpio_unindexed_bias_fails_closed.py`, and
`tests/test_sky_sc2_bias_provenance_gate.py`.

These exercise exact matching, mixed-group rejection, no double subtraction, correct units and variance propagation, missing/corrupt sidecars, strict SCORPIO raw-input refusal, and unindexed MasterBias rejection.

**Release gate:** repeat narrowly targeted Windows tests, then Golden, full fast regression, and Smoke on one pinned commit. Treat this implementation as unverified until those results return. The previous 596-fast-test success belongs to commit `2f190078e755a2c66ccc82729e97fc3b24817720`, not the new branch head.


## Follow-up: uncertainty of the *combined* MasterBias (pending Windows QA)

The initial indexed MasterBias variance handoff above transmitted the legacy
`(1.4826 MAD)^2` **single-exposure scatter** as though it were the uncertainty
of the combined master. That is not the correct statistical object when a
single MasterBias realization is subtracted from a science exposure.

The bias producer now uses
`src/scorpio_pipe/bias_uncertainty.py:master_bias_estimator_variance`
to calculate a per-pixel **estimator variance** from `N>=2` independent
Gaussian-approximate CCD bias measurements:

- For the un-clipped arithmetic mean,
  `Var(B_mean) = s^2/N`, where `s^2` is the unbiased (`ddof=1`)
  temporal sample variance in ADU².
- For the un-clipped sample median,
  `Var(B_median) = s^2 c_N` where `c_N` is the variance of the
  normal median for unit-variance independent samples. For `2<=N<=64`,
  `c_N` is computed using the exact Gaussian order-statistic beta
  marginal / adjacent-order-statistic joint distributions (with numerical
  integration); e.g. `c_2=0.5`, `c_3≈0.4486711046`,
  `c_4≈0.2981996185`. For `N>64` the documented asymptotic
  `π/(2N)` is used.
- A sigma-clipped mean has data-dependent selection. The producer
  estimates variance using the full temporal `s²` divided by the number
  of retained values, labels it `CLIPAPPX`, and marks pixels with fewer
  than two retained samples as `BADPIX`. This model is **approximate**
  and should undergo clip-threshold-specific injection/Monte Carlo
  studies before scientific certification.
- A single bias exposure can no longer masquerade as an empirical master
  with a zero-variance plane; `N=1` is rejected.
- FITS primary headers carry `SB_VMOD` (`MEANSEM`,
  `GAUSSMED`, `MEDASYMP`, `CLIPAPPX`), `SB_VFAC` where constant,
  `SB_NMIN`, `NBIAS` and `BUNIT=ADU^2`; the group index includes a
  variance-definition record. Corrected a separate FITS Header copy
  bug so `RN_SRC` and `NOISRC` remain visible with GAIN/RDNOISE.

**Crucial caveats:** The model assumes independent bias-frame read noise,
stationary electronics and an approximately Gaussian temporal distribution.
It does not represent common-mode drift, readout correlations, shutter/
temperature evolution, or the covariance across scientific exposures
created by subtracting the *same* MasterBias from each exposure. The
coadded stack variance must eventually retain this shared calibration term
instead of reducing it as though every exposure had an independent bias.
Those effects require their own science validation and covariance model.

**Compatibility concern:** Already-produced `master_bias_var.fits` files
from older runs contain the legacy temporal-scatter definition. They must
be regenerated before using the revised estimator interpretation; a
cache/schema migration guard and identification of all old artifacts are
still outstanding. This PR remains Draft.

Reference basis: MIT 18.650 "Order Statistics, Quantiles, and Sample Quantiles"
(https://math.mit.edu/~rmd/650/ordstats-quant.pdf), particularly the
Beta representation of continuous order statistics; N. Davison,
*Statistical Models*, sample quantile asymptotic normality. These are
the theoretical grounds for the normal-order-statistic factor rather
than a claim of empirical calibration of SCORPIO detector read noise.

**Current QA status:** The earlier green result was at
`4499d31f7352c093551e253a86f0db3730f26088`. The new
estimator implementation and its Monte Carlo tests have not yet been
executed in the user's Windows worktree.


## Cached MasterBias schema invalidation and shared calibration covariance (pending Windows QA)

The MasterBias index and MasterBias `VAR` format are now **scientifically versioned**:

- New products use `scorpio_pipe.master_bias_index` **schema_version=3**, with per-group `variance.definition=uncertainty_of_combined_master_bias`, `model`, `factor`, `min_effective_n` and FITS `SB_VMOD/SB_VFAC/SB_NMIN/NBIAS/BUNIT/SB_GID` consistency.
- During indexed MasterBias selection, `resolve_master_bias` validates the selected group's estimator-variance metadata against its actual FITS header; v1/v2 or missing schema version is rejected with an explicit request to **regenerate** MasterBias. The old v2 VAR was pixelwise bias-frame dispersion and cannot be silently relabeled as estimator VAR.
- Partially mocked minimal indexes without a named schema remain compatible with low-level historical tests. Actual `build_superbias` always emits the named index. Do not treat anonymous manually constructed indexes as scientific provenance.
- New `tests/test_masterbias_index_estimator_schema.py` verifies valid new selection and refuses old v1/v2/missing-version indexes, wrong methods, missing VAR, unit conflicts, wrong calibration GID and sample counts, and inconsistent variance factors.

**Next methodological issue: shared calibration covariance.** Let a science exposure have
`Y_i = S_i + n_i - (B_hat - B_true)`, where `n_i` is exposure-independent noise
and `B_hat` is the *same* combined MasterBias subtracted from all exposures. Then

`Cov(Y_i,Y_j) = δ_ij Var(n_i) + Var(B_hat)`,

under independence of `n_i` and `B_hat`. A weighted stack with normalized
weights `a_i` (sum a_i=1) has variance

`Var(sum_i a_i Y_i) = sum_i a_i^2 Var(n_i) + (sum_i a_i)^2 Var(B_hat)`.

The shared MasterBias term **does not shrink as 1/N science exposures**; the
present Stack2D treats all input VAR planes as independent, and will therefore
underestimate this common component when it enters each exposure VAR.

Rectification can map the shared pixelwise detector MasterBias error into
correlated wavelength/spatial pixels; input exposures with different
wavelength shifts and different normalization need a transformed covariance
term rather than naive per-pixel addition. The scientifically appropriate
next work package is a separate `independent_variance` / `shared_calibration`
uncertainty representation, propagation through Sky, Linearize and Stack2D,
and Monte Carlo with a reused MasterBias. **Do not claim this is solved by
the new estimator VAR map or by the passed synthetic Golden.**

Validation on real SCORPIO-1/2 data and independently verified readout-noise
benchmarks remain release blockers.

Latest index-v3 code/tests have **not yet passed Windows QA**; the prior 690
test executions passed at commit `3dac7434c69318373e253bbc855019d78d2cad4b`.
