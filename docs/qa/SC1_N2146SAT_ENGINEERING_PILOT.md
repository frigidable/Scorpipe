# N2146sat engineering pilot — stop at the first real pipeline gate

**Scope.** This is a one-time engineering run of actual N2146sat frames in a fresh
sibling run directory. It is **not** a publication-ready reduction, an approved
overscan correction, or a replacement of the existing `03_bias`.

### Read-only prerequisites

- Original run: `ScorpipeWork/06_12_2024/N2146sat_VPHG1200B_01`.
- Exactly 7 OBJ, early 20 BIAS, 3 FLAT, 2 NEON; original `calib.bias_policy=strict`, `flatfield.enabled=false`, `superneon.bias_sub=false`.
- The original source configuration and original `03_bias/master_bias*.fits` / index are SHA256-guarded before and after. The pilot **never** writes them.
- After writing the isolated config, the pilot immediately reloads it using the real Scorpipe loader and checks that the resolved run root and **every selected RAW file** match the originally audited selection. Any mismatch prevents the preparation manifest from being issued.
- Copying the source YAML deliberately **drops loader-derived root metadata** (`project_root`, `config_path`, `config_dir`, `work_dir_abs`, `_project_manifest`): Scorpipe reconstructs these from the pilot config path and its selected project manifest. This prevents false original-run references and stale path provenance. `work_dir` is explicitly set to the new pilot; RAW exposure paths remain immutable.
- `_91` (2026-11-09) is a retained failed engineering run: `manifest` and `superbias` passed, `cosmics` rejected `GROSSMED`; source hashes unchanged.
- `_92` (2026-11-09) is another retained failed engineering run: `manifest`, `superbias`, `cosmics`, `superneon` passed; `lineid_prepare` failed because the legacy CSV loader interpreted `# waveref: air` as the table header and produced NaN wavelengths; source hashes unchanged.
- **Never rerun or delete `_91`/`_92`.** Use the fresh sibling `N2146sat_VPHG1200B_93` after testing the LineID parser correction.
- Tests first: `tests/test_masterbias_estimator_uncertainty.py`, `tests/test_masterbias_gross_reject_optin.py`, `tests/test_sc1_engineering_pilot.py`, `tests/test_p5_lineid_long_filter.py`. On Windows, use `pytest --basetemp` pointing to a fresh dedicated directory in the isolated worktree if the default `%TEMP%/pytest-of-<user>` is not readable.

### Exact commands (from PR checkout, with `PYTHONPATH=<checkout>/src`)

```powershell
$source = "C:\Users\frigi\Documents\PhD\ScorpipeWork\06_12_2024\N2146sat_VPHG1200B_01\config.yaml"
$pilot = "C:\Users\frigi\Documents\PhD\ScorpipeWork\06_12_2024\N2146sat_VPHG1200B_93"

& $py scripts/qa/sc1_engineering_pilot.py --selftest
& $py scripts/qa/sc1_engineering_pilot.py prepare $source $pilot
# Inspect $pilot/config.yaml before the explicit engine run.
& $py scripts/qa/sc1_engineering_pilot.py run (Join-Path $pilot "config.yaml")
# Stop after lineid_prepare; inspect manual_pairs_auto.csv and curate hand_pairs.txt.
# Only AFTER scientific review of the line matches:
& $py scripts/qa/sc1_engineering_pilot.py resume (Join-Path $pilot "config.yaml") --lineid-reviewed
```

The pilot config activates only here:

```yaml
calib:
  bias_combine: median
  bias_sigma_clip: 0
  bias_gross_reject_adu: 50.0  # ADU, provisional
  bias_policy: strict
cosmics:
  enabled: true
  apply_to: [obj]
  bias_subtract: true
  method: auto
sky:
  failure_policy: strict
flatfield:
  enabled: false
superneon:
  bias_sub: false
```

### Task sequence

`manifest → superbias → cosmics → superneon → lineid_prepare → wavesolution → lsf_model → sky → sky_refine → linearize → stack2d → extract1d → qc_report` (13 distinct canonical tasks).

**Manual LineID gate:** this plan is executed in two parts. Phase `run` executes only the first five tasks, including `lineid_prepare`, and writes `sc1_engineering_pilot_lineid_phase.json` with status `ENGINEERING_PAUSED_FOR_LINEID_SCIENCE_HOLD`. The preparation step writes `manual_pairs_auto.csv`, *not* the `hand_pairs.txt` demanded by `wavesolution`. Review the physical line identifications and curate `hand_pairs.txt` in the same isolated wavesol directory; do not blindly convert/accept auto suggestions. Phase `resume --lineid-reviewed` checks five or more active unique pairs, broad pixel coverage, nominal wavelength coverage, and the SHA256 of the curated file before executing the remaining eight tasks. It refuses a changed config, incomplete phase-one report, or original protected-file mutation. A phase failure is final for that phase's report; do not silently retry without investigation.

No `superflat` or `flatfield` in this *engineering* subset: fast-mode FLAT/NEON (RATE 225 GAIN 3.5) cannot be bias-subtracted with slow-mode BIAS (RATE 145 GAIN 0.9). **Cosmics is required**: Sky refuses SCORPIO RAW frames without `BIASSUB=True`. The pilot runs `cosmics` only for the seven compatible slow-readout OBJ frames, with bias subtraction enabled. `method=auto` uses `stack_mad` for seven frames, which must be inspected for loss of astrophysical signal; a completed engineering run does not validate the resulting cosmic masks. `sky.failure_policy=strict` prohibits a soft-failure pass-through that would preserve an unsubtracted sky frame while reporting a completed stage. Failure to satisfy a real prerequisite is a **useful stop**, not permission to bypass preflight or use cross-mode calibration.

`biascorr` is **not** an independent engine task: it is an alias for `superbias`. Likewise, `arclineid` is an alias for `lineid_prepare`, and attempting to run both would skip the second invocation. Each must appear only once, under its canonical name. The harness now verifies that every planned task is canonical and that RAW selections and required pilot flags remain unchanged between preparation and run.

For each task the canonical `run_sequence_locked` engine is invoked with `force=false`, `qc_override=false`, `preflight_override=false`. On the first failed/not-OK task stop; phase-one errors are recorded in `sc1_engineering_pilot_lineid_phase.json`, whereas errors after the LineID gate are recorded in `sc1_engineering_pilot_result.json`. Retain the entire isolated run.

### Acceptance levels

1. **Code regression:** 56/56 targeted tests and pilot self-test passed on Windows at `d3bb0f4`; the newer LineID parser and manual-only preparation tests must pass before preparing `_93`.
2. **Engineering E2E:** phase one completes its five tasks and pauses for reviewed LineID; phase two completes its remaining eight tasks with `status=ok`; original protected hashes unchanged; inspect all produced products.
3. **Science:** **HOLD regardless of 1 or 2** until reliable bias uncertainty (including quantized ADC and time-dependent zero), wave/sky residuals, flux fidelity and spatial geometry are evaluated against the final extracted science product.

**GROSSMED contract (fix after `_91` stop):** The producer labels per-pixel retained-sample median uncertainty `GROSSMED`, with no global `SB_VFAC`. The consumer now recognizes this model with distinct sample-support checks, an exact comparison of index and FITS model/cutoff/rejection counts, provisional-estimator provenance and strict readout matching. It refuses gross-rejected bias groups with any unsupported pixels because Cosmics currently does not propagate MasterBias DQ. This is not a validation of uncertainty coverage. If `_92` reports insufficient support, stop and inspect the pixels; do not disable checks.

The 50 ADU gross threshold is from the audited N2146sat bias cohort, **not** an instrument-wide universal calibration. The finite-N Gaussian factor after gross rejection is provisional. Original MasterBias SCI/VAR/DQ must remain untouched.

### LineID CSV metadata parser and manual-only gate (fix after `_92`)

The packaged `neon_lines.csv` starts with `# waveref: air` and `# unit: angstrom` followed by its `lambda` column. The old `_load_neon_lines_csv` used `np.genfromtxt(names=True)` on the unfiltered input; NumPy consumed the first metadata comment as the names row, then coerced the actual `lambda` line to `NaN`. NumPy emitted `All-NaN slice encountered`, and automatic line matching failed without examining valid wavelengths. The new loader separates comment metadata from actual CSV rows before parsing, validates finite positive wavelengths, respects explicitly declared units, and refuses ambiguous malformed catalogues. It does **not** expand `pair_tol_A` or auto-accept any assignments.

If automatic pattern matching is inconclusive after a valid catalogue is loaded, `lineid_prepare` now writes the candidate peak template plus an explicit `MANUAL_ONLY_NO_RELIABLE_AUTO_PAIRS` report and a header-only auto-suggestions CSV. The stage can therefore complete **preparation** and stop at the mandatory manual LineID gate. This does not assert a valid wavelength solution; users must manually verify line identities and write `hand_pairs.txt` before `resume --lineid-reviewed`.

### 2026-11-09 Wave2D _93 physical failure: HOLD

The first five _93 stages passed; the reviewed Moiseev LineID list contains
19 pairs (SHA256 2d88b6948d9225326bcbf16125dfa844544da6a44ec1e5e8af6438975433d623).
The isolated wavesolution stage returned operational status ok, but the 2D
solution was scientifically wrong: 4.345 Å (4.99 px) RMS, 10.404 Å residual
P95, and four adjacent-line pairs merged for 94-100% of spatial rows.
Preserve the original _93 workdir as immutable evidence.

Root cause: _trace_one_line_xcorr used a broad 12 px search radius without a
row-to-row displacement limit, permitting jumps to brighter neighbours
10-13 px away. Its seed score was in raw-amplitude units while subsequent
scores were cross-correlation dot-products.

Fix: constrain candidate centres to at most 3 pixels per row by default,
while retaining the outer search window and allowing an explicitly configured
trace_max_step_px. The seed score is now correlation power. The separate
wave_trace_qc.py detects collapsed/reordered neighbouring trajectories,
missing manual seeds, and per-line support/bias failures. Whole-line
leave-one-line-out CV (Chebyshev 3x2) is an independent diagnostic; per-row
random CV would leak nearly identical samples from the same arc line.

Provisional fail-closed engineering limits (all in detector pixels):
2D RMS <= 1.0, residual absolute P95 <= 2.0, per-line median bias <= 1.5,
per-line used-sample fraction >= 0.65, whole-line CV RMS <= 1.25, and
whole-line CV worst median line bias <= 2.0. Reject any adjacent-line
collision fraction > 1% within 1.5 pixels. Store diagnostic JSON before
raising. A failed solution is rejected before lambda_map.fits is emitted.
These engineering checks are not publication-grade wavelength validation.

Next attempt: use new isolated N2146sat_VPHG1200B_94. Run regression tests
(including Golden wavesolution), prepare and run the five initial stages,
install the SHA-verified 19 manual pairs, and execute wavesolution alone.
Do not run LSF/Sky/Linearize or the legacy multi-stage Resume before a
scientific review of the new 2D solution and a sky-line zero-point check.

### 2026-11-09 v8 Windows regression check: 73 PASS, 1 FAIL (pre-run)

Pinned HEAD f15d032f5a6dba12677f3e58d5ae3f9b72468c80, isolated
r8 worktree. Targeted pytest completed 73/74 tests; only
test_rectification_model_records_science_signature_separately_from_arc failed.
The stage's final QC JSON referenced excluded_summary without defining it.
The old broad exception handler hid this pre-existing NameError; the new
fail-closed QC wrapper surfaced it. Because Check failed, no _94 directory
was prepared and no FITS reduction was attempted.

Correction: define exclusion provenance before applying the global exclude
policy, populate counts and dropped OBJ/ARC file paths only when exclusions
are present, and preserve null when no policy exclusions apply. The Golden
mini-night regression now asserts both done.json and wave_done.json are
written, the excluded_summary key is well-defined, and both 2D science
diagnostic JSON files exist. Use a fresh r9 detached worktree pinned to the
new reviewed HEAD; preserve failed r8 test evidence, all original _01 files
and earlier _91/_92/_93 pilot outputs. Never bypass or relax Wave2D QC to
make tests pass.
