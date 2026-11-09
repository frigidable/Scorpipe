# N2146sat engineering pilot — stop at the first real pipeline gate

**Scope.** This is a one-time engineering run of actual N2146sat frames in a fresh
sibling run directory. It is **not** a publication-ready reduction, an approved
overscan correction, or a replacement of the existing `03_bias`.

### Read-only prerequisites

- Original run: `ScorpipeWork/06_12_2024/N2146sat_VPHG1200B_01`.
- Exactly 7 OBJ, early 20 BIAS, 3 FLAT, 2 NEON; original `calib.bias_policy=strict`, `flatfield.enabled=false`, `superneon.bias_sub=false`.
- The original source configuration and original `03_bias/master_bias*.fits` / index are SHA256-guarded before and after. The pilot **never** writes them.
- Create a fresh run named e.g. `N2146sat_VPHG1200B_91` beside the original. Never reuse a previous pilot root.
- Tests first: `tests/test_masterbias_estimator_uncertainty.py`, `tests/test_masterbias_gross_reject_optin.py`, `tests/test_sc1_engineering_pilot.py`.

### Exact commands (from PR checkout, with `PYTHONPATH=<checkout>/src`)

```powershell
$source = "C:\Users\frigi\Documents\PhD\ScorpipeWork\06_12_2024\N2146sat_VPHG1200B_01\config.yaml"
$pilot = "C:\Users\frigi\Documents\PhD\ScorpipeWork\06_12_2024\N2146sat_VPHG1200B_91"

& $py scripts/qa/sc1_engineering_pilot.py --selftest
& $py scripts/qa/sc1_engineering_pilot.py prepare $source $pilot
# Inspect $pilot/config.yaml before the explicit engine run.
& $py scripts/qa/sc1_engineering_pilot.py run (Join-Path $pilot "config.yaml")
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

`manifest → superbias → cosmics → superneon → lineid_prepare → arclineid → wavesolution → lsf_model → sky → sky_refine → linearize → stack2d → extract1d → qc_report` (14 distinct canonical tasks).

No `superflat` or `flatfield` in this *engineering* subset: fast-mode FLAT/NEON (RATE 225 GAIN 3.5) cannot be bias-subtracted with slow-mode BIAS (RATE 145 GAIN 0.9). **Cosmics is required**: Sky refuses SCORPIO RAW frames without `BIASSUB=True`. The pilot runs `cosmics` only for the seven compatible slow-readout OBJ frames, with bias subtraction enabled. `method=auto` uses `stack_mad` for seven frames, which must be inspected for loss of astrophysical signal; a completed engineering run does not validate the resulting cosmic masks. `sky.failure_policy=strict` prohibits a soft-failure pass-through that would preserve an unsubtracted sky frame while reporting a completed stage. Failure to satisfy a real prerequisite is a **useful stop**, not permission to bypass preflight or use cross-mode calibration.

`biascorr` is **not** an independent engine task: it is an alias for `superbias`. Including both caused a deterministic duplicate/skip failure in the previous pilot plan. The harness now verifies that every planned task is canonical and that RAW selections and required pilot flags remain unchanged between preparation and run.

For each task the canonical `run_sequence_locked` engine is invoked with `force=false`, `qc_override=false`, `preflight_override=false`. On the first failed/not-OK task stop and inspect `sc1_engineering_pilot_result.json`; retain the entire isolated run.

### Acceptance levels

1. **Code regression:** all targeted tests pass on the Windows Python venv.
2. **Engineering E2E:** all planned 14 tasks complete with `status=ok`, original protected hashes unchanged; inspect produced products.
3. **Science:** **HOLD regardless of 1 or 2** until reliable bias uncertainty (including quantized ADC and time-dependent zero), wave/sky residuals, flux fidelity and spatial geometry are evaluated against the final extracted science product.

The 50 ADU gross threshold is from the audited N2146sat bias cohort, **not** an instrument-wide universal calibration. The finite-N Gaussian factor after gross rejection is provisional. Original MasterBias SCI/VAR/DQ must remain untouched.
