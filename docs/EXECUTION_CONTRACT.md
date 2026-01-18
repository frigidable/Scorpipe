# Execution Contract (ENGINE-first)

This document is the **law** for how Scorpipe executes pipeline stages.
The single source of truth is `scorpio_pipe.pipeline.engine`.

## Terms

- **run_root / work_dir**: the pipeline workspace directory for a single run.
- **task / stage**: a named processing step (e.g. `flatfield`, `sky`, `extract1d`).
- **canonical tasks**: the task names listed in `CANONICAL_TASKS`.

## Statuses

Each task writes a `done.json` with a normalized status:

- `ok` — task completed successfully.
- `warn` — task completed but with warnings (stage code may set this; the engine will keep the "worst" status).
- `skipped` — task was not executed because it was up-to-date.
- `blocked` — task was prevented from running by the QC gate.
- `cancelled` — cooperative cancellation was requested before the task.
- `fail` — task raised an error (including boundary contract violations).

The engine always writes/updates `done.json` even on crashes.

## Resume / force semantics

Given a task `T`:

1) **Completion**: `task_is_complete(cfg, T)` must be true (a minimal set of outputs exists on disk).
2) **Up-to-date**: the expected task hash is computed from:
   - the task name
   - the effective task config (`_stage_cfg_for_hash`)
   - declared task inputs (`_input_paths_for_hash`)
   - the **reference context id** (`manifest/reference_context.json`)

A task is considered **up-to-date** when it is complete *and* its stored hash matches the expected hash.

- With `resume=True` (default) and `force=False`: up-to-date tasks are **skipped**.
- With `force=True`: the task is **run** regardless of up-to-date state.

Planning and execution are consistent:

- `plan_sequence(...)` produces a run/skip plan using the same rules.
- `run_sequence(...)` executes according to this plan.

## QC gate

Before executing a task, the engine checks the QC gate (`scorpio_pipe.qc.gate`).

- If upstream stages wrote `ERROR`/`FATAL` QC flags, downstream tasks may be **blocked**.
- `qc_override=True` disables the gate (recorded in stage state).

The QC gate is the only *intended* downstream blocker.

## Stage state and where to look

The engine records execution metadata in two places:

### 1) `manifest/stage_state.json`

A machine-readable summary used for resume/force decisions and UI summaries.
It stores, per task:

- last known status (`ok`, `failed`, `skipped`, ...)
- the computed task hash
- timestamps / messages

### 2) `<stage_dir>/done.json`

A publication-grade per-task record.
Besides the status, it contains:

- `stage_hash`
- runner timing (`started_utc`, `finished_utc`, `duration_s`)
- `input_hashes` (best-effort)
- `outputs_list` (best-effort)
- `effective_config` for the task
- reference context id

Note: for historical reasons, some tasks store their main artifacts under the wavelength-solution folder
(`wavesol_dir(cfg)`), but `done.json` still lives under the canonical stage directory.

## Boundary product validation

After a task returns successfully, the engine may validate produced products against a strict boundary
contract:

- `contract_kind='mef'` → validate MEF FITS products
- `contract_kind='spec1d'` → validate 1D spectra products
- `contract_kind='lambda_map'` → validate wavelength map products

The set of files to validate is declared in `STAGE_SPECS[task].validate_globs`.
A violation raises `ProductContractError` and the task is marked as failed.

## StageSpec table (P0-A7)

`STAGE_SPECS` is the **execution contract metadata** for canonical tasks.
For each canonical task it declares:

- the expected output artifacts (`validate_globs`)
- the output format boundary kind (`contract_kind`)
- optional product keys (`output_keys`, aligned with `scorpio_pipe.products`)

CI enforces 100% coverage: `set(CANONICAL_TASKS) == set(STAGE_SPECS)`.
